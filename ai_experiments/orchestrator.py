"""Campaign orchestration: the auto-experiment loop.

`advance()` is one idempotent step of the loop: refresh active trials, record
results for finished ones, evaluate stopping conditions (target reached,
budget exhausted, wall clock), then plan and submit the next batch. The
monitor daemon calls it every tick; `iax campaign start` calls it once to
launch the first batch.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from ai_experiments.agents.runner import AgentRunner, CliAgentRunner
from ai_experiments.agents.strategy import AgentDecision, AgentStrategy
from ai_experiments.backends.base import ExperimentBackend
from ai_experiments.backends.factory import get_backend
from ai_experiments.improve.admission import admit_variant, variant_dir
from ai_experiments.improve.rounds import RoundLog, RoundRecord
from ai_experiments.planner.analysis import (
    best_trial,
    summarize_campaign,
)
from ai_experiments.planner.planner import build_trial_manifest, plan_next_params
from ai_experiments.planner.strategies import get_strategy
from ai_experiments.planner.validation import validate_params
from ai_experiments.preflight import goal_warnings, workload_warnings
from ai_experiments.recovery import recover_lost_trials
from ai_experiments.schemas import (
    CampaignState,
    GoalSpec,
    RunEvent,
    TrialRecord,
    utc_now,
)
from ai_experiments.stopping import (
    ACTIVE_TRIAL_STATES,
    FAILURE_STOP_REASONS,
    exhausted_reason,
    gpu_hours_spent,
    stop_reason,
)
from ai_experiments.store.campaign import CampaignStore
from ai_experiments.trial_sync import refresh_trials, update_best

if TYPE_CHECKING:
    from ai_experiments.daemon import TickReport
    from ai_experiments.improve.variants import VariantEdit, VariantRecord
    from ai_experiments.store import FilesystemRunStore

BackendFactory = Callable[[GoalSpec], ExperimentBackend]
AgentRunnerFactory = Callable[[GoalSpec, str], AgentRunner]


class CampaignOrchestrator:
    def __init__(
        self,
        run_store: FilesystemRunStore,
        campaign_store: CampaignStore | None = None,
        backend_factory: BackendFactory | None = None,
        address_resolver: Callable[[GoalSpec], str | None] | None = None,
        agent_runner_factory: AgentRunnerFactory | None = None,
    ) -> None:
        self.run_store = run_store
        self.campaign_store = campaign_store or CampaignStore(run_store.root)
        self._address_resolver = address_resolver or _default_address_resolver
        self._backend_factory = backend_factory or self._default_backend_factory
        self._agent_runner_factory = agent_runner_factory or self._default_agent_runner
        #: The agent decision made during the current `advance()`, if any.
        self.last_decision: AgentDecision | None = None
        #: Why the backend refused the submits of the current `advance()`. A
        #: backend that accepts nothing looks exactly like an exhausted search
        #: space from the campaign's side, and the two need opposite answers.
        self.last_submit_errors: list[str] = []

    def agent_runner(self, goal: GoalSpec, campaign_id: str) -> AgentRunner:
        """Build the agent this campaign talks to.

        One place, so every role — planner, reviewer — gets the same command,
        timeout and transcript directory.
        """
        return self._agent_runner_factory(goal, campaign_id)

    def _default_agent_runner(self, goal: GoalSpec, campaign_id: str) -> AgentRunner:
        return CliAgentRunner(
            goal.agent.command,
            transcript_dir=self.campaign_store.campaign_dir(campaign_id) / "agents",
            timeout_seconds=goal.agent.timeout_seconds,
        )

    def _default_backend_factory(self, goal: GoalSpec) -> ExperimentBackend:
        return get_backend(
            goal.backend,
            store=self.run_store,
            address=self._address_resolver(goal),
        )

    # -- lifecycle -----------------------------------------------------------

    def start(self, goal: GoalSpec) -> CampaignState:
        state = self.campaign_store.create_campaign(goal)
        self.campaign_store.append_event(
            state.campaign_id,
            RunEvent(message="campaign created", details={"goal": goal.goal}),
        )
        # Every trial runs this one workload. A campaign started from the API,
        # the server or the daemon never passes through the CLI's check, and a
        # workload that cannot start would otherwise fail max_trials times
        # with nothing saying why up front (#32).
        # Two different defects, two different events: one says the trials
        # will not run, the other says they will run and settle nothing.
        for message, warnings in (
            ("workload may not start", workload_warnings(goal)),
            ("goal may not settle anything", goal_warnings(goal)),
        ):
            if warnings:
                self.campaign_store.append_event(
                    state.campaign_id,
                    RunEvent(
                        level="warning",
                        message=message,
                        details={"warnings": warnings},
                    ),
                )
        return self.advance(state.campaign_id)

    def stop(self, campaign_id: str, reason: str = "user_requested") -> CampaignState:
        goal = self.campaign_store.read_goal(campaign_id)
        state = self.campaign_store.read_state(campaign_id)
        backend = self._backend_factory(goal)
        self._cancel_active(state, backend)
        state.status = "stopped"
        state.stop_reason = reason
        self.campaign_store.write_state(state)
        self.campaign_store.append_event(
            campaign_id,
            RunEvent(level="warning", message="campaign stopped", details={"reason": reason}),
        )
        return state

    def pause(self, campaign_id: str) -> CampaignState:
        """Stop scheduling new trials.

        Active trials keep running and stay monitored by the daemon. Resume with `resume()`.
        """
        state = self.campaign_store.read_state(campaign_id)
        if state.status != "running":
            raise ValueError(f"cannot pause a campaign in status '{state.status}'")
        state.status = "paused"
        self.campaign_store.write_state(state)
        self.campaign_store.append_event(campaign_id, RunEvent(message="campaign paused"))
        return state

    def resume(self, campaign_id: str) -> CampaignState:
        state = self.campaign_store.read_state(campaign_id)
        if state.status != "paused":
            raise ValueError(f"cannot resume a campaign in status '{state.status}'")
        state.status = "running"
        self.campaign_store.write_state(state)
        self.campaign_store.append_event(campaign_id, RunEvent(message="campaign resumed"))
        return self.advance(campaign_id)

    def edit_goal(self, campaign_id: str, new_goal: GoalSpec) -> GoalSpec:
        """Replace the campaign's goal (search space, budget, strategy, ...).

        Existing trial history is kept and feeds the strategy under the new
        goal. Changing the objective metric is rejected — recorded objective
        values would no longer be comparable; start a new campaign instead.
        """
        current = self.campaign_store.read_goal(campaign_id)
        if new_goal.objective.metric != current.objective.metric:
            raise ValueError(
                "editing the objective metric is not supported "
                f"('{current.objective.metric}' -> '{new_goal.objective.metric}'); "
                "start a new campaign instead"
            )
        campaign_dir = self.campaign_store.campaign_dir(campaign_id)
        (campaign_dir / "goal.yaml").write_text(new_goal.to_yaml())
        self.campaign_store.append_event(
            campaign_id,
            RunEvent(
                message="goal edited",
                details={
                    "search_space": {
                        name: spec.model_dump() for name, spec in new_goal.search_space.items()
                    },
                    "budget": new_goal.budget.model_dump(),
                    "strategy": new_goal.strategy.model_dump(),
                },
            ),
        )
        return new_goal

    def add_variant(
        self,
        campaign_id: str,
        edits: list[VariantEdit],
        *,
        hypothesis: str = "",
        rationale: str = "",
        parent: str | None = None,
    ) -> VariantRecord:
        """Materialize a code variant of the workload and smoke-check it.

        The smoke command's exit code, not the proposer, decides whether the
        variant may cost trials; see `admit_variant`.
        """
        return admit_variant(
            self.campaign_store,
            campaign_id,
            edits,
            hypothesis=hypothesis,
            rationale=rationale,
            parent=parent,
        )

    def suggest(
        self,
        campaign_id: str,
        params: dict[str, Any],
        note: str = "",
        variant_id: str | None = None,
    ) -> TrialRecord:
        """Queue an agent/human-suggested trial; submitted on the next advance.

        A suggestion is a proposal, not an override: it is rejected when the
        campaign can no longer run it, when the params are not in the search
        space, or when the trial budget is already committed (#13). Naming a
        ``variant_id`` runs it against that variant's copy of the workload —
        and only if the variant cleared its smoke check, since a round spent
        on code that cannot start teaches nothing.
        """
        state = self.campaign_store.read_state(campaign_id)
        if state.status not in {"running", "paused"}:
            raise ValueError(
                f"campaign {campaign_id} is {state.status}; it will never run a "
                "suggested trial — start a new campaign instead"
            )
        goal = self.campaign_store.read_goal(campaign_id)
        params = validate_params(goal.search_space, params)
        if len(state.trials) >= goal.budget.max_trials:
            raise ValueError(
                f"trial budget is spent ({len(state.trials)}/"
                f"{goal.budget.max_trials}); raise max_trials with "
                "`iax campaign edit` to make room"
            )
        if variant_id is not None:
            variant = self.campaign_store.read_variant(campaign_id, variant_id)
            if variant is None:
                raise ValueError(
                    f"unknown variant {variant_id}; `iax campaign variants "
                    f"{campaign_id}` lists the ones this campaign has"
                )
            if variant.smoke_ok is False:
                raise ValueError(
                    f"variant {variant_id} failed its smoke check and was "
                    "discarded; propose a fixed one instead of running it"
                )
        trial = TrialRecord(
            trial_id=f"t{len(state.trials):03d}",
            params=params,
            source="agent",
            variant_id=variant_id,
        )
        state.trials.append(trial)
        self.campaign_store.write_state(state)
        self.campaign_store.append_event(
            campaign_id,
            RunEvent(
                message="trial suggested",
                details={
                    "trial_id": trial.trial_id,
                    "params": params,
                    "note": note,
                    "variant_id": variant_id,
                },
            ),
        )
        return trial

    # -- the loop step ---------------------------------------------------------

    def advance(self, campaign_id: str, admit: bool = True) -> CampaignState:
        """Move the campaign one step.

        With ``admit=False`` it refreshes, scores and re-evaluates the stop
        condition but submits nothing and leaves `state.rounds` alone, so a
        caller can close a cohort and review it before paying for the next.
        """
        self.last_decision = None
        self.last_submit_errors = []
        state = self.campaign_store.read_state(campaign_id)
        if state.status in {"completed", "stopped", "failed", "paused"}:
            return state
        goal = self.campaign_store.read_goal(campaign_id)
        backend = self._backend_factory(goal)

        recover_lost_trials(state, backend, self.campaign_store, self.run_store)
        finished_now = refresh_trials(state, goal, backend, self.campaign_store, self.run_store)
        update_best(state, goal)

        reason = stop_reason(state, goal, gpu_hours_spent(state, goal, self.run_store))
        if reason:
            return self._finish(state, goal, backend, reason)

        if finished_now:
            self._record_evaluation(state, goal, finished_now)

        if not admit:
            if finished_now and goal.analysis.agent_review:
                self._request_agent_review(state, goal)
            self.campaign_store.write_state(state)
            return state

        submitted = self._fill_capacity(state, goal, backend)
        if submitted:
            state.rounds += 1
            self._record_proposal(state, goal, submitted)

        # The planner can run dry before `max_trials` — a grid or a small
        # discrete space has finitely many points. With nothing in flight and
        # nothing queued, no later tick can change that, so the campaign would
        # otherwise report `running` forever (#15).
        if not self._has_work(state):
            self.campaign_store.append_event(
                campaign_id,
                RunEvent(
                    level="warning",
                    message=(
                        "no trial could start"
                        if self.last_submit_errors
                        else "planner exhausted the search space"
                    ),
                    details={
                        "trials": len(state.trials),
                        "max_trials": goal.budget.max_trials,
                        "strategy": goal.strategy.name,
                        "agent_stop_reason": (
                            self.last_decision.stop_reason if self.last_decision is not None else ""
                        ),
                        "submit_errors": self.last_submit_errors,
                    },
                ),
            )
            return self._finish(
                state,
                goal,
                backend,
                exhausted_reason(submitted, self.last_submit_errors, self.last_decision),
            )

        if finished_now and goal.analysis.agent_review:
            self._request_agent_review(state, goal)

        self.campaign_store.write_state(state)
        return state

    def run_to_completion(
        self,
        state: CampaignState,
        tick: Callable[[], TickReport],
        on_tick: Callable[[TickReport], None],
        interval: int,
    ) -> CampaignState:
        """Drive `tick()` (typically `MonitorDaemon.tick`) until the campaign stops.

        Reports every tick via `on_tick`, sleeping `interval` seconds in between, and
        returns the campaign's final state once it is completed, stopped, or failed. A
        `KeyboardInterrupt` propagates to the caller mid-loop, leaving the campaign
        active for a later `MonitorDaemon` to resume.
        """
        while True:
            report = tick()
            on_tick(report)
            state = self.campaign_store.read_state(state.campaign_id)
            if state.status in {"completed", "stopped", "failed"}:
                return state
            time.sleep(interval)

    def reconcile(self, campaign_id: str) -> CampaignState:
        """Collect what already finished, and plan nothing new.

        `iax loop --max-rounds` stops between rounds, while the trials of the
        round it just paid for may already have finished. `advance` would
        submit another round; doing nothing leaves those trials `submitted`
        forever and throws away work that ran. This reads their results, and
        finishes the campaign when they turn out to have met the target.
        """
        self.last_decision = None
        self.last_submit_errors = []
        state = self.campaign_store.read_state(campaign_id)
        if state.status in {"completed", "stopped", "failed", "paused"}:
            return state
        goal = self.campaign_store.read_goal(campaign_id)
        backend = self._backend_factory(goal)

        recover_lost_trials(state, backend, self.campaign_store, self.run_store)
        finished_now = refresh_trials(state, goal, backend, self.campaign_store, self.run_store)
        update_best(state, goal)

        reason = stop_reason(state, goal, gpu_hours_spent(state, goal, self.run_store))
        if reason:
            return self._finish(state, goal, backend, reason)

        if finished_now:
            self._record_evaluation(state, goal, finished_now)
        self.campaign_store.write_state(state)
        return state

    def _has_work(self, state: CampaignState) -> bool:
        return any(t.status in ACTIVE_TRIAL_STATES or t.status == "planned" for t in state.trials)

    def _finish(
        self,
        state: CampaignState,
        goal: GoalSpec,
        backend: ExperimentBackend,
        reason: str,
    ) -> CampaignState:
        self._cancel_active(state, backend)
        state.status = "failed" if reason in FAILURE_STOP_REASONS else "completed"
        state.stop_reason = reason
        self.campaign_store.write_state(state)
        self.campaign_store.append_event(
            state.campaign_id,
            RunEvent(message="campaign finished", details={"reason": reason}),
        )
        if reason == "all_trials_failing":
            # The reason alone sends the reader to the runs directory. The
            # error is the only thing that tells them what to fix.
            self.campaign_store.append_event(
                state.campaign_id,
                RunEvent(
                    level="error",
                    message="campaign gave up: every trial failed",
                    details={
                        "failed_trials": len([t for t in state.trials if t.status == "failed"]),
                        "workload_error": self._last_trial_error(state),
                    },
                ),
            )
        self._write_summary(state, goal)
        return state

    # -- internals -------------------------------------------------------------

    def _last_trial_error(self, state: CampaignState) -> str | None:
        """Return what the workload said on its way out, for the giving-up event."""
        for trial in reversed(state.trials):
            if trial.status == "failed" and trial.error:
                return trial.error
        return None

    def _variant_dir(self, campaign_id: str, variant_id: str | None) -> str | None:
        return variant_dir(self.campaign_store, campaign_id, variant_id)

    def _rounds(self, campaign_id: str) -> RoundLog:
        return RoundLog(self.campaign_store.campaign_dir(campaign_id))

    def _record_proposal(
        self, state: CampaignState, goal: GoalSpec, submitted: list[TrialRecord]
    ) -> None:
        """Why this round exists, written the moment it is submitted."""
        decision = self.last_decision
        self._rounds(state.campaign_id).append(
            RoundRecord(
                campaign_id=state.campaign_id,
                round=state.rounds,
                stage="propose",
                strategy=goal.strategy.name,
                hypothesis=decision.hypothesis if decision else "",
                rationale=decision.rationale if decision else "",
                trial_ids=[t.trial_id for t in submitted],
                outcome={t.trial_id: t.params for t in submitted},
                agent_calls=state.agent_calls,
                used_fallback=bool(decision and decision.used_fallback),
                rejected=decision.rejected if decision else [],
            )
        )

    def _record_evaluation(
        self, state: CampaignState, goal: GoalSpec, finished: list[TrialRecord]
    ) -> None:
        """Record what the round actually measured, including what broke."""
        best = best_trial(state, goal.objective.mode)
        self._rounds(state.campaign_id).append(
            RoundRecord(
                campaign_id=state.campaign_id,
                round=state.rounds,
                stage="evaluate",
                strategy=goal.strategy.name,
                trial_ids=[t.trial_id for t in finished],
                outcome={
                    "metric": goal.objective.metric,
                    "values": {
                        t.trial_id: {
                            "status": t.status,
                            "objective_value": t.objective_value,
                            "error": t.error,
                        }
                        for t in finished
                    },
                    "best_so_far": (
                        {
                            "trial_id": best.trial_id,
                            "objective_value": best.objective_value,
                        }
                        if best
                        else None
                    ),
                },
                agent_calls=state.agent_calls,
            )
        )

    def _plan_params(
        self, state: CampaignState, goal: GoalSpec, count: int
    ) -> list[dict[str, Any]]:
        """Choose the next parameter assignments, from the agent or a strategy.

        `strategy: agent` is the only path that costs tokens, so it is the only
        one with a budget. Past `goal.agent.max_calls` the campaign keeps
        running on the fallback strategy rather than stopping: a spent agent
        budget is a reason to plan more cheaply, not a reason to give up.
        """
        if goal.strategy.name != "agent":
            return plan_next_params(goal, state.trials, count)

        if state.agent_calls >= goal.agent.max_calls:
            self.campaign_store.append_event(
                state.campaign_id,
                RunEvent(
                    level="warning",
                    message="agent call budget exhausted; planning with the fallback",
                    details={
                        "agent_calls": state.agent_calls,
                        "max_calls": goal.agent.max_calls,
                        "fallback": goal.strategy.fallback,
                    },
                ),
            )
            return get_strategy(goal.strategy.fallback).plan(goal, state.trials, count)

        strategy = AgentStrategy(
            self.agent_runner(goal, state.campaign_id),
            fallback=goal.strategy.fallback,
        )
        params = strategy.plan(goal, state.trials, count)
        state.agent_calls += 1
        decision = strategy.last_decision
        self.last_decision = decision
        self.campaign_store.append_event(
            state.campaign_id,
            RunEvent(
                level="warning" if decision.used_fallback else "info",
                message="agent planned a round",
                details={
                    "hypothesis": decision.hypothesis,
                    "rationale": decision.rationale,
                    "accepted": len(decision.accepted),
                    "rejected": len(decision.rejected),
                    "used_fallback": decision.used_fallback,
                    "fallback_reason": decision.fallback_reason,
                    "agent_error": decision.agent_error,
                },
            ),
        )
        return params

    def _fill_capacity(
        self, state: CampaignState, goal: GoalSpec, backend: ExperimentBackend
    ) -> list[TrialRecord]:
        return self._submit(state, goal, backend, self._queue(state, goal))

    def _queue(self, state: CampaignState, goal: GoalSpec) -> list[TrialRecord]:
        """Return the planned trials this tick may submit, newly planned ones included.

        Two separate caps apply. `max_parallel` caps concurrency; it says nothing
        about the budget, and trials queued by `suggest` are already in
        `state.trials`, so without the second cap a campaign could submit past
        `max_trials` (#13).
        """
        active = sum(1 for t in state.trials if t.status in ACTIVE_TRIAL_STATES)
        capacity = goal.budget.max_parallel - active
        if capacity <= 0:
            return []

        queue = [t for t in state.trials if t.status == "planned"]
        remaining_budget = goal.budget.max_trials - len(state.trials)
        batch_limit = goal.strategy.batch_size or goal.budget.max_parallel
        want_new = min(capacity - len(queue), remaining_budget, batch_limit)
        if want_new > 0:
            for params in self._plan_params(state, goal, want_new):
                trial = TrialRecord(trial_id=f"t{len(state.trials):03d}", params=params)
                state.trials.append(trial)
                queue.append(trial)

        committed = sum(1 for t in state.trials if t.status != "planned")
        allowance = max(goal.budget.max_trials - committed, 0)
        return queue[: min(capacity, allowance)]

    def _submit(
        self,
        state: CampaignState,
        goal: GoalSpec,
        backend: ExperimentBackend,
        queue: list[TrialRecord],
    ) -> list[TrialRecord]:
        """Hand each queued trial to the backend, recording the ones that failed."""
        submitted: list[TrialRecord] = []
        for trial in queue:
            try:
                manifest = build_trial_manifest(
                    goal,
                    trial.trial_id,
                    trial.params,
                    backend_address=self._address_resolver(goal),
                    working_dir=self._variant_dir(state.campaign_id, trial.variant_id),
                    variant_id=trial.variant_id,
                )
            except Exception as exc:
                # The goal cannot describe this trial. Nothing about the
                # backend is implied, so this must not read as one being down.
                self._fail_to_submit(state, trial, f"invalid trial manifest: {exc}")
                continue
            try:
                handle = backend.submit(manifest)
            except Exception as exc:
                self._fail_to_submit(state, trial, f"submit failed: {exc}")
                self.last_submit_errors.append(str(exc))
                continue
            trial.run_id = handle.run_id
            trial.status = "submitted"
            submitted.append(trial)
            # Persist before submitting the next one. A crash here used to
            # lose every submit of the tick; now it loses at most this one,
            # and `recover_lost_trials` rebuilds even that from the event
            # appended below (#5).
            self.campaign_store.append_event(
                state.campaign_id,
                RunEvent(
                    message="trial submitted",
                    details={
                        "trial_id": trial.trial_id,
                        "run_id": handle.run_id,
                        "params": trial.params,
                    },
                ),
            )
            self.campaign_store.write_state(state)
        return submitted

    def _fail_to_submit(self, state: CampaignState, trial: TrialRecord, error: str) -> None:
        trial.status = "failed"
        trial.error = error
        trial.completed_at = utc_now()
        self.campaign_store.append_event(
            state.campaign_id,
            RunEvent(
                level="error",
                message="trial submit failed",
                details={"trial_id": trial.trial_id, "error": error},
            ),
        )

    def _cancel_active(self, state: CampaignState, backend: ExperimentBackend) -> None:
        for trial in state.trials:
            if trial.status in ACTIVE_TRIAL_STATES and trial.run_id:
                with contextlib.suppress(Exception):
                    backend.cancel(trial.run_id)
                trial.status = "cancelled"
                trial.completed_at = utc_now()
            elif trial.status == "planned":
                trial.status = "cancelled"

    def _write_summary(self, state: CampaignState, goal: GoalSpec) -> None:
        summary = summarize_campaign(state, goal)
        path = self.campaign_store.campaign_dir(state.campaign_id) / "summary.json"
        path.write_text(summary.model_dump_json(indent=2))

    def _request_agent_review(self, state: CampaignState, goal: GoalSpec) -> None:
        from ai_experiments.monitoring.escalation import request_campaign_review

        request_campaign_review(self.run_store, state, goal)


def _default_address_resolver(goal: GoalSpec) -> str | None:
    if goal.backend != "ray":
        return None
    if goal.backend_address:
        return goal.backend_address
    if goal.cluster:
        from ai_experiments.clusters import resolve_cluster_address

        return resolve_cluster_address(goal.cluster)
    return None
