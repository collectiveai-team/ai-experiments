"""Campaign orchestration: the auto-experiment loop.

`advance()` is one idempotent step of the loop: refresh active trials, record
results for finished ones, evaluate stopping conditions (target reached,
budget exhausted, wall clock), then plan and submit the next batch. The
monitor daemon calls it every tick; `iax campaign start` calls it once to
launch the first batch.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Callable

from ai_experiments.agents.runner import AgentRunner, CliAgentRunner
from ai_experiments.agents.strategy import AgentDecision, AgentStrategy
from ai_experiments.backends.base import ExperimentBackend
from ai_experiments.backends.factory import get_backend
from ai_experiments.improve.rounds import RoundLog, RoundRecord
from ai_experiments.improve.variants import variants_root
from ai_experiments.planner.analysis import (
    best_trial,
    extract_objective,
    summarize_campaign,
)
from ai_experiments.planner.planner import build_trial_manifest, plan_next_params
from ai_experiments.planner.strategies import get_strategy
from ai_experiments.planner.validation import validate_params
from ai_experiments.preflight import workload_warnings
from ai_experiments.schemas import (
    CampaignState,
    GoalSpec,
    RunEvent,
    RunState,
    TrialRecord,
    TrialState,
    utc_now,
)
from ai_experiments.store import FilesystemRunStore
from ai_experiments.store.campaign import CampaignStore

ACTIVE_TRIAL_STATES: set[TrialState] = {"submitted", "running"}

#: Stop reasons that mean the campaign broke rather than finished.
FAILURE_STOP_REASONS = {"objective_not_reported"}

#: How many trials may complete without a usable objective before the
#: campaign is declared broken instead of merely unlucky.
MIN_TRIALS_BEFORE_CONTRACT_CHECK = 2

_RUN_TO_TRIAL: dict[RunState, TrialState] = {
    "submitted": "submitted",
    "running": "running",
    "completed": "completed",
    "failed": "failed",
    "cancelled": "cancelled",
}

BackendFactory = Callable[[GoalSpec], ExperimentBackend]
AgentRunnerFactory = Callable[[GoalSpec, str], AgentRunner]

#: Stop reason recorded when the agent itself says more trials cannot help.
AGENT_STOP_REASON = "agent_requested_stop"


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

    def agent_runner(self, goal: GoalSpec, campaign_id: str) -> AgentRunner:
        """The agent this campaign talks to. One place, so every role — planner,
        reviewer — gets the same command, timeout and transcript directory."""
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
        warnings = workload_warnings(goal)
        if warnings:
            self.campaign_store.append_event(
                state.campaign_id,
                RunEvent(
                    level="warning",
                    message="workload may not start",
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
            RunEvent(
                level="warning", message="campaign stopped", details={"reason": reason}
            ),
        )
        return state

    def pause(self, campaign_id: str) -> CampaignState:
        """Stop scheduling new trials; active trials keep running and stay
        monitored by the daemon. Resume with `resume()`."""
        state = self.campaign_store.read_state(campaign_id)
        if state.status != "running":
            raise ValueError(f"cannot pause a campaign in status '{state.status}'")
        state.status = "paused"
        self.campaign_store.write_state(state)
        self.campaign_store.append_event(
            campaign_id, RunEvent(message="campaign paused")
        )
        return state

    def resume(self, campaign_id: str) -> CampaignState:
        state = self.campaign_store.read_state(campaign_id)
        if state.status != "paused":
            raise ValueError(f"cannot resume a campaign in status '{state.status}'")
        state.status = "running"
        self.campaign_store.write_state(state)
        self.campaign_store.append_event(
            campaign_id, RunEvent(message="campaign resumed")
        )
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
                        name: spec.model_dump()
                        for name, spec in new_goal.search_space.items()
                    },
                    "budget": new_goal.budget.model_dump(),
                    "strategy": new_goal.strategy.model_dump(),
                },
            ),
        )
        return new_goal

    def suggest(
        self, campaign_id: str, params: dict[str, Any], note: str = ""
    ) -> TrialRecord:
        """Queue an agent/human-suggested trial; submitted on the next advance.

        A suggestion is a proposal, not an override: it is rejected when the
        campaign can no longer run it, when the params are not in the search
        space, or when the trial budget is already committed (#13).
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
        trial = TrialRecord(
            trial_id=f"t{len(state.trials):03d}",
            params=params,
            source="agent",
        )
        state.trials.append(trial)
        self.campaign_store.write_state(state)
        self.campaign_store.append_event(
            campaign_id,
            RunEvent(
                message="trial suggested",
                details={"trial_id": trial.trial_id, "params": params, "note": note},
            ),
        )
        return trial

    # -- the loop step ---------------------------------------------------------

    def advance(self, campaign_id: str) -> CampaignState:
        self.last_decision = None
        state = self.campaign_store.read_state(campaign_id)
        if state.status in {"completed", "stopped", "failed", "paused"}:
            return state
        goal = self.campaign_store.read_goal(campaign_id)
        backend = self._backend_factory(goal)

        finished_now = self._refresh_trials(state, goal, backend)
        self._update_best(state, goal)

        stop_reason = self._stop_reason(state, goal)
        if stop_reason:
            return self._finish(state, goal, backend, stop_reason)

        if finished_now:
            self._record_evaluation(state, goal, finished_now)

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
                    message="planner exhausted the search space",
                    details={
                        "trials": len(state.trials),
                        "max_trials": goal.budget.max_trials,
                        "strategy": goal.strategy.name,
                        "agent_stop_reason": (
                            self.last_decision.stop_reason
                            if self.last_decision is not None
                            else ""
                        ),
                    },
                ),
            )
            exhausted_reason = (
                AGENT_STOP_REASON
                if self.last_decision is not None and self.last_decision.stop
                else "search_space_exhausted"
            )
            return self._finish(state, goal, backend, exhausted_reason)

        if finished_now and goal.analysis.agent_review:
            self._request_agent_review(state, goal)

        self.campaign_store.write_state(state)
        return state

    def _has_work(self, state: CampaignState) -> bool:
        return any(
            t.status in ACTIVE_TRIAL_STATES or t.status == "planned"
            for t in state.trials
        )

    def _finish(
        self,
        state: CampaignState,
        goal: GoalSpec,
        backend: ExperimentBackend,
        stop_reason: str,
    ) -> CampaignState:
        self._cancel_active(state, backend)
        state.status = "failed" if stop_reason in FAILURE_STOP_REASONS else "completed"
        state.stop_reason = stop_reason
        self.campaign_store.write_state(state)
        self.campaign_store.append_event(
            state.campaign_id,
            RunEvent(message="campaign finished", details={"reason": stop_reason}),
        )
        self._write_summary(state, goal)
        return state

    # -- internals -------------------------------------------------------------

    def _refresh_trials(
        self, state: CampaignState, goal: GoalSpec, backend: ExperimentBackend
    ) -> list[TrialRecord]:
        finished: list[TrialRecord] = []
        for trial in state.trials:
            if trial.status not in ACTIVE_TRIAL_STATES or not trial.run_id:
                continue
            try:
                run_status = backend.inspect(trial.run_id)
            except Exception as exc:
                self.campaign_store.append_event(
                    state.campaign_id,
                    RunEvent(
                        level="warning",
                        message="trial inspect failed",
                        details={"trial_id": trial.trial_id, "error": str(exc)},
                    ),
                )
                continue
            mapped = _RUN_TO_TRIAL.get(run_status.status)
            if mapped is None:
                continue
            trial.status = mapped
            if mapped in {"completed", "failed", "cancelled"}:
                trial.completed_at = run_status.completed_at or utc_now()
                trial.error = run_status.error
                trial.gpu_hours = _trial_gpu_hours(
                    goal,
                    run_status.started_at or run_status.submitted_at,
                    trial.completed_at,
                )
                reading = extract_objective(
                    self.run_store, trial.run_id, goal.objective
                )
                trial.objective_value = reading.value
                trial.final_metrics = reading.final_metrics
                miss = reading.miss_message(goal.objective.metric)
                if miss and mapped == "completed":
                    # A trial that ran to completion without producing the
                    # objective is a broken contract, not a bad result. Say so
                    # on the trial; scoring it `null` in silence burns the
                    # whole budget with no explanation (#11).
                    trial.error = miss
                    self.campaign_store.append_event(
                        state.campaign_id,
                        RunEvent(
                            level="warning",
                            message="trial reported no usable objective",
                            details={
                                "trial_id": trial.trial_id,
                                "objective_metric": goal.objective.metric,
                                "observed_metrics": reading.observed_metrics,
                                "reason": reading.miss_reason,
                            },
                        ),
                    )
                finished.append(trial)
                self.campaign_store.append_event(
                    state.campaign_id,
                    RunEvent(
                        message="trial finished",
                        details={
                            "trial_id": trial.trial_id,
                            "status": mapped,
                            "objective_value": reading.value,
                            "params": trial.params,
                        },
                    ),
                )
        return finished

    def _update_best(self, state: CampaignState, goal: GoalSpec) -> None:
        best = best_trial(state, goal.objective.mode)
        state.best_trial_id = best.trial_id if best else None

    def _stop_reason(self, state: CampaignState, goal: GoalSpec) -> str | None:
        contract_broken = self._objective_contract_broken(state)
        if contract_broken:
            return contract_broken

        best = best_trial(state, goal.objective.mode)
        target = goal.objective.target
        if best is not None and target is not None and best.objective_value is not None:
            reached = (
                best.objective_value >= target
                if goal.objective.mode == "max"
                else best.objective_value <= target
            )
            if reached:
                return "target_reached"

        if goal.budget.max_hours is not None:
            created = state.created_at
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            age_hours = (utc_now() - created).total_seconds() / 3600
            if age_hours > goal.budget.max_hours:
                return "max_hours_exceeded"

        if goal.budget.max_gpu_hours is not None:
            if self.gpu_hours_spent(state, goal) >= goal.budget.max_gpu_hours:
                return "gpu_hours_exhausted"

        active = [t for t in state.trials if t.status in ACTIVE_TRIAL_STATES]
        planned = [t for t in state.trials if t.status == "planned"]
        if len(state.trials) >= goal.budget.max_trials and not active and not planned:
            return "budget_exhausted"
        return None

    def _objective_contract_broken(self, state: CampaignState) -> str | None:
        """Stop early when the workload never reports the objective.

        Continuing would spend the whole budget on trials that can only score
        ``null`` — the campaign would then end as a "successful"
        ``budget_exhausted`` with no best trial and no explanation (#11).
        """
        completed = [t for t in state.trials if t.status == "completed"]
        if len(completed) < MIN_TRIALS_BEFORE_CONTRACT_CHECK:
            return None
        if any(t.objective_value is not None for t in completed):
            return None
        return "objective_not_reported"

    def gpu_hours_spent(self, state: CampaignState, goal: GoalSpec) -> float:
        """GPU-hours consumed so far: recorded for finished trials, a live
        estimate (started -> now) for trials still running."""
        total = sum(t.gpu_hours or 0.0 for t in state.trials)
        for trial in state.trials:
            if trial.status in ACTIVE_TRIAL_STATES and trial.run_id:
                status = self.run_store.read_status(trial.run_id)
                live = _trial_gpu_hours(
                    goal, status.started_at or status.submitted_at, utc_now()
                )
                total += live or 0.0
        return total

    def _variant_dir(self, campaign_id: str, variant_id: str | None) -> str | None:
        """Where a trial's workload variant lives, if it has one."""
        if not variant_id:
            return None
        root = variants_root(self.campaign_store.campaign_dir(campaign_id)) / variant_id
        if not root.is_dir():
            raise ValueError(f"variant {variant_id} is missing from {root}")
        return str(root)

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
        """What the round actually measured, including what broke."""
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
        """The next parameter assignments, from the agent or from a strategy.

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

        # `capacity` caps concurrency; it says nothing about the budget. Trials
        # queued by `suggest` are already in `state.trials`, so without this
        # second cap a campaign could submit past `max_trials` (#13).
        committed = sum(1 for t in state.trials if t.status != "planned")
        allowance = max(goal.budget.max_trials - committed, 0)

        submitted: list[TrialRecord] = []
        for trial in queue[: min(capacity, allowance)]:
            try:
                manifest = build_trial_manifest(
                    goal,
                    trial.trial_id,
                    trial.params,
                    backend_address=self._address_resolver(goal),
                    working_dir=self._variant_dir(state.campaign_id, trial.variant_id),
                    variant_id=trial.variant_id,
                )
                handle = backend.submit(manifest)
            except Exception as exc:
                trial.status = "failed"
                trial.error = f"submit failed: {exc}"
                trial.completed_at = utc_now()
                self.campaign_store.append_event(
                    state.campaign_id,
                    RunEvent(
                        level="error",
                        message="trial submit failed",
                        details={"trial_id": trial.trial_id, "error": str(exc)},
                    ),
                )
                continue
            trial.run_id = handle.run_id
            trial.status = "submitted"
            submitted.append(trial)
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
        return submitted

    def _cancel_active(self, state: CampaignState, backend: ExperimentBackend) -> None:
        for trial in state.trials:
            if trial.status in ACTIVE_TRIAL_STATES and trial.run_id:
                try:
                    backend.cancel(trial.run_id)
                except Exception:
                    pass
                trial.status = "cancelled"
                trial.completed_at = utc_now()
            elif trial.status == "planned":
                trial.status = "cancelled"

    def _write_summary(self, state: CampaignState, goal: GoalSpec) -> None:
        summary = summarize_campaign(state, goal)
        path = self.campaign_store.campaign_dir(state.campaign_id) / "summary.json"
        path.write_text(json.dumps(summary, indent=2))

    def _request_agent_review(self, state: CampaignState, goal: GoalSpec) -> None:
        """Drop a review request for an agent session — analysis beyond the
        built-in strategy (e.g. reshaping the search space) costs tokens, so
        it is opt-in via ``analysis.agent_review`` and file-based."""
        from ai_experiments.monitoring.escalation import (
            CAMPAIGN_PREFIX,
            CampaignReview,
        )

        escalations = self.run_store.root / "_escalations"
        escalations.mkdir(parents=True, exist_ok=True)
        review = CampaignReview(
            campaign_id=state.campaign_id,
            summary=summarize_campaign(state, goal),
            note=(
                "Review trial history; queue better trials via "
                "`iax campaign suggest <campaign_id> --params '{...}'` "
                "or stop via `iax campaign stop <campaign_id>`."
            ),
        )
        (escalations / f"{CAMPAIGN_PREFIX}{state.campaign_id}.json").write_text(
            json.dumps(review.model_dump(mode="json"), indent=2)
        )


def _trial_gpu_hours(
    goal: GoalSpec, started: datetime | None, completed: datetime | None
) -> float | None:
    if goal.resources.gpus <= 0 or started is None or completed is None:
        return 0.0 if goal.resources.gpus <= 0 else None
    if started.tzinfo is None:
        started = started.replace(tzinfo=timezone.utc)
    if completed.tzinfo is None:
        completed = completed.replace(tzinfo=timezone.utc)
    hours = max((completed - started).total_seconds(), 0.0) / 3600
    return hours * goal.resources.gpus


def _default_address_resolver(goal: GoalSpec) -> str | None:
    if goal.backend != "ray":
        return None
    if goal.backend_address:
        return goal.backend_address
    if goal.cluster:
        from ai_experiments.clusters import resolve_cluster_address

        return resolve_cluster_address(goal.cluster)
    return None
