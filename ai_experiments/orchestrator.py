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

from ai_experiments.backends.base import ExperimentBackend
from ai_experiments.backends.factory import get_backend
from ai_experiments.planner.analysis import (
    best_trial,
    extract_objective,
    summarize_campaign,
)
from ai_experiments.planner.planner import build_trial_manifest, plan_next_params
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


class CampaignOrchestrator:
    def __init__(
        self,
        run_store: FilesystemRunStore,
        campaign_store: CampaignStore | None = None,
        backend_factory: BackendFactory | None = None,
        address_resolver: Callable[[GoalSpec], str | None] | None = None,
    ) -> None:
        self.run_store = run_store
        self.campaign_store = campaign_store or CampaignStore(run_store.root)
        self._address_resolver = address_resolver or _default_address_resolver
        self._backend_factory = backend_factory or self._default_backend_factory

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
        """Queue an agent/human-suggested trial; submitted on the next advance."""
        state = self.campaign_store.read_state(campaign_id)
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

        submitted = self._fill_capacity(state, goal, backend)
        if submitted:
            state.rounds += 1

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
                    },
                ),
            )
            return self._finish(state, goal, backend, "search_space_exhausted")

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
            for params in plan_next_params(goal, state.trials, want_new):
                trial = TrialRecord(trial_id=f"t{len(state.trials):03d}", params=params)
                state.trials.append(trial)
                queue.append(trial)

        submitted: list[TrialRecord] = []
        for trial in queue[:capacity]:
            try:
                manifest = build_trial_manifest(
                    goal,
                    trial.trial_id,
                    trial.params,
                    backend_address=self._address_resolver(goal),
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
        escalations = self.run_store.root / "_escalations"
        escalations.mkdir(parents=True, exist_ok=True)
        payload = {
            "type": "campaign_review",
            "created_at": utc_now().isoformat(),
            "campaign_id": state.campaign_id,
            "summary": summarize_campaign(state, goal),
            "note": (
                "Review trial history; queue better trials via "
                "`iax campaign suggest <campaign_id> --params '{...}'` "
                "or stop via `iax campaign stop <campaign_id>`."
            ),
        }
        (escalations / f"campaign_{state.campaign_id}.json").write_text(
            json.dumps(payload, indent=2)
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
