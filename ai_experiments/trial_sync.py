"""Reading the backend's answers back into the campaign's trial records.

A trial record is the campaign's copy of something the backend owns: whether the
run is still going, what it scored, why it broke. Bringing the two back into
agreement is its own job, and keeping it out of `advance()` leaves the loop free
to be about what to *do* with the answer rather than how to obtain it.

Everything here takes its stores explicitly, so a sync can be driven in a test
without standing up an orchestrator.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ai_experiments.planner.analysis import best_trial, extract_objective
from ai_experiments.schemas import RunEvent, utc_now
from ai_experiments.stopping import ACTIVE_TRIAL_STATES, trial_gpu_hours, trial_wall_hours

if TYPE_CHECKING:
    from ai_experiments.backends.base import ExperimentBackend
    from ai_experiments.schemas import (
        CampaignState,
        GoalSpec,
        RunState,
        RunStatus,
        TrialRecord,
        TrialState,
    )
    from ai_experiments.store import FilesystemRunStore
    from ai_experiments.store.campaign import CampaignStore

#: How a run's state reads as a trial's state. A run state missing from this map
#: is one the trial has no opinion about yet, so the trial is left as it was.
RUN_TO_TRIAL: dict[RunState, TrialState] = {
    "submitted": "submitted",
    "running": "running",
    "completed": "completed",
    "failed": "failed",
    "cancelled": "cancelled",
}

#: Trial states that mean the backend is done with the run, one way or another.
TERMINAL_TRIAL_STATES: set[TrialState] = {"completed", "failed", "cancelled"}


def refresh_trials(
    state: CampaignState,
    goal: GoalSpec,
    backend: ExperimentBackend,
    campaign_store: CampaignStore,
    run_store: FilesystemRunStore,
) -> list[TrialRecord]:
    """Poll every trial still in flight and return the ones that have finished."""
    finished: list[TrialRecord] = []
    for trial in state.trials:
        if trial.status not in ACTIVE_TRIAL_STATES or not trial.run_id:
            continue
        run_status = _inspect(state, backend, campaign_store, trial.trial_id, trial.run_id)
        if run_status is None:
            continue
        mapped = RUN_TO_TRIAL.get(run_status.status)
        if mapped is None:
            continue
        trial.status = mapped
        if mapped in TERMINAL_TRIAL_STATES:
            _record_finish(state, goal, campaign_store, run_store, trial, trial.run_id, run_status)
            finished.append(trial)
    return finished


def update_best(state: CampaignState, goal: GoalSpec) -> None:
    """Point the campaign at its best trial so far, or at nothing if it has none."""
    best = best_trial(state, goal.objective.mode)
    state.best_trial_id = best.trial_id if best else None


def _inspect(
    state: CampaignState,
    backend: ExperimentBackend,
    campaign_store: CampaignStore,
    trial_id: str,
    run_id: str,
) -> RunStatus | None:
    """Ask the backend about one run, or say so on the event log and give up.

    A backend that cannot answer about one trial should not sink the tick: the
    next one will ask again.
    """
    try:
        return backend.inspect(run_id)
    except Exception as exc:
        campaign_store.append_event(
            state.campaign_id,
            RunEvent(
                level="warning",
                message="trial inspect failed",
                details={"trial_id": trial_id, "error": str(exc)},
            ),
        )
        return None


def _record_finish(
    state: CampaignState,
    goal: GoalSpec,
    campaign_store: CampaignStore,
    run_store: FilesystemRunStore,
    trial: TrialRecord,
    run_id: str,
    run_status: RunStatus,
) -> None:
    """Write a finished run's cost, score and outcome onto its trial."""
    trial.completed_at = run_status.completed_at or utc_now()
    trial.error = run_status.error
    started = run_status.started_at or run_status.submitted_at
    trial.gpu_hours = trial_gpu_hours(goal, started, trial.completed_at)
    trial.wall_hours = trial_wall_hours(started, trial.completed_at)
    reading = extract_objective(run_store, run_id, goal.objective)
    trial.objective_value = reading.value
    trial.objective_stderr = reading.stderr
    trial.objective_observations = reading.n_observations
    trial.final_metrics = reading.final_metrics
    miss = reading.miss_message(goal.objective.metric)
    if miss and trial.status == "completed":
        # A trial that ran to completion without producing the objective is a
        # broken contract, not a bad result. Say so on the trial; scoring it
        # `null` in silence burns the whole budget with no explanation (#11).
        trial.error = miss
        campaign_store.append_event(
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
    campaign_store.append_event(
        state.campaign_id,
        RunEvent(
            message="trial finished",
            details={
                "trial_id": trial.trial_id,
                "status": trial.status,
                "objective_value": reading.value,
                "params": trial.params,
            },
        ),
    )
