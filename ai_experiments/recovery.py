"""Rebuilding a campaign's trials from its event log after a crash.

`advance` submits real runs and persists the state after each one, so a crash
in between leaves a run executing that no trial references (#5). The campaign's
own event log is the record that survives: a "trial submitted" event carries
the trial id, the run id and the params, and it is appended *before* the state
is written. Everything here reads that log back.

The functions take their stores explicitly rather than hanging off the
orchestrator: recovery is a pure function of what was written down, and keeping
it that way is what makes it testable without a live campaign.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ai_experiments.schemas import RunEvent, TrialRecord
from ai_experiments.stopping import ACTIVE_TRIAL_STATES

if TYPE_CHECKING:
    from ai_experiments.backends.base import ExperimentBackend
    from ai_experiments.schemas import CampaignState
    from ai_experiments.store import FilesystemRunStore
    from ai_experiments.store.campaign import CampaignStore


def recover_lost_trials(
    state: CampaignState,
    backend: ExperimentBackend,
    campaign_store: CampaignStore,
    run_store: FilesystemRunStore,
) -> None:
    """Reconcile the state file with the runs the backend actually holds.

    Without this, the next tick would re-plan a lost trial -- same seed, same
    params -- and submit it again: two runs for one trial, one of them never
    scored, never cancelled, invisible in `campaign status`. A trial the crash
    lost is rebuilt from the event log, and a run that some other trial has
    already replaced is cancelled rather than left burning a GPU.
    """
    known = {trial.trial_id: trial for trial in state.trials}
    for event in campaign_store.read_events(state.campaign_id):
        if event.message != "trial submitted":
            continue
        trial_id = str(event.details.get("trial_id", ""))
        run_id = str(event.details.get("run_id", ""))
        if not trial_id or not run_id:
            continue
        trial = known.get(trial_id)
        if trial is None:
            restored = TrialRecord(
                trial_id=trial_id,
                params=dict(event.details.get("params") or {}),
                run_id=run_id,
                status="submitted",
                submitted_at=event.timestamp,
            )
            state.trials.append(restored)
            known[trial_id] = restored
            log_recovery(state, campaign_store, "recovered a trial the crash lost", restored)
        elif trial.run_id != run_id:
            cancel_orphan(state, backend, campaign_store, run_store, trial_id, run_id)
    state.trials.sort(key=lambda trial: trial.trial_id)


def log_recovery(
    state: CampaignState,
    campaign_store: CampaignStore,
    message: str,
    trial: TrialRecord,
) -> None:
    """Record a recovery on the campaign's event log, where an operator reads it."""
    campaign_store.append_event(
        state.campaign_id,
        RunEvent(
            level="warning",
            message=message,
            details={"trial_id": trial.trial_id, "run_id": trial.run_id},
        ),
    )


def cancel_orphan(
    state: CampaignState,
    backend: ExperimentBackend,
    campaign_store: CampaignStore,
    run_store: FilesystemRunStore,
    trial_id: str,
    run_id: str,
) -> None:
    """Stop a run that a trial no longer claims.

    Idempotent: a run that has already stopped is left alone, so a replayed
    event cancels once.
    """
    try:
        status = run_store.read_status(run_id)
    except FileNotFoundError:
        return
    if status.status not in ACTIVE_TRIAL_STATES:
        return
    backend.cancel(run_id)
    campaign_store.append_event(
        state.campaign_id,
        RunEvent(
            level="warning",
            message="cancelled a run no trial claims",
            details={"trial_id": trial_id, "run_id": run_id},
        ),
    )
