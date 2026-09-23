"""Syncing a finished run onto its trial: what the results channel is allowed to mean."""

from __future__ import annotations

from typing import Literal

from ai_experiments.planner.planner import build_trial_manifest
from ai_experiments.schemas import ObjectiveSpec, ResultRecord, TrialRecord
from ai_experiments.store import FilesystemRunStore
from ai_experiments.store.campaign import CampaignStore
from ai_experiments.trial_sync import refresh_trials
from tests.conftest import FakeBackend, _goal


class _TwoResults(FakeBackend):
    """A workload that declares one result per fold, as a fold loop does."""

    def inspect(self, run_id):
        status = self.store.read_status(run_id)
        if status.status in {"submitted", "running"}:
            for value in (0.4, 0.6):
                self.store.append_result(run_id, ResultRecord(values={"loss": value}))
            status = self.store.update_status(run_id, status="completed")
        return status


def _synced(tmp_path, aggregate: Literal["best", "mean", "bootstrap"]):
    store = FilesystemRunStore(tmp_path / "runs", capture_repro=False)
    campaign_store = CampaignStore(store.root)
    goal = _goal(objective=ObjectiveSpec(metric="loss", mode="min", aggregate=aggregate))
    state = campaign_store.create_campaign(goal)
    backend = _TwoResults(store)
    handle = backend.submit(build_trial_manifest(goal, "t000", {"x": 1.0}))
    state.trials.append(
        TrialRecord(trial_id="t000", params={"x": 1.0}, status="submitted", run_id=handle.run_id)
    )
    campaign_store.write_state(state)

    refresh_trials(state, goal, backend, campaign_store, store)
    return state, campaign_store.read_events(state.campaign_id)


def test_several_results_are_observations_under_mean(tmp_path):
    state, events = _synced(tmp_path, "mean")

    assert state.trials[0].objective_value == 0.5
    assert state.trials[0].objective_observations == 2
    assert not [e for e in events if "more than one result" in e.message]


def test_several_results_under_best_are_scored_and_warned_about(tmp_path):
    """`best` takes one of them; the rest are a contract the workload did not read."""
    state, events = _synced(tmp_path, "best")

    assert state.trials[0].objective_value == 0.4
    surplus = [e for e in events if "more than one result" in e.message]
    assert len(surplus) == 1
    assert surplus[0].level == "warning"
    assert surplus[0].details["results"] == 2
