"""What a campaign does when the tick that submitted a trial never finished.

A campaign is meant to run overnight, unattended. The events it must survive
-- SIGKILL, OOM, a host reboot -- all land in the same place: between the
moment the backend accepts a run and the moment the state file records it.
"""

from __future__ import annotations

import pytest

from ai_experiments.orchestrator import CampaignOrchestrator
from ai_experiments.schemas import BudgetSpec, ExperimentManifest, RunEvent
from ai_experiments.store import FilesystemRunStore
from ai_experiments.store.campaign import CampaignStore
from tests.test_orchestrator import FakeBackend, _goal


class CrashingCampaignStore(CampaignStore):
    """A store whose `write_state` dies once, the way a SIGKILL would."""

    def __init__(self, root, *, crash_when=lambda state: False) -> None:
        super().__init__(root)
        self.crash_when = crash_when

    def write_state(self, state):
        if self.crash_when(state):
            self.crash_when = lambda state: False  # only the one death
            raise SystemExit("supervisor killed between submit and write_state")
        super().write_state(state)


def _orchestrator(tmp_path, campaign_store=None):
    store = FilesystemRunStore(tmp_path / "runs")
    backend = FakeBackend(store)
    campaign_store = campaign_store or CampaignStore(store.root)
    orchestrator = CampaignOrchestrator(
        store, campaign_store, backend_factory=lambda goal: backend
    )
    return orchestrator, backend, store


def test_a_crash_after_submit_does_not_duplicate_the_trial(tmp_path):
    """The run the crash orphaned is adopted, not re-submitted (#5).

    The old behaviour: the state file never learned about run #1, the next
    tick re-planned the same trial with the same seed, and two runs burned
    resources for one trial -- one of them never scored, never cancelled,
    invisible in `campaign status`.
    """
    store = FilesystemRunStore(tmp_path / "runs")
    campaign_store = CrashingCampaignStore(
        store.root, crash_when=lambda state: bool(state.trials)
    )
    orchestrator, backend, store = _orchestrator(tmp_path, campaign_store)
    goal = _goal(budget=BudgetSpec(max_trials=2, max_parallel=1))
    state = campaign_store.create_campaign(goal)

    with pytest.raises(SystemExit):
        orchestrator.advance(state.campaign_id)

    orphaned_run = _submitted_runs(campaign_store, state.campaign_id)["t000"]
    lost = campaign_store.read_state(state.campaign_id)
    assert lost.trials == []  # the state file never heard about it

    recovered = orchestrator.advance(state.campaign_id)

    first = next(t for t in recovered.trials if t.trial_id == "t000")
    assert first.run_id == orphaned_run, "the crashed trial was submitted twice"
    run_ids = [t.run_id for t in recovered.trials]
    assert len(set(run_ids)) == len(run_ids)  # one run per trial, no orphan


def _submitted_runs(campaign_store, campaign_id) -> dict[str, str]:
    return {
        str(event.details["trial_id"]): str(event.details["run_id"])
        for event in campaign_store.read_events(campaign_id)
        if event.message == "trial submitted"
    }


def test_a_recovered_trial_keeps_its_params_and_still_scores(tmp_path):
    """The adopted run has to finish the campaign, not just stop the double."""
    store = FilesystemRunStore(tmp_path / "runs")
    campaign_store = CrashingCampaignStore(
        store.root, crash_when=lambda state: bool(state.trials)
    )
    orchestrator, backend, store = _orchestrator(tmp_path, campaign_store)
    goal = _goal(budget=BudgetSpec(max_trials=3, max_parallel=1))
    state = campaign_store.create_campaign(goal)
    with pytest.raises(SystemExit):
        orchestrator.advance(state.campaign_id)

    for _ in range(20):
        state = orchestrator.advance(state.campaign_id)
        if state.status == "completed":
            break

    assert state.status == "completed"
    assert len(state.trials) == 3
    assert all(t.objective_value is not None for t in state.trials)
    assert len(backend.submitted) == 3  # one run per trial, no orphan


def test_each_submit_is_persisted_before_the_next_one(tmp_path):
    """A crash mid-batch must lose at most the submit it was making."""
    store = FilesystemRunStore(tmp_path / "runs")
    seen: list[int] = []

    class RecordingStore(CampaignStore):
        def write_state(self, state):
            seen.append(sum(1 for t in state.trials if t.run_id))
            super().write_state(state)

    campaign_store = RecordingStore(store.root)
    orchestrator, backend, store = _orchestrator(tmp_path, campaign_store)
    goal = _goal(budget=BudgetSpec(max_trials=4, max_parallel=4))
    state = campaign_store.create_campaign(goal)

    orchestrator.advance(state.campaign_id)

    # one write per submit, each seeing one more run_id than the last
    assert [1, 2, 3, 4] == [count for count in seen if count][:4]


class UnreachableBackend(FakeBackend):
    def submit(self, manifest):
        raise ConnectionError("Failed to connect to Ray at http://127.0.0.1:8265")


def test_an_unreachable_backend_is_not_an_exhausted_search_space(tmp_path):
    """The two need opposite responses: widen the goal, or start the cluster (#36)."""
    store = FilesystemRunStore(tmp_path / "runs")
    campaign_store = CampaignStore(store.root)
    backend = UnreachableBackend(store)
    orchestrator = CampaignOrchestrator(
        store, campaign_store, backend_factory=lambda goal: backend
    )
    state = orchestrator.start(_goal())

    state = orchestrator.advance(state.campaign_id)

    assert state.stop_reason == "backend_unavailable"
    assert state.status == "failed"
    assert "Ray" in (state.trials[0].error or "")


def test_a_search_space_that_really_is_exhausted_still_says_so(tmp_path):
    """The pin on the fix above."""
    orchestrator, backend, store = _orchestrator(tmp_path)
    goal = _goal(
        search_space={"x": {"type": "choice", "values": [1.0, 2.0]}},
        budget=BudgetSpec(max_trials=10, max_parallel=2),
        strategy={"name": "grid"},
    )
    state = orchestrator.start(goal)

    for _ in range(20):
        state = orchestrator.advance(state.campaign_id)
        if state.status != "running":
            break

    assert state.stop_reason == "search_space_exhausted"


def test_a_run_that_no_trial_claims_is_cancelled(tmp_path):
    """A duplicate that already happened must not keep burning a GPU."""
    store = FilesystemRunStore(tmp_path / "runs")
    campaign_store = CampaignStore(store.root)
    orchestrator, backend, store = _orchestrator(tmp_path, campaign_store)
    state = orchestrator.start(_goal(budget=BudgetSpec(max_trials=2, max_parallel=1)))
    state = orchestrator.advance(state.campaign_id)
    claimed = state.trials[0]

    # An earlier submit for the same trial that the state never recorded.
    orphan = backend.submit(
        ExperimentManifest(
            experiment="orphan",
            workload=state and _goal().workload,  # type: ignore[arg-type]
        )
    )
    campaign_store.append_event(
        state.campaign_id,
        _submitted_event(claimed.trial_id, orphan.run_id, claimed.params),
    )

    orchestrator.advance(state.campaign_id)

    assert orphan.run_id in backend.cancelled
    assert store.read_status(orphan.run_id).status == "cancelled"


def _submitted_event(trial_id: str, run_id: str, params: dict) -> RunEvent:
    return RunEvent(
        message="trial submitted",
        details={"trial_id": trial_id, "run_id": run_id, "params": params},
    )


def test_the_loop_exits_3_when_no_trial_could_start(tmp_path):
    """Exit 4 says the work ran and missed. Nothing ran here (#36)."""
    import yaml
    from typer.testing import CliRunner

    from ai_experiments.cli import app

    goal_path = tmp_path / "goal.yaml"
    goal = _goal().model_dump(mode="json")
    goal["backend"] = "ray"
    goal["backend_address"] = "http://127.0.0.1:59999"  # nothing listens here
    goal["budget"] = {"max_trials": 2, "max_parallel": 1}
    goal_path.write_text(yaml.safe_dump(goal))

    result = CliRunner().invoke(
        app,
        [
            "loop",
            str(goal_path),
            "--runs-dir",
            str(tmp_path / "runs"),
            "--interval",
            "0",
        ],
    )

    assert result.exit_code == 3, result.stdout
    assert "backend_unavailable" in result.stdout
