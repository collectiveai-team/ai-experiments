from __future__ import annotations

from typing import Callable

from ai_experiments.backends.base import ExperimentBackend
from ai_experiments.monitoring.rules import diagnose_run
from ai_experiments.orchestrator import CampaignOrchestrator
from ai_experiments.schemas import (
    BudgetSpec,
    DiagnosisReport,
    ExperimentManifest,
    GoalSpec,
    MetricPoint,
    ObjectiveSpec,
    RunEvent,
    RunHandle,
    RunStatus,
    StrategySpec,
    WorkloadSpec,
    utc_now,
)
from ai_experiments.store import FilesystemRunStore
from ai_experiments.store.campaign import CampaignStore


class FakeBackend(ExperimentBackend):
    """Runs 'complete' instantly; the objective is a deterministic function
    of the submitted params, recorded as a metric on inspect."""

    def __init__(
        self,
        store: FilesystemRunStore,
        objective_fn: Callable[[dict], float] | None = None,
    ) -> None:
        self.store = store
        self.objective_fn = objective_fn or (lambda p: (p["x"] - 2.0) ** 2)
        self.submitted: list[ExperimentManifest] = []
        self.cancelled: list[str] = []

    def submit(self, manifest: ExperimentManifest) -> RunHandle:
        run_id, run_dir = self.store.create_run(manifest)
        self.submitted.append(manifest)
        handle = RunHandle(
            run_id=run_id,
            backend="local",
            status="submitted",
            status_uri=str(self.store.status_path(run_id)),
            run_dir=str(run_dir),
        )
        self.store.write_handle(handle)
        return handle

    def inspect(self, run_id: str) -> RunStatus:
        status = self.store.read_status(run_id)
        if status.status in {"submitted", "running"}:
            manifest = self.store.read_manifest(run_id)
            assert manifest is not None
            params = manifest.metadata["params"]
            value = self.objective_fn(params)
            self.store.append_metric(
                run_id, MetricPoint(step=1, values={"loss": value})
            )
            status = self.store.update_status(
                run_id, status="completed", completed_at=utc_now()
            )
        return status

    def logs(self, run_id: str, tail: int = 200) -> list[RunEvent]:
        return []

    def cancel(self, run_id: str) -> None:
        self.cancelled.append(run_id)
        self.store.update_status(run_id, status="cancelled", completed_at=utc_now())

    def diagnose(self, run_id: str) -> DiagnosisReport:
        return diagnose_run(self.store, run_id)


def _goal(**overrides: object) -> GoalSpec:
    data: dict = {
        "goal": "minimize (x-2)^2",
        "name": "quadratic",
        "objective": ObjectiveSpec(metric="loss", mode="min"),
        "search_space": {"x": {"type": "uniform", "low": -5.0, "high": 5.0}},
        "workload": WorkloadSpec(entrypoint="python toy.py"),
        "budget": BudgetSpec(max_trials=6, max_parallel=2),
        "strategy": StrategySpec(name="adaptive", seed=3),
    }
    data.update(overrides)
    return GoalSpec(**data)


def _orchestrator(tmp_path) -> tuple[CampaignOrchestrator, FakeBackend]:
    store = FilesystemRunStore(tmp_path / "runs")
    backend = FakeBackend(store)
    orchestrator = CampaignOrchestrator(
        store, CampaignStore(store.root), backend_factory=lambda goal: backend
    )
    return orchestrator, backend


def test_campaign_runs_to_budget_exhaustion(tmp_path):
    orchestrator, backend = _orchestrator(tmp_path)
    state = orchestrator.start(_goal())

    for _ in range(20):
        state = orchestrator.advance(state.campaign_id)
        if state.status == "completed":
            break

    assert state.status == "completed"
    assert state.stop_reason == "budget_exhausted"
    assert len(state.trials) == 6
    assert all(t.status == "completed" for t in state.trials)
    assert state.best_trial_id is not None
    best = next(t for t in state.trials if t.trial_id == state.best_trial_id)
    assert best.objective_value == min(
        t.objective_value for t in state.trials if t.objective_value is not None
    )
    summary = (
        orchestrator.campaign_store.campaign_dir(state.campaign_id) / "summary.json"
    )
    assert summary.exists()


def test_campaign_stops_when_target_reached(tmp_path):
    orchestrator, backend = _orchestrator(tmp_path)
    backend.objective_fn = lambda p: 0.0  # every trial hits the target
    state = orchestrator.start(
        _goal(objective=ObjectiveSpec(metric="loss", mode="min", target=0.5))
    )

    state = orchestrator.advance(state.campaign_id)

    assert state.status == "completed"
    assert state.stop_reason == "target_reached"
    assert len(state.trials) < 6  # stopped early, budget unspent


def test_max_parallel_caps_in_flight_trials(tmp_path):
    orchestrator, backend = _orchestrator(tmp_path)
    orchestrator.start(_goal())

    # FakeBackend completes runs on inspect, so the first batch is exactly
    # the parallelism cap.
    assert len(backend.submitted) == 2


def test_suggested_trial_is_submitted_first(tmp_path):
    orchestrator, backend = _orchestrator(tmp_path)
    state = orchestrator.start(_goal())
    orchestrator.suggest(state.campaign_id, {"x": 1.99}, note="agent hunch")

    state = orchestrator.advance(state.campaign_id)

    suggested = [t for t in state.trials if t.source == "agent"]
    assert len(suggested) == 1
    assert suggested[0].status in {"submitted", "running", "completed"}
    assert any(m.metadata["params"] == {"x": 1.99} for m in backend.submitted)


def test_stop_cancels_active_trials(tmp_path):
    store = FilesystemRunStore(tmp_path / "runs")
    backend = FakeBackend(store)
    # Backend that never completes: inspect leaves runs in 'submitted'.
    backend.inspect = lambda run_id: store.read_status(run_id)  # type: ignore[method-assign]
    orchestrator = CampaignOrchestrator(
        store, CampaignStore(store.root), backend_factory=lambda goal: backend
    )
    state = orchestrator.start(_goal())
    assert any(t.status == "submitted" for t in state.trials)

    state = orchestrator.stop(state.campaign_id)

    assert state.status == "stopped"
    assert backend.cancelled
    assert all(t.status in {"cancelled", "failed"} for t in state.trials)


def test_pause_blocks_advance_and_resume_restarts(tmp_path):
    orchestrator, backend = _orchestrator(tmp_path)
    state = orchestrator.start(_goal())
    submitted_before = len(backend.submitted)

    paused = orchestrator.pause(state.campaign_id)
    assert paused.status == "paused"

    after_pause = orchestrator.advance(state.campaign_id)
    assert after_pause.status == "paused"
    assert len(backend.submitted) == submitted_before  # nothing new scheduled

    resumed = orchestrator.resume(state.campaign_id)
    assert resumed.status in {"running", "completed"}
    assert len(backend.submitted) > submitted_before


def test_pause_requires_running_campaign(tmp_path):
    import pytest

    orchestrator, _ = _orchestrator(tmp_path)
    state = orchestrator.start(_goal())
    orchestrator.stop(state.campaign_id)

    with pytest.raises(ValueError):
        orchestrator.pause(state.campaign_id)


def test_edit_goal_mid_flight(tmp_path):
    import pytest

    orchestrator, _ = _orchestrator(tmp_path)
    state = orchestrator.start(_goal())

    narrowed = _goal(
        search_space={"x": {"type": "uniform", "low": 1.5, "high": 2.5}},
    )
    orchestrator.edit_goal(state.campaign_id, narrowed)

    stored = orchestrator.campaign_store.read_goal(state.campaign_id)
    assert stored.search_space["x"].low == 1.5  # type: ignore[union-attr]

    renamed_metric = _goal(objective=ObjectiveSpec(metric="other", mode="min"))
    with pytest.raises(ValueError, match="objective metric"):
        orchestrator.edit_goal(state.campaign_id, renamed_metric)


def test_gpu_hours_recorded_and_budget_stops_campaign(tmp_path):
    from ai_experiments.schemas import ResourceSpec

    orchestrator, _ = _orchestrator(tmp_path)
    goal = _goal(resources=ResourceSpec(gpus=2))
    state = orchestrator.start(goal)
    for _ in range(20):
        state = orchestrator.advance(state.campaign_id)
        if state.status == "completed":
            break
    finished = [t for t in state.trials if t.status == "completed"]
    assert finished and all(
        t.gpu_hours is not None and t.gpu_hours >= 0 for t in finished
    )

    # A zero GPU-hour budget halts a fresh campaign on its first step.
    capped = _goal(
        budget=BudgetSpec(max_trials=6, max_parallel=2, max_gpu_hours=0.0),
        resources=ResourceSpec(gpus=2),
    )
    capped_state = orchestrator.start(capped)
    assert capped_state.status == "completed"
    assert capped_state.stop_reason == "gpu_hours_exhausted"


def test_failed_trials_recorded_and_loop_continues(tmp_path):
    orchestrator, backend = _orchestrator(tmp_path)
    original_inspect = backend.inspect
    failures = {"count": 0}

    def flaky_inspect(run_id: str) -> RunStatus:
        status = backend.store.read_status(run_id)
        if status.status in {"submitted", "running"} and failures["count"] == 0:
            failures["count"] += 1
            return backend.store.update_status(
                run_id, status="failed", error="boom", completed_at=utc_now()
            )
        return original_inspect(run_id)

    backend.inspect = flaky_inspect  # type: ignore[method-assign]
    state = orchestrator.start(_goal())
    for _ in range(20):
        state = orchestrator.advance(state.campaign_id)
        if state.status == "completed":
            break

    assert state.status == "completed"
    assert sum(1 for t in state.trials if t.status == "failed") == 1
    assert sum(1 for t in state.trials if t.status == "completed") == 5


def test_failed_trial_never_wins_the_campaign(tmp_path):
    """A crashed trial can report a metric before dying. Its value must not
    end the campaign as `target_reached` (#12)."""
    orchestrator, backend = _orchestrator(tmp_path)

    def dying_inspect(run_id: str) -> RunStatus:
        status = backend.store.read_status(run_id)
        if status.status in {"submitted", "running"}:
            backend.store.append_metric(
                run_id, MetricPoint(step=1, values={"loss": 0.0})
            )
            return backend.store.update_status(
                run_id, status="failed", error="OOM", completed_at=utc_now()
            )
        return status

    backend.inspect = dying_inspect  # type: ignore[method-assign]
    state = orchestrator.start(
        _goal(objective=ObjectiveSpec(metric="loss", mode="min", target=0.5))
    )
    state = orchestrator.advance(state.campaign_id)

    assert any(t.status == "failed" for t in state.trials)
    assert state.stop_reason != "target_reached"
    assert state.best_trial_id is None


def test_failed_trial_value_still_visible_to_the_agent(tmp_path):
    """Excluding failures from `best` must not hide them from the history the
    agent reasons over."""
    from ai_experiments.planner.analysis import summarize_campaign

    orchestrator, backend = _orchestrator(tmp_path)
    goal = _goal()
    state = orchestrator.start(goal)
    state.trials[0].status = "failed"
    state.trials[0].objective_value = 0.001
    state.trials[0].error = "OOM"

    summary = summarize_campaign(state, goal)

    failed = [
        h for h in summary["history"] if h["trial_id"] == state.trials[0].trial_id
    ]
    assert failed and failed[0]["status"] == "failed"
    assert failed[0]["error"] == "OOM"
