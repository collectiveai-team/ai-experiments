from __future__ import annotations

from ai_experiments.orchestrator import CampaignOrchestrator
from ai_experiments.schemas import BudgetSpec, ObjectiveSpec, RunStatus, utc_now
from ai_experiments.store import FilesystemRunStore
from ai_experiments.store.campaign import CampaignStore


def _orchestrator(tmp_path, fake_backend_factory):
    store = FilesystemRunStore(tmp_path / "runs")
    backend = fake_backend_factory(store)
    orchestrator = CampaignOrchestrator(
        store, CampaignStore(store.root), backend_factory=lambda goal: backend
    )
    return orchestrator, backend


def test_campaign_runs_to_budget_exhaustion(tmp_path, fake_backend_factory, goal_factory):
    orchestrator, _backend = _orchestrator(tmp_path, fake_backend_factory)
    state = orchestrator.start(goal_factory())

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
    summary = orchestrator.campaign_store.campaign_dir(state.campaign_id) / "summary.json"
    assert summary.exists()


def test_campaign_stops_when_target_reached(tmp_path, fake_backend_factory, goal_factory):
    orchestrator, backend = _orchestrator(tmp_path, fake_backend_factory)
    backend.objective_fn = lambda p: 0.0  # every trial hits the target
    state = orchestrator.start(
        goal_factory(objective=ObjectiveSpec(metric="loss", mode="min", target=0.5))
    )

    state = orchestrator.advance(state.campaign_id)

    assert state.status == "completed"
    assert state.stop_reason == "target_reached"
    assert len(state.trials) < 6  # stopped early, budget unspent


def test_max_parallel_caps_in_flight_trials(tmp_path, fake_backend_factory, goal_factory):
    orchestrator, backend = _orchestrator(tmp_path, fake_backend_factory)
    orchestrator.start(goal_factory())

    # FakeBackend completes runs on inspect, so the first batch is exactly
    # the parallelism cap.
    assert len(backend.submitted) == 2


def test_suggested_trial_is_submitted_first(tmp_path, fake_backend_factory, goal_factory):
    orchestrator, backend = _orchestrator(tmp_path, fake_backend_factory)
    state = orchestrator.start(goal_factory())
    orchestrator.suggest(state.campaign_id, {"x": 1.99}, note="agent hunch")

    state = orchestrator.advance(state.campaign_id)

    suggested = [t for t in state.trials if t.source == "agent"]
    assert len(suggested) == 1
    assert suggested[0].status in {"submitted", "running", "completed"}
    assert any(m.metadata["params"] == {"x": 1.99} for m in backend.submitted)


def test_stop_cancels_active_trials(tmp_path, fake_backend_factory, goal_factory):
    store = FilesystemRunStore(tmp_path / "runs")
    backend = fake_backend_factory(store)
    # Backend that never completes: inspect leaves runs in 'submitted'.
    backend.inspect = store.read_status  # type: ignore[method-assign]
    orchestrator = CampaignOrchestrator(
        store, CampaignStore(store.root), backend_factory=lambda goal: backend
    )
    state = orchestrator.start(goal_factory())
    assert any(t.status == "submitted" for t in state.trials)

    state = orchestrator.stop(state.campaign_id)

    assert state.status == "stopped"
    assert backend.cancelled
    assert all(t.status in {"cancelled", "failed"} for t in state.trials)


def test_pause_blocks_advance_and_resume_restarts(tmp_path, fake_backend_factory, goal_factory):
    orchestrator, backend = _orchestrator(tmp_path, fake_backend_factory)
    state = orchestrator.start(goal_factory())
    submitted_before = len(backend.submitted)

    paused = orchestrator.pause(state.campaign_id)
    assert paused.status == "paused"

    after_pause = orchestrator.advance(state.campaign_id)
    assert after_pause.status == "paused"
    assert len(backend.submitted) == submitted_before  # nothing new scheduled

    resumed = orchestrator.resume(state.campaign_id)
    assert resumed.status in {"running", "completed"}
    assert len(backend.submitted) > submitted_before


def test_pause_requires_running_campaign(tmp_path, fake_backend_factory, goal_factory):
    import pytest

    orchestrator, _ = _orchestrator(tmp_path, fake_backend_factory)
    state = orchestrator.start(goal_factory())
    orchestrator.stop(state.campaign_id)

    with pytest.raises(ValueError, match=r"^cannot pause a campaign in status"):
        orchestrator.pause(state.campaign_id)


def test_edit_goal_mid_flight(tmp_path, fake_backend_factory, goal_factory):
    import pytest

    orchestrator, _ = _orchestrator(tmp_path, fake_backend_factory)
    state = orchestrator.start(goal_factory())

    narrowed = goal_factory(
        search_space={"x": {"type": "uniform", "low": 1.5, "high": 2.5}},
    )
    orchestrator.edit_goal(state.campaign_id, narrowed)

    stored = orchestrator.campaign_store.read_goal(state.campaign_id)
    assert stored.search_space["x"].low == 1.5  # type: ignore[union-attr]

    renamed_metric = goal_factory(objective=ObjectiveSpec(metric="other", mode="min"))
    with pytest.raises(ValueError, match="objective metric"):
        orchestrator.edit_goal(state.campaign_id, renamed_metric)


def test_gpu_hours_recorded_and_budget_stops_campaign(tmp_path, fake_backend_factory, goal_factory):
    from ai_experiments.schemas import ResourceSpec

    orchestrator, _ = _orchestrator(tmp_path, fake_backend_factory)
    goal = goal_factory(resources=ResourceSpec(gpus=2))
    state = orchestrator.start(goal)
    for _ in range(20):
        state = orchestrator.advance(state.campaign_id)
        if state.status == "completed":
            break
    finished = [t for t in state.trials if t.status == "completed"]
    assert finished
    assert all(t.gpu_hours is not None and t.gpu_hours >= 0 for t in finished)

    # A zero GPU-hour budget halts a fresh campaign on its first step.
    capped = goal_factory(
        budget=BudgetSpec(max_trials=6, max_parallel=2, max_gpu_hours=0.0),
        resources=ResourceSpec(gpus=2),
    )
    capped_state = orchestrator.start(capped)
    assert capped_state.status == "completed"
    assert capped_state.stop_reason == "gpu_hours_exhausted"


def test_failed_trials_recorded_and_loop_continues(tmp_path, fake_backend_factory, goal_factory):
    orchestrator, backend = _orchestrator(tmp_path, fake_backend_factory)
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
    state = orchestrator.start(goal_factory())
    for _ in range(20):
        state = orchestrator.advance(state.campaign_id)
        if state.status == "completed":
            break

    assert state.status == "completed"
    assert sum(1 for t in state.trials if t.status == "failed") == 1
    assert sum(1 for t in state.trials if t.status == "completed") == 5
