from __future__ import annotations

import pytest

from ai_experiments.orchestrator import CampaignOrchestrator
from ai_experiments.schemas import (
    BudgetSpec,
    MetricPoint,
    ObjectiveSpec,
    RunStatus,
    StrategySpec,
    utc_now,
)
from ai_experiments.store import FilesystemRunStore
from ai_experiments.store.campaign import CampaignStore
from tests.conftest import FakeBackend, _goal


def _orchestrator(tmp_path, fake_backend_factory: type[FakeBackend] = FakeBackend):
    """Build an orchestrator wired to one in-memory backend, and return both.

    The factory defaults to the conftest fake so a test that needs no other
    backend can call this without taking the fixture as a parameter.
    """
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


def test_failed_trial_never_wins_the_campaign(tmp_path):
    """Only trials that ran to completion are eligible (#12).

    A crashed trial can report a metric before dying; that value must not
    end the campaign as `target_reached`.
    """
    orchestrator, backend = _orchestrator(tmp_path)

    def dying_inspect(run_id: str) -> RunStatus:
        status = backend.store.read_status(run_id)
        if status.status in {"submitted", "running"}:
            backend.store.append_metric(run_id, MetricPoint(step=1, values={"loss": 0.0}))
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
    """A failed trial is excluded from `best`.

    But it must still appear in the history the agent reasons over.
    """
    from ai_experiments.planner.analysis import summarize_campaign

    orchestrator, _backend = _orchestrator(tmp_path)
    goal = _goal()
    state = orchestrator.start(goal)
    state.trials[0].status = "failed"
    state.trials[0].objective_value = 0.001
    state.trials[0].error = "OOM"

    summary = summarize_campaign(state, goal)

    failed = [h for h in summary.history if h.trial_id == state.trials[0].trial_id]
    assert failed
    assert failed[0].status == "failed"
    assert failed[0].error == "OOM"


def test_typod_objective_metric_is_reported_on_the_trial(tmp_path):
    """A workload can report metrics without reporting *the* objective metric (#11).

    When that happens, the trial and the events must say so.
    """
    orchestrator, _backend = _orchestrator(tmp_path)
    state = orchestrator.start(_goal(objective=ObjectiveSpec(metric="val_loss")))
    state = orchestrator.advance(state.campaign_id)

    finished = [t for t in state.trials if t.status in {"completed", "failed"}]
    assert finished
    assert all(t.objective_value is None for t in finished)
    assert all("val_loss" in (t.error or "") for t in finished)
    assert all("loss" in (t.error or "") for t in finished)

    events = orchestrator.campaign_store.read_events(state.campaign_id)
    assert any(e.level == "warning" and "objective" in e.message for e in events)


def test_objective_never_reported_stops_the_campaign_early(tmp_path):
    """The budget must not be burned silently on a metric nobody reports."""
    orchestrator, _backend = _orchestrator(tmp_path)
    state = orchestrator.start(
        _goal(
            objective=ObjectiveSpec(metric="val_loss"),
            budget=BudgetSpec(max_trials=20, max_parallel=2),
        )
    )
    for _ in range(20):
        state = orchestrator.advance(state.campaign_id)
        if state.status != "running":
            break

    assert state.status == "failed"
    assert state.stop_reason == "objective_not_reported"
    assert len(state.trials) < 20


def test_workload_reporting_nothing_is_distinguished_from_a_typo(tmp_path):
    """No result at all is a different diagnosis from the wrong metric name."""
    orchestrator, backend = _orchestrator(tmp_path)

    def silent_inspect(run_id: str) -> RunStatus:
        status = backend.store.read_status(run_id)
        if status.status in {"submitted", "running"}:
            return backend.store.update_status(run_id, status="completed", completed_at=utc_now())
        return status

    backend.inspect = silent_inspect  # type: ignore[method-assign]
    state = orchestrator.start(_goal())
    state = orchestrator.advance(state.campaign_id)

    finished = [t for t in state.trials if t.status == "completed"]
    assert finished
    assert all("no result" in (t.error or "") for t in finished)


def test_exhausted_search_space_ends_the_campaign(tmp_path):
    """A grid smaller than max_trials must terminate (#15).

    Otherwise it sits in `running` forever, with the planner returning []
    every tick.
    """
    orchestrator, _backend = _orchestrator(tmp_path)
    state = orchestrator.start(
        _goal(
            search_space={"x": {"type": "choice", "values": [1.0, 2.0, 3.0]}},
            budget=BudgetSpec(max_trials=50, max_parallel=2),
            strategy=StrategySpec(name="grid", seed=1),
        )
    )
    for _ in range(20):
        state = orchestrator.advance(state.campaign_id)
        if state.status != "running":
            break

    assert state.status == "completed"
    assert state.stop_reason == "search_space_exhausted"
    assert len(state.trials) == 3

    events = orchestrator.campaign_store.read_events(state.campaign_id)
    assert any("search space" in e.message for e in events)


def test_suggest_rejects_params_outside_the_search_space(tmp_path):
    """An unknown key must be rejected before it reaches the workload (#13).

    Otherwise it becomes a literal --not_a_param flag on the workload's
    command line and crashes it as a failed trial.
    """
    from ai_experiments.planner.validation import ParamValidationError

    orchestrator, _backend = _orchestrator(tmp_path)
    state = orchestrator.start(_goal())

    with pytest.raises(ParamValidationError):
        orchestrator.suggest(state.campaign_id, {"x": 1.0, "not_a_param": 42})
    with pytest.raises(ParamValidationError):
        orchestrator.suggest(state.campaign_id, {"x": 99.0})

    assert not [
        t
        for t in orchestrator.campaign_store.read_state(state.campaign_id).trials
        if t.source == "agent"
    ]


def test_suggest_respects_max_trials(tmp_path):
    orchestrator, backend = _orchestrator(tmp_path)
    state = orchestrator.start(_goal(budget=BudgetSpec(max_trials=1, max_parallel=1)))

    with pytest.raises(ValueError, match="budget"):
        orchestrator.suggest(state.campaign_id, {"x": 1.0})

    for _ in range(5):
        state = orchestrator.advance(state.campaign_id)
        if state.status != "running":
            break
    assert len(backend.submitted) == 1


def test_suggest_rejects_a_finished_campaign(tmp_path):
    orchestrator, _backend = _orchestrator(tmp_path)
    state = orchestrator.start(_goal())
    orchestrator.stop(state.campaign_id)

    with pytest.raises(ValueError, match="stopped"):
        orchestrator.suggest(state.campaign_id, {"x": 1.0})


def test_agent_review_round_keeps_the_escalation_inbox_readable(tmp_path):
    """End-to-end guard for #4.

    A campaign with agent_review on must not poison `iax escalations`.
    """
    from ai_experiments.monitoring.escalation import CampaignReview, list_escalations

    orchestrator, _backend = _orchestrator(tmp_path)
    state = orchestrator.start(_goal(analysis={"agent_review": True}))
    state = orchestrator.advance(state.campaign_id)

    items = list_escalations(orchestrator.run_store)
    assert [i.kind for i in items] == ["campaign"]
    review = items[0]
    assert isinstance(review, CampaignReview)
    assert review.summary["campaign_id"] == state.campaign_id


class FailingBackend(FakeBackend):
    """Every workload dies the same way, the way a mis-spelled flag dies."""

    MESSAGE = "train.py: error: unrecognized arguments: --x 0.5"

    def inspect(self, run_id: str) -> RunStatus:
        status = self.store.read_status(run_id)
        if status.status in {"submitted", "running"}:
            status = self.store.update_status(
                run_id,
                status="failed",
                error=f"workload exited with code 2: {self.MESSAGE}",
                completed_at=utc_now(),
            )
        return status


def _failing(tmp_path) -> tuple[CampaignOrchestrator, FailingBackend]:
    store = FilesystemRunStore(tmp_path / "runs")
    backend = FailingBackend(store)
    return (
        CampaignOrchestrator(
            store, CampaignStore(store.root), backend_factory=lambda goal: backend
        ),
        backend,
    )


def test_a_campaign_whose_every_trial_fails_gives_up_early(tmp_path):
    """A broken workload fails identically every time.

    Spending the whole budget to learn that costs real compute and ends as a
    "successful" budget_exhausted with no best trial.
    """
    orchestrator, _ = _failing(tmp_path)
    state = orchestrator.start(_goal(budget=BudgetSpec(max_trials=40, max_parallel=2)))

    for _ in range(40):
        state = orchestrator.advance(state.campaign_id)
        if state.status in {"completed", "failed"}:
            break

    assert state.status == "failed"
    assert state.stop_reason == "all_trials_failing"
    assert len(state.trials) < 40


def test_the_halt_carries_what_the_workload_actually_said(tmp_path):
    """The reason is useless without the error; that is the whole point."""
    orchestrator, _ = _failing(tmp_path)
    state = orchestrator.start(_goal(budget=BudgetSpec(max_trials=40, max_parallel=2)))

    for _ in range(40):
        state = orchestrator.advance(state.campaign_id)
        if state.status in {"completed", "failed"}:
            break

    events = orchestrator.campaign_store.read_events(state.campaign_id)
    gave_up = [e for e in events if e.level == "error"]
    assert gave_up
    assert FailingBackend.MESSAGE in str(gave_up[-1].details)


def test_a_campaign_that_recovers_is_not_halted(tmp_path):
    """Some failures are flaky. Only an unbroken streak means broken."""
    store = FilesystemRunStore(tmp_path / "runs")
    backend = FakeBackend(store)
    calls = {"n": 0}
    original = backend.inspect

    def flaky(run_id: str) -> RunStatus:
        calls["n"] += 1
        status = store.read_status(run_id)
        if calls["n"] <= 2 and status.status in {"submitted", "running"}:
            return store.update_status(
                run_id, status="failed", error="transient", completed_at=utc_now()
            )
        return original(run_id)

    backend.inspect = flaky  # type: ignore[method-assign]
    orchestrator = CampaignOrchestrator(
        store, CampaignStore(store.root), backend_factory=lambda goal: backend
    )
    state = orchestrator.start(_goal(budget=BudgetSpec(max_trials=8, max_parallel=1)))

    for _ in range(30):
        state = orchestrator.advance(state.campaign_id)
        if state.status in {"completed", "failed"}:
            break

    assert state.stop_reason != "all_trials_failing"
