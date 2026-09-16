"""The unattended path is the one that most needs supervision.

`iax loop` (`run_loop`) drives a campaign with nobody watching between
rounds. `supervise_once`, extracted from the daemon's per-tick run check,
must run on every loop iteration against the campaign's in-flight trials --
and, called directly, must behave exactly as the daemon's own check does.
"""

from __future__ import annotations

from datetime import timedelta

from ai_experiments.daemon import MonitorDaemon, SupervisionReport, supervise_once
from ai_experiments.loop import run_loop
from ai_experiments.orchestrator import ACTIVE_TRIAL_STATES, CampaignOrchestrator
from ai_experiments.schemas import (
    BudgetSpec,
    ExperimentManifest,
    GoalSpec,
    MonitorPolicy,
    ObjectiveSpec,
    RunHandle,
    StrategySpec,
    WorkloadSpec,
    utc_now,
)
from ai_experiments.store import FilesystemRunStore
from ai_experiments.store.campaign import CampaignStore
from tests.test_orchestrator import FakeBackend


def _store(tmp_path) -> FilesystemRunStore:
    return FilesystemRunStore(tmp_path / "runs")


def _goal(**overrides) -> GoalSpec:
    data: dict = {
        "goal": "minimize (x-2)^2",
        "name": "loop-supervision",
        "objective": ObjectiveSpec(metric="loss", mode="min"),
        "search_space": {"x": {"type": "uniform", "low": -5.0, "high": 5.0}},
        "workload": WorkloadSpec(entrypoint="python toy.py"),
        "budget": BudgetSpec(max_trials=6, max_parallel=2),
        "strategy": StrategySpec(name="adaptive", seed=3),
    }
    data.update(overrides)
    return GoalSpec(**data)


def test_the_loop_supervises_the_trials_it_is_driving(tmp_path, monkeypatch):
    """The call site: every iteration hands supervise_once the run ids of
    trials still in flight -- not the finished ones, not the campaign id."""
    calls: list[list[str]] = []

    def _spy(run_store, run_ids, notifier=None):
        calls.append(list(run_ids))
        return SupervisionReport()

    monkeypatch.setattr("ai_experiments.loop.supervise_once", _spy)

    class SlowBackend(FakeBackend):
        def inspect(self, run_id):  # never finishes -- keeps trials in flight
            return self.store.read_status(run_id)

    store = _store(tmp_path)
    orchestrator = CampaignOrchestrator(
        store,
        CampaignStore(store.root),
        backend_factory=lambda goal: SlowBackend(store),
    )

    report = run_loop(
        _goal(),
        store,
        orchestrator=orchestrator,
        interval_seconds=0,
        max_seconds=0.2,
    )

    assert calls, "run_loop never supervised the runs it was driving"

    state = CampaignStore(store.root).read_state(report.campaign_id)
    expected = {t.run_id for t in state.trials if t.status in ACTIVE_TRIAL_STATES}
    assert expected, (
        "the campaign must still have trials in flight for this test to mean anything"
    )
    assert set(calls[-1]) == expected


def _running_run(
    store: FilesystemRunStore,
    monitoring: MonitorPolicy,
    **status_overrides: object,
) -> str:
    manifest = ExperimentManifest(
        experiment="loop-supervision-test",
        backend="local",
        workload=WorkloadSpec(entrypoint="python train.py"),
        monitoring=monitoring,
    )
    run_id, run_dir = store.create_run(manifest)
    base: dict[str, object] = {
        "started_at": utc_now() - timedelta(minutes=10),
        "details": {"heartbeat_at": utc_now().isoformat()},
    }
    base.update(status_overrides)
    store.write_handle(
        RunHandle(
            run_id=run_id,
            backend="local",
            status="running",
            status_uri=str(store.status_path(run_id)),
            run_dir=str(run_dir),
        )
    )
    store.update_status(run_id, **base)
    return run_id


def test_supervise_once_auto_kills_a_timed_out_run_like_the_daemon_does(tmp_path):
    """A behaviour test, not just a call-site spy: driven directly, without
    a daemon in sight, supervise_once must reach the same verdict the daemon
    path reaches for the same fatal condition."""
    store = _store(tmp_path)
    run_id = _running_run(
        store, MonitorPolicy(timeout_seconds=60, auto_kill=True), pid=None
    )

    report = supervise_once(store, [run_id])

    assert len(report.actions) == 1
    assert report.actions[0].run_id == run_id
    assert report.actions[0].action == "auto_killed"
    assert "timeout_exceeded" in report.actions[0].reasons
    assert store.read_status(run_id).status == "cancelled"
    assert report.errors == []


def test_supervise_once_reports_a_run_whose_check_raises_instead_of_hiding_it(
    tmp_path, monkeypatch
):
    """A backend that raises on every diagnose used to fill TickReport.errors
    before the extraction. supervise_once must still surface it -- a pass that
    saw nothing wrong must not be indistinguishable from one that blew up on
    every run it looked at."""
    store = _store(tmp_path)
    run_id = _running_run(store, MonitorPolicy())

    def _boom(run_store, run_id, ladder):
        raise RuntimeError("backend exploded")

    monkeypatch.setattr("ai_experiments.daemon._check_run", _boom)

    report = supervise_once(store, [run_id])

    assert report.actions == []
    assert len(report.errors) == 1
    assert run_id in report.errors[0]
    assert "backend exploded" in report.errors[0]


def test_the_daemons_tick_still_reports_a_run_whose_check_raises(tmp_path, monkeypatch):
    """The same fault, reached through MonitorDaemon.tick() instead of
    supervise_once directly -- the daemon must still fold supervise_once's
    errors into its own TickReport.errors."""
    store = _store(tmp_path)
    run_id = _running_run(store, MonitorPolicy())

    def _boom(run_store, run_id, ladder):
        raise RuntimeError("backend exploded")

    monkeypatch.setattr("ai_experiments.daemon._check_run", _boom)

    daemon = MonitorDaemon(store)
    tick_report = daemon.tick()

    assert len(tick_report.errors) == 1
    assert run_id in tick_report.errors[0]
    assert "backend exploded" in tick_report.errors[0]


def test_the_loops_report_carries_a_fatal_action_supervision_took(tmp_path):
    """A loop iteration over a run the rules call fatal ends with that action
    in report.supervision -- the loop must not throw away what it did.

    The run is backdated the same way `_running_run` backdates one, rather
    than relying on a real-time timeout race: a budget of exactly one trial
    means that once supervision cancels it, the campaign has no more work and
    finishes on its own, so the loop needs no `max_rounds` guess to stop.
    """

    class SlowBackend(FakeBackend):
        def inspect(self, run_id):  # never finishes on its own
            return self.store.read_status(run_id)

    store = _store(tmp_path)
    orchestrator = CampaignOrchestrator(
        store,
        CampaignStore(store.root),
        backend_factory=lambda goal: SlowBackend(store),
    )

    goal = _goal(
        monitoring=MonitorPolicy(timeout_seconds=60, auto_kill=True),
        budget=BudgetSpec(max_trials=1, max_parallel=1),
    )
    state = orchestrator.start(goal)
    (run_id,) = [t.run_id for t in state.trials if t.run_id]
    store.update_status(run_id, started_at=utc_now() - timedelta(minutes=10))

    report = run_loop(
        goal,
        store,
        orchestrator=orchestrator,
        campaign_id=state.campaign_id,
        interval_seconds=0,
        max_seconds=5.0,
    )

    assert report.supervision, "the loop's fatal action never made it into the report"
    assert any(action.action == "auto_killed" for action in report.supervision)
