"""The unattended path is the one that most needs supervision.

`iax loop` (`run_loop`) drives a campaign with nobody watching between
rounds. `supervise_once`, extracted from the daemon's per-tick run check,
must run on every loop iteration against the campaign's in-flight trials --
and, called directly, must behave exactly as the daemon's own check does.
"""

from __future__ import annotations

import json
from datetime import timedelta

from ai_experiments.agents.contracts import AgentResult
from ai_experiments.daemon import MonitorDaemon
from ai_experiments.loop import _apply_changes, run_loop
from ai_experiments.monitoring.escalation import CAMPAIGN_PREFIX, CampaignReview
from ai_experiments.monitoring.supervision import SupervisionReport, supervise_once
from ai_experiments.orchestrator import ACTIVE_TRIAL_STATES, CampaignOrchestrator
from ai_experiments.schemas import (
    BudgetSpec,
    CampaignState,
    ExperimentManifest,
    GoalSpec,
    LogUniformParam,
    MonitorPolicy,
    ObjectiveSpec,
    RunHandle,
    StrategySpec,
    UniformParam,
    WorkloadSpec,
    utc_now,
)
from ai_experiments.store import FilesystemRunStore
from ai_experiments.store.campaign import CampaignStore
from tests.conftest import FakeBackend


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
    """Every iteration hands supervise_once the run ids of trials still in flight.

    Not the finished ones, not the campaign id.

    Task 12 moved this call to right after the cohort-closing
    `advance(admit=False)`, before the trailing `advance()` that admits the
    next cohort -- reviewing what that trailing call just submitted is
    exactly the timing hole this branch closes. So a trial it submits is not
    supervised until the *next* iteration's refresh, by design. Each call is
    therefore checked against the campaign's active trials as they stood the
    instant that call was made, not against the campaign's final trials,
    which can include one submitted after the last call ever happened.
    """
    calls: list[list[str]] = []
    active_at_call: list[set[str]] = []

    # The orchestrator's backend_factory constructs a fresh backend on every
    # advance()/reconcile() call, so "first run id ever seen" cannot live on
    # `self` -- it has to survive across instances, hence this closure list.
    finished_run_id: list[str] = []

    class SlowBackend(FakeBackend):
        """The first run id ever inspected finishes for real; every later one stays in flight.

        Across every instance this backend factory constructs -- a genuinely mixed campaign state,
        so the active-state filter has something to filter out.
        """

        def inspect(self, run_id):
            if not finished_run_id:
                finished_run_id.append(run_id)
            if run_id == finished_run_id[0]:
                return super().inspect(run_id)
            return self.store.read_status(run_id)

    store = _store(tmp_path)
    campaign_store = CampaignStore(store.root)
    orchestrator = CampaignOrchestrator(
        store,
        campaign_store,
        backend_factory=lambda goal: SlowBackend(store),
    )

    def _spy(run_store, run_ids, notifier=None):
        calls.append(list(run_ids))
        campaigns = list(campaign_store.list_campaigns())
        state = campaign_store.read_state(campaigns[0]) if campaigns else None
        active_at_call.append(
            {t.run_id for t in state.trials if t.status in ACTIVE_TRIAL_STATES and t.run_id}
            if state is not None
            else set()
        )
        return SupervisionReport()

    monkeypatch.setattr("ai_experiments.loop.supervise_once", _spy)

    report = run_loop(
        _goal(),
        store,
        orchestrator=orchestrator,
        interval_seconds=0,
        max_seconds=0.2,
    )

    assert calls, "run_loop never supervised the runs it was driving"

    state = campaign_store.read_state(report.campaign_id)
    finished = {t.run_id for t in state.trials if t.status not in ACTIVE_TRIAL_STATES}
    assert finished, (
        "the fixture must finish at least one trial, or this test cannot tell "
        "the active-state filter from no filter at all"
    )
    assert active_at_call[-1], (
        "the campaign must still have trials in flight for this test to mean anything"
    )
    assert set(calls[-1]) == active_at_call[-1]


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
    """A behaviour test, not just a call-site spy.

    Driven directly, without a daemon in sight, supervise_once must reach the same verdict the
    daemon path reaches for the same fatal condition.
    """
    store = _store(tmp_path)
    run_id = _running_run(store, MonitorPolicy(timeout_seconds=60, auto_kill=True), pid=None)

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
    """A backend that raises on every diagnose must still surface in the errors.

    It used to fill TickReport.errors before the extraction. supervise_once must still surface it --
    a pass that saw nothing wrong must not be indistinguishable from one that blew up on every run
    it looked at.
    """
    store = _store(tmp_path)
    run_id = _running_run(store, MonitorPolicy())

    def _boom(run_store, run_id, ladder):
        raise RuntimeError("backend exploded")

    monkeypatch.setattr("ai_experiments.monitoring.supervision._check_run", _boom)

    report = supervise_once(store, [run_id])

    assert report.actions == []
    assert len(report.errors) == 1
    assert run_id in report.errors[0]
    assert "backend exploded" in report.errors[0]


def test_the_daemons_tick_still_reports_a_run_whose_check_raises(tmp_path, monkeypatch):
    """The same fault, reached through MonitorDaemon.tick() instead of supervise_once directly.

    The daemon must still fold supervise_once's errors into its own TickReport.errors.
    """
    store = _store(tmp_path)
    run_id = _running_run(store, MonitorPolicy())

    def _boom(run_store, run_id, ladder):
        raise RuntimeError("backend exploded")

    monkeypatch.setattr("ai_experiments.monitoring.supervision._check_run", _boom)

    daemon = MonitorDaemon(store)
    tick_report = daemon.tick()

    assert len(tick_report.errors) == 1
    assert run_id in tick_report.errors[0]
    assert "backend exploded" in tick_report.errors[0]


def test_the_loops_report_carries_a_fatal_action_supervision_took(tmp_path):
    """A fatal action taken during a loop iteration ends up in report.supervision.

    The loop must not throw away what it did.

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


def _run_a_reviewed_loop(tmp_path, reviewer):
    """Run a loop with one trial in flight at a time, reviewed between rounds.

    Driven entirely by an injected reviewer -- the fixture Task 12's timing test needs.
    """
    store = _store(tmp_path)
    orchestrator = CampaignOrchestrator(
        store,
        CampaignStore(store.root),
        backend_factory=lambda goal: FakeBackend(store),
        agent_runner_factory=lambda goal, campaign_id: reviewer,
    )
    goal = _goal(
        analysis={"review_between_rounds": True},
        budget=BudgetSpec(max_trials=6, max_parallel=1),
    )
    return run_loop(goal, store, orchestrator=orchestrator, interval_seconds=0)


def _evidence_lines(prompt: str) -> int:
    """How many *scored* trials the review prompt shows.

    `review_brief`'s evidence block lists one ``"- <trial_id> ..."`` line per
    trial that already has an objective value or an error -- a trial that was
    merely submitted this tick has neither, so it cannot inflate this count.
    Scoped to the text after the "Evidence so far" marker so a `-` in some
    other section (e.g. a negative search-space bound) can never be mistaken
    for one.
    """
    _, _, evidence = prompt.partition("Evidence so far")
    return evidence.count("\n- ")


def test_the_review_sees_the_cohort_before_the_next_one_is_submitted(tmp_path):
    """An agent that says 'stop' must be able to stop something.

    Reviewing after `advance` had already filled capacity meant the verdict
    arrived when the next cohort was submitted and paid for.
    """
    submitted_at_review: list[int] = []

    class _Reviewer:
        def run(self, prompt, *, role="planner"):
            submitted_at_review.append(_evidence_lines(prompt))
            return AgentResult(ok=True, payload={"verdict": "stop", "reason": "done"})

    report = _run_a_reviewed_loop(tmp_path, _Reviewer())

    assert report.loop_stop == "agent_review_stop"
    assert report.trials == submitted_at_review[0], (
        "the review ran after a new cohort had already been submitted"
    )


def test_advance_with_admit_false_submits_nothing(tmp_path):
    """`admit=False` must refresh and score, never call `_fill_capacity`.

    Even when there is spare capacity and budget to fill it with.
    """
    store = _store(tmp_path)
    orchestrator = CampaignOrchestrator(
        store,
        CampaignStore(store.root),
        backend_factory=lambda goal: FakeBackend(store),
    )
    goal = _goal(
        budget=BudgetSpec(max_trials=6, max_parallel=2),
        strategy=StrategySpec(name="adaptive", seed=3, batch_size=1),
    )
    state = orchestrator.start(goal)
    before = len(state.trials)
    assert before < goal.budget.max_parallel, (
        "the fixture needs spare capacity, or this call proves nothing"
    )

    state = orchestrator.advance(state.campaign_id, admit=False)

    assert len(state.trials) == before


def test_admit_false_still_escalates_a_finished_trial_under_agent_review(tmp_path):
    """`analysis.agent_review` must survive the cohort-closing `admit=False` pass.

    Not just the trailing `admit=True` one.

    `finished_now` is scored on the `admit=False` call and is always empty by
    the time the trailing `admit=True` call runs, so firing the escalation
    only from the block below `_fill_capacity` left this channel dead for
    every campaign `run_loop` drives with `analysis.agent_review` on -- the
    method was never broken, only unreachable. This asserts on the file
    `_request_agent_review` actually writes, not on a spy, because a spy on
    `_request_agent_review` would have kept passing through that regression.
    """
    store = _store(tmp_path)
    orchestrator = CampaignOrchestrator(
        store,
        CampaignStore(store.root),
        backend_factory=lambda goal: FakeBackend(store),
    )
    goal = _goal(
        analysis={"agent_review": True},
        budget=BudgetSpec(max_trials=2, max_parallel=1),
        strategy=StrategySpec(name="adaptive", seed=3, batch_size=1),
    )

    report = run_loop(goal, store, orchestrator=orchestrator, interval_seconds=0)

    path = store.root / "_escalations" / f"{CAMPAIGN_PREFIX}{report.campaign_id}.json"
    assert path.exists(), (
        "a trial finished under analysis.agent_review but run_loop never escalated it"
    )
    review = CampaignReview(**json.loads(path.read_text()))
    assert review.campaign_id == report.campaign_id


def _started_campaign(tmp_path, goal: GoalSpec) -> tuple[CampaignOrchestrator, CampaignState]:
    """Build an orchestrator with one live campaign, for exercising `_apply_changes`.

    Directly against the real `edit_goal` validation path.
    """
    store = _store(tmp_path)
    orchestrator = CampaignOrchestrator(
        store,
        CampaignStore(store.root),
        backend_factory=lambda goal: FakeBackend(store),
    )
    state = orchestrator.start(goal)
    return orchestrator, state


def test_the_agent_can_widen_the_search_space(tmp_path):
    goal = _goal()
    orchestrator, state = _started_campaign(tmp_path, goal)

    _apply_changes(
        orchestrator,
        state,
        goal,
        {
            "suggested_changes": {
                "search_space": {"lr": {"type": "loguniform", "low": 1e-5, "high": 1.0}}
            }
        },
    )

    updated = orchestrator.campaign_store.read_goal(state.campaign_id)
    assert updated.search_space == {
        "x": UniformParam(type="uniform", low=-5.0, high=5.0),
        "lr": LogUniformParam(type="loguniform", low=1e-5, high=1.0),
    }, "the merge must keep the pre-existing parameter, not just add the new one"


def test_the_agent_cannot_widen_its_own_budget(tmp_path):
    """An optimizer asked to stay under a ceiling will ask to raise it."""
    goal = _goal(budget=BudgetSpec(max_trials=6, max_parallel=2))
    orchestrator, state = _started_campaign(tmp_path, goal)

    _apply_changes(
        orchestrator,
        state,
        goal,
        {"suggested_changes": {"budget": {"max_trials": goal.budget.max_trials * 100}}},
    )

    updated = orchestrator.campaign_store.read_goal(state.campaign_id)
    assert updated.budget == goal.budget, (
        "the whole budget must round-trip unchanged, not just max_trials"
    )


def test_a_rejected_budget_change_is_recorded(tmp_path):
    goal = _goal()
    orchestrator, state = _started_campaign(tmp_path, goal)
    payload = {"suggested_changes": {"budget": {"max_gpu_hours": 10_000}}}

    _apply_changes(orchestrator, state, goal, payload)

    events = orchestrator.campaign_store.read_events(state.campaign_id)
    refusals = [event for event in events if "budget" in event.details.get("refused", {})]
    assert refusals, "the refused budget change was never recorded in the campaign's events"
    assert refusals[0].details["refused"] == payload["suggested_changes"]


def test_a_malformed_suggested_changes_is_recorded(tmp_path):
    """A malformed `suggested_changes` must not vanish.

    A review that returns it as something other than a mapping is malformed, not merely unhelpful.
    """
    goal = _goal()
    orchestrator, state = _started_campaign(tmp_path, goal)

    _apply_changes(orchestrator, state, goal, {"suggested_changes": "widen everything"})

    events = orchestrator.campaign_store.read_events(state.campaign_id)
    malformed = [event for event in events if "suggested_changes" in event.details]
    assert malformed, "a malformed suggested_changes was never recorded"
    assert malformed[0].details["suggested_changes"] == "widen everything"


def test_an_absent_suggested_changes_records_nothing(tmp_path):
    """A review with nothing to suggest is the normal case, not a malformed one.

    It must stay silent rather than add log noise.
    """
    goal = _goal()
    orchestrator, state = _started_campaign(tmp_path, goal)
    before = orchestrator.campaign_store.read_events(state.campaign_id)

    _apply_changes(orchestrator, state, goal, {})

    assert orchestrator.campaign_store.read_events(state.campaign_id) == before


def test_the_loop_reports_what_supervision_could_not_do(tmp_path, monkeypatch):
    """The loop lifts `errors` from `supervise_once` beside `actions`.

    Dropping the other half means an unattended `iax loop` whose backend raises on every
    `diagnose()` returns a report that looks exactly like a night on which everything was fine.
    """

    class StuckBackend(FakeBackend):
        """Nothing ever finishes, so there are always trials to supervise."""

        def inspect(self, run_id):
            return self.store.read_status(run_id)

        def diagnose(self, run_id):
            raise RuntimeError("ray dashboard unreachable")

    store = _store(tmp_path)
    orchestrator = CampaignOrchestrator(
        store,
        CampaignStore(store.root),
        backend_factory=lambda goal: StuckBackend(store),
    )
    monkeypatch.setattr(
        "ai_experiments.monitoring.supervision.backend_for_run",
        lambda run_store, run_id: StuckBackend(run_store),
    )

    report = run_loop(
        _goal(),
        store,
        orchestrator=orchestrator,
        interval_seconds=0,
        max_seconds=0.2,
    )

    assert report.supervision_errors, (
        "every diagnose() raised and the loop's report says supervision had nothing to report"
    )
    assert any("ray dashboard unreachable" in error for error in report.supervision_errors)


def test_a_healthy_loop_reports_no_supervision_errors(tmp_path):
    """The empty list has to mean "nothing went wrong", not "nobody looked"."""
    store = _store(tmp_path)
    orchestrator = CampaignOrchestrator(
        store,
        CampaignStore(store.root),
        backend_factory=lambda goal: FakeBackend(store),
    )

    report = run_loop(
        _goal(),
        store,
        orchestrator=orchestrator,
        interval_seconds=0,
        max_seconds=5,
    )

    assert report.supervision_errors == []
