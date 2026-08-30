"""`iax loop`: one goal in, one answer out.

This is the command an agent drives. It must terminate on its own, say
plainly whether the target was reached, and never report success for a
campaign that only ran out of budget.
"""

from __future__ import annotations

import json
import sys

import yaml
from typer.testing import CliRunner

from ai_experiments.agents import StubAgentRunner
from ai_experiments.cli import app
from ai_experiments.loop import run_loop
from ai_experiments.orchestrator import CampaignOrchestrator
from ai_experiments.schemas import GoalSpec
from ai_experiments.store import FilesystemRunStore
from ai_experiments.store.campaign import CampaignStore
from tests.test_orchestrator import FakeBackend

runner = CliRunner()


def _goal(**overrides) -> GoalSpec:
    data: dict = {
        "goal": "minimize (x-2)^2",
        "name": "loop",
        "objective": {"metric": "loss", "mode": "min", "target": 0.01},
        "search_space": {"x": {"type": "uniform", "low": -5.0, "high": 5.0}},
        "workload": {"entrypoint": "python toy.py"},
        "budget": {"max_trials": 40, "max_parallel": 2},
        "strategy": {"name": "adaptive", "seed": 5},
    }
    data.update(overrides)
    return GoalSpec(**data)


def _harness(tmp_path, agent_runner=None):
    store = FilesystemRunStore(tmp_path / "runs")
    backend = FakeBackend(store)
    orchestrator = CampaignOrchestrator(
        store,
        CampaignStore(store.root),
        backend_factory=lambda goal: backend,
        agent_runner_factory=(lambda goal, cid: agent_runner) if agent_runner else None,
    )
    return orchestrator, store


def test_the_loop_reaches_the_target_and_says_so(tmp_path):
    orchestrator, store = _harness(tmp_path)

    report = run_loop(_goal(), store, orchestrator=orchestrator, interval_seconds=0)

    assert report.target_reached
    assert report.stop_reason == "target_reached"
    assert report.best is not None
    assert report.best["objective_value"] <= 0.01
    assert report.rounds >= 1
    assert report.history


def test_the_loop_does_not_claim_success_when_the_budget_runs_out(tmp_path):
    """An unattended loop that reports success it did not earn is worse than none."""
    orchestrator, store = _harness(tmp_path)
    goal = _goal(objective={"metric": "loss", "mode": "min", "target": 1e-12})

    report = run_loop(goal, store, orchestrator=orchestrator, interval_seconds=0)

    assert not report.target_reached
    assert report.stop_reason != "target_reached"
    assert report.best is not None


def test_max_rounds_stops_the_loop_without_stopping_the_campaign(tmp_path):
    orchestrator, store = _harness(tmp_path)

    report = run_loop(
        _goal(), store, orchestrator=orchestrator, max_rounds=2, interval_seconds=0
    )

    assert report.loop_stop == "max_rounds"
    assert report.rounds <= 2
    state = CampaignStore(store.root).read_state(report.campaign_id)
    assert state.status == "running"


def test_a_stopped_loop_resumes_where_it_left_off(tmp_path):
    orchestrator, store = _harness(tmp_path)
    first = run_loop(
        _goal(), store, orchestrator=orchestrator, max_rounds=1, interval_seconds=0
    )

    second = run_loop(
        _goal(),
        store,
        orchestrator=orchestrator,
        campaign_id=first.campaign_id,
        interval_seconds=0,
    )

    assert second.campaign_id == first.campaign_id
    assert second.trials >= first.trials
    assert second.target_reached


def test_the_loop_never_sleeps_more_than_it_iterates(tmp_path):
    orchestrator, store = _harness(tmp_path)
    slept: list[float] = []

    run_loop(
        _goal(),
        store,
        orchestrator=orchestrator,
        interval_seconds=7.0,
        sleep=slept.append,
        max_rounds=3,
    )

    assert all(pause == 7.0 for pause in slept)


def test_the_agent_review_can_end_a_hopeless_campaign(tmp_path):
    agent = StubAgentRunner(
        [
            {
                "verdict": "stop",
                "reason": "the workload ignores x, so no trial can move the loss",
            }
        ]
    )
    orchestrator, store = _harness(tmp_path, agent_runner=agent)
    goal = _goal(
        objective={"metric": "loss", "mode": "min", "target": 1e-12},
        analysis={"review_between_rounds": True},
    )

    report = run_loop(goal, store, orchestrator=orchestrator, interval_seconds=0)

    assert report.loop_stop == "agent_review_stop"
    assert not report.target_reached
    assert report.reviews[0]["verdict"] == "stop"
    assert "ignores x" in report.reviews[0]["reason"]


def test_a_review_the_agent_botches_does_not_stop_a_healthy_campaign(tmp_path):
    agent = StubAgentRunner(["I have no idea, honestly."])
    orchestrator, store = _harness(tmp_path, agent_runner=agent)
    goal = _goal(analysis={"review_between_rounds": True})

    report = run_loop(goal, store, orchestrator=orchestrator, interval_seconds=0)

    assert report.target_reached
    assert report.reviews[0]["agent_error"]


def test_an_accepted_review_can_widen_the_search_space(tmp_path):
    agent = StubAgentRunner(
        [
            {
                "verdict": "change_goal",
                "reason": "the optimum is outside the current bounds",
                "suggested_changes": {
                    "search_space": {
                        "x": {"type": "uniform", "low": -50.0, "high": 50.0}
                    }
                },
            }
        ]
    )
    orchestrator, store = _harness(tmp_path, agent_runner=agent)
    goal = _goal(
        analysis={"review_between_rounds": True, "apply_agent_changes": True},
        budget={"max_trials": 4, "max_parallel": 1},
    )

    report = run_loop(goal, store, orchestrator=orchestrator, interval_seconds=0)

    edited = CampaignStore(store.root).read_goal(report.campaign_id)
    assert edited.search_space["x"].high == 50.0


def test_a_review_cannot_change_the_objective_metric(tmp_path):
    """Recorded values would stop being comparable; the goal must survive."""
    agent = StubAgentRunner(
        [
            {
                "verdict": "change_goal",
                "reason": "measure accuracy instead",
                "suggested_changes": {"objective": {"metric": "accuracy"}},
            }
        ]
    )
    orchestrator, store = _harness(tmp_path, agent_runner=agent)
    goal = _goal(
        analysis={"review_between_rounds": True, "apply_agent_changes": True},
        budget={"max_trials": 4, "max_parallel": 1},
    )

    report = run_loop(goal, store, orchestrator=orchestrator, interval_seconds=0)

    assert CampaignStore(store.root).read_goal(report.campaign_id).objective.metric == (
        "loss"
    )


def test_reviews_are_recorded_as_round_records(tmp_path):
    from ai_experiments.improve.rounds import RoundLog

    agent = StubAgentRunner([{"verdict": "continue", "reason": "still improving"}])
    orchestrator, store = _harness(tmp_path, agent_runner=agent)

    report = run_loop(
        _goal(analysis={"review_between_rounds": True}),
        store,
        orchestrator=orchestrator,
        interval_seconds=0,
    )

    records = RoundLog(
        CampaignStore(store.root).campaign_dir(report.campaign_id)
    ).read()
    reviews = [r for r in records if r.stage == "review"]
    assert reviews and reviews[0].outcome["verdict"] == "continue"


def _real_goal_file(tmp_path):
    script = tmp_path / "train.py"
    script.write_text(
        "import json, os, sys\n"
        "params = json.loads(os.environ['IAX_PARAMS'])\n"
        "loss = (params['x'] - 2.0) ** 2\n"
        "print('IAX_METRIC ' + json.dumps({'step': 1, 'loss': loss}), flush=True)\n"
    )
    goal = {
        "goal": "minimize (x-2)^2",
        "name": "loop-cli",
        "objective": {"metric": "loss", "mode": "min", "target": 0.25},
        "search_space": {"x": {"type": "uniform", "low": 1.0, "high": 3.0}},
        "workload": {
            "entrypoint": sys.executable,
            "args": [str(script)],
            "working_dir": str(tmp_path),
        },
        "budget": {"max_trials": 12, "max_parallel": 2},
        "strategy": {"name": "adaptive", "seed": 11},
        "monitoring": {"interval_seconds": 1, "stuck_after_minutes": 5},
    }
    path = tmp_path / "goal.yaml"
    path.write_text(yaml.safe_dump(goal))
    return path


def test_loop_command_runs_a_real_local_campaign_to_target(tmp_path):
    """The whole loop over the local backend, no stubs below the CLI."""
    result = runner.invoke(
        app,
        [
            "loop",
            str(_real_goal_file(tmp_path)),
            "--runs-dir",
            str(tmp_path / "runs"),
            "--interval",
            "0",
            "--max-seconds",
            "120",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert report["target_reached"]
    assert report["best"]["objective_value"] <= 0.25
    assert report["campaign_id"].startswith("cmp_")


def test_loop_command_exits_four_when_the_target_is_missed(tmp_path):
    path = _real_goal_file(tmp_path)
    goal = yaml.safe_load(path.read_text())
    goal["objective"]["target"] = 1e-12
    goal["budget"]["max_trials"] = 2
    path.write_text(yaml.safe_dump(goal))

    result = runner.invoke(
        app,
        [
            "loop",
            str(path),
            "--runs-dir",
            str(tmp_path / "runs"),
            "--interval",
            "0",
            "--max-seconds",
            "120",
        ],
    )

    assert result.exit_code == 4
    assert "NOT reached" in result.stdout


def test_loop_command_rejects_an_invalid_goal(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("goal: only a goal\n")

    result = runner.invoke(app, ["loop", str(path)])

    assert result.exit_code == 2


def test_max_rounds_collects_the_round_it_already_paid_for(tmp_path):
    """A trial that finished before the limit must not be thrown away.

    `FakeBackend` completes a run on the next inspect, exactly like a real
    backend whose process ends between two ticks. Stopping without that final
    read left the trial `submitted` forever, and its measured value unread
    (found by a real campaign on vjepa-experiment: a GPU trial ran, completed,
    and never entered the leaderboard).
    """
    orchestrator, store = _harness(tmp_path)
    goal = _goal(objective={"metric": "loss", "mode": "min", "target": 1e-12})

    report = run_loop(
        goal, store, orchestrator=orchestrator, max_rounds=1, interval_seconds=0
    )

    state = CampaignStore(store.root).read_state(report.campaign_id)
    assert state.trials
    assert [t.status for t in state.trials] == ["completed"] * len(state.trials)
    assert all(t.objective_value is not None for t in state.trials)
    assert report.pending_trials == []
    assert report.best is not None


def test_the_report_names_the_trials_it_could_not_collect(tmp_path):
    """A campaign with work in flight is not an answer, and must not read as one."""

    class SlowBackend(FakeBackend):
        def inspect(self, run_id):  # never finishes
            return self.store.read_status(run_id)

    store = FilesystemRunStore(tmp_path / "runs")
    orchestrator = CampaignOrchestrator(
        store,
        CampaignStore(store.root),
        backend_factory=lambda goal: SlowBackend(store),
    )

    report = run_loop(
        _goal(), store, orchestrator=orchestrator, max_rounds=1, interval_seconds=0
    )

    assert report.loop_stop == "max_rounds"
    assert report.pending_trials
    assert not report.target_reached


def test_reconcile_finishes_a_campaign_whose_last_trial_met_the_target(tmp_path):
    """The target can be met by the very round the limit interrupted."""
    orchestrator, store = _harness(tmp_path)

    run_loop(
        _goal(), store, orchestrator=orchestrator, max_rounds=1, interval_seconds=0
    )
    campaign_id = CampaignStore(store.root).list_campaigns()[0]
    state = orchestrator.reconcile(campaign_id)

    assert state.status in {"running", "completed"}
    assert all(t.status != "submitted" for t in state.trials)
