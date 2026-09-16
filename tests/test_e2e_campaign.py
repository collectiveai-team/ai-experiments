"""End-to-end: a real campaign on the local backend with a toy objective.

Each trial is a real detached subprocess that reports IAX_METRIC progress
lines and one IAX_RESULT; the orchestrator plans, submits, collects, and
stops on budget exhaustion.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import textwrap
from pathlib import Path

import pytest

from ai_experiments.orchestrator import CampaignOrchestrator
from ai_experiments.report import parse_result_line
from ai_experiments.schemas import (
    BudgetSpec,
    GoalSpec,
    MonitorPolicy,
    ObjectiveSpec,
    StrategySpec,
    WorkloadSpec,
)
from ai_experiments.store import FilesystemRunStore
from ai_experiments.store.campaign import CampaignStore

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"

TOY_SCRIPT = textwrap.dedent(
    """
    import argparse, json, sys

    parser = argparse.ArgumentParser()
    parser.add_argument("--x", type=float, required=True)
    args = parser.parse_args()

    for step in range(3):
        loss = (args.x - 2.0) ** 2 + 0.1 / (step + 1)
        print("IAX_METRIC " + json.dumps({"step": step, "loss": loss}))
        sys.stdout.flush()

    print("IAX_RESULT " + json.dumps({"loss": loss}))
    sys.stdout.flush()
    """
)


def test_local_campaign_end_to_end(tmp_path):
    script = tmp_path / "toy_train.py"
    script.write_text(TOY_SCRIPT)

    goal = GoalSpec(
        goal="minimize (x-2)^2 on a real subprocess",
        name="e2e-quadratic",
        objective=ObjectiveSpec(metric="loss", mode="min"),
        search_space={"x": {"type": "uniform", "low": 0.0, "high": 4.0}},
        workload=WorkloadSpec(
            entrypoint=f"{sys.executable} {script}",
            working_dir=str(tmp_path),
        ),
        budget=BudgetSpec(max_trials=3, max_parallel=2),
        strategy=StrategySpec(name="random", seed=42),
        monitoring=MonitorPolicy(interval_seconds=1, stuck_after_minutes=5),
    )

    store = FilesystemRunStore(tmp_path / "runs")
    orchestrator = CampaignOrchestrator(store, CampaignStore(store.root))
    state = orchestrator.start(goal)

    deadline = time.monotonic() + 90
    while state.status != "completed" and time.monotonic() < deadline:
        time.sleep(0.3)
        state = orchestrator.advance(state.campaign_id)

    assert state.status == "completed", f"campaign stuck: {state.model_dump()}"
    assert state.stop_reason == "budget_exhausted"
    completed = [t for t in state.trials if t.status == "completed"]
    assert len(completed) == 3

    for trial in completed:
        assert trial.run_id is not None
        assert trial.objective_value is not None
        # objective = best observed loss for the x the workload actually saw
        # (CLI args are rendered with .6g precision)
        x_seen = float(format(trial.params["x"], ".6g"))
        expected = (x_seen - 2.0) ** 2 + 0.1 / 3
        assert trial.objective_value == pytest.approx(expected, rel=1e-6)
        assert len(store.read_metrics(trial.run_id)) == 3

    assert state.best_trial_id is not None
    best = next(t for t in state.trials if t.trial_id == state.best_trial_id)
    assert best.objective_value == min(t.objective_value for t in completed)


def test_the_shipped_example_scores_only_from_its_evaluator(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    env = {**os.environ, "IAX_WORK_DIR": str(work)}

    train = subprocess.run(
        [
            sys.executable,
            str(EXAMPLES / "toy_train.py"),
            "--steps",
            "3",
            "--sleep",
            "0",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert train.returncode == 0
    assert "IAX_RESULT" not in train.stdout
    assert "IAX_METRIC" in train.stdout

    evaluate = subprocess.run(
        [sys.executable, str(EXAMPLES / "toy_evaluate.py")],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert evaluate.returncode == 0
    model = json.loads((work / "model.json").read_text())
    results = [parse_result_line(line) for line in evaluate.stdout.splitlines()]
    results = [values for values in results if values]
    assert len(results) == 1, "the evaluator must declare exactly one result"
    assert results[0]["loss"] == pytest.approx((model["x"] - 2.0) ** 2)


def test_the_shipped_goal_declares_two_phases():
    goal = GoalSpec.from_yaml(EXAMPLES / "goal_toy.yaml")

    phases = dict(goal.workload.phases())
    assert list(phases) == ["train", "evaluate"]
    assert "toy_train.py" in phases["train"]
    assert "toy_evaluate.py" in phases["evaluate"]
    assert not goal.workload.args, (
        "top-level args are appended to every phase, the evaluator included"
    )


def test_the_harness_carries_the_handoff_between_the_two_phases(tmp_path):
    """The trainer's artifact, not the trainer's word, is what gets scored."""
    goal = GoalSpec(
        goal="the shipped two-phase example, driven by the harness",
        name="e2e-two-phase",
        objective=ObjectiveSpec(metric="loss", mode="min"),
        search_space={"lr": {"type": "uniform", "low": 0.1, "high": 0.4}},
        workload=WorkloadSpec(
            entrypoint=f"{sys.executable} {EXAMPLES / 'toy_train.py'}",
            train=f"{sys.executable} {EXAMPLES / 'toy_train.py'} --steps 5 --sleep 0",
            evaluate=f"{sys.executable} {EXAMPLES / 'toy_evaluate.py'}",
            working_dir=str(tmp_path),
        ),
        budget=BudgetSpec(max_trials=2, max_parallel=2),
        strategy=StrategySpec(name="random", seed=42),
        monitoring=MonitorPolicy(interval_seconds=1, stuck_after_minutes=5),
    )

    store = FilesystemRunStore(tmp_path / "runs")
    orchestrator = CampaignOrchestrator(store, CampaignStore(store.root))
    state = orchestrator.start(goal)

    deadline = time.monotonic() + 90
    while state.status != "completed" and time.monotonic() < deadline:
        time.sleep(0.3)
        state = orchestrator.advance(state.campaign_id)

    assert state.status == "completed", f"campaign stuck: {state.model_dump()}"
    completed = [t for t in state.trials if t.status == "completed"]
    assert len(completed) == 2

    for trial in completed:
        # The scored number has to be derivable from the artifact the trainer
        # wrote -- that is the whole point of running the evaluator separately.
        model = json.loads(
            (store.run_dir(trial.run_id) / "work" / "model.json").read_text()
        )
        assert trial.objective_value == pytest.approx((model["x"] - 2.0) ** 2)
        assert len(store.read_results(trial.run_id)) == 1
        assert len(store.read_metrics(trial.run_id)) == 5
        assert not [
            event
            for event in store.read_events(trial.run_id)
            if "train phase" in event.message
        ]
