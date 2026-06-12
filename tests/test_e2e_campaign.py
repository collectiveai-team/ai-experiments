"""End-to-end: a real campaign on the local backend with a toy objective.

Each trial is a real detached subprocess that reports IAX_METRIC lines; the
orchestrator plans, submits, collects, and stops on budget exhaustion.
"""

from __future__ import annotations

import sys
import time
import textwrap

import pytest

from ai_experiments.orchestrator import CampaignOrchestrator
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
