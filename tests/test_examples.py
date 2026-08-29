"""The shipped examples must run, or they teach the wrong thing.

`examples/` is what a new user copies and what the skills point at. A goal
file that no longer validates, or a workload that stopped printing metrics,
is a broken first experience — so the suite runs them.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from ai_experiments.schemas import GoalSpec

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
GOAL_FILES = sorted(EXAMPLES.glob("goal_*.yaml"))


@pytest.mark.parametrize("path", GOAL_FILES, ids=lambda p: p.stem)
def test_every_example_goal_validates(path):
    goal = GoalSpec.from_yaml(path)
    assert goal.objective.metric
    assert goal.search_space


@pytest.mark.parametrize("path", GOAL_FILES, ids=lambda p: p.stem)
def test_every_example_goal_points_at_a_workload_that_exists(path):
    goal = GoalSpec.from_yaml(path)
    script = next(
        (part for part in goal.workload.entrypoint.split() if part.endswith(".py")),
        None,
    )
    if script is None:
        script = next((a for a in goal.workload.args if a.endswith(".py")), None)
    assert script is not None, f"{path.name} names no python workload"
    assert (EXAMPLES.parent / script).is_file()


def _load(script: str):
    """Import an example workload without making `examples/` a package."""
    spec = importlib.util.spec_from_file_location(script[:-3], EXAMPLES / script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(script: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(EXAMPLES / script), *args],
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_the_agentic_workload_reports_the_metric_the_goal_asks_for():
    goal = GoalSpec.from_yaml(EXAMPLES / "goal_agentic.yaml")

    result = _run(
        "agentic_train.py",
        "--lr",
        "0.0025",
        "--width",
        "256",
        "--depth",
        "4",
        "--batch",
        "32",
        "--steps",
        "5",
        "--sleep",
        "0",
    )

    assert result.returncode == 0, result.stderr
    points = [
        json.loads(line.split("IAX_METRIC ", 1)[1])
        for line in result.stdout.splitlines()
        if line.startswith("IAX_METRIC ")
    ]
    assert points
    assert all(goal.objective.metric in point for point in points)


def test_the_agentic_workload_fails_loudly_on_a_configuration_that_cannot_run():
    """The failure is the lesson: a planner that cannot read it learns nothing."""
    result = _run(
        "agentic_train.py",
        "--width",
        "512",
        "--depth",
        "8",
        "--batch",
        "128",
        "--steps",
        "1",
        "--sleep",
        "0",
    )

    assert result.returncode != 0
    assert "out of memory" in result.stderr
    assert "width=512" in result.stderr


def test_the_agentic_target_is_reachable_inside_its_own_search_space():
    """A target no feasible configuration can meet makes the example a trap."""
    import random

    from ai_experiments.planner.search_space import sample

    workload = _load("agentic_train.py")
    goal = GoalSpec.from_yaml(EXAMPLES / "goal_agentic.yaml")
    rng = random.Random(0)
    feasible = []
    for _ in range(2000):
        params = sample(goal.search_space, rng)
        fits = (
            workload.memory_gb(params["width"], params["depth"], params["batch"])
            <= workload.DEVICE_MEMORY_GB
        )
        if fits:
            feasible.append(
                workload.loss_surface(params["lr"], params["width"], params["depth"])
            )

    assert feasible, "no configuration in the search space fits the device"
    assert min(feasible) < goal.objective.target


def test_a_failed_trial_carries_the_reason_the_workload_gave(tmp_path):
    """The whole point of `strategy: agent` is a planner that reads failures."""
    import sys as _sys

    from ai_experiments.orchestrator import CampaignOrchestrator
    from ai_experiments.store import FilesystemRunStore
    from ai_experiments.store.campaign import CampaignStore

    script = tmp_path / "fails.py"
    script.write_text(
        "import sys\n"
        "print('loading data', flush=True)\n"
        "raise RuntimeError('out of memory: needs 25.2 GB')\n"
    )
    goal = GoalSpec(
        goal="fail on purpose",
        name="failing",
        objective={"metric": "loss", "mode": "min"},
        search_space={"x": {"type": "uniform", "low": 0.0, "high": 1.0}},
        workload={
            "entrypoint": _sys.executable,
            "args": [str(script)],
            "working_dir": str(tmp_path),
        },
        budget={"max_trials": 1, "max_parallel": 1},
        monitoring={"interval_seconds": 1, "stuck_after_minutes": 5},
    )
    store = FilesystemRunStore(tmp_path / "runs")
    orchestrator = CampaignOrchestrator(store, CampaignStore(store.root))
    state = orchestrator.start(goal)

    for _ in range(60):
        state = orchestrator.advance(state.campaign_id)
        if state.trials[0].status in {"failed", "cancelled"}:
            break
        time.sleep(0.2)

    assert state.trials[0].status == "failed"
    assert "out of memory" in (state.trials[0].error or "")
