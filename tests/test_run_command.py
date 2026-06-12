"""`iax run` — the single-command experiment loop."""

from __future__ import annotations

import sys
import textwrap

import yaml
from typer.testing import CliRunner

from ai_experiments.cli import app

runner = CliRunner()

TOY_SCRIPT = textwrap.dedent(
    """
    import argparse, json, sys

    parser = argparse.ArgumentParser()
    parser.add_argument("--x", type=float, required=True)
    args = parser.parse_args()
    print("IAX_METRIC " + json.dumps({"step": 0, "loss": (args.x - 2.0) ** 2}))
    sys.stdout.flush()
    """
)


def _goal_file(tmp_path) -> str:
    script = tmp_path / "toy.py"
    script.write_text(TOY_SCRIPT)
    goal = {
        "goal": "minimize quadratic via iax run",
        "name": "run-cmd",
        "objective": {"metric": "loss", "mode": "min"},
        "search_space": {"x": {"type": "uniform", "low": 0.0, "high": 4.0}},
        "workload": {
            "entrypoint": f"{sys.executable} {script}",
            "working_dir": str(tmp_path),
        },
        "budget": {"max_trials": 2, "max_parallel": 2},
        "strategy": {"name": "random", "seed": 1},
    }
    path = tmp_path / "goal.yaml"
    path.write_text(yaml.safe_dump(goal))
    return str(path)


def test_run_command_drives_campaign_to_completion(tmp_path):
    goal_path = _goal_file(tmp_path)

    result = runner.invoke(
        app,
        [
            "run",
            goal_path,
            "--no-serve",
            "--interval",
            "1",
            "--runs-dir",
            str(tmp_path / "runs"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "budget_exhausted" in result.output
    assert "Best:" in result.output


def test_run_command_rejects_invalid_goal(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("goal: ''\n")

    result = runner.invoke(app, ["run", str(bad), "--no-serve"])

    assert result.exit_code == 1
    assert "invalid goal" in result.output
