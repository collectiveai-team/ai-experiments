"""The environment a workload inherits, and where its output is."""

from __future__ import annotations

from ai_experiments.schemas import ExperimentManifest, WorkloadSpec
from ai_experiments.worker import workload_env


def _manifest(**workload) -> ExperimentManifest:
    return ExperimentManifest(
        experiment="env",
        workload=WorkloadSpec(entrypoint="python train.py", **workload),
    )


def test_the_harness_virtualenv_does_not_follow_the_workload():
    """`uv run` in a workload warns and ignores an inherited VIRTUAL_ENV.

    The harness runs inside its own virtualenv. A workload that manages its
    own environment resolves against the wrong interpreter, or at best
    prints the warning on every trial of every campaign.
    """
    env = workload_env(
        {"VIRTUAL_ENV": "/harness/.venv", "PATH": "/usr/bin"}, _manifest()
    )

    assert "VIRTUAL_ENV" not in env
    assert env["PATH"] == "/usr/bin"


def test_a_workload_that_wants_a_virtualenv_asks_for_one():
    env = workload_env(
        {"VIRTUAL_ENV": "/harness/.venv"},
        _manifest(env={"VIRTUAL_ENV": "/workload/.venv"}),
    )

    assert env["VIRTUAL_ENV"] == "/workload/.venv"


def test_the_workload_env_still_wins_over_the_inherited_one():
    env = workload_env({"SEED": "1"}, _manifest(env={"SEED": "7"}))

    assert env["SEED"] == "7"
