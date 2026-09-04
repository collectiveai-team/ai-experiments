"""`iax new` must emit files that the harness accepts without editing (#19).

The wheel ships only `ai_experiments`, so a pip-installed user has no
`examples/` to copy. These tests are the guard that the shipped templates stay
in step with the schemas: a renamed or removed field breaks them here, not at
the user's first `iax validate`.
"""

from __future__ import annotations

import os
import subprocess
import sys
import sysconfig
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ai_experiments import scaffold
from ai_experiments.cli import app
from ai_experiments.preflight import workload_warnings
from ai_experiments.schemas import ExperimentManifest, GoalSpec

runner = CliRunner()


def test_manifest_template_validates_against_the_schema(tmp_path):
    path = tmp_path / "experiment.yaml"
    scaffold.write("manifest", path)

    manifest = ExperimentManifest.from_yaml(path)

    assert manifest.experiment
    assert manifest.workload.entrypoint


def test_goal_template_validates_against_the_schema(tmp_path):
    path = tmp_path / "goal.yaml"
    scaffold.write("goal", path)

    goal = GoalSpec.from_yaml(path)

    assert goal.search_space
    assert goal.objective.metric == "loss"
    # The goal validator mirrors the objective onto monitoring; a template that
    # names a metric nobody monitors would score nothing.
    assert goal.monitoring.objective_metric == "loss"


@pytest.mark.parametrize("kind", ["manifest", "goal"])
def test_template_entrypoint_resolves_without_the_harness_venv(
    tmp_path, kind, monkeypatch
):
    """A template that validates but cannot spawn is still a broken scaffold.

    `entrypoint: python` passed every check here and then failed at the first
    trial with `FileNotFoundError: 'python'`. PEP 394 only guarantees
    `python3`, so a bare `python` is absent on macOS and on Debian without
    `python-is-python3` -- it resolves only inside an activated venv, which is
    why the suite never saw it: pytest runs under `uv run`, so `.venv/bin` is
    on PATH and every name resolves.

    The supervisor spawns the workload with the PATH it inherited, and a user
    who runs `uv tool install ai-experiments` or `.venv/bin/iax` has no
    `.venv/bin` there. So drop it before checking: the shipped entrypoint has
    to resolve on the machine's own PATH, not on the harness's.
    """
    venv_bin = Path(sysconfig.get_path("scripts"))
    monkeypatch.setenv(
        "PATH",
        os.pathsep.join(
            entry
            for entry in os.environ["PATH"].split(os.pathsep)
            if entry and Path(entry) != venv_bin
        ),
    )
    path = tmp_path / f"{kind}.yaml"
    scaffold.write(kind, path)
    schema = ExperimentManifest if kind == "manifest" else GoalSpec

    assert workload_warnings(schema.from_yaml(path)) == []


def test_workload_template_runs_and_reports_metrics(tmp_path):
    path = tmp_path / "train.py"
    scaffold.write("workload", path)
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()

    result = subprocess.run(
        [sys.executable, str(path)],
        capture_output=True,
        text=True,
        env={
            "PATH": "/usr/bin:/bin",
            "IAX_PARAMS": '{"lr": 0.05}',
            "IAX_ARTIFACTS_DIR": str(artifacts),
        },
        timeout=60,
    )

    assert result.returncode == 0
    metric_lines = [
        line for line in result.stdout.splitlines() if line.startswith("IAX_METRIC ")
    ]
    assert len(metric_lines) == 100
    assert (artifacts / "result.json").exists()


def test_workload_template_reads_the_params_the_harness_injects(tmp_path):
    """A scaffold that ignored IAX_PARAMS would make every trial identical."""
    path = tmp_path / "train.py"
    scaffold.write("workload", path)

    def final_loss(lr: str) -> float:
        import json

        out = subprocess.run(
            [sys.executable, str(path)],
            capture_output=True,
            text=True,
            env={"PATH": "/usr/bin:/bin", "IAX_PARAMS": f'{{"lr": {lr}}}'},
            timeout=60,
        ).stdout
        last = [ln for ln in out.splitlines() if ln.startswith("IAX_METRIC ")][-1]
        return json.loads(last[len("IAX_METRIC ") :])["loss"]

    assert final_loss("0.001") > final_loss("0.05")


@pytest.mark.parametrize("kind", ["manifest", "goal", "workload"])
def test_new_writes_the_file_and_names_the_next_step(tmp_path, kind):
    target = tmp_path / "out" / "thing"

    result = runner.invoke(app, ["new", kind, str(target)])

    assert result.exit_code == 0
    assert target.read_text()
    assert "Next:" in result.stdout


@pytest.mark.parametrize("kind", ["manifest", "goal", "workload"])
def test_new_into_a_directory_uses_the_conventional_name(tmp_path, kind):
    result = runner.invoke(app, ["new", kind, str(tmp_path)])

    assert result.exit_code == 0
    assert (tmp_path / scaffold.DEFAULT_NAMES[kind]).exists()


def test_new_refuses_to_clobber_without_force(tmp_path):
    target = tmp_path / "goal.yaml"
    target.write_text("keep me\n")

    result = runner.invoke(app, ["new", "goal", str(target)])

    assert result.exit_code == 2  # invalid input, per the CLI error contract
    assert target.read_text() == "keep me\n"

    forced = runner.invoke(app, ["new", "goal", str(target), "--force"])

    assert forced.exit_code == 0
    assert "goal:" in target.read_text()


def test_new_manifest_from_run_reproduces_the_submitted_manifest(tmp_path):
    from ai_experiments.store import FilesystemRunStore

    store = FilesystemRunStore(tmp_path / "runs")
    original = ExperimentManifest(
        experiment="already-ran",
        workload={"entrypoint": sys.executable, "args": ["-c", "pass"]},
    )
    run_id, _ = store.create_run(original)

    target = tmp_path / "copy.yaml"
    result = runner.invoke(
        app,
        [
            "new",
            "manifest",
            str(target),
            "--from-run",
            run_id,
            "--runs-dir",
            str(tmp_path / "runs"),
        ],
    )

    assert result.exit_code == 0
    assert ExperimentManifest.from_yaml(target).experiment == "already-ran"


def test_new_manifest_from_an_unknown_run_is_not_found(tmp_path):
    result = runner.invoke(
        app,
        [
            "new",
            "manifest",
            str(tmp_path / "x.yaml"),
            "--from-run",
            "run_nope",
            "--runs-dir",
            str(tmp_path / "runs"),
        ],
    )

    assert result.exit_code == 1


def test_the_scaffolded_goal_and_workload_reach_their_target(tmp_path):
    """`iax new` twice, then `iax loop`, is a new user's first five minutes.

    If that misses its target, the templates teach that the tool does not
    work. So the pair is held to reaching it.
    """
    from ai_experiments.api import goal_from_yaml, run_loop

    scaffold.write("workload", tmp_path / "train.py", force=False)
    scaffold.write("goal", tmp_path / "goal.yaml", force=False)

    goal = goal_from_yaml(tmp_path / "goal.yaml")
    goal.workload.working_dir = str(tmp_path)
    goal.workload.entrypoint = sys.executable
    goal.workload.args = ["train.py"]

    report = run_loop(
        goal, runs_dir=tmp_path / "runs", interval_seconds=0, max_seconds=180
    )

    assert report.target_reached, report.stop_reason
    assert report.best["objective_value"] <= goal.objective.target
