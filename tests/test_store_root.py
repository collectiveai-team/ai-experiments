"""One project, one run store, whatever directory the command ran in (#21).

An agent submits from the repo root and asks for status three commands later,
from wherever the shell happens to be. The store used to resolve against the
cwd, so the second command read an empty directory and reported the run as
unknown. These tests hold the walk-up resolution and the error text that makes
a mismatch self-diagnosing.
"""

from __future__ import annotations

import json
import os
import sys
import time

import pytest
import yaml
from typer.testing import CliRunner

from ai_experiments.cli import app
from ai_experiments.store.filesystem import (
    DEFAULT_RUNS_SUBDIR,
    FilesystemRunStore,
    default_runs_root,
)

runner = CliRunner()


@pytest.fixture(autouse=True)
def _no_env_override(monkeypatch):
    monkeypatch.delenv("IAX_RUNS_DIR", raising=False)


def _project(tmp_path, marker: str = "pyproject.toml"):
    project = tmp_path / "project"
    (project / "src" / "deep").mkdir(parents=True)
    (project / marker).touch() if marker != ".git" else (project / ".git").mkdir()
    return project


@pytest.mark.parametrize("marker", ["pyproject.toml", ".git"])
def test_a_subdirectory_resolves_to_the_project_store(tmp_path, marker):
    project = _project(tmp_path, marker)

    assert default_runs_root(project / "src" / "deep") == default_runs_root(project)
    assert default_runs_root(project) == project / DEFAULT_RUNS_SUBDIR


def test_an_existing_store_beats_a_nearer_marker(tmp_path):
    """A package inside a repo has its own pyproject; the runs are still one store."""
    project = _project(tmp_path)
    (project / DEFAULT_RUNS_SUBDIR).mkdir(parents=True)
    inner = project / "src" / "deep"
    (inner / "pyproject.toml").touch()

    assert default_runs_root(inner) == project / DEFAULT_RUNS_SUBDIR


def test_an_explicit_env_override_still_wins(tmp_path, monkeypatch):
    project = _project(tmp_path)
    monkeypatch.setenv("IAX_RUNS_DIR", str(tmp_path / "elsewhere"))

    assert default_runs_root(project / "src") == tmp_path / "elsewhere"


def test_the_resolved_root_is_absolute(tmp_path, monkeypatch):
    """A relative root cannot be printed in an error the user can act on."""
    project = _project(tmp_path)
    monkeypatch.chdir(project / "src")

    assert FilesystemRunStore().root.is_absolute()
    assert FilesystemRunStore("runs").root == project / "src" / "runs"


def _manifest(project):
    script = project / "train.py"
    script.write_text('print(\'IAX_METRIC {"step": 1, "loss": 0.5}\', flush=True)\n')
    path = project / "experiment.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "experiment": "cwd-smoke",
                "backend": "local",
                "workload": {
                    "entrypoint": sys.executable,
                    "args": [str(script)],
                    "working_dir": str(project),
                },
                "artifacts": {"output_dir": str(project / "outputs")},
            }
        )
    )
    return path


def test_a_run_submitted_from_the_root_is_visible_from_a_subdirectory(
    tmp_path, monkeypatch
):
    """The bug, end to end: submit here, ask over there, get the same run."""
    project = _project(tmp_path)
    manifest = _manifest(project)

    monkeypatch.chdir(project)
    submitted = runner.invoke(app, ["submit", str(manifest), "--json"])
    assert submitted.exit_code == 0, submitted.stdout
    run_id = json.loads(submitted.stdout)["run_id"]

    monkeypatch.chdir(project / "src" / "deep")
    status = None
    for _ in range(50):
        result = runner.invoke(app, ["status", run_id, "--json"])
        assert result.exit_code == 0, result.stdout
        status = json.loads(result.stdout)
        if status["status"] in {"completed", "failed"}:
            break
        time.sleep(0.1)

    assert status is not None and status["status"] == "completed"
    assert run_id in runner.invoke(app, ["runs"]).stdout


def test_an_unknown_run_names_the_store_it_looked_in(tmp_path, monkeypatch):
    project = _project(tmp_path)
    monkeypatch.chdir(project)

    result = runner.invoke(app, ["status", "run_nope"])

    assert result.exit_code == 1
    assert str(project / DEFAULT_RUNS_SUBDIR) in result.stderr


def test_an_empty_listing_names_the_store_on_stderr(tmp_path, monkeypatch):
    """`iax runs --json` must stay parseable while still saying where it looked."""
    project = _project(tmp_path)
    monkeypatch.chdir(project)

    result = runner.invoke(app, ["runs", "--json"])

    assert result.exit_code == 0
    assert json.loads(result.stdout) == []
    assert str(project / DEFAULT_RUNS_SUBDIR) in result.stderr


def test_the_daemon_and_the_cli_agree_on_the_store(tmp_path, monkeypatch):
    """Two processes in different directories must drive the same campaign."""
    project = _project(tmp_path)
    monkeypatch.chdir(project / "src")
    from_sub = FilesystemRunStore().root

    monkeypatch.chdir(project)
    assert FilesystemRunStore().root == from_sub
    assert os.path.commonpath([from_sub, project]) == str(project)
