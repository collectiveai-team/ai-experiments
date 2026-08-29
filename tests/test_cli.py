from __future__ import annotations

import json
import sys
import time

import yaml
from typer.testing import CliRunner

from ai_experiments.cli import app


runner = CliRunner()


def _manifest(tmp_path):
    script = tmp_path / "train.py"
    script.write_text("print('metric accuracy=1.0')\n")
    manifest = {
        "experiment": "smoke",
        "backend": "local",
        "workload": {
            "entrypoint": sys.executable,
            "args": [str(script)],
            "working_dir": str(tmp_path),
        },
        "artifacts": {"output_dir": str(tmp_path / "outputs")},
    }
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(manifest))
    return path


def _sleep_manifest(tmp_path):
    script = tmp_path / "sleep_train.py"
    script.write_text(
        "import time\nprint('started', flush=True)\ntime.sleep(2)\nprint('done')\n"
    )
    manifest = {
        "experiment": "sleep",
        "backend": "local",
        "workload": {
            "entrypoint": sys.executable,
            "args": [str(script)],
            "working_dir": str(tmp_path),
        },
        "monitoring": {"interval_seconds": 60, "stuck_after_minutes": 30},
    }
    path = tmp_path / "sleep_experiment.yaml"
    path.write_text(yaml.safe_dump(manifest))
    return path


def test_validate_manifest(tmp_path):
    result = runner.invoke(app, ["validate", str(_manifest(tmp_path))])
    assert result.exit_code == 0
    assert "Manifest valid" in result.stdout


def test_validate_rejects_malformed_backend_address(tmp_path):
    manifest = {
        "experiment": "bad-address",
        "backend": "ray",
        "backend_address": "ray://cluster",
        "workload": {"entrypoint": sys.executable},
    }
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(manifest))

    result = runner.invoke(app, ["validate", str(path)])

    assert result.exit_code == 2  # invalid input, per the CLI error contract
    assert "invalid manifest" in result.stderr


def test_submit_status_and_diagnose_local_run(tmp_path):
    runs_dir = tmp_path / "runs"
    result = runner.invoke(
        app,
        ["submit", str(_manifest(tmp_path)), "--runs-dir", str(runs_dir), "--json"],
    )
    assert result.exit_code == 0
    handle = json.loads(result.stdout)

    status = None
    for _ in range(30):
        status_result = runner.invoke(
            app,
            ["status", handle["run_id"], "--runs-dir", str(runs_dir), "--json"],
        )
        assert status_result.exit_code == 0
        status = json.loads(status_result.stdout)
        if status["status"] == "completed":
            break
        time.sleep(0.1)

    assert status is not None
    assert status["status"] == "completed"

    diagnosis = runner.invoke(
        app,
        ["diagnose", handle["run_id"], "--runs-dir", str(runs_dir), "--json"],
    )
    assert diagnosis.exit_code == 0
    report = json.loads(diagnosis.stdout)
    assert report["decision"]["decision"] == "training_complete"


def test_monitor_is_quiet_while_waiting(tmp_path):
    runs_dir = tmp_path / "runs"
    result = runner.invoke(
        app,
        [
            "submit",
            str(_sleep_manifest(tmp_path)),
            "--runs-dir",
            str(runs_dir),
            "--json",
        ],
    )
    assert result.exit_code == 0
    handle = json.loads(result.stdout)

    monitor = runner.invoke(
        app,
        [
            "monitor",
            handle["run_id"],
            "--runs-dir",
            str(runs_dir),
            "--json",
            "--quiet-when-waiting",
        ],
    )
    assert monitor.exit_code == 0
    assert monitor.stdout == ""

    status = None
    for _ in range(40):
        status_result = runner.invoke(
            app,
            ["status", handle["run_id"], "--runs-dir", str(runs_dir), "--json"],
        )
        assert status_result.exit_code == 0
        status = json.loads(status_result.stdout)
        if status["status"] == "completed":
            break
        time.sleep(0.1)
    assert status is not None
    assert status["status"] == "completed"

    complete_monitor = runner.invoke(
        app,
        [
            "monitor",
            handle["run_id"],
            "--runs-dir",
            str(runs_dir),
            "--json",
            "--quiet-when-waiting",
        ],
    )
    assert complete_monitor.exit_code == 0
    report = json.loads(complete_monitor.stdout)
    assert report["decision"]["decision"] == "training_complete"


def test_campaign_suggest_rejects_bad_params(tmp_path):
    """A suggestion outside the search space exits 2, not 0 (#13)."""
    from ai_experiments.orchestrator import CampaignOrchestrator
    from ai_experiments.schemas import GoalSpec
    from ai_experiments.store import FilesystemRunStore

    runs = tmp_path / "runs"
    goal = GoalSpec(
        goal="minimize loss",
        name="cli-suggest",
        objective={"metric": "loss"},
        search_space={"x": {"type": "uniform", "low": 0.0, "high": 1.0}},
        workload={"entrypoint": "python -c pass"},
        budget={"max_trials": 4, "max_parallel": 1},
    )
    store = FilesystemRunStore(runs)
    state = CampaignOrchestrator(store).start(goal)

    result = runner.invoke(
        app,
        [
            "campaign",
            "suggest",
            state.campaign_id,
            "--params",
            '{"x": 42.0}',
            "--runs-dir",
            str(runs),
        ],
    )
    assert result.exit_code == 2
    assert "rejected" in result.output
