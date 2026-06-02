from __future__ import annotations

import json
import sys
import time

import yaml
from typer.testing import CliRunner

from industrial_ai_experiments.cli import app


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
    script.write_text("import time\nprint('started', flush=True)\ntime.sleep(2)\nprint('done')\n")
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
        ["submit", str(_sleep_manifest(tmp_path)), "--runs-dir", str(runs_dir), "--json"],
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
