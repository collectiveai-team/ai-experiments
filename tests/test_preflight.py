"""Pre-submit checks and the supervisor log that used to be unreachable."""

from __future__ import annotations

import json
import sys
import time

import yaml
from typer.testing import CliRunner

from ai_experiments.cli import app
from ai_experiments.preflight import workload_warnings
from ai_experiments.schemas import ExperimentManifest, WorkloadSpec

runner = CliRunner()


def _manifest(entrypoint: str, working_dir: str) -> ExperimentManifest:
    return ExperimentManifest(
        experiment="preflight",
        workload=WorkloadSpec(entrypoint=entrypoint, working_dir=working_dir),
    )


def _manifest_file(tmp_path, entrypoint: str, working_dir: str):
    path = tmp_path / "experiment.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "experiment": "preflight",
                "backend": "local",
                "workload": {"entrypoint": entrypoint, "working_dir": working_dir},
            }
        )
    )
    return path


def test_no_warnings_for_a_runnable_workload(tmp_path):
    assert workload_warnings(_manifest(sys.executable, str(tmp_path))) == []


def test_warns_when_the_entrypoint_is_not_on_path(tmp_path):
    warnings = workload_warnings(_manifest("definitely-not-a-binary", str(tmp_path)))

    assert warnings == ["entrypoint 'definitely-not-a-binary' is not on PATH"]


def test_warns_when_the_working_dir_does_not_exist(tmp_path):
    warnings = workload_warnings(_manifest(sys.executable, str(tmp_path / "nope")))

    assert any("working_dir does not exist" in warning for warning in warnings)


def test_warns_when_a_relative_entrypoint_is_not_executable(tmp_path):
    (tmp_path / "train.sh").write_text("#!/bin/sh\n")  # exists, not executable

    warnings = workload_warnings(_manifest("./train.sh", str(tmp_path)))

    assert any("not an executable file" in warning for warning in warnings)


def test_a_relative_entrypoint_is_resolved_against_the_working_dir(tmp_path):
    script = tmp_path / "train.sh"
    script.write_text("#!/bin/sh\n")
    script.chmod(0o755)

    assert workload_warnings(_manifest("./train.sh", str(tmp_path))) == []


def test_validate_warns_but_still_succeeds(tmp_path):
    path = _manifest_file(tmp_path, "definitely-not-a-binary", str(tmp_path))

    result = runner.invoke(app, ["validate", str(path)])

    assert result.exit_code == 0
    assert "Manifest valid" in result.stdout
    assert "is not on PATH" in result.stderr


def test_validate_strict_fails_on_warnings(tmp_path):
    path = _manifest_file(tmp_path, "definitely-not-a-binary", str(tmp_path))

    result = runner.invoke(app, ["validate", str(path), "--strict"])

    assert result.exit_code == 1
    assert "is not on PATH" in result.stderr


def test_validate_strict_passes_a_runnable_workload(tmp_path):
    path = _manifest_file(tmp_path, sys.executable, str(tmp_path))

    result = runner.invoke(app, ["validate", str(path), "--strict"])

    assert result.exit_code == 0


def test_logs_worker_surfaces_the_supervisor_log(tmp_path):
    """The supervisor's traceback used to be readable only by knowing the run
    store's layout."""
    runs_dir = tmp_path / "runs"
    path = _manifest_file(tmp_path, "definitely-not-a-binary", str(tmp_path))
    submitted = runner.invoke(
        app, ["submit", str(path), "--runs-dir", str(runs_dir), "--json"]
    )
    assert submitted.exit_code == 0
    run_id = json.loads(submitted.stdout)["run_id"]

    state = None
    for _ in range(300):
        status = runner.invoke(
            app, ["status", run_id, "--runs-dir", str(runs_dir), "--json"]
        )
        state = json.loads(status.stdout)["status"]
        if state == "failed":
            break
        time.sleep(0.05)
    assert state == "failed"

    result = runner.invoke(
        app, ["logs", run_id, "--worker", "--runs-dir", str(runs_dir)]
    )

    assert result.exit_code == 0
    assert "FileNotFoundError" in result.stdout


def test_logs_worker_reports_a_missing_log(tmp_path):
    from ai_experiments.store import FilesystemRunStore

    store = FilesystemRunStore(tmp_path / "runs", capture_repro=False)
    run_id, _ = store.create_run(_manifest(sys.executable, str(tmp_path)))

    result = runner.invoke(
        app, ["logs", run_id, "--worker", "--runs-dir", str(store.root)]
    )

    assert result.exit_code == 1
    assert "no worker log" in result.stderr


# -- submit and campaign start run the same check (#32) ---------------------


def _goal_file(tmp_path, entrypoint: str, working_dir: str):
    path = tmp_path / "goal.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "goal": "minimize loss",
                "name": "preflight",
                "objective": {"metric": "loss", "mode": "min", "target": 0.0},
                "search_space": {"x": {"type": "uniform", "low": 0.0, "high": 1.0}},
                "workload": {"entrypoint": entrypoint, "working_dir": working_dir},
                "budget": {"max_trials": 1, "max_parallel": 1},
            }
        )
    )
    return path


def test_submit_warns_on_stderr_and_still_submits(tmp_path):
    """The warning must not reach stdout: `--json` output is parsed."""
    path = _manifest_file(tmp_path, "definitely-not-a-binary", str(tmp_path))

    result = runner.invoke(
        app, ["submit", str(path), "--runs-dir", str(tmp_path / "runs"), "--json"]
    )

    assert result.exit_code == 0
    assert "is not on PATH" in result.stderr
    assert json.loads(result.stdout)["run_id"]


def test_submit_is_quiet_about_a_runnable_workload(tmp_path):
    path = _manifest_file(tmp_path, sys.executable, str(tmp_path))

    result = runner.invoke(
        app, ["submit", str(path), "--runs-dir", str(tmp_path / "runs")]
    )

    assert result.exit_code == 0
    assert "Warning" not in result.stderr


def test_submit_strict_refuses_and_creates_no_run(tmp_path):
    runs_dir = tmp_path / "runs"
    path = _manifest_file(tmp_path, "definitely-not-a-binary", str(tmp_path))

    result = runner.invoke(
        app, ["submit", str(path), "--runs-dir", str(runs_dir), "--strict"]
    )

    assert result.exit_code == 1
    assert "is not on PATH" in result.stderr
    listed = runner.invoke(app, ["runs", "--runs-dir", str(runs_dir), "--json"])
    assert json.loads(listed.stdout) == []


def test_campaign_start_warns_once_before_it_spends_the_budget(tmp_path):
    path = _goal_file(tmp_path, "definitely-not-a-binary", str(tmp_path))

    result = runner.invoke(
        app,
        ["campaign", "start", str(path), "--runs-dir", str(tmp_path / "runs")],
    )

    assert result.exit_code == 0
    assert result.stderr.count("is not on PATH") == 1


def test_campaign_start_strict_refuses_and_creates_no_campaign(tmp_path):
    runs_dir = tmp_path / "runs"
    path = _goal_file(tmp_path, "definitely-not-a-binary", str(tmp_path))

    result = runner.invoke(
        app,
        ["campaign", "start", str(path), "--runs-dir", str(runs_dir), "--strict"],
    )

    assert result.exit_code == 1
    listed = runner.invoke(
        app, ["campaign", "list", "--runs-dir", str(runs_dir), "--json"]
    )
    assert json.loads(listed.stdout) == []


def test_a_campaign_records_the_warning_where_an_agent_reads_it(tmp_path):
    """A campaign started from the API never passes through the CLI."""
    from ai_experiments.orchestrator import CampaignOrchestrator
    from ai_experiments.schemas import GoalSpec
    from ai_experiments.store import FilesystemRunStore
    from ai_experiments.store.campaign import CampaignStore

    store = FilesystemRunStore(tmp_path / "runs", capture_repro=False)
    campaign_store = CampaignStore(store.root)
    goal = GoalSpec.from_yaml(_goal_file(tmp_path, "not-a-binary", str(tmp_path)))

    state = CampaignOrchestrator(store, campaign_store).start(goal)

    messages = [e.message for e in campaign_store.read_events(state.campaign_id)]
    assert "workload may not start" in messages


def test_a_runnable_campaign_records_no_warning(tmp_path):
    from ai_experiments.orchestrator import CampaignOrchestrator
    from ai_experiments.schemas import GoalSpec
    from ai_experiments.store import FilesystemRunStore
    from ai_experiments.store.campaign import CampaignStore

    store = FilesystemRunStore(tmp_path / "runs", capture_repro=False)
    campaign_store = CampaignStore(store.root)
    goal = GoalSpec.from_yaml(_goal_file(tmp_path, sys.executable, str(tmp_path)))

    state = CampaignOrchestrator(store, campaign_store).start(goal)

    messages = [e.message for e in campaign_store.read_events(state.campaign_id)]
    assert "workload may not start" not in messages
