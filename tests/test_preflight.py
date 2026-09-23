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

    assert result.exit_code == 2  # invalid input, per the CLI error contract
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

    assert result.exit_code == 2
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

    assert result.exit_code == 2
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
    # A bare interpreter is not a runnable *campaign* workload: the goal's
    # search space puts `--x` on every trial's command line, and only a
    # script that declares it can take one.
    script = tmp_path / "train.py"
    script.write_text(
        "import argparse\n"
        "p = argparse.ArgumentParser()\n"
        'p.add_argument("--x", type=float)\n'
        "p.parse_args()\n"
    )
    goal = GoalSpec.from_yaml(
        _goal_file(tmp_path, f"{sys.executable} {script}", str(tmp_path))
    )

    state = CampaignOrchestrator(store, campaign_store).start(goal)

    messages = [e.message for e in campaign_store.read_events(state.campaign_id)]
    assert "workload may not start" not in messages


# --- the search space the workload will actually be handed ---------------

_ACCEPTS = """\
import argparse
p = argparse.ArgumentParser()
p.add_argument("--label-source")
p.add_argument("--window-days", type=int)
p.parse_args()
"""

_REJECTS = """\
import argparse
p = argparse.ArgumentParser()
p.add_argument("--label-source")
p.parse_args()
"""


def _goal_with(tmp_path, script: str, space: dict, **workload):
    from ai_experiments.schemas import (
        BudgetSpec,
        GoalSpec,
        ObjectiveSpec,
        WorkloadSpec as W,
    )

    path = tmp_path / "train.py"
    path.write_text(script)
    return GoalSpec(
        goal="probe",
        name="probe",
        objective=ObjectiveSpec(metric="loss", mode="min"),
        search_space=space,
        workload=W(
            entrypoint=f"{sys.executable} {path}",
            working_dir=str(tmp_path),
            **workload,
        ),
        budget=BudgetSpec(max_trials=1, max_parallel=1),
    )


def test_a_workload_that_declares_every_flag_draws_no_warning(tmp_path):
    goal = _goal_with(
        tmp_path,
        _ACCEPTS,
        {
            "label_source": {"type": "choice", "values": ["a"]},
            "window_days": {"type": "choice", "values": [30]},
        },
    )
    assert workload_warnings(goal) == []


def test_a_key_the_workload_cannot_parse_is_named_with_its_flag(tmp_path):
    """The campaign's most expensive failure, caught before the first trial.

    Every trial gets every search space key on its command line, so one key
    the parser never declared fails all of them identically.
    """
    goal = _goal_with(
        tmp_path,
        _REJECTS,
        {
            "label_source": {"type": "choice", "values": ["a"]},
            "window_days": {"type": "choice", "values": [30]},
        },
    )
    warnings = workload_warnings(goal)

    assert len(warnings) == 1
    assert "window_days" in warnings[0]
    assert "--window-days" in warnings[0]
    assert "label_source" not in warnings[0]


def test_the_probe_asks_the_workload_not_the_launcher(tmp_path):
    """`uv run` and `python -m` are launchers; their help is not the answer.

    With the script in `args` the entrypoint alone is an interpreter, and
    `python --help` lists python's own long options -- enough to look like a
    parser that declared something, and none of it the workload's. Every
    search space key then reads as undeclared, on every campaign start.
    """
    goal = _goal_with(
        tmp_path,
        _ACCEPTS,
        {"window_days": {"type": "choice", "values": [30]}},
    )
    goal.workload.args = [str(tmp_path / "train.py")]
    goal.workload.entrypoint = sys.executable

    assert workload_warnings(goal) == []


def test_the_probe_respects_the_workload_flag_style(tmp_path):
    goal = _goal_with(
        tmp_path,
        _ACCEPTS,
        {"label_source": {"type": "choice", "values": ["a"]}},
        flag_style="underscore",
    )
    warnings = workload_warnings(goal)

    assert len(warnings) == 1
    assert "--label_source" in warnings[0]


def test_a_workload_with_no_usable_help_is_given_the_benefit_of_the_doubt(tmp_path):
    """Absence of a `--help` is not evidence the flags are wrong."""
    goal = _goal_with(
        tmp_path,
        "import sys; sys.exit(3)",
        {"label_source": {"type": "choice", "values": ["a"]}},
    )
    assert workload_warnings(goal) == []


def test_a_manifest_has_no_search_space_to_check(tmp_path):
    """`workload_warnings` still answers for a plain manifest."""
    assert workload_warnings(_manifest(sys.executable, str(tmp_path))) == []
