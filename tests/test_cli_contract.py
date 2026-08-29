"""The CLI error contract that an agent driving `iax` branches on (#18, #16).

Every failing command must exit with a documented code and, under `--json`,
print one `{"error", "code"}` object on stdout. Before this contract, an
unknown id surfaced as a python traceback and a bad manifest exited 1 exactly
like a missing run — an agent could not tell the two apart.
"""

from __future__ import annotations

import json

import pytest
import yaml
from typer.testing import CliRunner

from ai_experiments.cli import app
from ai_experiments.cli_support import (
    EXIT_INVALID_INPUT,
    EXIT_NOT_FOUND,
)
from ai_experiments.orchestrator import CampaignOrchestrator
from ai_experiments.schemas import GoalSpec
from ai_experiments.store import FilesystemRunStore

runner = CliRunner()


def _campaign(runs_dir) -> str:
    goal = GoalSpec(
        goal="minimize loss",
        name="contract",
        objective={"metric": "loss"},
        search_space={"x": {"type": "uniform", "low": 0.0, "high": 1.0}},
        workload={"entrypoint": "python -c pass"},
        budget={"max_trials": 2, "max_parallel": 1},
    )
    return CampaignOrchestrator(FilesystemRunStore(runs_dir)).start(goal).campaign_id


UNKNOWN_RUN_COMMANDS = [
    ["status", "run_nope"],
    ["logs", "run_nope"],
    ["diagnose", "run_nope"],
    ["monitor", "run_nope"],
    ["metrics", "run_nope"],
    ["artifacts", "run_nope"],
    ["cancel", "run_nope"],
    ["repro", "run_nope"],
]


@pytest.mark.parametrize("command", UNKNOWN_RUN_COMMANDS, ids=lambda c: c[0])
def test_unknown_run_is_a_json_not_found(tmp_path, command):
    result = runner.invoke(
        app, [*command, "--runs-dir", str(tmp_path / "runs"), "--json"]
    )

    assert result.exit_code == EXIT_NOT_FOUND
    payload = json.loads(result.stdout)
    assert payload["code"] == "not_found"
    assert "run_nope" in payload["error"]
    assert payload["details"] == {"run": "run_nope"}


UNKNOWN_CAMPAIGN_COMMANDS = [
    ["campaign", "status"],
    ["campaign", "advance"],
    ["campaign", "stop"],
    ["campaign", "pause"],
    ["campaign", "resume"],
]


@pytest.mark.parametrize(
    "command", UNKNOWN_CAMPAIGN_COMMANDS, ids=lambda c: "-".join(c)
)
def test_unknown_campaign_is_a_json_not_found(tmp_path, command):
    result = runner.invoke(
        app, [*command, "camp_nope", "--runs-dir", str(tmp_path / "runs"), "--json"]
    )

    assert result.exit_code == EXIT_NOT_FOUND
    payload = json.loads(result.stdout)
    assert payload["code"] == "not_found"
    assert payload["details"] == {"campaign": "camp_nope"}


def test_unknown_run_without_json_writes_a_line_to_stderr(tmp_path):
    result = runner.invoke(app, ["status", "run_nope", "--runs-dir", str(tmp_path)])

    assert result.exit_code == EXIT_NOT_FOUND
    assert result.stdout == ""
    assert "unknown run 'run_nope'" in result.stderr


def test_invalid_manifest_exits_two_not_one(tmp_path):
    """`not_found` and `invalid_input` used to share exit 1 (#16)."""
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump({"experiment": "no-workload"}))

    result = runner.invoke(app, ["validate", str(path)])

    assert result.exit_code == EXIT_INVALID_INPUT
    assert "invalid manifest" in result.stderr


def test_missing_manifest_file_is_invalid_input(tmp_path):
    result = runner.invoke(app, ["validate", str(tmp_path / "absent.yaml")])

    assert result.exit_code == EXIT_INVALID_INPUT


def test_suggest_reports_every_search_space_violation_as_json(tmp_path):
    runs = tmp_path / "runs"
    campaign_id = _campaign(runs)

    result = runner.invoke(
        app,
        [
            "campaign",
            "suggest",
            campaign_id,
            "--params",
            '{"x": 42.0, "typo": 1}',
            "--runs-dir",
            str(runs),
            "--json",
        ],
    )

    assert result.exit_code == EXIT_INVALID_INPUT
    payload = json.loads(result.stdout)
    assert payload["code"] == "invalid_input"
    assert "x" in payload["error"] and "typo" in payload["error"]


def test_malformed_params_json_is_invalid_input(tmp_path):
    runs = tmp_path / "runs"
    campaign_id = _campaign(runs)

    result = runner.invoke(
        app,
        [
            "campaign",
            "suggest",
            campaign_id,
            "--params",
            "not json",
            "--runs-dir",
            str(runs),
            "--json",
        ],
    )

    assert result.exit_code == EXIT_INVALID_INPUT
    assert json.loads(result.stdout)["code"] == "invalid_input"


def test_accepted_suggestion_prints_the_trial_as_json(tmp_path):
    runs = tmp_path / "runs"
    campaign_id = _campaign(runs)

    result = runner.invoke(
        app,
        [
            "campaign",
            "suggest",
            campaign_id,
            "--params",
            '{"x": 0.25}',
            "--runs-dir",
            str(runs),
            "--json",
        ],
    )

    assert result.exit_code == 0
    trial = json.loads(result.stdout)
    assert trial["params"] == {"x": 0.25}
    assert trial["trial_id"]
