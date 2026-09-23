"""`iax campaign variant`: the way an agent's code reaches a campaign.

The edits arrive as whole files on disk, not as text on a command line, so
what runs is exactly what was written and reviewable. The exit code is the
smoke check's, so a caller branching on it never runs a broken variant.
"""

from __future__ import annotations

import json
import sys

from typer.testing import CliRunner

from ai_experiments.cli import app
from ai_experiments.orchestrator import CampaignOrchestrator
from ai_experiments.schemas import GoalSpec
from ai_experiments.store import FilesystemRunStore

runner = CliRunner()

PASSES = "import pathlib, sys; sys.exit(0 if pathlib.Path('train.py').exists() else 1)"


def _campaign(tmp_path, smoke: str = PASSES):
    source = tmp_path / "workload"
    source.mkdir()
    (source / "train.py").write_text("LOSS = 1.0\n")
    runs = tmp_path / "runs"
    goal = GoalSpec(
        goal="minimize loss",
        name="cli-variant",
        objective={"metric": "loss"},
        search_space={"x": {"type": "uniform", "low": 0.0, "high": 1.0}},
        workload={"entrypoint": "python train.py"},
        budget={"max_trials": 4, "max_parallel": 1},
        variants={
            "enabled": True,
            "source_dir": str(source),
            "editable_paths": ["*.py"],
            "smoke_command": [sys.executable, "-c", smoke],
        },
    )
    state = CampaignOrchestrator(FilesystemRunStore(runs)).start(goal)
    return state.campaign_id, runs


def test_a_variant_is_created_from_files_on_disk(tmp_path):
    campaign_id, runs = _campaign(tmp_path)
    edit = tmp_path / "new_train.py"
    edit.write_text("LOSS = 0.5\n")

    result = runner.invoke(
        app,
        [
            "campaign",
            "variant",
            campaign_id,
            "--edit",
            f"train.py={edit}",
            "--hypothesis",
            "a smaller constant",
            "--runs-dir",
            str(runs),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    record = json.loads(result.stdout)
    assert record["smoke_ok"] is True
    assert record["edited_paths"] == ["train.py"]
    assert record["hypothesis"] == "a smaller constant"


def test_a_variant_that_fails_its_smoke_check_exits_two_and_says_why(tmp_path):
    campaign_id, runs = _campaign(
        tmp_path, smoke="import sys; sys.stderr.write('boom\\n'); sys.exit(3)"
    )
    edit = tmp_path / "new_train.py"
    edit.write_text("LOSS = 0.5\n")

    result = runner.invoke(
        app,
        [
            "campaign",
            "variant",
            campaign_id,
            "--edit",
            f"train.py={edit}",
            "--runs-dir",
            str(runs),
        ],
    )

    assert result.exit_code == 2
    assert "boom" in result.stderr


def test_variants_are_listed_with_their_smoke_verdict(tmp_path):
    campaign_id, runs = _campaign(tmp_path)
    edit = tmp_path / "new_train.py"
    edit.write_text("LOSS = 0.5\n")
    runner.invoke(
        app,
        [
            "campaign",
            "variant",
            campaign_id,
            "--edit",
            f"train.py={edit}",
            "--runs-dir",
            str(runs),
            "--json",
        ],
    )

    result = runner.invoke(app, ["campaign", "variants", campaign_id, "--runs-dir", str(runs)])

    assert result.exit_code == 0, result.output
    assert "smoke ok" in result.output


def test_an_edit_whose_source_file_is_missing_exits_two(tmp_path):
    campaign_id, runs = _campaign(tmp_path)

    result = runner.invoke(
        app,
        [
            "campaign",
            "variant",
            campaign_id,
            "--edit",
            f"train.py={tmp_path}/nope.py",
            "--runs-dir",
            str(runs),
        ],
    )

    assert result.exit_code == 2
    assert "nope.py" in result.stderr


def test_a_suggestion_can_name_a_variant(tmp_path):
    campaign_id, runs = _campaign(tmp_path)
    edit = tmp_path / "new_train.py"
    edit.write_text("LOSS = 0.5\n")
    created = runner.invoke(
        app,
        [
            "campaign",
            "variant",
            campaign_id,
            "--edit",
            f"train.py={edit}",
            "--runs-dir",
            str(runs),
            "--json",
        ],
    )
    variant_id = json.loads(created.stdout)["variant_id"]

    result = runner.invoke(
        app,
        [
            "campaign",
            "suggest",
            campaign_id,
            "--params",
            '{"x": 0.5}',
            "--variant",
            variant_id,
            "--runs-dir",
            str(runs),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["variant_id"] == variant_id


def test_a_rejected_variant_left_on_disk_names_the_directory(tmp_path, monkeypatch):
    """`and was discarded` has to be true when it is printed.

    When the delete fails, the caller is told where the leftover is instead.
    """
    import shutil

    campaign_id, runs = _campaign(tmp_path, smoke="import sys; sys.exit(3)")
    edit = tmp_path / "new_train.py"
    edit.write_text("LOSS = 0.5\n")

    def refuse(path):
        raise OSError(16, "Device or resource busy")

    monkeypatch.setattr(shutil, "rmtree", refuse)
    result = runner.invoke(
        app,
        [
            "campaign",
            "variant",
            campaign_id,
            "--edit",
            f"train.py={edit}",
            "--runs-dir",
            str(runs),
        ],
    )

    assert result.exit_code == 2, result.output
    assert "could not be removed" in result.output
    assert "Device or resource busy" in result.output

    listed = runner.invoke(app, ["campaign", "variants", campaign_id, "--runs-dir", str(runs)])
    assert "left on disk" in listed.output
