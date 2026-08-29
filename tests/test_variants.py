"""Workload variants: change the code, never the user's tree.

A variant is where an unattended loop is most dangerous — it writes files an
agent proposed. These tests pin the two properties that make it safe: writes
stay inside the sandbox, and a variant that cannot start never costs a round.
"""

from __future__ import annotations

import sys

import pytest

from ai_experiments.improve.variants import (
    VariantEdit,
    VariantError,
    discard_variant,
    materialize_variant,
    smoke_check,
    variants_root,
)
from ai_experiments.schemas import VariantSpec


def _workload(tmp_path):
    source = tmp_path / "workload"
    (source / "pkg").mkdir(parents=True)
    (source / "train.py").write_text("LOSS = 1.0\n")
    (source / "pkg" / "model.py").write_text("def build():\n    return None\n")
    (source / ".git").mkdir()
    (source / ".git" / "objects").write_text("x" * 1000)
    return source


def test_a_variant_copies_the_workload_and_applies_the_edit(tmp_path):
    source = _workload(tmp_path)

    record = materialize_variant(
        tmp_path / "campaign",
        source,
        [VariantEdit(path="train.py", content="LOSS = 0.5\n")],
        hypothesis="the loss constant is wrong",
    )

    root = variants_root(tmp_path / "campaign") / record.variant_id
    assert (root / "train.py").read_text() == "LOSS = 0.5\n"
    assert (root / "pkg" / "model.py").exists()
    assert record.edited_paths == ["train.py"]
    assert record.hypothesis == "the loss constant is wrong"


def test_the_original_workload_is_never_touched(tmp_path):
    source = _workload(tmp_path)

    materialize_variant(
        tmp_path / "campaign",
        source,
        [VariantEdit(path="train.py", content="LOSS = 0.5\n")],
    )

    assert (source / "train.py").read_text() == "LOSS = 1.0\n"


def test_a_variant_skips_git_and_virtualenvs(tmp_path):
    source = _workload(tmp_path)

    record = materialize_variant(tmp_path / "campaign", source, [])

    assert not (
        variants_root(tmp_path / "campaign") / record.variant_id / ".git"
    ).exists()


def test_a_new_file_can_be_created_inside_the_variant(tmp_path):
    source = _workload(tmp_path)

    record = materialize_variant(
        tmp_path / "campaign",
        source,
        [VariantEdit(path="pkg/loss.py", content="def loss():\n    return 0.0\n")],
    )

    assert (
        variants_root(tmp_path / "campaign") / record.variant_id / "pkg" / "loss.py"
    ).exists()


@pytest.mark.parametrize(
    "bad_path",
    ["../escaped.py", "../../etc/passwd", "/etc/passwd", "pkg/../../out.py", ""],
)
def test_an_edit_that_escapes_the_variant_is_rejected(tmp_path, bad_path):
    source = _workload(tmp_path)

    with pytest.raises(VariantError):
        materialize_variant(
            tmp_path / "campaign",
            source,
            [VariantEdit(path=bad_path, content="pwned\n")],
        )

    assert not (tmp_path / "escaped.py").exists()
    assert not (tmp_path / "out.py").exists()


def test_editable_paths_narrow_what_a_variant_may_write(tmp_path):
    source = _workload(tmp_path)
    spec = VariantSpec(enabled=True, editable_paths=["pkg/*.py"])

    with pytest.raises(VariantError) as caught:
        materialize_variant(
            tmp_path / "campaign",
            source,
            [VariantEdit(path="train.py", content="LOSS = 0\n")],
            spec,
        )

    assert "editable_paths" in str(caught.value)


def test_a_missing_source_directory_is_reported_not_guessed(tmp_path):
    with pytest.raises(VariantError):
        materialize_variant(tmp_path / "campaign", tmp_path / "absent", [])


def test_the_smoke_check_passes_a_variant_that_starts(tmp_path):
    source = _workload(tmp_path)
    record = materialize_variant(tmp_path / "campaign", source, [])
    spec = VariantSpec(
        enabled=True, smoke_command=[sys.executable, "-c", "import train"]
    )

    checked = smoke_check(record, spec)

    assert checked.smoke_ok is True


def test_the_smoke_check_catches_a_variant_that_cannot_import(tmp_path):
    """Without this, every trial of the round fails identically and teaches nothing."""
    source = _workload(tmp_path)
    record = materialize_variant(
        tmp_path / "campaign",
        source,
        [VariantEdit(path="train.py", content="def broken(:\n")],
    )
    spec = VariantSpec(
        enabled=True, smoke_command=[sys.executable, "-c", "import train"]
    )

    checked = smoke_check(record, spec)

    assert checked.smoke_ok is False
    assert "SyntaxError" in checked.smoke_output


def test_the_smoke_check_times_out_instead_of_hanging(tmp_path):
    source = _workload(tmp_path)
    record = materialize_variant(tmp_path / "campaign", source, [])
    spec = VariantSpec(
        enabled=True,
        smoke_command=[sys.executable, "-c", "import time; time.sleep(30)"],
        smoke_timeout_seconds=1,
    )

    checked = smoke_check(record, spec)

    assert checked.smoke_ok is False
    assert "timed out" in checked.smoke_output


def test_no_smoke_command_means_unknown_not_passed(tmp_path):
    source = _workload(tmp_path)
    record = materialize_variant(tmp_path / "campaign", source, [])

    checked = smoke_check(record, VariantSpec(enabled=True))

    assert checked.smoke_ok is None


def test_discarding_a_variant_removes_only_that_variant(tmp_path):
    source = _workload(tmp_path)
    keep = materialize_variant(tmp_path / "campaign", source, [])
    drop = materialize_variant(tmp_path / "campaign", source, [])

    discard_variant(drop)

    assert not (variants_root(tmp_path / "campaign") / drop.variant_id).exists()
    assert (variants_root(tmp_path / "campaign") / keep.variant_id).exists()


def test_a_trial_manifest_points_at_the_variant(tmp_path):
    from ai_experiments.planner.planner import build_trial_manifest
    from ai_experiments.schemas import GoalSpec

    source = _workload(tmp_path)
    record = materialize_variant(tmp_path / "campaign", source, [])
    goal = GoalSpec(
        goal="minimize loss",
        name="variant-goal",
        objective={"metric": "loss"},
        search_space={"lr": {"type": "uniform", "low": 0.0, "high": 1.0}},
        workload={"entrypoint": "python", "args": ["train.py"], "working_dir": "."},
    )

    manifest = build_trial_manifest(
        goal, "t000", {"lr": 0.1}, working_dir=record.root, variant_id=record.variant_id
    )

    assert manifest.workload.working_dir == record.root
    assert manifest.metadata["variant_id"] == record.variant_id
