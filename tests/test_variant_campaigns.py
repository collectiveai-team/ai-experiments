"""The production path for code variants: propose, smoke-check, run.

`materialize_variant` and `smoke_check` were reachable only from their own
unit tests, so a campaign could run a variant nobody could create. These
tests pin the way in, and the rule that makes it safe to leave unattended:
the smoke command's exit code decides whether a variant is allowed to cost a
round — not the agent that wrote it.
"""

from __future__ import annotations

import sys

import pytest

from ai_experiments.improve.variants import VariantEdit
from ai_experiments.schemas import (
    BudgetSpec,
    GoalSpec,
    ObjectiveSpec,
    StrategySpec,
    VariantSpec,
    WorkloadSpec,
)
from ai_experiments.store import FilesystemRunStore
from ai_experiments.store.campaign import CampaignStore
from tests.test_orchestrator import FakeBackend

from ai_experiments.orchestrator import CampaignOrchestrator

PASSES = "import pathlib, sys; sys.exit(0 if pathlib.Path('train.py').exists() else 1)"


def _source(tmp_path):
    source = tmp_path / "workload"
    source.mkdir()
    (source / "train.py").write_text("LOSS = 1.0\n")
    (source / "notes.txt").write_text("not code\n")
    return source


def _goal(tmp_path, **variant_overrides) -> GoalSpec:
    variants = {
        "enabled": True,
        "source_dir": str(_source(tmp_path)),
        "editable_paths": ["*.py"],
        "smoke_command": [sys.executable, "-c", PASSES],
    }
    variants.update(variant_overrides)
    return GoalSpec(
        goal="minimize (x-2)^2",
        name="quadratic",
        objective=ObjectiveSpec(metric="loss", mode="min"),
        search_space={"x": {"type": "uniform", "low": -5.0, "high": 5.0}},
        workload=WorkloadSpec(entrypoint="python train.py"),
        budget=BudgetSpec(max_trials=6, max_parallel=2),
        strategy=StrategySpec(name="adaptive", seed=3),
        variants=VariantSpec(**variants),
    )


def _orchestrator(tmp_path):
    store = FilesystemRunStore(tmp_path / "runs")
    backend = FakeBackend(store)
    return (
        CampaignOrchestrator(
            store, CampaignStore(store.root), backend_factory=lambda goal: backend
        ),
        backend,
    )


def test_a_goal_that_did_not_enable_variants_refuses_one(tmp_path):
    orchestrator, _ = _orchestrator(tmp_path)
    state = orchestrator.start(_goal(tmp_path, enabled=False))

    with pytest.raises(ValueError, match="variants.enabled"):
        orchestrator.add_variant(
            state.campaign_id, [VariantEdit(path="train.py", content="LOSS = 0.5\n")]
        )


def test_an_accepted_variant_is_recorded_and_readable(tmp_path):
    orchestrator, _ = _orchestrator(tmp_path)
    state = orchestrator.start(_goal(tmp_path))

    record = orchestrator.add_variant(
        state.campaign_id,
        [VariantEdit(path="train.py", content="LOSS = 0.5\n")],
        hypothesis="a smaller constant",
    )

    assert record.smoke_ok is True
    assert record.edited_paths == ["train.py"]
    stored = orchestrator.campaign_store.read_variants(state.campaign_id)
    assert [r.variant_id for r in stored] == [record.variant_id]
    assert stored[0].hypothesis == "a smaller constant"


def test_a_variant_that_fails_its_smoke_check_is_discarded_but_not_forgotten(tmp_path):
    """The directory goes; the record stays, so the next proposal can read
    why this one did not run."""
    orchestrator, _ = _orchestrator(tmp_path)
    state = orchestrator.start(
        _goal(tmp_path, smoke_command=[sys.executable, "-c", "raise SystemExit(3)"])
    )

    record = orchestrator.add_variant(
        state.campaign_id, [VariantEdit(path="train.py", content="LOSS = 0.5\n")]
    )

    assert record.smoke_ok is False
    from pathlib import Path

    assert not Path(record.root).exists()
    assert record.discarded is True
    assert orchestrator.campaign_store.read_variants(state.campaign_id) == [record]


def test_an_edit_outside_the_goals_allowlist_is_refused(tmp_path):
    orchestrator, _ = _orchestrator(tmp_path)
    state = orchestrator.start(_goal(tmp_path))

    with pytest.raises(ValueError, match="editable_paths"):
        orchestrator.add_variant(
            state.campaign_id, [VariantEdit(path="notes.txt", content="hi\n")]
        )


def test_a_suggested_trial_runs_against_the_variants_copy(tmp_path):
    orchestrator, backend = _orchestrator(tmp_path)
    state = orchestrator.start(_goal(tmp_path))
    record = orchestrator.add_variant(
        state.campaign_id, [VariantEdit(path="train.py", content="LOSS = 0.5\n")]
    )

    trial = orchestrator.suggest(
        state.campaign_id, {"x": 1.0}, variant_id=record.variant_id
    )
    orchestrator.advance(state.campaign_id)

    assert trial.variant_id == record.variant_id
    submitted = next(m for m in backend.submitted if m.metadata["params"] == {"x": 1.0})
    assert submitted.workload.working_dir == record.root


def test_a_suggestion_naming_an_unknown_variant_is_refused(tmp_path):
    orchestrator, _ = _orchestrator(tmp_path)
    state = orchestrator.start(_goal(tmp_path))

    with pytest.raises(ValueError, match="var_nope"):
        orchestrator.suggest(state.campaign_id, {"x": 1.0}, variant_id="var_nope")


def test_a_suggestion_naming_a_variant_that_failed_its_smoke_check_is_refused(tmp_path):
    orchestrator, _ = _orchestrator(tmp_path)
    state = orchestrator.start(
        _goal(tmp_path, smoke_command=[sys.executable, "-c", "raise SystemExit(3)"])
    )
    record = orchestrator.add_variant(
        state.campaign_id, [VariantEdit(path="train.py", content="LOSS = 0.5\n")]
    )

    with pytest.raises(ValueError, match="smoke"):
        orchestrator.suggest(
            state.campaign_id, {"x": 1.0}, variant_id=record.variant_id
        )
