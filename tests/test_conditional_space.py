"""Dimensions that only exist for some of the trials.

A space that lists `learning_rate`, `max_leaf_nodes`, `max_bins` and
`min_samples_leaf` alongside `model: [hist_gradient_boosting, logreg]` spends
part of its budget drawing four gradient-boosting knobs for logistic
regressions that ignore all of them. Those trials are duplicates the
deduplicator cannot see, and the search reads their spread as evidence that
the knobs do nothing.
"""

from __future__ import annotations

import random

import pytest

from ai_experiments.planner.search_space import (
    active_space,
    grid_points,
    perturb,
    sample,
)
from ai_experiments.planner.validation import ParamValidationError, validate_params
from ai_experiments.schemas import GoalSpec, ObjectiveSpec, WorkloadSpec

SPACE = {
    "model": {"type": "choice", "values": ["hist_gb", "logreg"]},
    "max_leaf_nodes": {
        "type": "int",
        "low": 8,
        "high": 64,
        "when": {"model": ["hist_gb"]},
    },
    "learning_rate": {"type": "loguniform", "low": 1e-3, "high": 1.0},
}


def _space():
    return GoalSpec(
        goal="g",
        name="n",
        objective=ObjectiveSpec(metric="loss", mode="min"),
        search_space=SPACE,
        workload=WorkloadSpec(entrypoint="python train.py"),
    ).search_space


def test_a_conditional_dimension_is_drawn_only_when_its_condition_holds():
    space = _space()
    rng = random.Random(0)  # noqa: S311  # test fixture, not security

    draws = [sample(space, rng) for _ in range(50)]

    assert any(d["model"] == "logreg" for d in draws)
    assert any(d["model"] == "hist_gb" for d in draws)
    for d in draws:
        assert ("max_leaf_nodes" in d) == (d["model"] == "hist_gb")
        assert "learning_rate" in d


def test_the_active_space_is_what_a_given_assignment_actually_has():
    space = _space()

    assert set(active_space(space, {"model": "logreg"})) == {"model", "learning_rate"}
    assert set(active_space(space, {"model": "hist_gb"})) == set(space)


def test_a_grid_does_not_multiply_out_knobs_the_model_ignores():
    """The logreg half of the grid is one point per learning rate.

    Not one per (learning rate x leaf count) it will never read.
    """
    points = grid_points(_space(), resolution=2)

    logreg = [p for p in points if p["model"] == "logreg"]
    assert logreg
    assert all("max_leaf_nodes" not in p for p in logreg)
    assert len({tuple(sorted(p.items())) for p in logreg}) == len(logreg)
    assert all("max_leaf_nodes" in p for p in points if p["model"] == "hist_gb")


def test_perturbing_across_the_condition_drops_and_adds_the_dimension():
    """A neighbour that flips `model` stops carrying a dead knob.

    One that flips back has to draw it fresh rather than resurrect a stale value.
    """
    space = _space()
    rng = random.Random(1)  # noqa: S311  # test fixture, not security

    moved = [
        perturb(
            space,
            {"model": "hist_gb", "max_leaf_nodes": 31, "learning_rate": 0.1},
            rng,
            scale=1.0,
        )
        for _ in range(40)
    ]

    for params in moved:
        assert ("max_leaf_nodes" in params) == (params["model"] == "hist_gb")


def test_an_assignment_missing_an_inactive_key_is_valid():
    coerced = validate_params(_space(), {"model": "logreg", "learning_rate": 0.1})

    assert set(coerced) == {"model", "learning_rate"}


def test_an_assignment_that_sets_an_inactive_key_is_rejected():
    """Sending `--max-leaf-nodes` to a logistic regression is the bug the condition prevents.

    Accepting it silently puts a parameter in the trial record that had no effect on the run.
    """
    with pytest.raises(ParamValidationError) as excinfo:
        validate_params(
            _space(),
            {"model": "logreg", "learning_rate": 0.1, "max_leaf_nodes": 31},
        )

    assert "max_leaf_nodes" in str(excinfo.value)


def test_an_active_key_is_still_required():
    with pytest.raises(ParamValidationError) as excinfo:
        validate_params(_space(), {"model": "hist_gb", "learning_rate": 0.1})

    assert "max_leaf_nodes" in str(excinfo.value)


def test_a_condition_on_a_key_the_space_does_not_define_is_refused():
    with pytest.raises(ValueError, match="typo"):
        GoalSpec(
            goal="g",
            name="n",
            objective=ObjectiveSpec(metric="loss", mode="min"),
            search_space={
                "lr": {
                    "type": "uniform",
                    "low": 0.0,
                    "high": 1.0,
                    "when": {"typo": ["x"]},
                }
            },
            workload=WorkloadSpec(entrypoint="python train.py"),
        )


def test_a_condition_on_a_conditional_key_is_refused():
    """One level deep keeps the sampling order obvious.

    Unconditional keys first, then everything that depends on them. Chains would need a topological
    sort and a cycle check for no use anyone has asked for.
    """
    with pytest.raises(ValueError, match="itself conditional"):
        GoalSpec(
            goal="g",
            name="n",
            objective=ObjectiveSpec(metric="loss", mode="min"),
            search_space={
                "model": {"type": "choice", "values": ["a", "b"]},
                "mid": {
                    "type": "choice",
                    "values": ["x"],
                    "when": {"model": ["a"]},
                },
                "leaf": {
                    "type": "choice",
                    "values": ["y"],
                    "when": {"mid": ["x"]},
                },
            },
            workload=WorkloadSpec(entrypoint="python train.py"),
        )
