"""Checks on the goal itself — the half of a campaign that runs before it does.

Everything here is about a definition that is *valid* and still wrong: the
schema accepts it, the campaign runs to the end, and the result cannot be
believed. These fire at `iax campaign validate` and at campaign start, where
fixing them still costs nothing.
"""

from __future__ import annotations

import sys

from ai_experiments.preflight import goal_warnings
from ai_experiments.schemas import (
    GoalSpec,
    ObjectiveSpec,
    SuccessCriteria,
    VariantSpec,
    WorkloadSpec,
)


def _goal(*, objective=None, space=None, criteria=None, variants=None) -> GoalSpec:
    return GoalSpec(
        goal="find a better window",
        name="probe",
        objective=objective or ObjectiveSpec(metric="loss", mode="min"),
        search_space=space or {"lr": {"type": "loguniform", "low": 1e-4, "high": 1e-1}},
        workload=WorkloadSpec(entrypoint=sys.executable),
        success_criteria=criteria or SuccessCriteria(min_objective=0.1),
        variants=variants or VariantSpec(),
    )


def test_a_goal_that_never_says_what_success_means_is_flagged():
    """The failure this catches is not a crash: the campaign runs, produces a
    best trial, and there is no written-down rule that says whether that
    number was the point. The answer then gets decided afterwards, by whoever
    is reading, which is how a max-of-24 becomes a headline.
    """
    warnings = goal_warnings(_goal(criteria=SuccessCriteria()))

    assert any("success" in w for w in warnings)


def test_a_dimension_that_changes_the_data_needs_a_baseline():
    """Two trials that fit different label sets are not competing on the same
    problem. The one with more positives scores higher without being more
    predictable, so the search optimises the slice instead of the model.
    """
    warnings = goal_warnings(
        _goal(
            space={
                "label_source": {
                    "type": "choice",
                    "values": ["work_orders", "manual"],
                    "changes_data": True,
                }
            }
        )
    )

    assert any("label_source" in w and "baseline_metric" in w for w in warnings)


def test_a_declared_baseline_settles_the_comparability_question():
    warnings = goal_warnings(
        _goal(
            objective=ObjectiveSpec(
                metric="pr_auc", baseline_metric="baseline_pr_auc", mode="max"
            ),
            space={
                "label_source": {
                    "type": "choice",
                    "values": ["work_orders", "manual"],
                    "changes_data": True,
                }
            },
        )
    )

    assert not any("baseline_metric" in w for w in warnings)


def test_a_knob_that_only_changes_the_fit_needs_no_baseline():
    assert goal_warnings(_goal()) == []


def test_asking_for_separation_without_measuring_spread_can_never_pass():
    """`aggregate: best` produces no interval, so the criterion is unmeetable
    by construction — a campaign that would report failure whatever it found.
    """
    warnings = goal_warnings(_goal(criteria=SuccessCriteria(require_separation=True)))

    assert any("aggregate" in w for w in warnings)


def test_asking_to_beat_a_baseline_that_is_not_declared_can_never_pass():
    warnings = goal_warnings(
        _goal(criteria=SuccessCriteria(require_beats_baseline=True))
    )

    assert any("baseline_metric" in w for w in warnings)


def test_a_coherent_statistical_goal_draws_no_warning():
    warnings = goal_warnings(
        _goal(
            objective=ObjectiveSpec(
                metric="pr_auc",
                baseline_metric="baseline_pr_auc",
                mode="max",
                aggregate="mean",
            ),
            space={
                "window_days": {
                    "type": "choice",
                    "values": [90, 180],
                    "changes_data": True,
                }
            },
            criteria=SuccessCriteria(
                min_observations=5,
                require_separation=True,
                require_beats_baseline=True,
            ),
        )
    )

    assert warnings == []


# --- where they surface ---------------------------------------------------


def _goal_file(tmp_path, **overrides):
    import yaml

    body = {
        "goal": "find a better window",
        "name": "probe",
        "objective": {"metric": "pr_auc", "mode": "max"},
        "search_space": {
            "label_source": {
                "type": "choice",
                "values": ["work_orders", "manual"],
                "changes_data": True,
            }
        },
        "workload": {"entrypoint": sys.executable},
    }
    body.update(overrides)
    path = tmp_path / "goal.yaml"
    path.write_text(yaml.safe_dump(body))
    return path


def test_campaign_validate_says_what_is_wrong_with_the_question(tmp_path):
    """Validate is the last moment fixing this is free."""
    from typer.testing import CliRunner

    from ai_experiments.cli import app

    result = CliRunner().invoke(
        app, ["campaign", "validate", str(_goal_file(tmp_path))]
    )

    assert result.exit_code == 0
    assert "baseline_metric" in result.stderr
    assert "success_criteria" in result.stderr


def test_a_goal_that_proves_nothing_can_be_refused_outright(tmp_path):
    """`--strict` is how CI and an agent driving the CLI stop instead of
    reading warnings nobody will read."""
    from typer.testing import CliRunner

    from ai_experiments.cli import app

    result = CliRunner().invoke(
        app, ["campaign", "validate", str(_goal_file(tmp_path)), "--strict"]
    )

    assert result.exit_code == 2


def test_variants_without_a_smoke_command_are_flagged():
    """Nothing else stands between an agent's edit and a whole round of
    trials: with no smoke command the harness records `smoke_ok: None` and
    runs it anyway, so a variant that cannot import costs the round and
    teaches nothing."""
    warnings = goal_warnings(
        _goal(variants=VariantSpec(enabled=True, editable_paths=["*.py"]))
    )

    assert any("smoke_command" in w for w in warnings)


def test_variants_that_may_write_anywhere_are_flagged():
    """`editable_paths` empty means every file in the copied workload is
    writable, including the evaluation the variant is scored by."""
    warnings = goal_warnings(
        _goal(
            variants=VariantSpec(enabled=True, smoke_command=[sys.executable, "-c", ""])
        )
    )

    assert any("editable_paths" in w for w in warnings)


def test_a_goal_without_variants_says_nothing_about_them():
    assert not any("smoke_command" in w for w in goal_warnings(_goal()))
