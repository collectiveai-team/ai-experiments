from __future__ import annotations

import pytest

from ai_experiments.planner.validation import ParamValidationError, validate_params
from ai_experiments.schemas import GoalSpec

SPACE = GoalSpec(
    goal="g",
    name="n",
    objective={"metric": "loss"},
    search_space={
        "lr": {"type": "loguniform", "low": 1e-5, "high": 1e-2},
        "layers": {"type": "int", "low": 1, "high": 4},
        "opt": {"type": "choice", "values": ["adam", "sgd"]},
        "dropout": {"type": "uniform", "low": 0.0, "high": 0.5},
    },
    workload={"entrypoint": "python t.py"},
).search_space


def test_valid_params_round_trip():
    params = {"lr": 1e-3, "layers": 2, "opt": "adam", "dropout": 0.1}
    assert validate_params(SPACE, params) == params


def test_integer_valued_float_is_coerced():
    result = validate_params(
        SPACE, {"lr": 1e-3, "layers": 2.0, "opt": "sgd", "dropout": 0.0}
    )
    assert result["layers"] == 2
    assert isinstance(result["layers"], int)


def test_unknown_key_is_rejected():
    with pytest.raises(ParamValidationError, match="unknown parameter"):
        validate_params(
            SPACE,
            {"lr": 1e-3, "layers": 2, "opt": "adam", "dropout": 0.1, "not_a_param": 42},
        )


def test_missing_key_is_rejected():
    with pytest.raises(ParamValidationError, match="missing parameter"):
        validate_params(SPACE, {"lr": 1e-3})


def test_out_of_range_values_are_rejected():
    with pytest.raises(ParamValidationError) as exc:
        validate_params(SPACE, {"lr": 1.0, "layers": 9, "opt": "adam", "dropout": 0.1})
    assert len(exc.value.violations) == 2


def test_choice_outside_the_set_is_rejected():
    with pytest.raises(ParamValidationError, match="not one of"):
        validate_params(
            SPACE, {"lr": 1e-3, "layers": 2, "opt": "rmsprop", "dropout": 0.1}
        )


def test_non_integer_layers_is_rejected():
    with pytest.raises(ParamValidationError, match="not an integer"):
        validate_params(
            SPACE, {"lr": 1e-3, "layers": 2.5, "opt": "adam", "dropout": 0.1}
        )


def test_every_violation_is_reported_at_once():
    with pytest.raises(ParamValidationError) as exc:
        validate_params(SPACE, {"nope": 1})
    assert any("unknown" in v for v in exc.value.violations)
    assert any("missing" in v for v in exc.value.violations)
