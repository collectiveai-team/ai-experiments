"""Validate a parameter assignment against a goal's search space.

Anything that proposes a trial — a human with ``iax campaign suggest``, an
agent strategy, the REST API — goes through here. Without it an unknown key
becomes a literal ``--not_a_param 42`` on the workload's command line and
surfaces much later as a *failed trial* rather than immediately as a rejected
proposal (#13).
"""

from __future__ import annotations

import math
from typing import Any

from ai_experiments.schemas import (
    ChoiceParam,
    IntParam,
    LogUniformParam,
    ParamSpec,
    UniformParam,
)


class ParamValidationError(ValueError):
    """Raised with every violation found, not just the first."""

    def __init__(self, violations: list[str]) -> None:
        self.violations = violations
        super().__init__("; ".join(violations))


def validate_params(
    space: dict[str, ParamSpec], params: dict[str, Any]
) -> dict[str, Any]:
    """Return `params` coerced to the space's types, or raise.

    Missing keys are rejected too: a partial assignment silently inherits the
    workload's own defaults for the rest, which makes the trial's recorded
    params a lie about what ran.
    """
    violations: list[str] = []
    unknown = sorted(set(params) - set(space))
    if unknown:
        violations.append(
            f"unknown parameter(s) {unknown}; the search space defines {sorted(space)}"
        )
    missing = sorted(set(space) - set(params))
    if missing:
        violations.append(f"missing parameter(s) {missing}")

    coerced: dict[str, Any] = {}
    for name, spec in space.items():
        if name not in params:
            continue
        try:
            coerced[name] = _coerce(name, spec, params[name])
        except ValueError as exc:
            violations.append(str(exc))

    if violations:
        raise ParamValidationError(violations)
    return coerced


def _coerce(name: str, spec: ParamSpec, value: Any) -> Any:
    if isinstance(spec, ChoiceParam):
        if value in spec.values:
            return value
        raise ValueError(f"{name}={value!r} is not one of {spec.values}")

    if isinstance(spec, IntParam):
        as_int = _as_int(name, value)
        if not spec.low <= as_int <= spec.high:
            raise ValueError(f"{name}={value!r} is outside [{spec.low}, {spec.high}]")
        return as_int

    if isinstance(spec, (UniformParam, LogUniformParam)):
        as_float = _as_float(name, value)
        if not spec.low <= as_float <= spec.high:
            raise ValueError(f"{name}={value!r} is outside [{spec.low}, {spec.high}]")
        return as_float

    raise ValueError(f"{name}: unsupported param spec {spec!r}")


def _as_int(name: str, value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name}={value!r} is not an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    raise ValueError(f"{name}={value!r} is not an integer")


def _as_float(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name}={value!r} is not a number")
    as_float = float(value)
    if not math.isfinite(as_float):
        raise ValueError(f"{name}={value!r} is not finite")
    return as_float
