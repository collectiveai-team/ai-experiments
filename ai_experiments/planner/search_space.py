"""Sampling, grid expansion, and perturbation over a search space."""

from __future__ import annotations

import itertools
import math
from typing import TYPE_CHECKING, Any

from ai_experiments.schemas import (
    ChoiceParam,
    IntParam,
    LogUniformParam,
    ParamSpec,
    UniformParam,
)

if TYPE_CHECKING:
    import random


def applies(spec: ParamSpec, params: dict[str, Any]) -> bool:
    """Whether this dimension exists for a trial with these parameters."""
    return all(params.get(name) in values for name, values in spec.when.items())


def active_space(  # ast-grep-ignore: no-dict-return-annotation
    space: dict[str, ParamSpec], params: dict[str, Any]
) -> dict[str, ParamSpec]:
    """Keep only the dimensions a trial with these parameters actually has.

    A gradient-boosting knob is not a dimension of a logistic regression
    trial; it is a value the run will never read. Everything downstream —
    sampling, the grid, validation, the command line — asks this first.
    """
    return {name: spec for name, spec in space.items() if applies(spec, params)}


def _partition(
    space: dict[str, ParamSpec],
) -> tuple[dict[str, ParamSpec], dict[str, ParamSpec]]:
    """Unconditional dimensions, then the ones that depend on them.

    Conditions are one level deep by construction (``GoalSpec`` refuses a
    condition on a conditional key), so this ordering is all the dependency
    resolution anyone needs.
    """
    free = {name: spec for name, spec in space.items() if not spec.when}
    return free, {name: spec for name, spec in space.items() if spec.when}


def sample(  # ast-grep-ignore: no-dict-return-annotation
    space: dict[str, ParamSpec], rng: random.Random
) -> dict[str, Any]:
    """Draw one random parameter assignment.

    Keys are the user's own search-space parameter names (from `space`), not
    a fixed schema -- a sampled hyperparameter assignment, one value per
    parameter the goal's search space defines.
    """
    free, conditional = _partition(space)
    params = {name: _sample_param(spec, rng) for name, spec in free.items()}
    params.update(
        {
            name: _sample_param(spec, rng)
            for name, spec in conditional.items()
            if applies(spec, params)
        }
    )
    return params


def _sample_param(spec: ParamSpec, rng: random.Random) -> Any:
    if isinstance(spec, ChoiceParam):
        return rng.choice(spec.values)
    if isinstance(spec, UniformParam):
        return rng.uniform(spec.low, spec.high)
    if isinstance(spec, LogUniformParam):
        return math.exp(rng.uniform(math.log(spec.low), math.log(spec.high)))
    if isinstance(spec, IntParam):
        return rng.randint(spec.low, spec.high)
    raise TypeError(f"unsupported param spec: {spec!r}")


def grid_points(space: dict[str, ParamSpec], resolution: int = 4) -> list[dict[str, Any]]:
    """Expand the space into a full grid (continuous params get `resolution` steps).

    Conditional dimensions multiply out only over the points that have them,
    so a two-model grid does not spend half its points on combinations one of
    the models cannot read.
    """
    free, conditional = _partition(space)
    points: list[dict[str, Any]] = []
    for base in _product(free, resolution):
        extra = {name: spec for name, spec in conditional.items() if applies(spec, base)}
        points.extend({**base, **combo} for combo in _product(extra, resolution))
    return points


def _product(space: dict[str, ParamSpec], resolution: int) -> list[dict[str, Any]]:
    names = sorted(space)
    axes = [_grid_axis(space[name], resolution) for name in names]
    return [dict(zip(names, combo, strict=False)) for combo in itertools.product(*axes)]


def _grid_axis(spec: ParamSpec, resolution: int) -> list[Any]:
    if isinstance(spec, ChoiceParam):
        return list(spec.values)
    if isinstance(spec, IntParam):
        span = spec.high - spec.low
        if span < resolution:
            return list(range(spec.low, spec.high + 1))
        return sorted({spec.low + round(i * span / (resolution - 1)) for i in range(resolution)})
    if isinstance(spec, UniformParam):
        step = (spec.high - spec.low) / (resolution - 1)
        return [spec.low + i * step for i in range(resolution)]
    if isinstance(spec, LogUniformParam):
        log_low, log_high = math.log(spec.low), math.log(spec.high)
        step = (log_high - log_low) / (resolution - 1)
        return [math.exp(log_low + i * step) for i in range(resolution)]
    raise TypeError(f"unsupported param spec: {spec!r}")


def perturb(  # ast-grep-ignore: no-dict-return-annotation
    space: dict[str, ParamSpec],
    base: dict[str, Any],
    rng: random.Random,
    scale: float = 0.2,
) -> dict[str, Any]:
    """Sample a neighbor of `base`.

    Gaussian moves for numeric params (log-space for loguniform), a re-draw with
    probability `scale` for choices. Keys are the user's own search-space
    parameter names, not a fixed schema. A conditional dimension whose
    condition the neighbour no longer meets is dropped rather than carried.
    """
    free, conditional = _partition(space)
    result: dict[str, Any] = {}
    for name, spec in free.items():
        value = base.get(name)
        result[name] = (
            _sample_param(spec, rng) if value is None else _perturb_param(spec, value, rng, scale)
        )
    for name, spec in conditional.items():
        if not applies(spec, result):
            # The neighbour flipped across the condition: this knob is not a
            # dimension here, and carrying the old value over would put a
            # parameter in the trial record that the run never reads.
            continue
        value = base.get(name)
        result[name] = (
            _sample_param(spec, rng) if value is None else _perturb_param(spec, value, rng, scale)
        )
    return result


def _perturb_param(spec: ParamSpec, value: Any, rng: random.Random, scale: float) -> Any:
    if isinstance(spec, ChoiceParam):
        if len(spec.values) > 1 and rng.random() < scale:
            return rng.choice([v for v in spec.values if v != value])
        return value
    if isinstance(spec, UniformParam):
        span = spec.high - spec.low
        moved = float(value) + rng.gauss(0, scale * span)
        return min(max(moved, spec.low), spec.high)
    if isinstance(spec, LogUniformParam):
        log_span = math.log(spec.high) - math.log(spec.low)
        moved = math.exp(math.log(float(value)) + rng.gauss(0, scale * log_span))
        return min(max(moved, spec.low), spec.high)
    if isinstance(spec, IntParam):
        span = max(spec.high - spec.low, 1)
        moved = round(int(value) + rng.gauss(0, max(scale * span, 1.0)))
        return min(max(moved, spec.low), spec.high)
    raise TypeError(f"unsupported param spec: {spec!r}")


def params_key(params: dict[str, Any]) -> str:
    """Stable identity for duplicate detection."""
    return repr(sorted(params.items()))
