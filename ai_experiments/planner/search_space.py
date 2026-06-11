"""Sampling, grid expansion, and perturbation over a search space."""

from __future__ import annotations

import itertools
import math
import random
from typing import Any

from ai_experiments.schemas import (
    ChoiceParam,
    IntParam,
    LogUniformParam,
    ParamSpec,
    UniformParam,
)


def sample(space: dict[str, ParamSpec], rng: random.Random) -> dict[str, Any]:
    """Draw one random parameter assignment."""
    return {name: _sample_param(spec, rng) for name, spec in space.items()}


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


def grid_points(
    space: dict[str, ParamSpec], resolution: int = 4
) -> list[dict[str, Any]]:
    """Expand the space into a full grid (continuous params get `resolution` steps)."""
    names = sorted(space)
    axes = [_grid_axis(space[name], resolution) for name in names]
    return [dict(zip(names, combo)) for combo in itertools.product(*axes)]


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


def perturb(
    space: dict[str, ParamSpec],
    base: dict[str, Any],
    rng: random.Random,
    scale: float = 0.2,
) -> dict[str, Any]:
    """Sample a neighbor of `base`: gaussian moves for numeric params (log-space
    for loguniform), a re-draw with probability `scale` for choices."""
    result: dict[str, Any] = {}
    for name, spec in space.items():
        value = base.get(name)
        if value is None:
            result[name] = _sample_param(spec, rng)
        else:
            result[name] = _perturb_param(spec, value, rng, scale)
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
