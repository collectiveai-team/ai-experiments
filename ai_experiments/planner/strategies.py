"""Planning strategies: which parameter assignments to try next.

All strategies are deterministic given (goal.strategy.seed, prior trials), so
re-planning after a crash reproduces the same decisions.
"""

from __future__ import annotations

import math
import random
from typing import Any, Protocol

from ai_experiments.planner.search_space import grid_points, params_key, perturb, sample
from ai_experiments.schemas import GoalSpec, TrialRecord

MAX_SAMPLE_ATTEMPTS = 50


class Strategy(Protocol):
    def plan(
        self, goal: GoalSpec, trials: list[TrialRecord], count: int
    ) -> list[dict[str, Any]]:
        """Return up to `count` new parameter assignments given trial history."""
        ...


def get_strategy(name: str) -> Strategy:
    strategies: dict[str, Strategy] = {
        "grid": GridStrategy(),
        "random": RandomStrategy(),
        "adaptive": AdaptiveStrategy(),
    }
    if name not in strategies:
        raise ValueError(f"unknown strategy: {name}")
    return strategies[name]


def _seen_keys(trials: list[TrialRecord]) -> set[str]:
    return {params_key(t.params) for t in trials}


def _rng(goal: GoalSpec, trials: list[TrialRecord]) -> random.Random:
    return random.Random(f"{goal.strategy.seed}:{len(trials)}")


def _fresh_samples(
    goal: GoalSpec,
    trials: list[TrialRecord],
    count: int,
    draw: Any,
) -> list[dict[str, Any]]:
    """Collect `count` assignments from `draw()`, skipping duplicates."""
    seen = _seen_keys(trials)
    picked: list[dict[str, Any]] = []
    attempts = 0
    while len(picked) < count and attempts < MAX_SAMPLE_ATTEMPTS * count:
        attempts += 1
        params = draw()
        key = params_key(params)
        if key in seen:
            continue
        seen.add(key)
        picked.append(params)
    return picked


class RandomStrategy:
    def plan(
        self, goal: GoalSpec, trials: list[TrialRecord], count: int
    ) -> list[dict[str, Any]]:
        rng = _rng(goal, trials)
        return _fresh_samples(
            goal, trials, count, lambda: sample(goal.search_space, rng)
        )


class GridStrategy:
    def plan(
        self, goal: GoalSpec, trials: list[TrialRecord], count: int
    ) -> list[dict[str, Any]]:
        seen = _seen_keys(trials)
        remaining = [
            params
            for params in grid_points(goal.search_space, goal.strategy.grid_resolution)
            if params_key(params) not in seen
        ]
        return remaining[:count]


class AdaptiveStrategy:
    """Explore randomly, then exploit by perturbing around the best trials.

    Warmup: pure random until a few results exist. Afterwards each new trial
    is, with probability `strategy.exploration`, a fresh random sample, and
    otherwise a gaussian perturbation of one of the `strategy.top_k` best
    completed trials.
    """

    WARMUP_RESULTS = 3

    def plan(
        self, goal: GoalSpec, trials: list[TrialRecord], count: int
    ) -> list[dict[str, Any]]:
        rng = _rng(goal, trials)
        scored = [
            t
            for t in trials
            if t.objective_value is not None and math.isfinite(t.objective_value)
        ]
        if len(scored) < self.WARMUP_RESULTS:
            return _fresh_samples(
                goal, trials, count, lambda: sample(goal.search_space, rng)
            )

        reverse = goal.objective.mode == "max"
        ranked = sorted(
            scored,
            key=lambda t: t.objective_value,
            reverse=reverse,  # type: ignore[arg-type, return-value]
        )
        top = ranked[: max(goal.strategy.top_k, 1)]

        def draw() -> dict[str, Any]:
            if rng.random() < goal.strategy.exploration:
                return sample(goal.search_space, rng)
            anchor = rng.choice(top)
            return perturb(goal.search_space, anchor.params, rng)

        return _fresh_samples(goal, trials, count, draw)
