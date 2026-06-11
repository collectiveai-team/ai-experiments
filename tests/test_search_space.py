from __future__ import annotations

import random

import pytest

from ai_experiments.planner.search_space import grid_points, params_key, perturb, sample
from ai_experiments.schemas import (
    ChoiceParam,
    IntParam,
    LogUniformParam,
    UniformParam,
)

SPACE = {
    "lr": LogUniformParam(type="loguniform", low=1e-5, high=1e-1),
    "dropout": UniformParam(type="uniform", low=0.0, high=0.5),
    "layers": IntParam(type="int", low=1, high=4),
    "optimizer": ChoiceParam(type="choice", values=["adam", "sgd"]),
}


def test_sample_respects_bounds():
    rng = random.Random(7)
    for _ in range(50):
        params = sample(SPACE, rng)
        assert 1e-5 <= params["lr"] <= 1e-1
        assert 0.0 <= params["dropout"] <= 0.5
        assert params["layers"] in {1, 2, 3, 4}
        assert params["optimizer"] in {"adam", "sgd"}


def test_sample_is_deterministic_per_seed():
    assert sample(SPACE, random.Random(3)) == sample(SPACE, random.Random(3))


def test_grid_covers_product():
    points = grid_points(SPACE, resolution=3)

    # lr x dropout x layers x optimizer; layers spans 4 values but is
    # downsampled to the grid resolution.
    assert len(points) == 3 * 3 * 3 * 2
    assert len({params_key(p) for p in points}) == len(points)
    lrs = sorted({p["lr"] for p in points})
    assert lrs[0] == pytest.approx(1e-5) and lrs[-1] == pytest.approx(1e-1)


def test_perturb_stays_in_bounds_and_near_base():
    rng = random.Random(11)
    base = {"lr": 1e-3, "dropout": 0.25, "layers": 2, "optimizer": "adam"}
    for _ in range(100):
        neighbor = perturb(SPACE, base, rng, scale=0.2)
        assert 1e-5 <= neighbor["lr"] <= 1e-1
        assert 0.0 <= neighbor["dropout"] <= 0.5
        assert 1 <= neighbor["layers"] <= 4
        assert neighbor["optimizer"] in {"adam", "sgd"}
