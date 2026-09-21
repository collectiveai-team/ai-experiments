from __future__ import annotations

import json

from ai_experiments.planner.planner import build_trial_manifest, plan_next_params
from ai_experiments.planner.search_space import params_key
from ai_experiments.schemas import (
    BudgetSpec,
    GoalSpec,
    ObjectiveSpec,
    StrategySpec,
    TrialRecord,
    WorkloadSpec,
)


def _goal(**overrides: object) -> GoalSpec:
    data: dict = {
        "goal": "minimize toy loss",
        "name": "toy",
        "objective": ObjectiveSpec(metric="loss", mode="min"),
        "search_space": {
            "lr": {"type": "loguniform", "low": 1e-4, "high": 1e-1},
            "layers": {"type": "int", "low": 1, "high": 3},
        },
        "workload": WorkloadSpec(entrypoint="python train.py", args=["--lr", "{lr}"]),
        "budget": BudgetSpec(max_trials=6, max_parallel=2),
        "strategy": StrategySpec(name="random", seed=1),
    }
    data.update(overrides)
    return GoalSpec(**data)


def test_manifest_substitutes_placeholders_and_appends_rest():
    goal = _goal()
    manifest = build_trial_manifest(goal, "t000", {"lr": 0.001, "layers": 2})

    assert manifest.workload.args[:2] == ["--lr", "0.001"]
    assert manifest.workload.args[2:] == ["--layers", "2"]
    assert manifest.experiment == "toy/t000"
    assert json.loads(manifest.workload.env["IAX_PARAMS"]) == {
        "lr": 0.001,
        "layers": 2,
    }
    assert manifest.metadata["trial_id"] == "t000"
    assert manifest.metadata["params"] == {"lr": 0.001, "layers": 2}


def test_goal_propagates_objective_metric_to_monitoring():
    goal = _goal()
    manifest = build_trial_manifest(goal, "t000", {"lr": 0.001, "layers": 2})

    assert manifest.monitoring.objective_metric == "loss"


def test_random_planning_avoids_duplicates_and_respects_count():
    goal = _goal()
    first = plan_next_params(goal, [], 3)
    trials = [
        TrialRecord(trial_id=f"t{i:03d}", params=params)
        for i, params in enumerate(first)
    ]
    second = plan_next_params(goal, trials, 3)

    keys = {params_key(p) for p in first} | {params_key(p) for p in second}
    assert len(first) == 3 and len(second) == 3
    assert len(keys) == 6


def test_grid_strategy_exhausts_grid():
    goal = _goal(
        strategy=StrategySpec(name="grid", grid_resolution=2),
        search_space={
            "lr": {"type": "choice", "values": [0.1, 0.01]},
            "layers": {"type": "int", "low": 1, "high": 2},
        },
    )
    first = plan_next_params(goal, [], 10)
    assert len(first) == 4

    trials = [
        TrialRecord(trial_id=f"t{i:03d}", params=params)
        for i, params in enumerate(first)
    ]
    assert plan_next_params(goal, trials, 10) == []


def test_adaptive_strategy_exploits_best_region():
    goal = _goal(
        strategy=StrategySpec(name="adaptive", seed=5, exploration=0.0, top_k=1)
    )
    trials = [
        TrialRecord(
            trial_id="t000",
            params={"lr": 0.001, "layers": 2},
            status="completed",
            objective_value=0.1,
        ),
        TrialRecord(
            trial_id="t001",
            params={"lr": 0.05, "layers": 1},
            status="completed",
            objective_value=5.0,
        ),
        TrialRecord(
            trial_id="t002",
            params={"lr": 0.09, "layers": 3},
            status="completed",
            objective_value=9.0,
        ),
    ]
    planned = plan_next_params(goal, trials, 4)

    assert len(planned) == 4
    # With exploration=0 and top_k=1 every new trial perturbs the best (lr=0.001);
    # perturbations live in log-space, so they stay well below the bad region.
    for params in planned:
        assert params["lr"] < 0.05


def test_underscore_params_reach_the_workload_as_hyphen_flags():
    """`argparse` declares `--label-source`; a search space key cannot.

    Search space keys are Python identifiers, so a two-word parameter is
    `label_source`. Every CLI convention spells the flag `--label-source`,
    and argparse rejects long options it did not declare, so emitting the
    key verbatim fails every trial of an otherwise correct workload.
    """
    goal = _goal(
        search_space={"label_source": {"type": "choice", "values": ["a", "b"]}},
        workload=WorkloadSpec(entrypoint="python train.py"),
    )
    manifest = build_trial_manifest(goal, "t000", {"label_source": "a"})

    assert manifest.workload.args == ["--label-source", "a"]


def test_only_one_spelling_of_a_flag_is_ever_sent():
    """Sending both spellings to be safe is the bug, not the fix.

    argparse rejects any long option it did not declare, so a workload that
    accepts `--label-source` dies on the `--label_source` sent beside it.
    """
    goal = _goal(
        search_space={"label_source": {"type": "choice", "values": ["a"]}},
        workload=WorkloadSpec(entrypoint="python train.py"),
    )
    args = build_trial_manifest(goal, "t000", {"label_source": "a"}).workload.args

    assert args.count("a") == 1
    assert "--label_source" not in args


def test_a_workload_can_ask_for_the_underscore_spelling():
    goal = _goal(
        search_space={"label_source": {"type": "choice", "values": ["a"]}},
        workload=WorkloadSpec(entrypoint="python train.py", flag_style="underscore"),
    )
    args = build_trial_manifest(goal, "t000", {"label_source": "a"}).workload.args

    assert args == ["--label_source", "a"]


def test_placeholders_keep_the_python_name():
    """The flag style is about flags; `{name}` refers to the search space key."""
    goal = _goal(
        search_space={"label_source": {"type": "choice", "values": ["a"]}},
        workload=WorkloadSpec(
            entrypoint="python train.py", args=["--src", "{label_source}"]
        ),
    )
    args = build_trial_manifest(goal, "t000", {"label_source": "a"}).workload.args

    assert args == ["--src", "a"]


def test_iax_params_keeps_the_search_space_spelling():
    """The env carries the assignment, so it uses the goal's own names."""
    goal = _goal(
        search_space={"label_source": {"type": "choice", "values": ["a"]}},
        workload=WorkloadSpec(entrypoint="python train.py"),
    )
    env = build_trial_manifest(goal, "t000", {"label_source": "a"}).workload.env

    assert json.loads(env["IAX_PARAMS"]) == {"label_source": "a"}
