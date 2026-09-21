"""The brief must carry enough evidence for an agent to plan a better round."""

from __future__ import annotations

from ai_experiments.agents.prompts import review_brief, round_brief
from ai_experiments.schemas import GoalSpec

GOAL = GoalSpec(
    goal="Get validation loss under 0.05",
    name="brief",
    objective={"metric": "val_loss", "mode": "min", "target": 0.05},
    search_space={
        "lr": {"type": "loguniform", "low": 0.0001, "high": 0.1},
        "layers": {"type": "int", "low": 2, "high": 8},
    },
    workload={"entrypoint": "python train.py"},
    budget={"max_trials": 10, "max_parallel": 2},
)


def _summary(history: list[dict], best: dict | None = None) -> dict:
    return {"trials_total": len(history), "history": history, "best": best}


def test_round_brief_states_the_goal_the_space_and_the_contract():
    brief = round_brief(GOAL, _summary([]), max_trials=3)

    assert "Get validation loss under 0.05" in brief
    assert "val_loss" in brief and "0.05" in brief
    assert "loguniform" in brief and "layers" in brief
    assert '"trials"' in brief
    assert "return between 1 and 3 trials" in brief


def test_round_brief_shows_failures_and_their_errors():
    """The agent cannot fix a broken workload it is never told about."""
    history = [
        {
            "trial_id": "t1",
            "status": "failed",
            "objective_value": None,
            "params": {"lr": 0.1, "layers": 8},
            "error": "objective metric 'val_loss' was never reported; observed: loss",
        },
        {
            "trial_id": "t2",
            "status": "completed",
            "objective_value": 0.3,
            "params": {"lr": 0.01, "layers": 4},
            "error": None,
        },
    ]

    brief = round_brief(GOAL, _summary(history, best=history[1]), max_trials=2)

    assert "t1" in brief and "failed" in brief
    assert "was never reported" in brief
    assert "t2" in brief and "0.3" in brief


def test_round_brief_says_so_when_nothing_has_finished():
    brief = round_brief(GOAL, _summary([]), max_trials=4)

    assert "no trial has finished yet" in brief


def test_review_brief_asks_for_a_verdict():
    brief = review_brief(GOAL, _summary([]))

    assert '"verdict"' in brief
    assert "change_goal" in brief


def test_the_brief_says_the_score_is_a_lift_when_it_is_one():
    """An agent told to maximise `pr_auc` will reason about published PR-AUCs
    and propose whatever raises the base rate. The number it is actually
    moving is the lift over the trial's own baseline.
    """
    goal = GOAL.model_copy(
        update={
            "objective": GOAL.objective.model_copy(
                update={
                    "metric": "pr_auc",
                    "baseline_metric": "baseline_pr_auc",
                    "mode": "max",
                    "aggregate": "mean",
                    "target": None,
                }
            )
        }
    )

    brief = round_brief(goal, _summary([]), max_trials=3)

    assert "baseline_pr_auc" in brief
    assert "averaged" in brief


def test_the_brief_spells_out_which_keys_only_apply_to_some_trials():
    """Without this the agent proposes every key for every trial, and the
    orchestrator rejects the assignment it just paid an agent call for."""
    goal = GOAL.model_copy(
        update={
            "search_space": GoalSpec(
                goal="g",
                name="n",
                objective={"metric": "val_loss", "mode": "min"},
                search_space={
                    "model": {"type": "choice", "values": ["hist_gb", "logreg"]},
                    "max_leaf_nodes": {
                        "type": "int",
                        "low": 8,
                        "high": 64,
                        "when": {"model": ["hist_gb"]},
                    },
                },
                workload={"entrypoint": "python train.py"},
            ).search_space
        }
    )

    brief = round_brief(goal, _summary([]), max_trials=3)

    assert "max_leaf_nodes" in brief
    assert "`when`" in brief or "when" in brief
    assert "omit" in brief
