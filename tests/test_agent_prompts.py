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
