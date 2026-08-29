"""`strategy: agent` must be useful when the agent works and harmless when it does not."""

from __future__ import annotations

from ai_experiments.agents import AgentResult, StubAgentRunner
from ai_experiments.agents.strategy import AgentStrategy
from ai_experiments.schemas import GoalSpec, TrialRecord

GOAL = GoalSpec(
    goal="Get loss under 0.05",
    name="agent-strategy",
    objective={"metric": "loss", "mode": "min", "target": 0.05},
    search_space={
        "lr": {"type": "uniform", "low": 0.0, "high": 1.0},
        "layers": {"type": "int", "low": 2, "high": 8},
    },
    workload={"entrypoint": "python train.py"},
    budget={"max_trials": 20, "max_parallel": 2},
    strategy={"name": "agent", "seed": 1},
)


def test_valid_proposals_become_the_round():
    runner = StubAgentRunner(
        [
            {
                "hypothesis": "the learning rate is too high",
                "rationale": "bracket it an order of magnitude lower",
                "trials": [
                    {"params": {"lr": 0.01, "layers": 4}},
                    {"params": {"lr": 0.05, "layers": 6}},
                ],
            }
        ]
    )
    strategy = AgentStrategy(runner)

    params = strategy.plan(GOAL, [], count=2)

    assert params == [{"lr": 0.01, "layers": 4}, {"lr": 0.05, "layers": 6}]
    assert strategy.last_decision.hypothesis.startswith("the learning rate")
    assert not strategy.last_decision.used_fallback


def test_out_of_range_proposals_are_dropped_not_submitted():
    runner = StubAgentRunner(
        [
            {
                "trials": [
                    {"params": {"lr": 42.0, "layers": 4}},
                    {"params": {"lr": 0.2, "layers": 4}},
                ]
            }
        ]
    )
    strategy = AgentStrategy(runner)

    params = strategy.plan(GOAL, [], count=2)

    assert params == [{"lr": 0.2, "layers": 4}]
    assert len(strategy.last_decision.rejected) == 1
    assert "lr" in strategy.last_decision.rejected[0]["reason"]


def test_a_repeat_of_an_existing_trial_is_dropped():
    """Re-running the same point spends budget and teaches nothing."""
    tried = TrialRecord(trial_id="t000", params={"lr": 0.2, "layers": 4})
    runner = StubAgentRunner(
        [
            {
                "trials": [
                    {"params": {"lr": 0.2, "layers": 4}},
                    {"params": {"lr": 0.3, "layers": 4}},
                ]
            }
        ]
    )
    strategy = AgentStrategy(runner)

    params = strategy.plan(GOAL, [tried], count=2)

    assert params == [{"lr": 0.3, "layers": 4}]
    assert "already tried" in strategy.last_decision.rejected[0]["reason"]


def test_a_failing_agent_falls_back_instead_of_stalling_the_campaign():
    runner = StubAgentRunner([AgentResult(error="agent exited 7: rate limited")])
    strategy = AgentStrategy(runner, fallback="random")

    params = strategy.plan(GOAL, [], count=3)

    assert len(params) == 3
    assert strategy.last_decision.used_fallback
    assert "rate limited" in strategy.last_decision.agent_error


def test_all_invalid_proposals_fall_back_too():
    runner = StubAgentRunner([{"trials": [{"params": {"lr": 99.0, "layers": 99}}]}])
    strategy = AgentStrategy(runner, fallback="random")

    params = strategy.plan(GOAL, [], count=2)

    assert len(params) == 2
    assert strategy.last_decision.used_fallback
    assert "no usable trials" in strategy.last_decision.fallback_reason


def test_a_reply_without_json_falls_back():
    runner = StubAgentRunner(["I would try a smaller learning rate."])
    strategy = AgentStrategy(runner, fallback="random")

    params = strategy.plan(GOAL, [], count=1)

    assert len(params) == 1
    assert strategy.last_decision.used_fallback


def test_the_agent_can_end_the_round_by_asking_to_stop():
    runner = StubAgentRunner(
        [{"stop": True, "rationale": "the space is exhausted at this resolution"}]
    )
    strategy = AgentStrategy(runner)

    params = strategy.plan(GOAL, [], count=3)

    assert params == []
    assert strategy.last_decision.stop
    assert "exhausted" in strategy.last_decision.stop_reason
    assert not strategy.last_decision.used_fallback


def test_more_proposals_than_asked_for_are_truncated():
    runner = StubAgentRunner(
        [{"trials": [{"params": {"lr": i / 10, "layers": 4}} for i in range(1, 8)]}]
    )
    strategy = AgentStrategy(runner)

    params = strategy.plan(GOAL, [], count=2)

    assert len(params) == 2


def test_the_brief_the_agent_sees_names_the_goal_and_the_history():
    runner = StubAgentRunner([{"trials": [{"params": {"lr": 0.1, "layers": 3}}]}])
    strategy = AgentStrategy(runner)
    failed = TrialRecord(
        trial_id="t000",
        params={"lr": 0.9, "layers": 8},
        status="failed",
        error="workload exited 1",
    )

    strategy.plan(GOAL, [failed], count=1)

    brief = runner.prompts[0]
    assert "Get loss under 0.05" in brief
    assert "t000" in brief and "workload exited 1" in brief
