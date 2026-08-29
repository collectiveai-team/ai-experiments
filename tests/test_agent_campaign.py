"""A campaign whose rounds an agent plans, end to end, with no real agent.

The point of `strategy: agent` is that the loop keeps its own guarantees: the
budget still binds, a broken agent still yields progress, and every proposal
is still validated against the search space before a trial is submitted.
"""

from __future__ import annotations

import json

from ai_experiments.agents import AgentResult, StubAgentRunner
from ai_experiments.orchestrator import AGENT_STOP_REASON, CampaignOrchestrator
from ai_experiments.schemas import GoalSpec
from ai_experiments.store import FilesystemRunStore
from ai_experiments.store.campaign import CampaignStore
from tests.test_orchestrator import FakeBackend


def _agent_goal(**overrides) -> GoalSpec:
    data: dict = {
        "goal": "minimize (x-2)^2",
        "name": "agent-quadratic",
        "objective": {"metric": "loss", "mode": "min", "target": 0.01},
        "search_space": {"x": {"type": "uniform", "low": -5.0, "high": 5.0}},
        "workload": {"entrypoint": "python toy.py"},
        "budget": {"max_trials": 6, "max_parallel": 1},
        "strategy": {"name": "agent", "seed": 3, "fallback": "random"},
    }
    data.update(overrides)
    return GoalSpec(**data)


def _harness(tmp_path, runner):
    store = FilesystemRunStore(tmp_path / "runs")
    backend = FakeBackend(store)
    orchestrator = CampaignOrchestrator(
        store,
        CampaignStore(store.root),
        backend_factory=lambda goal: backend,
        agent_runner_factory=lambda goal, campaign_id: runner,
    )
    return orchestrator, backend, store


def _drive(orchestrator, state, limit=20):
    for _ in range(limit):
        state = orchestrator.advance(state.campaign_id)
        if state.status != "running":
            break
    return state


def test_the_agent_walks_the_campaign_to_its_target(tmp_path):
    """Each reply moves x closer to 2, where the objective hits its target."""
    runner = StubAgentRunner(
        [
            {"hypothesis": "start wide", "trials": [{"params": {"x": -4.0}}]},
            {"hypothesis": "too low", "trials": [{"params": {"x": 0.0}}]},
            {"hypothesis": "closer", "trials": [{"params": {"x": 2.05}}]},
        ]
    )
    orchestrator, _, _ = _harness(tmp_path, runner)

    state = _drive(orchestrator, orchestrator.start(_agent_goal()))

    assert state.status == "completed"
    assert state.stop_reason == "target_reached"
    assert [t.params["x"] for t in state.trials][:3] == [-4.0, 0.0, 2.05]
    assert state.agent_calls >= 3


def test_the_brief_grows_with_the_evidence(tmp_path):
    runner = StubAgentRunner(
        [
            {"trials": [{"params": {"x": -4.0}}]},
            {"trials": [{"params": {"x": 0.0}}]},
        ]
    )
    orchestrator, _, _ = _harness(tmp_path, runner)

    _drive(orchestrator, orchestrator.start(_agent_goal()), limit=3)

    assert "no trial has finished yet" in runner.prompts[0]
    assert "t000" in runner.prompts[1]


def test_a_dead_agent_still_produces_a_finished_campaign(tmp_path):
    runner = StubAgentRunner([AgentResult(error="agent command could not run")])
    orchestrator, backend, _ = _harness(tmp_path, runner)

    state = _drive(orchestrator, orchestrator.start(_agent_goal()))

    assert state.status == "completed"
    assert len(state.trials) > 0
    assert all(t.params["x"] == t.params["x"] for t in state.trials)


def test_an_out_of_range_proposal_never_reaches_the_backend(tmp_path):
    runner = StubAgentRunner([{"trials": [{"params": {"x": 500.0}}]}])
    orchestrator, backend, _ = _harness(tmp_path, runner)

    _drive(orchestrator, orchestrator.start(_agent_goal()), limit=2)

    submitted = [m.metadata["params"]["x"] for m in backend.submitted]
    assert submitted
    assert all(-5.0 <= x <= 5.0 for x in submitted)


def test_the_agent_can_end_the_campaign(tmp_path):
    runner = StubAgentRunner(
        [{"stop": True, "rationale": "the workload ignores x entirely"}]
    )
    orchestrator, _, _ = _harness(tmp_path, runner)

    state = _drive(orchestrator, orchestrator.start(_agent_goal()))

    assert state.status == "completed"
    assert state.stop_reason == AGENT_STOP_REASON
    assert state.trials == []


def test_the_agent_call_budget_binds(tmp_path):
    """An unattended loop that keeps asking is a loop that keeps spending."""
    runner = StubAgentRunner([{"trials": [{"params": {"x": 4.0}}]}])
    orchestrator, _, _ = _harness(tmp_path, runner)
    goal = _agent_goal(agent={"command": "unused", "max_calls": 2})

    state = _drive(orchestrator, orchestrator.start(goal))

    assert state.agent_calls == 2
    assert len(runner.prompts) == 2
    # The campaign kept planning on the fallback rather than stalling.
    assert len(state.trials) > 2


def test_the_agents_reasoning_lands_in_the_campaign_events(tmp_path):
    runner = StubAgentRunner(
        [
            {
                "hypothesis": "x is far from the optimum",
                "rationale": "bracket the minimum from the left",
                "trials": [{"params": {"x": -1.0}}],
            }
        ]
    )
    orchestrator, _, store = _harness(tmp_path, runner)

    state = orchestrator.start(_agent_goal())

    events = [
        json.loads(line)
        for line in (
            CampaignStore(store.root).campaign_dir(state.campaign_id) / "events.jsonl"
        )
        .read_text()
        .splitlines()
    ]
    planned = [e for e in events if e["message"] == "agent planned a round"]
    assert planned
    assert planned[0]["details"]["hypothesis"] == "x is far from the optimum"
    assert planned[0]["details"]["used_fallback"] is False
