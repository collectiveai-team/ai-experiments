"""`strategy: agent` — the agent proposes the next round.

This is the one strategy that is not pure, so it lives here and not under
``planner/`` (CONVENTIONS.md §3): it shells out to an agent, which is I/O.

The contract with the agent is deliberately narrow. It reads the goal, the
search space and every trial so far — failures included — and returns params.
The harness validates each proposal against the search space and drops what
does not fit. If nothing survives, or the agent is down, over budget, or
talking nonsense, the campaign falls back to a built-in strategy and records
why. An agent outage must slow a campaign down, never stop it.
"""

from __future__ import annotations

from typing import Any, Callable

from pydantic import BaseModel, Field

from ai_experiments.agents.contracts import AgentResult
from ai_experiments.agents.prompts import round_brief
from ai_experiments.agents.runner import AgentRunner
from ai_experiments.planner.analysis import summarize_trials
from ai_experiments.planner.search_space import params_key
from ai_experiments.planner.strategies import get_strategy
from ai_experiments.planner.validation import ParamValidationError, validate_params
from ai_experiments.schemas import GoalSpec, TrialRecord


class AgentDecision(BaseModel):
    """What the agent said, and what the harness did with it.

    Persisted into the round record, so "why did it try that?" has an answer
    the morning after.
    """

    hypothesis: str = ""
    rationale: str = ""
    stop: bool = False
    stop_reason: str = ""
    accepted: list[dict[str, Any]] = Field(default_factory=list)
    rejected: list[dict[str, Any]] = Field(default_factory=list)
    used_fallback: bool = False
    fallback_reason: str = ""
    agent_error: str = ""
    raw: str = ""


class AgentStrategy:
    """Plans a round by asking an agent, with a built-in strategy as backstop."""

    def __init__(
        self,
        runner: AgentRunner,
        fallback: str = "adaptive",
        on_result: Callable[[AgentResult], None] | None = None,
    ) -> None:
        self.runner = runner
        self.fallback = fallback
        self.on_result = on_result
        self.last_decision = AgentDecision()

    def plan(
        self, goal: GoalSpec, trials: list[TrialRecord], count: int
    ) -> list[dict[str, Any]]:
        if count <= 0:
            return []
        brief = round_brief(goal, summarize_trials(trials, goal), count)
        result = self.runner.run(brief, role="planner")
        if self.on_result is not None:
            self.on_result(result)

        decision = AgentDecision(raw=result.raw)
        if not result.ok:
            decision.agent_error = result.error
            return self._fall_back(goal, trials, count, decision, result.error)

        payload = result.payload
        decision.hypothesis = str(payload.get("hypothesis", ""))[:2000]
        decision.rationale = str(payload.get("rationale", ""))[:4000]
        if payload.get("stop") is True:
            decision.stop = True
            decision.stop_reason = decision.rationale or "the agent asked to stop"
            self.last_decision = decision
            return []

        accepted = self._accept(goal, trials, payload.get("trials"), count, decision)
        if accepted:
            decision.accepted = accepted
            self.last_decision = decision
            return accepted
        return self._fall_back(
            goal, trials, count, decision, "the agent proposed no usable trials"
        )

    def _accept(
        self,
        goal: GoalSpec,
        trials: list[TrialRecord],
        proposals: Any,
        count: int,
        decision: AgentDecision,
    ) -> list[dict[str, Any]]:
        if not isinstance(proposals, list):
            decision.rejected.append(
                {"params": proposals, "reason": "'trials' was not a list"}
            )
            return []
        seen = {params_key(t.params) for t in trials}
        accepted: list[dict[str, Any]] = []
        for proposal in proposals[: count * 4]:
            params = proposal.get("params") if isinstance(proposal, dict) else proposal
            if not isinstance(params, dict):
                decision.rejected.append(
                    {"params": proposal, "reason": "no 'params' object"}
                )
                continue
            try:
                coerced = validate_params(goal.search_space, params)
            except ParamValidationError as exc:
                decision.rejected.append({"params": params, "reason": str(exc)})
                continue
            key = params_key(coerced)
            if key in seen:
                decision.rejected.append(
                    {"params": coerced, "reason": "already tried in this campaign"}
                )
                continue
            seen.add(key)
            accepted.append(coerced)
            if len(accepted) >= count:
                break
        return accepted

    def _fall_back(
        self,
        goal: GoalSpec,
        trials: list[TrialRecord],
        count: int,
        decision: AgentDecision,
        reason: str,
    ) -> list[dict[str, Any]]:
        decision.used_fallback = True
        decision.fallback_reason = reason
        params = get_strategy(self.fallback).plan(goal, trials, count)
        decision.accepted = params
        self.last_decision = decision
        return params
