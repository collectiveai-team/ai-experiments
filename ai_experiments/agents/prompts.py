"""The briefs the harness sends an agent, and the replies it will accept.

One prompt per role. Each one states the goal, the evidence so far — including
the trials that *failed* and why, which is where most of the signal is — and
the exact JSON the harness will parse back. Prose outside the JSON is allowed
and ignored; agents reason better when they are allowed to think first.
"""

from __future__ import annotations

import json
from typing import Any

from ai_experiments.schemas import GoalSpec

PLANNER_CONTRACT = """Reply with one JSON object, last thing in your message:

{
  "hypothesis": "one sentence: what you think limits the objective right now",
  "rationale": "why these trials test it",
  "trials": [{"params": {"<name>": <value>}, "note": "what this one probes"}],
  "stop": false
}

Rules:
- every key in "params" must be a parameter of the search space, and every
  parameter must be present;
- values must respect the bounds and choices given below; out-of-range trials
  are dropped and wasted;
- return between 1 and {max_trials} trials;
- set "stop": true only when more trials cannot help (the space is exhausted,
  or the workload itself is broken) and say why in "rationale"."""

REVIEW_CONTRACT = """Reply with one JSON object, last thing in your message:

{
  "verdict": "continue" | "stop" | "change_goal",
  "reason": "one sentence",
  "observations": ["what the evidence actually shows"],
  "suggested_changes": {"search_space": {}, "budget": {}}
}

Use "stop" when further trials cannot reach the target, and "change_goal" when
the search space or the budget is what blocks it."""


def round_brief(goal: GoalSpec, summary: dict[str, Any], max_trials: int) -> str:
    """Ask the agent for the next batch of trials."""
    return "\n\n".join(
        [
            "You are planning the next round of an automated experiment campaign.",
            f"Goal: {goal.goal}",
            _objective_block(goal),
            _search_space_block(goal),
            _budget_block(goal, summary),
            _evidence_block(summary),
            PLANNER_CONTRACT.replace("{max_trials}", str(max_trials)),
        ]
    )


def review_brief(goal: GoalSpec, summary: dict[str, Any]) -> str:
    """Ask the agent whether the campaign is still worth running."""
    return "\n\n".join(
        [
            "You are reviewing an automated experiment campaign between rounds.",
            f"Goal: {goal.goal}",
            _objective_block(goal),
            _search_space_block(goal),
            _budget_block(goal, summary),
            _evidence_block(summary),
            REVIEW_CONTRACT,
        ]
    )


def _objective_block(goal: GoalSpec) -> str:
    target = (
        f", target {goal.objective.target}"
        if goal.objective.target is not None
        else ", no explicit target"
    )
    return (
        f"Objective: {goal.objective.mode}imize `{goal.objective.metric}`{target}. "
        'The workload reports it on stdout as `IAX_METRIC {"step": n, '
        f'"{goal.objective.metric}": value}}`.'
    )


def _search_space_block(goal: GoalSpec) -> str:
    space = {
        name: spec.model_dump(mode="json", exclude_none=True)
        for name, spec in goal.search_space.items()
    }
    return "Search space:\n" + json.dumps(space, indent=2)


def _budget_block(goal: GoalSpec, summary: dict[str, Any]) -> str:
    used = summary.get("trials_total", 0)
    return (
        f"Budget: {used} of {goal.budget.max_trials} trials used, "
        f"{goal.budget.max_parallel} may run at once."
    )


def _evidence_block(summary: dict[str, Any]) -> str:
    history = summary.get("history") or []
    if not history:
        return "Evidence: no trial has finished yet. Propose a spread that covers the space."
    lines = ["Evidence so far (every trial, failures included):"]
    for entry in history:
        value = entry.get("objective_value")
        scored = f"{value:.6g}" if isinstance(value, (int, float)) else "no value"
        line = f"- {entry.get('trial_id')} [{entry.get('status')}] {scored} {json.dumps(entry.get('params', {}))}"
        if entry.get("error"):
            line += f" -- error: {entry['error']}"
        lines.append(line)
    best = summary.get("best")
    if best:
        lines.append(
            f"Best completed trial: {best['trial_id']} = {best['objective_value']} "
            f"at {json.dumps(best['params'])}"
        )
    return "\n".join(lines)
