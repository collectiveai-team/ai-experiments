"""The briefs the harness sends an agent, and the replies it will accept.

One prompt per role. Each one states the goal, the evidence so far — including
the trials that *failed* and why, which is where most of the signal is — and
the exact JSON the harness will parse back. Prose outside the JSON is allowed
and ignored; agents reason better when they are allowed to think first.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
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
  "verdict": "continue" | "stop" | "change_goal" | "needs_change",
  "reason": "one sentence",
  "observations": ["what the evidence actually shows"],
  "suggested_changes": {"search_space": {}},
  "change": {"title": "", "files": [], "acceptance": ""}
}

Use "stop" when further trials cannot reach the target, and "change_goal" when
the search space is what blocks it. "search_space" is the only section a
verdict can change: the budget is the ceiling the person running this campaign
set, and a request to raise it is refused and recorded rather than applied. If
the budget is what blocks the target, say so in "reason" and let them decide;
you may redistribute effort inside it, but you may not widen it.

Use "needs_change" only when no choice of parameters can help, because the
defect is in the code — every trial failing on the same error, a workload that
declares no result, a harness that returns NaN at the edge of the space. It
stops the campaign and sends a ticket to a developer, so it costs more than a
wrong "continue". Fill "change": point "files" at what the evidence names, and
write "acceptance" as the single check that would prove the fix."""


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
    objective = goal.objective
    lines = [
        f"Objective: {objective.mode}imize `{objective.metric}`{target}. "
        "The workload declares it on stdout as "
        f'`IAX_RESULT {{"{objective.metric}": value}}`, from its evaluate '
        "phase. That declared result is the only thing scored; `IAX_METRIC` lines "
        "are progress only and are never scored."
    ]
    if objective.baseline_metric:
        # Scores are lifts, not levels. An agent that does not know this
        # proposes whatever raises the base rate.
        lines.append(
            f"Scores are the lift `{objective.metric} - "
            f"{objective.baseline_metric}`, paired within one declared result, so "
            "a change that raises both is worth nothing."
        )
    if objective.aggregate == "mean":
        lines.append(
            "A trial's score is averaged over all its declared results, not taken "
            "from the best one, and carries a standard error; a difference "
            "smaller than that error is not evidence."
        )
    return " ".join(lines)


def _search_space_block(goal: GoalSpec) -> str:
    space = {
        name: spec.model_dump(mode="json", exclude_none=True)
        for name, spec in goal.search_space.items()
    }
    block = "Search space:\n" + json.dumps(space, indent=2)
    conditional = sorted(name for name, spec in goal.search_space.items() if spec.when)
    if conditional:
        block += (
            f"\n{', '.join(conditional)} carry a `when` condition: include each "
            "one only in the trials whose other parameters satisfy it, and omit "
            "it everywhere else. An assignment that sets a key whose condition "
            "does not hold is rejected."
        )
    return block


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
        params = json.dumps(entry.get("params", {}))
        line = f"- {entry.get('trial_id')} [{entry.get('status')}] {scored} {params}"
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
