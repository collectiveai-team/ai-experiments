"""`iax loop`: one goal in, one answer out.

Everything else in this package is a piece of the loop — plan, submit,
monitor, analyze, replan. This module is the loop itself, so that "give the
harness a goal and let it work" is one call, from a chat session or from
python, with no daemon to supervise and no ticking to script.

One iteration is one :meth:`CampaignOrchestrator.advance`. Between rounds, and
only when asked, the agent reviews: it can end a campaign it judges hopeless
instead of spending the whole budget proving it, and — with
``analysis.apply_agent_changes`` — widen a search space that provably cannot
contain the answer.

The loop always terminates. It stops on a terminal campaign status, on
``max_rounds``, or on ``max_seconds``, and it reports which of those it was.
"""

from __future__ import annotations

import time
from typing import Any, Callable

from pydantic import BaseModel, Field

from ai_experiments.agents.contracts import AgentResult
from ai_experiments.agents.prompts import review_brief
from ai_experiments.agents.runner import AgentRunner
from ai_experiments.improve.rounds import RoundLog, RoundRecord
from ai_experiments.orchestrator import ACTIVE_TRIAL_STATES, CampaignOrchestrator
from ai_experiments.planner.analysis import summarize_campaign
from ai_experiments.schemas import CampaignState, GoalSpec
from ai_experiments.store import FilesystemRunStore

TERMINAL_STATUSES = {"completed", "stopped", "failed"}

#: Why the loop returned. Only ``target_reached`` means the goal was met.
LoopStop = str


class LoopReport(BaseModel):
    """What the loop did, in the shape a caller can act on without parsing prose."""

    campaign_id: str
    status: str
    stop_reason: str | None = None
    target_reached: bool = False
    rounds: int = 0
    trials: int = 0
    agent_calls: int = 0
    elapsed_seconds: float = 0.0
    loop_stop: LoopStop = "campaign_finished"
    #: Trials still in flight when the loop returned. Non-empty means the
    #: campaign has unread work: resume it before you conclude anything.
    pending_trials: list[str] = Field(default_factory=list)
    objective: dict[str, Any] = Field(default_factory=dict)
    best: dict[str, Any] | None = None
    history: list[dict[str, Any]] = Field(default_factory=list)
    reviews: list[dict[str, Any]] = Field(default_factory=list)


def run_loop(
    goal: GoalSpec,
    store: FilesystemRunStore,
    *,
    orchestrator: CampaignOrchestrator | None = None,
    campaign_id: str | None = None,
    max_rounds: int | None = None,
    max_seconds: float | None = None,
    interval_seconds: float = 5.0,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], float] = time.monotonic,
    on_state: Callable[[CampaignState], None] | None = None,
) -> LoopReport:
    """Drive one campaign to its goal, its budget, or the limits given here.

    ``campaign_id`` resumes an existing campaign instead of starting a new one,
    which is what makes an interrupted overnight loop restartable.
    """
    orchestrator = orchestrator or CampaignOrchestrator(store)
    started = now()

    if campaign_id is None:
        state = orchestrator.start(goal)
    else:
        state = orchestrator.advance(campaign_id)

    reviews: list[dict[str, Any]] = []
    loop_stop = "campaign_finished"
    iterations = 0

    while state.status not in TERMINAL_STATUSES:
        if on_state is not None:
            on_state(state)
        if max_rounds is not None and state.rounds >= max_rounds:
            loop_stop = "max_rounds"
            break
        if max_seconds is not None and now() - started >= max_seconds:
            loop_stop = "max_seconds"
            break

        iterations += 1
        rounds_before = state.rounds
        if interval_seconds > 0 and iterations > 1:
            sleep(interval_seconds)
        state = orchestrator.advance(state.campaign_id)

        if state.rounds > rounds_before and state.status not in TERMINAL_STATUSES:
            verdict = _review(orchestrator, state, reviews)
            if verdict == "stop":
                state = orchestrator.stop(state.campaign_id, "agent_review_stop")
                loop_stop = "agent_review_stop"
                break

    if loop_stop in {"max_rounds", "max_seconds"}:
        # The last round was submitted and paid for. Leaving without reading it
        # loses a finished trial and leaves the campaign claiming work in
        # flight that nothing will ever collect.
        state = orchestrator.reconcile(state.campaign_id)

    if on_state is not None:
        on_state(state)

    pending = [t.trial_id for t in state.trials if t.status in ACTIVE_TRIAL_STATES]
    goal = orchestrator.campaign_store.read_goal(state.campaign_id)
    summary = summarize_campaign(state, goal)
    return LoopReport(
        campaign_id=state.campaign_id,
        status=state.status,
        stop_reason=state.stop_reason,
        target_reached=state.stop_reason == "target_reached",
        rounds=state.rounds,
        trials=len(state.trials),
        agent_calls=state.agent_calls,
        elapsed_seconds=round(now() - started, 3),
        loop_stop=loop_stop,
        pending_trials=pending,
        objective=summary["objective"],
        best=summary["best"],
        history=summary["history"],
        reviews=reviews,
    )


def _review(
    orchestrator: CampaignOrchestrator,
    state: CampaignState,
    reviews: list[dict[str, Any]],
) -> str:
    """Ask the agent whether the campaign is still worth running.

    Returns the verdict, or ``""`` when no review happened — reviews are
    opt-in, budgeted like any other agent call, and a failed review never
    stops a campaign that is otherwise making progress.
    """
    goal = orchestrator.campaign_store.read_goal(state.campaign_id)
    if not goal.analysis.review_between_rounds:
        return ""
    if state.agent_calls >= goal.agent.max_calls:
        return ""

    runner: AgentRunner = orchestrator.agent_runner(goal, state.campaign_id)
    summary = summarize_campaign(state, goal)
    result = runner.run(review_brief(goal, summary), role="reviewer")
    state.agent_calls += 1
    orchestrator.campaign_store.write_state(state)

    record = _review_record(state, goal, result)
    RoundLog(orchestrator.campaign_store.campaign_dir(state.campaign_id)).append(record)
    reviews.append(record.outcome)
    if not result.ok:
        return ""

    verdict = str(result.payload.get("verdict", ""))
    if verdict == "change_goal" and goal.analysis.apply_agent_changes:
        _apply_changes(orchestrator, state, goal, result.payload)
    return verdict


def _review_record(
    state: CampaignState, goal: GoalSpec, result: AgentResult
) -> RoundRecord:
    payload = result.payload if result.ok else {}
    return RoundRecord(
        campaign_id=state.campaign_id,
        round=state.rounds,
        stage="review",
        strategy=goal.strategy.name,
        rationale=str(payload.get("reason", "")),
        agent_calls=state.agent_calls,
        outcome={
            "verdict": payload.get("verdict", ""),
            "reason": payload.get("reason", ""),
            "observations": payload.get("observations", []),
            "suggested_changes": payload.get("suggested_changes", {}),
            "agent_error": result.error,
        },
    )


def _apply_changes(
    orchestrator: CampaignOrchestrator,
    state: CampaignState,
    goal: GoalSpec,
    payload: dict[str, Any],
) -> None:
    """Merge an accepted review's changes into the goal.

    Only the search space and the budget can move, and only through the same
    validation `iax campaign edit` uses. An invalid suggestion is recorded and
    dropped: the campaign continues under the goal it already has.
    """
    changes = payload.get("suggested_changes")
    if not isinstance(changes, dict):
        return
    data = goal.model_dump(mode="json")
    for key in ("search_space", "budget"):
        section = changes.get(key)
        if isinstance(section, dict) and section:
            data[key] = {**data[key], **section}
    try:
        orchestrator.edit_goal(state.campaign_id, GoalSpec(**data))
    except Exception as exc:
        orchestrator.campaign_store.append_event(
            state.campaign_id,
            _rejected_change_event(str(exc)),
        )


def _rejected_change_event(error: str):
    from ai_experiments.schemas import RunEvent

    return RunEvent(
        level="warning",
        message="agent review suggested an invalid goal change; keeping the current goal",
        details={"error": error},
    )
