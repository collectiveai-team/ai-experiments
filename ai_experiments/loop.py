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

import hashlib
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from ai_experiments.agents.prompts import review_brief
from ai_experiments.improve.rounds import RoundLog, RoundRecord
from ai_experiments.monitoring.escalation import ChangeRequest, record_change_request
from ai_experiments.orchestrator import ACTIVE_TRIAL_STATES, CampaignOrchestrator
from ai_experiments.planner.analysis import summarize_campaign
from ai_experiments.responses import (  # noqa: TC001  # LoopReport field types, resolved at class creation
    BestTrialSummary,
    CampaignHistoryEntry,
    CampaignVerdict,
    SuccessReport,
)
from ai_experiments.schemas import CampaignState, GoalSpec, ObjectiveSpec

if TYPE_CHECKING:
    from collections.abc import Callable

    from ai_experiments.agents.contracts import AgentResult
    from ai_experiments.agents.runner import AgentRunner
    from ai_experiments.store import FilesystemRunStore

TERMINAL_STATUSES = {"completed", "stopped", "failed"}

#: The campaign stopped because the code, not the search, is what blocks it.
BLOCKED_ON_CHANGE = "blocked_on_change"

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
    #: Set when the loop stopped because the blocker is a defect, not the search.
    change_request: dict[str, Any] | None = None
    #: Always set by `_report`; optional only so a report can be constructed
    #: in a test without restating the whole objective.
    objective: ObjectiveSpec | None = None
    best: BestTrialSummary | None = None
    #: Whether the best trial is actually distinguishable from the runner-up
    #: and from its baseline. A caller that reads `best` alone reports the
    #: winner of a raffle; see `campaign_verdict`.
    verdict: CampaignVerdict | None = None
    #: Whether the campaign cleared the bar its goal declared. ``met`` is
    #: ``None`` when the goal declared none, which is not a pass.
    success: SuccessReport | None = None
    history: list[CampaignHistoryEntry] = Field(default_factory=list)
    reviews: list[dict[str, Any]] = Field(default_factory=list)


@dataclass
class _LoopControls:
    """One `run_loop` call's limits, clock, and progress callback.

    All of them are the caller's, not the campaign's, which is what lets a test
    drive the same loop on a fake clock that never sleeps.
    """

    max_rounds: int | None
    max_seconds: float | None
    interval_seconds: float
    sleep: Callable[[float], None]
    now: Callable[[], float]
    started: float
    on_state: Callable[[CampaignState], None] | None

    def notify(self, state: CampaignState) -> None:
        """Show the caller the state the loop is about to act on, if it asked."""
        if self.on_state is not None:
            self.on_state(state)

    def pause(self, iterations: int) -> None:
        """Wait out the poll interval before every iteration except the first."""
        if self.interval_seconds > 0 and iterations > 1:
            self.sleep(self.interval_seconds)

    def limit_reached(self, state: CampaignState) -> str | None:
        """Name the caller-supplied limit the loop has just hit, or None."""
        if self.max_rounds is not None and state.rounds >= self.max_rounds:
            return "max_rounds"
        if self.max_seconds is not None and self.now() - self.started >= self.max_seconds:
            return "max_seconds"
        return None


@dataclass
class _LoopOutcome:
    """Where a step of the loop left the campaign, and why it would stop.

    ``loop_stop`` is None while the campaign is still worth advancing; anything
    else is the answer `run_loop` reports as its own.
    """

    state: CampaignState
    loop_stop: str | None = None
    change_request: ChangeRequest | None = None


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
    state = orchestrator.start(goal) if campaign_id is None else orchestrator.advance(campaign_id)

    controls = _LoopControls(
        max_rounds=max_rounds,
        max_seconds=max_seconds,
        interval_seconds=interval_seconds,
        sleep=sleep,
        now=now,
        started=started,
        on_state=on_state,
    )
    reviews: list[dict[str, Any]] = []
    outcome = _advance_until_done(orchestrator, store, state, controls, reviews)
    state = outcome.state

    if outcome.loop_stop in {"max_rounds", "max_seconds"}:
        # The last round was submitted and paid for. Leaving without reading it
        # loses a finished trial and leaves the campaign claiming work in
        # flight that nothing will ever collect.
        state = orchestrator.reconcile(state.campaign_id)

    controls.notify(state)
    return _report(orchestrator, state, outcome, reviews, elapsed=round(now() - started, 3))


def _advance_until_done(
    orchestrator: CampaignOrchestrator,
    store: FilesystemRunStore,
    state: CampaignState,
    controls: _LoopControls,
    reviews: list[dict[str, Any]],
) -> _LoopOutcome:
    """Advance the campaign until it ends, a limit is hit, or a review stops it.

    This is the loop and nothing else: a round's decisions live in
    `_act_on_review`, the caller's limits in `_LoopControls`.
    """
    iterations = 0
    while state.status not in TERMINAL_STATUSES:
        controls.notify(state)
        reached = controls.limit_reached(state)
        if reached is not None:
            return _LoopOutcome(state, reached)

        iterations += 1
        rounds_before = state.rounds
        controls.pause(iterations)
        state = orchestrator.advance(state.campaign_id)

        if state.rounds <= rounds_before or state.status in TERMINAL_STATUSES:
            continue
        outcome = _act_on_review(orchestrator, store, state, reviews)
        state = outcome.state
        if outcome.loop_stop is not None:
            return outcome
    return _LoopOutcome(state, "campaign_finished")


def _act_on_review(
    orchestrator: CampaignOrchestrator,
    store: FilesystemRunStore,
    state: CampaignState,
    reviews: list[dict[str, Any]],
) -> _LoopOutcome:
    """Review the round that just finished, and act on the verdict it returns.

    ``stop`` ends the campaign on the agent's judgement; ``needs_change`` files
    the ticket first, because no parameter fixes a defect and the rest of the
    budget would only buy more copies of the same failure.
    """
    review = _review(orchestrator, state, reviews)
    verdict = str(review.get("verdict", ""))
    if verdict == "stop":
        stopped = orchestrator.stop(state.campaign_id, "agent_review_stop")
        return _LoopOutcome(stopped, "agent_review_stop")
    if verdict == "needs_change":
        change = _change_request(state, review)
        record_change_request(store, change)
        stopped = orchestrator.stop(state.campaign_id, BLOCKED_ON_CHANGE)
        return _LoopOutcome(stopped, "needs_change", change)
    return _LoopOutcome(state)


def _report(
    orchestrator: CampaignOrchestrator,
    state: CampaignState,
    outcome: _LoopOutcome,
    reviews: list[dict[str, Any]],
    *,
    elapsed: float,
) -> LoopReport:
    """Turn the finished campaign into the answer `run_loop`'s caller acts on."""
    pending = [t.trial_id for t in state.trials if t.status in ACTIVE_TRIAL_STATES]
    goal = orchestrator.campaign_store.read_goal(state.campaign_id)
    summary = summarize_campaign(state, goal)
    change = outcome.change_request
    return LoopReport(
        campaign_id=state.campaign_id,
        status=state.status,
        stop_reason=state.stop_reason,
        target_reached=state.stop_reason == "target_reached",
        rounds=state.rounds,
        trials=len(state.trials),
        agent_calls=state.agent_calls,
        elapsed_seconds=elapsed,
        loop_stop=outcome.loop_stop or "campaign_finished",
        pending_trials=pending,
        change_request=change.model_dump(mode="json") if change is not None else None,
        objective=summary.objective,
        best=summary.best,
        verdict=summary.verdict,
        success=summary.success,
        history=summary.history,
        reviews=reviews,
    )


def _review(  # ast-grep-ignore: no-dict-return-annotation
    orchestrator: CampaignOrchestrator,
    state: CampaignState,
    reviews: list[dict[str, Any]],
) -> dict[str, Any]:
    """Ask the agent whether the campaign is still worth running.

    Returns the reply, or ``{}`` when no review happened — reviews are opt-in,
    budgeted like any other agent call, and a failed review never stops a
    campaign that is otherwise making progress.

    The reply is the agent's own JSON, passed through unchanged from
    ``AgentResult.payload``. There is no shape here for us to declare, only one
    an agent may or may not have honoured, so it stays a raw mapping and each
    reader validates the keys it uses.
    """
    goal = orchestrator.campaign_store.read_goal(state.campaign_id)
    if not goal.analysis.review_between_rounds:
        return {}  # ast-grep-ignore: no-dict-literal-return  # reviews are off, no reply
    if state.agent_calls >= goal.agent.max_calls:
        return {}  # ast-grep-ignore: no-dict-literal-return  # out of agent budget, no reply

    runner: AgentRunner = orchestrator.agent_runner(goal, state.campaign_id)
    summary = summarize_campaign(state, goal)
    # Serialised at the prompt boundary: the brief builders read one shape of
    # evidence, and `round_brief`'s caller already hands them `summarize_trials`
    # output. Typing one of the two and not the other would split that contract.
    brief = review_brief(goal, summary.model_dump(mode="json"))
    result = runner.run(brief, role="reviewer")
    state.agent_calls += 1
    orchestrator.campaign_store.write_state(state)

    record = _review_record(state, goal, result)
    RoundLog(orchestrator.campaign_store.campaign_dir(state.campaign_id)).append(record)
    reviews.append(record.outcome)
    if not result.ok:
        return {}  # ast-grep-ignore: no-dict-literal-return  # the call failed, no reply

    verdict = str(result.payload.get("verdict", ""))
    if verdict == "change_goal" and goal.analysis.apply_agent_changes:
        _apply_changes(orchestrator, state, goal, result.payload)
    return result.payload


def _change_request(state: CampaignState, review: dict[str, Any]) -> ChangeRequest:
    """Turn a `needs_change` verdict into a ticket a development flow can take.

    The agent supplies the diagnosis; the evidence comes from the campaign
    record, so a reader can check the claim instead of trusting it. The failed
    trials are the evidence that matters: a defect the search cannot route
    around shows up as the same error, trial after trial.
    """
    change = review.get("change") or {}
    failed = [t for t in state.trials if t.status == "failed"]
    title = str(change.get("title") or "").strip() or (
        f"{state.name}: the campaign cannot proceed without a code change"
    )
    files = [str(f) for f in (change.get("files") or [])]
    # A digest over what the ticket is about, not when it was raised: the same
    # defect escalated twice must produce the same key, so a connector can
    # refuse to start a second development flow for it.
    digest = hashlib.sha256("\x00".join([title, *sorted(files)]).encode()).hexdigest()
    return ChangeRequest(
        campaign_id=state.campaign_id,
        title=title,
        rationale=str(review.get("reason", "")),
        files=files,
        trial_ids=[t.trial_id for t in failed],
        run_ids=[t.run_id for t in failed if t.run_id],
        error_tail=_error_tail(failed),
        acceptance=str(change.get("acceptance", "")),
        source_key=f"iax:{state.campaign_id}:{digest[:12]}",
    )


#: How much workload output a ticket carries. Enough to name the failure,
#: little enough that an inbox stays readable.
ERROR_TAIL_CHARS = 2000


def _error_tail(failed: list[Any]) -> str:
    """Return the errors of the failed trials, newest last, bounded in size."""
    lines = [f"{t.trial_id}: {t.error}" for t in failed if t.error]
    return "\n".join(lines)[-ERROR_TAIL_CHARS:]


def _review_record(state: CampaignState, goal: GoalSpec, result: AgentResult) -> RoundRecord:
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
