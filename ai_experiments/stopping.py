"""How a campaign ends: the documented reasons, and the checks that pick one.

A campaign ends with one word, and that word is the whole answer for the agent
reading `summary.json`. Keeping the vocabulary and the checks that emit it in
one module means the two cannot drift: `STOP_REASONS` below is the contract the
skills and the README document, and `tests/test_stop_reasons.py` reads this file
to prove nothing emits a word it does not explain.

Every check is a pure function of the campaign's recorded state, so the
orchestrator decides *when* to ask and this module decides *what the answer is*.
"""

from __future__ import annotations

from datetime import timezone
from typing import TYPE_CHECKING

from ai_experiments.planner.analysis import best_trial
from ai_experiments.schemas import utc_now

if TYPE_CHECKING:
    from datetime import datetime

    from ai_experiments.agents.strategy import AgentDecision
    from ai_experiments.schemas import CampaignState, GoalSpec, TrialRecord, TrialState
    from ai_experiments.store import FilesystemRunStore

#: Trial states the campaign is still waiting on. A trial in any other state
#: has already produced its answer, so the budget checks below count it as
#: spent rather than as work in flight.
ACTIVE_TRIAL_STATES: set[TrialState] = {"submitted", "running"}

#: Stop reasons that mean the campaign broke rather than finished.
FAILURE_STOP_REASONS = {"objective_not_reported", "backend_unavailable"}

#: How many trials may complete without a usable objective before the
#: campaign is declared broken instead of merely unlucky.
MIN_TRIALS_BEFORE_CONTRACT_CHECK = 2

#: Stop reason recorded when the agent itself says more trials cannot help.
AGENT_STOP_REASON = "agent_requested_stop"

#: Every reason a campaign can end with, and what the reader should do about
#: it. This is the contract the skills and the README document, so a new reason
#: added without a line here fails `tests/test_stop_reasons.py`.
STOP_REASONS: dict[str, str] = {
    "target_reached": "the objective target was met; the best trial is the answer",
    "budget_exhausted": "max_trials were run without reaching the target",
    "max_hours_exceeded": "budget.max_hours elapsed",
    "gpu_hours_exhausted": "budget.max_gpu_hours were spent",
    "search_space_exhausted": "the planner ran out of points; widen the goal",
    "backend_unavailable": "no trial could be submitted; start the cluster",
    "objective_not_reported": "trials ran but never reported the objective metric",
    AGENT_STOP_REASON: "the reviewing agent judged the campaign hopeless",
    "user_requested": "`iax campaign stop`",
}


def stop_reason(state: CampaignState, goal: GoalSpec, gpu_hours_spent: float) -> str | None:
    """Return the first stop condition that fires, in priority order.

    A broken objective contract outranks everything else: continuing would
    spend the rest of the budget on trials that can only score `null`.
    After that `target_reached` outranks the budget/time checks, so a
    campaign that has both hit its target and exhausted its budget reports
    `target_reached`.

    `gpu_hours_spent` is passed in rather than computed here because the live
    figure needs the run store, which is the orchestrator's to hold.
    """
    return (
        objective_contract_broken(state)
        or target_reached_reason(state, goal)
        or max_hours_reason(state, goal)
        or gpu_hours_reason(goal, gpu_hours_spent)
        or budget_exhausted_reason(state, goal)
    )


def exhausted_reason(
    submitted: list[TrialRecord],
    submit_errors: list[str],
    decision: AgentDecision | None,
) -> str:
    """Name the reason the campaign has nothing left to do.

    `search_space_exhausted` means the planner ran out of points, and the
    answer is to widen the goal. A backend that refused every submit ran
    out of nothing, and the answer is to start the cluster (#36).
    """
    if submit_errors and not submitted:
        return "backend_unavailable"
    if decision is not None and decision.stop:
        return AGENT_STOP_REASON
    return "search_space_exhausted"


def objective_contract_broken(state: CampaignState) -> str | None:
    """Stop early when the workload never reports the objective.

    Continuing would spend the whole budget on trials that can only score
    ``null`` — the campaign would then end as a "successful"
    ``budget_exhausted`` with no best trial and no explanation (#11).
    """
    completed = [t for t in state.trials if t.status == "completed"]
    if len(completed) < MIN_TRIALS_BEFORE_CONTRACT_CHECK:
        return None
    if any(t.objective_value is not None for t in completed):
        return None
    return "objective_not_reported"


def target_reached_reason(state: CampaignState, goal: GoalSpec) -> str | None:
    """Report whether the best trial so far has met the objective's target."""
    best = best_trial(state, goal.objective.mode)
    target = goal.objective.target
    if best is None or target is None or best.objective_value is None:
        return None
    reached = (
        best.objective_value >= target
        if goal.objective.mode == "max"
        else best.objective_value <= target
    )
    return "target_reached" if reached else None


def max_hours_reason(state: CampaignState, goal: GoalSpec) -> str | None:
    """Report whether the campaign has outlived `budget.max_hours`."""
    if goal.budget.max_hours is None:
        return None
    # `created_at` may be persisted without a timezone; normalise before comparing.
    created = state.created_at
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    age_hours = (utc_now() - created).total_seconds() / 3600
    if age_hours > goal.budget.max_hours:
        return "max_hours_exceeded"
    return None


def gpu_hours_spent(state: CampaignState, goal: GoalSpec, run_store: FilesystemRunStore) -> float:
    """GPU-hours consumed so far.

    Recorded for finished trials, a live estimate (started -> now) for trials still
    running -- which is why this one needs the run store while its siblings do not.
    """
    total = sum(t.gpu_hours or 0.0 for t in state.trials)
    for trial in state.trials:
        if trial.status in ACTIVE_TRIAL_STATES and trial.run_id:
            status = run_store.read_status(trial.run_id)
            live = trial_gpu_hours(goal, status.started_at or status.submitted_at, utc_now())
            total += live or 0.0
    return total


def gpu_hours_reason(goal: GoalSpec, spent: float) -> str | None:
    """Report whether `spent` has reached `budget.max_gpu_hours`."""
    if goal.budget.max_gpu_hours is not None and spent >= goal.budget.max_gpu_hours:
        return "gpu_hours_exhausted"
    return None


def budget_exhausted_reason(state: CampaignState, goal: GoalSpec) -> str | None:
    """Report whether `budget.max_trials` have been run with none still in flight."""
    active = [t for t in state.trials if t.status in ACTIVE_TRIAL_STATES]
    planned = [t for t in state.trials if t.status == "planned"]
    if len(state.trials) >= goal.budget.max_trials and not active and not planned:
        return "budget_exhausted"
    return None


def trial_gpu_hours(
    goal: GoalSpec, started: datetime | None, completed: datetime | None
) -> float | None:
    """GPU-hours one trial consumed between two timestamps.

    `None` means "not answerable yet" — a trial that reserves GPUs but has not
    started cannot be costed. A goal that asks for no GPUs always costs 0.0.
    """
    if goal.resources.gpus <= 0 or started is None or completed is None:
        return 0.0 if goal.resources.gpus <= 0 else None
    if started.tzinfo is None:
        started = started.replace(tzinfo=timezone.utc)
    if completed.tzinfo is None:
        completed = completed.replace(tzinfo=timezone.utc)
    hours = max((completed - started).total_seconds(), 0.0) / 3600
    return hours * goal.resources.gpus
