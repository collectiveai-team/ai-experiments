"""Post-run analysis: objective extraction and campaign summaries."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

from ai_experiments.schemas import (
    BestTrialSummary,
    BudgetSummary,
    CampaignHistoryEntry,
    CampaignSummary,
)

if TYPE_CHECKING:
    from ai_experiments.schemas import CampaignState, GoalSpec, ObjectiveSpec, TrialRecord
    from ai_experiments.store import FilesystemRunStore


def extract_objective(
    store: FilesystemRunStore, run_id: str, objective: ObjectiveSpec
) -> tuple[float | None, dict[str, float]]:
    """Best observed objective value for a run, plus the final metric snapshot."""
    metrics = store.read_metrics(run_id)
    values = [
        point.values[objective.metric]
        for point in metrics
        if objective.metric in point.values and math.isfinite(point.values[objective.metric])
    ]
    best = None
    if values:
        best = max(values) if objective.mode == "max" else min(values)
    final = dict(metrics[-1].values) if metrics else {}
    return best, final


def is_improvement(candidate: float, incumbent: float | None, mode: str) -> bool:
    if incumbent is None:
        return True
    return candidate > incumbent if mode == "max" else candidate < incumbent


def best_trial(state: CampaignState, mode: str) -> TrialRecord | None:
    scored = [
        t
        for t in state.trials
        if t.objective_value is not None and math.isfinite(t.objective_value)
    ]
    if not scored:
        return None

    def _key(trial: TrialRecord) -> float:
        # scored is filtered above to non-None, finite objective_value; cast makes
        # that already-proven invariant visible to the type checker.
        value = cast("float", trial.objective_value)
        return -value if mode == "max" else value

    return min(scored, key=_key)


def summarize_campaign(state: CampaignState, goal: GoalSpec) -> CampaignSummary:
    by_status: dict[str, int] = {}
    for trial in state.trials:
        by_status[trial.status] = by_status.get(trial.status, 0) + 1
    best = best_trial(state, goal.objective.mode)
    history = [
        CampaignHistoryEntry(
            trial_id=t.trial_id,
            objective_value=t.objective_value,
            params=t.params,
        )
        for t in state.trials
        if t.objective_value is not None
    ]
    gpu_hours = sum(t.gpu_hours or 0.0 for t in state.trials)
    cost = gpu_hours * goal.budget.gpu_hour_rate if goal.budget.gpu_hour_rate is not None else None
    return CampaignSummary(
        campaign_id=state.campaign_id,
        name=state.name,
        goal=state.goal,
        status=state.status,
        stop_reason=state.stop_reason,
        gpu_hours=round(gpu_hours, 4),
        estimated_cost=round(cost, 2) if cost is not None else None,
        budget=BudgetSummary(
            max_trials=goal.budget.max_trials,
            max_gpu_hours=goal.budget.max_gpu_hours,
            gpu_hour_rate=goal.budget.gpu_hour_rate,
        ),
        # copy, don't alias: `budget` above is a fresh BudgetSummary, and a summary that
        # shared the goal's ObjectiveSpec instance would let a mutation of either reach the
        # other. pydantic v2 stores the instance as-is, so the copy has to be explicit.
        objective=goal.objective.model_copy(),
        rounds=state.rounds,
        trials_by_status=by_status,
        trials_total=len(state.trials),
        best=(
            BestTrialSummary(
                trial_id=best.trial_id,
                run_id=best.run_id,
                objective_value=best.objective_value,
                params=best.params,
            )
            if best
            else None
        ),
        history=history,
    )
