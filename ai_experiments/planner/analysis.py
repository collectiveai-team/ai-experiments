"""Post-run analysis: objective extraction and campaign summaries."""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, Field

from ai_experiments.schemas import CampaignState, GoalSpec, ObjectiveSpec, TrialRecord
from ai_experiments.store import FilesystemRunStore


class ObjectiveReading(BaseModel):
    """What a finished run actually reported about the objective.

    ``value`` alone cannot distinguish "the workload never reported" from
    "the objective metric is misspelled" from "every value was NaN" — three
    failures with three different fixes. ``miss_reason`` names which one it
    was, so the orchestrator can put it on the trial instead of scoring
    ``null`` in silence (#11).
    """

    value: float | None = None
    final_metrics: dict[str, float] = Field(default_factory=dict)
    observed_metrics: list[str] = Field(default_factory=list)
    miss_reason: Literal["no_metrics", "metric_absent", "not_finite"] | None = None

    def miss_message(self, metric: str) -> str | None:
        if self.miss_reason is None:
            return None
        if self.miss_reason == "no_metrics":
            return (
                "no metrics reported: the workload printed no IAX_METRIC lines, "
                f"so objective '{metric}' could not be scored"
            )
        observed = ", ".join(self.observed_metrics) or "(none)"
        if self.miss_reason == "metric_absent":
            return (
                f"objective metric '{metric}' was never reported; "
                f"observed metrics: {observed}"
            )
        return (
            f"objective metric '{metric}' was reported but never finite "
            f"(NaN/inf only); observed metrics: {observed}"
        )


def extract_objective(
    store: FilesystemRunStore, run_id: str, objective: ObjectiveSpec
) -> ObjectiveReading:
    """Best observed objective value for a run, plus why it is missing."""
    metrics = store.read_metrics(run_id)
    if not metrics:
        return ObjectiveReading(miss_reason="no_metrics")

    observed = sorted({name for point in metrics for name in point.values})
    final = dict(metrics[-1].values)
    if objective.metric not in observed:
        return ObjectiveReading(
            final_metrics=final,
            observed_metrics=observed,
            miss_reason="metric_absent",
        )

    values = [
        point.values[objective.metric]
        for point in metrics
        if objective.metric in point.values
        and math.isfinite(point.values[objective.metric])
    ]
    if not values:
        return ObjectiveReading(
            final_metrics=final,
            observed_metrics=observed,
            miss_reason="not_finite",
        )
    best = max(values) if objective.mode == "max" else min(values)
    return ObjectiveReading(value=best, final_metrics=final, observed_metrics=observed)


def is_improvement(candidate: float, incumbent: float | None, mode: str) -> bool:
    if incumbent is None:
        return True
    return candidate > incumbent if mode == "max" else candidate < incumbent


def best_of(trials: list[TrialRecord], mode: str) -> TrialRecord | None:
    """The best *completed* trial.

    A crashed trial can report a good value moments before it dies — an OOM
    kill mid-epoch, a diverging run that prints one lucky step. Letting such a
    value win would end the campaign on a result nobody can reproduce, so only
    trials that ran to completion are eligible (#12). Failed trials stay in
    the history the agent reasons over; they just cannot be the answer.
    """
    scored = [
        t
        for t in trials
        if t.status == "completed"
        and t.objective_value is not None
        and math.isfinite(t.objective_value)
    ]
    if not scored:
        return None
    key = (
        (lambda t: -t.objective_value)
        if mode == "max"
        else (lambda t: t.objective_value)
    )  # type: ignore[operator]
    return min(scored, key=key)  # type: ignore[arg-type]


def best_trial(state: CampaignState, mode: str) -> TrialRecord | None:
    return best_of(state.trials, mode)


def trial_history(trials: list[TrialRecord]) -> list[dict[str, Any]]:
    """Every trial an agent can learn from: scored ones and failures alike."""
    return [
        {
            "trial_id": t.trial_id,
            "status": t.status,
            "objective_value": t.objective_value,
            "params": t.params,
            "error": t.error,
        }
        for t in trials
        if t.objective_value is not None or t.error is not None
    ]


def summarize_trials(trials: list[TrialRecord], goal: GoalSpec) -> dict[str, Any]:
    """The evidence block an agent plans from, without a CampaignState.

    A strategy sees trials, not campaigns. This is the same history and best
    trial that ``summarize_campaign`` reports, so the agent and the dashboard
    never disagree about what happened.
    """
    best = best_of(trials, goal.objective.mode)
    return {
        "trials_total": len(trials),
        "history": trial_history(trials),
        "best": (
            {
                "trial_id": best.trial_id,
                "run_id": best.run_id,
                "objective_value": best.objective_value,
                "params": best.params,
            }
            if best
            else None
        ),
    }


def summarize_campaign(state: CampaignState, goal: GoalSpec) -> dict[str, Any]:
    by_status: dict[str, int] = {}
    for trial in state.trials:
        by_status[trial.status] = by_status.get(trial.status, 0) + 1
    best = best_trial(state, goal.objective.mode)
    history = trial_history(state.trials)
    gpu_hours = sum(t.gpu_hours or 0.0 for t in state.trials)
    cost = (
        gpu_hours * goal.budget.gpu_hour_rate
        if goal.budget.gpu_hour_rate is not None
        else None
    )
    return {
        "campaign_id": state.campaign_id,
        "name": state.name,
        "goal": state.goal,
        "status": state.status,
        "stop_reason": state.stop_reason,
        "gpu_hours": round(gpu_hours, 4),
        "estimated_cost": round(cost, 2) if cost is not None else None,
        "budget": {
            "max_trials": goal.budget.max_trials,
            "max_gpu_hours": goal.budget.max_gpu_hours,
            "gpu_hour_rate": goal.budget.gpu_hour_rate,
        },
        "objective": {
            "metric": goal.objective.metric,
            "mode": goal.objective.mode,
            "target": goal.objective.target,
        },
        "rounds": state.rounds,
        "trials_by_status": by_status,
        "trials_total": len(state.trials),
        "best": (
            {
                "trial_id": best.trial_id,
                "run_id": best.run_id,
                "objective_value": best.objective_value,
                "params": best.params,
            }
            if best
            else None
        ),
        "history": history,
    }
