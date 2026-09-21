"""Post-run analysis: objective extraction and campaign summaries."""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterator
from typing import Any, Literal

from pydantic import BaseModel, Field

from ai_experiments.schemas import (
    CampaignState,
    GoalSpec,
    MetricPoint,
    ObjectiveSpec,
    SuccessCriteria,
    TrialRecord,
)
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
    #: Standard error of ``value`` across the observations it averages, and
    #: ``None`` whenever the spread is not measurable — a ``best`` objective,
    #: or a single observation. Reporting 0.0 there would claim a certainty
    #: one sample cannot support.
    stderr: float | None = None
    #: How many observations carried a usable value.
    n_observations: int = 0
    final_metrics: dict[str, float] = Field(default_factory=dict)
    observed_metrics: list[str] = Field(default_factory=list)
    miss_reason: (
        Literal["no_metrics", "metric_absent", "not_finite", "baseline_absent"] | None
    ) = None

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
        if self.miss_reason == "baseline_absent":
            return (
                f"objective metric '{metric}' was reported but its baseline "
                f"never was, so the trial has no comparable score; "
                f"observed metrics: {observed}"
            )
        return (
            f"objective metric '{metric}' was reported but never finite "
            f"(NaN/inf only); observed metrics: {observed}"
        )


def extract_objective(
    store: FilesystemRunStore, run_id: str, objective: ObjectiveSpec
) -> ObjectiveReading:
    """The run's objective value, its uncertainty, and why it is missing."""
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

    values = list(_scored(metrics, objective))
    if not values:
        # A baseline the workload never reported is its own failure: scoring
        # the raw metric instead would rank this trial on a different scale
        # from the rest, which is the whole thing `baseline_metric` prevents.
        reason = (
            "baseline_absent"
            if objective.baseline_metric is not None
            and objective.baseline_metric not in observed
            else "not_finite"
        )
        return ObjectiveReading(
            final_metrics=final,
            observed_metrics=observed,
            miss_reason=reason,
        )
    value, stderr = _aggregate(values, objective)
    return ObjectiveReading(
        value=value,
        stderr=stderr,
        n_observations=len(values),
        final_metrics=final,
        observed_metrics=observed,
    )


def _aggregate(
    values: list[float], objective: ObjectiveSpec
) -> tuple[float, float | None]:
    """One score out of many observations, and how sure of it we are.

    The standard error is only defined for ``mean``: under ``best`` the
    observations are stages of one run, not samples of one quantity, so
    their spread measures training progress rather than uncertainty.
    """
    if objective.aggregate == "best":
        return (max(values) if objective.mode == "max" else min(values)), None
    mean = statistics.fmean(values)
    if len(values) < 2:
        return mean, None
    return mean, statistics.stdev(values) / math.sqrt(len(values))


def _scored(metrics: list[MetricPoint], objective: ObjectiveSpec) -> Iterator[float]:
    """Every observation that carries a usable value of the objective.

    With a baseline, the two metrics are paired *within one observation*.
    Taking the best metric and the best baseline separately would produce a
    lift no single observation ever achieved.
    """
    for point in metrics:
        if objective.metric not in point.values:
            continue
        value = point.values[objective.metric]
        if objective.baseline_metric is not None:
            if objective.baseline_metric not in point.values:
                continue
            value -= point.values[objective.baseline_metric]
        if math.isfinite(value):
            yield value


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


#: Two-sided 95% normal quantile. The intervals below are normal
#: approximations over a handful of folds, so they are indicative, not exact:
#: their job is to stop a campaign announcing a lead the data cannot carry,
#: and for that a slightly narrow interval still beats no interval at all.
Z95 = 1.959963984540054


def confidence_interval(
    value: float | None, stderr: float | None
) -> tuple[float, float] | None:
    """The 95% interval around a score, or ``None`` when nothing measured it."""
    if value is None or stderr is None:
        return None
    return (value - Z95 * stderr, value + Z95 * stderr)


def campaign_verdict(state: CampaignState, goal: GoalSpec) -> dict[str, Any]:
    """Whether the campaign actually found anything, decided in code.

    Two questions a report must not leave to the reader. First, is the best
    trial distinguishable from the second best, or is the ranking noise? The
    best of many noisy trials beats its runner-up by construction, so a
    headline that names a winner without this check is reporting the
    selection, not a result. Second, when the objective is a lift over a
    baseline the trial reported itself, does that lift clear zero?

    ``None`` on either answer means *not measured* — the objective takes the
    best observation and so carries no spread — which is a different claim
    from ``False``, and the report must keep them apart.
    """
    mode = goal.objective.mode
    ranked = sorted(
        (t for t in state.trials if t.status == "completed" and t.objective_value is not None),
        key=lambda t: t.objective_value,  # type: ignore[arg-type,return-value]
        reverse=mode == "max",
    )
    best = ranked[0] if ranked else None
    runner_up = ranked[1] if len(ranked) > 1 else None

    margin: float | None = None
    separated: bool | None = None
    if best is not None and runner_up is not None:
        assert best.objective_value is not None and runner_up.objective_value is not None
        margin = (
            best.objective_value - runner_up.objective_value
            if mode == "max"
            else runner_up.objective_value - best.objective_value
        )
        if best.objective_stderr is not None and runner_up.objective_stderr is not None:
            spread = math.hypot(best.objective_stderr, runner_up.objective_stderr)
            separated = margin > Z95 * spread

    beats_baseline: bool | None = None
    if goal.objective.baseline_metric is not None and best is not None:
        interval = confidence_interval(best.objective_value, best.objective_stderr)
        if interval is not None:
            beats_baseline = interval[0] > 0 if mode == "max" else interval[1] < 0

    return {
        "best_trial_id": best.trial_id if best else None,
        "runner_up_trial_id": runner_up.trial_id if runner_up else None,
        "margin": margin,
        "separated": separated,
        "beats_baseline": beats_baseline,
        "confidence": 0.95,
    }




def evaluate_success(
    best: dict[str, Any] | None,
    verdict: dict[str, Any],
    criteria: SuccessCriteria,
    mode: str,
) -> dict[str, Any]:
    """Whether the campaign cleared the bar it set itself, and what it missed.

    Every check reports the evidence it wanted, so a failure reads as an
    instruction: raise the folds, set a baseline, run more trials. The one
    rule that is easy to get wrong is that *unmeasured* fails a criterion
    that asks for measurement — a campaign cannot satisfy "show me the lead
    is real" by never looking.
    """
    if not criteria.declared:
        return {"declared": False, "met": None, "unmet": []}
    if best is None or best.get("objective_value") is None:
        return {
            "declared": True,
            "met": False,
            "unmet": ["no trial produced a usable objective value"],
        }

    unmet: list[str] = []
    value = best["objective_value"]
    if criteria.min_objective is not None:
        cleared = (
            value >= criteria.min_objective
            if mode == "max"
            else value <= criteria.min_objective
        )
        if not cleared:
            direction = "at least" if mode == "max" else "at most"
            unmet.append(
                f"objective {value:.6g} does not reach the required "
                f"{direction} {criteria.min_objective:.6g}"
            )

    if criteria.min_observations is not None:
        n = best.get("n_observations") or 0
        if n < criteria.min_observations:
            unmet.append(
                f"the score rests on {n} observation{'' if n == 1 else 's'}, "
                f"fewer than the required {criteria.min_observations}"
            )

    if criteria.require_separation:
        separated = verdict.get("separated")
        if separated is None:
            unmet.append(
                "separation from the runner-up was never measured; the "
                "objective needs `aggregate: mean` and a second scored trial"
            )
        elif not separated:
            unmet.append(
                f"the lead over {verdict.get('runner_up_trial_id')} is within "
                "the noise at 95% confidence"
            )

    if criteria.require_beats_baseline:
        beats = verdict.get("beats_baseline")
        if beats is None:
            unmet.append(
                "the comparison against the baseline was never measured; the "
                "objective needs `baseline_metric` and `aggregate: mean`"
            )
        elif not beats:
            unmet.append("the interval does not clear the baseline")

    return {"declared": True, "met": not unmet, "unmet": unmet}


def result_lines(summary: dict[str, Any]) -> list[str]:
    """The campaign's finding, in the words a report should use.

    One place decides how a verdict is spoken, so the CLI, the loop and the
    dashboard cannot disagree about whether a campaign found something. The
    rules it encodes: a score is never printed without its interval when one
    exists, a lead inside the noise is named as such instead of being
    announced, and nothing is claimed about a comparison nobody measured.
    """
    best = summary.get("best")
    if not best:
        return []
    verdict = summary.get("verdict") or {}
    metric = summary["objective"]["metric"]
    baseline = summary["objective"].get("baseline_metric")
    scale = f"{metric} lift over {baseline}" if baseline else metric

    score = f"{best['objective_value']:.6g}"
    if best.get("stderr") is not None:
        score += f" ± {best['stderr']:.3g}"
        if best.get("n_observations"):
            score += f" (SE over {best['n_observations']} observations)"
    lines = [f"{best['trial_id']}: {scale} = {score}"]

    runner_up = verdict.get("runner_up_trial_id")
    separated = verdict.get("separated")
    if runner_up and separated is not None:
        margin = verdict.get("margin")
        lines.append(
            f"lead of {margin:.3g} beats {runner_up} at 95% confidence"
            if separated
            else f"lead of {margin:.3g} is not distinguishable from {runner_up} "
            f"at 95% confidence; the ranking is within the noise"
        )

    beats = verdict.get("beats_baseline")
    if beats is not None:
        lines.append(
            "the interval clears the baseline"
            if beats
            else "the interval does not clear the baseline: this campaign has "
            "not shown the model beats it"
        )

    success = summary.get("success") or {}
    if not success.get("declared"):
        lines.append(
            "no success criteria were declared, so this result cannot be "
            "called a success or a failure"
        )
    elif success["met"]:
        lines.append("success criteria met")
    else:
        lines.append("success criteria NOT met: " + "; ".join(success["unmet"]))
    return lines


def summarize_campaign(state: CampaignState, goal: GoalSpec) -> dict[str, Any]:
    by_status: dict[str, int] = {}
    for trial in state.trials:
        by_status[trial.status] = by_status.get(trial.status, 0) + 1
    best = best_trial(state, goal.objective.mode)
    best_block = (
        {
            "trial_id": best.trial_id,
            "run_id": best.run_id,
            "objective_value": best.objective_value,
            "stderr": best.objective_stderr,
            "n_observations": best.objective_observations,
            "ci95": confidence_interval(best.objective_value, best.objective_stderr),
            "params": best.params,
        }
        if best
        else None
    )
    verdict = campaign_verdict(state, goal)
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
        "created_at": state.created_at.isoformat(),
        "last_advanced_at": state.updated_at.isoformat(),
        "gpu_hours": round(gpu_hours, 4),
        "estimated_cost": round(cost, 2) if cost is not None else None,
        "budget": {
            "max_trials": goal.budget.max_trials,
            "max_gpu_hours": goal.budget.max_gpu_hours,
            "gpu_hour_rate": goal.budget.gpu_hour_rate,
        },
        "objective": {
            "metric": goal.objective.metric,
            "baseline_metric": goal.objective.baseline_metric,
            "aggregate": goal.objective.aggregate,
            "mode": goal.objective.mode,
            "target": goal.objective.target,
        },
        "rounds": state.rounds,
        "agent_calls": state.agent_calls,
        "trials_by_status": by_status,
        "trials_total": len(state.trials),
        "best": best_block,
        "verdict": verdict,
        "success": evaluate_success(
            best_block, verdict, goal.success_criteria, goal.objective.mode
        ),
        "history": history,
    }
