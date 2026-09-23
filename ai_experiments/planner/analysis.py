"""Post-run analysis: objective extraction and campaign summaries."""

from __future__ import annotations

import math
import statistics
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

from pydantic import BaseModel, Field

from ai_experiments.responses import (
    BestTrialSummary,
    BudgetSummary,
    CampaignHistoryEntry,
    CampaignSummary,
    CampaignVerdict,
    SuccessReport,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

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
    miss_reason: Literal["no_metrics", "metric_absent", "not_finite", "baseline_absent"] | None = (
        None
    )

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
            return f"objective metric '{metric}' was never reported; observed metrics: {observed}"
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
    """Read the run's objective value, its uncertainty, and why it is missing."""
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
            if objective.baseline_metric is not None and objective.baseline_metric not in observed
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


def _aggregate(values: list[float], objective: ObjectiveSpec) -> tuple[float, float | None]:
    """One score out of many observations, and how sure of it we are.

    Three kinds of observation need three answers. Under ``best`` they are
    stages of one run, so their spread measures training progress rather than
    uncertainty and no standard error is defined. Under ``mean`` they are
    independent evaluations of one configuration, so the error of their
    average shrinks with their count. Under ``bootstrap`` they are resamples
    of a *single* evaluation: their spread already is the standard error of
    the statistic, and dividing it again would shrink the interval by exactly
    the factor the resampling exists to expose — while the count is a
    computational knob, so confidence would become something a workload could
    buy by resampling longer.
    """
    if objective.aggregate == "best":
        return (max(values) if objective.mode == "max" else min(values)), None
    mean = statistics.fmean(values)
    if len(values) < 2:
        return mean, None
    spread = statistics.stdev(values)
    if objective.aggregate == "bootstrap":
        return mean, spread
    return mean, spread / math.sqrt(len(values))


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
    """Return the best *completed* trial.

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

    def _key(trial: TrialRecord) -> float:
        # scored is filtered above to non-None, finite objective_value; cast makes
        # that already-proven invariant visible to the type checker.
        value = cast("float", trial.objective_value)
        return -value if mode == "max" else value

    return min(scored, key=_key)


def best_trial(state: CampaignState, mode: str) -> TrialRecord | None:
    return best_of(state.trials, mode)


def trial_history(trials: list[TrialRecord]) -> list[CampaignHistoryEntry]:
    """Every trial an agent can learn from: scored ones and failures alike."""
    return [
        CampaignHistoryEntry(
            trial_id=t.trial_id,
            status=t.status,
            objective_value=t.objective_value,
            params=t.params,
            error=t.error,
        )
        for t in trials
        if t.objective_value is not None or t.error is not None
    ]


def summarize_trials(  # ast-grep-ignore: no-dict-return-annotation
    trials: list[TrialRecord], goal: GoalSpec
) -> dict[str, Any]:
    """Return the evidence block an agent plans from, without a CampaignState.

    A strategy sees trials, not campaigns. This is the same history and best
    trial that ``summarize_campaign`` reports, so the agent and the dashboard
    never disagree about what happened.

    Serialised on purpose: this is the prompt boundary. Its only consumer is
    ``agents.prompts.round_brief``, whose other caller hands it
    ``summarize_campaign(...).model_dump(mode="json")`` — one shape of evidence
    for both briefs. Typing one side and not the other would split that
    contract, so both stay JSON-shaped until the pair is typed together.
    """
    best = best_of(trials, goal.objective.mode)
    return {  # ast-grep-ignore: no-dict-literal-return  # prompt boundary, see the docstring
        "trials_total": len(trials),
        "history": [entry.model_dump(mode="json") for entry in trial_history(trials)],
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


def confidence_interval(value: float | None, stderr: float | None) -> tuple[float, float] | None:
    """Return the 95% interval around a score, or ``None`` when nothing measured it."""
    if value is None or stderr is None:
        return None
    return (value - Z95 * stderr, value + Z95 * stderr)


def campaign_verdict(state: CampaignState, goal: GoalSpec) -> CampaignVerdict:
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
        key=lambda t: cast("float", t.objective_value),
        reverse=mode == "max",
    )
    best = ranked[0] if ranked else None
    runner_up = ranked[1] if len(ranked) > 1 else None

    margin, separated = _separation(best, runner_up, mode)

    beats_baseline: bool | None = None
    if goal.objective.baseline_metric is not None and best is not None:
        interval = confidence_interval(best.objective_value, best.objective_stderr)
        if interval is not None:
            beats_baseline = interval[0] > 0 if mode == "max" else interval[1] < 0

    return CampaignVerdict(
        best_trial_id=best.trial_id if best else None,
        runner_up_trial_id=runner_up.trial_id if runner_up else None,
        margin=margin,
        separated=separated,
        beats_baseline=beats_baseline,
        confidence=0.95,
    )


def _separation(
    best: TrialRecord | None, runner_up: TrialRecord | None, mode: str
) -> tuple[float | None, bool | None]:
    """Measure the best trial's lead over the runner-up, and whether it clears the noise."""
    if best is None or runner_up is None:
        return None, None
    best_value = cast("float", best.objective_value)
    runner_value = cast("float", runner_up.objective_value)
    margin = best_value - runner_value if mode == "max" else runner_value - best_value
    if best.objective_stderr is None or runner_up.objective_stderr is None:
        return margin, None
    spread = math.hypot(best.objective_stderr, runner_up.objective_stderr)
    return margin, margin > Z95 * spread


def evaluate_success(
    best: BestTrialSummary | None,
    verdict: CampaignVerdict,
    criteria: SuccessCriteria,
    mode: str,
) -> SuccessReport:
    """Whether the campaign cleared the bar it set itself, and what it missed.

    Every check reports the evidence it wanted, so a failure reads as an
    instruction: raise the folds, set a baseline, run more trials. The one
    rule that is easy to get wrong is that *unmeasured* fails a criterion
    that asks for measurement — a campaign cannot satisfy "show me the lead
    is real" by never looking.
    """
    if not criteria.declared:
        return SuccessReport(declared=False, met=None, unmet=[])
    if best is None or best.objective_value is None:
        return SuccessReport(
            declared=True,
            met=False,
            unmet=["no trial produced a usable objective value"],
        )

    unmet = [
        *_objective_shortfall(best.objective_value, criteria, mode),
        *_observation_shortfall(best.n_observations or 0, criteria),
        *_separation_shortfall(verdict, criteria),
        *_baseline_shortfall(verdict, criteria),
    ]
    return SuccessReport(declared=True, met=not unmet, unmet=unmet)


def _objective_shortfall(value: float, criteria: SuccessCriteria, mode: str) -> list[str]:
    if criteria.min_objective is None:
        return []
    cleared = value >= criteria.min_objective if mode == "max" else value <= criteria.min_objective
    if cleared:
        return []
    direction = "at least" if mode == "max" else "at most"
    return [
        f"objective {value:.6g} does not reach the required "
        f"{direction} {criteria.min_objective:.6g}"
    ]


def _observation_shortfall(n: int, criteria: SuccessCriteria) -> list[str]:
    if criteria.min_observations is None or n >= criteria.min_observations:
        return []
    return [
        f"the score rests on {n} observation{'' if n == 1 else 's'}, "
        f"fewer than the required {criteria.min_observations}"
    ]


def _separation_shortfall(verdict: CampaignVerdict, criteria: SuccessCriteria) -> list[str]:
    if not criteria.require_separation:
        return []
    if verdict.separated is None:
        return [
            "separation from the runner-up was never measured; the "
            "objective needs `aggregate: mean` or `bootstrap`, and a second "
            "scored trial"
        ]
    if not verdict.separated:
        return [f"the lead over {verdict.runner_up_trial_id} is within the noise at 95% confidence"]
    return []


def _baseline_shortfall(verdict: CampaignVerdict, criteria: SuccessCriteria) -> list[str]:
    if not criteria.require_beats_baseline:
        return []
    if verdict.beats_baseline is None:
        return [
            "the comparison against the baseline was never measured; the "
            "objective needs `baseline_metric`, and `aggregate: mean` or "
            "`bootstrap`"
        ]
    if not verdict.beats_baseline:
        return ["the interval does not clear the baseline"]
    return []


class ResultView(Protocol):
    """What `result_lines` needs to speak a verdict: a summary or a loop report."""

    @property
    def objective(self) -> ObjectiveSpec | None: ...
    @property
    def best(self) -> BestTrialSummary | None: ...
    @property
    def verdict(self) -> CampaignVerdict | None: ...
    @property
    def success(self) -> SuccessReport | None: ...


def result_lines(summary: ResultView) -> list[str]:
    """Speak the campaign's finding, in the words a report should use.

    One place decides how a verdict is spoken, so the CLI, the loop and the
    dashboard cannot disagree about whether a campaign found something. The
    rules it encodes: a score is never printed without its interval when one
    exists, a lead inside the noise is named as such instead of being
    announced, and nothing is claimed about a comparison nobody measured.
    """
    best = summary.best
    if best is None or best.objective_value is None or summary.objective is None:
        return []
    metric = summary.objective.metric
    baseline = summary.objective.baseline_metric
    scale = f"{metric} lift over {baseline}" if baseline else metric

    score = f"{best.objective_value:.6g}"
    if best.stderr is not None:
        score += f" ± {best.stderr:.3g}"
        if best.n_observations:
            score += f" (SE over {best.n_observations} observations)"
    lines = [f"{best.trial_id}: {scale} = {score}"]
    lines.extend(_verdict_lines(summary.verdict))
    lines.extend(_success_lines(summary.success))
    return lines


def _verdict_lines(verdict: CampaignVerdict | None) -> list[str]:
    if verdict is None:
        return []
    lines: list[str] = []
    if verdict.runner_up_trial_id and verdict.separated is not None:
        margin = cast("float", verdict.margin)
        lines.append(
            f"lead of {margin:.3g} beats {verdict.runner_up_trial_id} at 95% confidence"
            if verdict.separated
            else f"lead of {margin:.3g} is not distinguishable from "
            f"{verdict.runner_up_trial_id} at 95% confidence; the ranking is within the noise"
        )
    if verdict.beats_baseline is not None:
        lines.append(
            "the interval clears the baseline"
            if verdict.beats_baseline
            else "the interval does not clear the baseline: this campaign has "
            "not shown the model beats it"
        )
    return lines


def _success_lines(success: SuccessReport | None) -> list[str]:
    if success is None or not success.declared:
        return [
            "no success criteria were declared, so this result cannot be "
            "called a success or a failure"
        ]
    if success.met:
        return ["success criteria met"]
    return ["success criteria NOT met: " + "; ".join(success.unmet)]


def summarize_campaign(state: CampaignState, goal: GoalSpec) -> CampaignSummary:
    by_status: dict[str, int] = {}
    for trial in state.trials:
        by_status[trial.status] = by_status.get(trial.status, 0) + 1
    best = best_trial(state, goal.objective.mode)
    best_block = (
        BestTrialSummary(
            trial_id=best.trial_id,
            run_id=best.run_id,
            objective_value=best.objective_value,
            stderr=best.objective_stderr,
            n_observations=best.objective_observations,
            ci95=confidence_interval(best.objective_value, best.objective_stderr),
            params=best.params,
        )
        if best
        else None
    )
    verdict = campaign_verdict(state, goal)
    history = trial_history(state.trials)
    gpu_hours = sum(t.gpu_hours or 0.0 for t in state.trials)
    wall_hours = sum(t.wall_hours or 0.0 for t in state.trials)
    cost = gpu_hours * goal.budget.gpu_hour_rate if goal.budget.gpu_hour_rate is not None else None
    return CampaignSummary(
        campaign_id=state.campaign_id,
        name=state.name,
        goal=state.goal,
        status=state.status,
        stop_reason=state.stop_reason,
        created_at=state.created_at.isoformat(),
        last_advanced_at=state.updated_at.isoformat(),
        gpu_hours=round(gpu_hours, 4),
        wall_hours=round(wall_hours, 4),
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
        agent_calls=state.agent_calls,
        trials_by_status=by_status,
        trials_total=len(state.trials),
        best=best_block,
        verdict=verdict,
        success=evaluate_success(best_block, verdict, goal.success_criteria, goal.objective.mode),
        history=history,
    )
