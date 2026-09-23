"""Reading a trial's score out of what the workload reported."""

from __future__ import annotations

from ai_experiments.planner.analysis import (
    extract_objective,
    result_lines,
    summarize_campaign,
)
from ai_experiments.schemas import (
    CampaignState,
    ExperimentManifest,
    GoalSpec,
    MetricPoint,
    ObjectiveSpec,
    SuccessCriteria,
    TrialRecord,
    WorkloadSpec,
)
from ai_experiments.store import FilesystemRunStore


def _run(tmp_path, points: list[dict[str, float]]):
    store = FilesystemRunStore(tmp_path / "runs", capture_repro=False)
    run_id, _ = store.create_run(
        ExperimentManifest(
            experiment="analysis", workload=WorkloadSpec(entrypoint="python train.py")
        )
    )
    for step, values in enumerate(points):
        store.append_metric(run_id, MetricPoint(step=step, values=values))
    return store, run_id


def test_the_best_observation_is_the_score(tmp_path):
    store, run_id = _run(tmp_path, [{"pr_auc": 0.4}, {"pr_auc": 0.7}, {"pr_auc": 0.5}])

    reading = extract_objective(store, run_id, ObjectiveSpec(metric="pr_auc", mode="max"))

    assert reading.value == 0.7


def test_a_baseline_makes_trials_on_different_data_comparable(tmp_path):
    """Score the lift over the baseline the trial itself reported.

    Raw PR-AUC is not comparable when a search space dimension moves the base rate: the trial with
    more positives starts higher without being more predictable.
    """
    store, run_id = _run(tmp_path, [{"pr_auc": 0.58, "baseline_pr_auc": 0.50}])

    reading = extract_objective(
        store,
        run_id,
        ObjectiveSpec(metric="pr_auc", baseline_metric="baseline_pr_auc", mode="max"),
    )

    assert reading.value is not None
    assert abs(reading.value - 0.08) < 1e-9


def test_the_lift_pairs_the_two_metrics_within_one_observation(tmp_path):
    """Best metric minus best baseline is a number no trial ever achieved."""
    store, run_id = _run(
        tmp_path,
        [
            {"pr_auc": 0.58, "baseline_pr_auc": 0.50},  # lift 0.08
            {"pr_auc": 0.55, "baseline_pr_auc": 0.30},  # lift 0.25  <- the best
        ],
    )

    reading = extract_objective(
        store,
        run_id,
        ObjectiveSpec(metric="pr_auc", baseline_metric="baseline_pr_auc", mode="max"),
    )

    assert reading.value is not None
    assert abs(reading.value - 0.25) < 1e-9  # not 0.58 - 0.30 = 0.28


def test_an_observation_missing_the_baseline_is_ignored(tmp_path):
    store, run_id = _run(
        tmp_path,
        [{"pr_auc": 0.90}, {"pr_auc": 0.55, "baseline_pr_auc": 0.30}],
    )

    reading = extract_objective(
        store,
        run_id,
        ObjectiveSpec(metric="pr_auc", baseline_metric="baseline_pr_auc", mode="max"),
    )

    assert reading.value is not None
    assert abs(reading.value - 0.25) < 1e-9


def test_a_trial_that_never_reported_the_baseline_does_not_score(tmp_path):
    """Scoring it on the raw metric would rank it against a different scale."""
    store, run_id = _run(tmp_path, [{"pr_auc": 0.90}])

    reading = extract_objective(
        store,
        run_id,
        ObjectiveSpec(metric="pr_auc", baseline_metric="baseline_pr_auc", mode="max"),
    )

    assert reading.value is None
    assert reading.miss_reason == "baseline_absent"


# --- folds are not epochs -------------------------------------------------


def test_folds_are_averaged_not_maximized(tmp_path):
    """Two kinds of run report many observations, and they need opposite aggregations.

    Epochs are successive states of one model, so the best is the answer.
    Folds are independent evaluations of the *same* configuration, so the
    best is max-of-k: biased upward by exactly the noise the folds exist to
    measure. `aggregate: mean` is how a cross-validated workload says which
    kind it is.
    """
    store, run_id = _run(tmp_path, [{"pr_auc": 0.30}, {"pr_auc": 0.90}, {"pr_auc": 0.60}])

    reading = extract_objective(
        store, run_id, ObjectiveSpec(metric="pr_auc", mode="max", aggregate="mean")
    )

    assert reading.value is not None
    assert abs(reading.value - 0.60) < 1e-9


def test_an_averaged_score_carries_how_much_the_folds_disagreed(tmp_path):
    """A mean without its spread is the number that hides a dead fold."""
    store, run_id = _run(tmp_path, [{"pr_auc": 0.30}, {"pr_auc": 0.90}, {"pr_auc": 0.60}])

    reading = extract_objective(
        store, run_id, ObjectiveSpec(metric="pr_auc", mode="max", aggregate="mean")
    )

    assert reading.n_observations == 3
    assert reading.stderr is not None
    # sd = 0.3 over 3 samples -> se = 0.3/sqrt(3)
    assert abs(reading.stderr - 0.3 / 3**0.5) < 1e-9


def test_one_observation_has_no_measurable_spread(tmp_path):
    """Reporting 0.0 would claim certainty a single fold cannot support."""
    store, run_id = _run(tmp_path, [{"pr_auc": 0.5}])

    reading = extract_objective(
        store, run_id, ObjectiveSpec(metric="pr_auc", mode="max", aggregate="mean")
    )

    assert reading.value == 0.5
    assert reading.n_observations == 1
    assert reading.stderr is None


def test_bootstrap_replicates_keep_their_spread_undivided(tmp_path):
    """Resamples of one evaluation are not independent evaluations.

    The spread of the replicates already *is* the standard error of the
    statistic. Dividing it by the square root of their count would shrink the
    interval by exactly the factor the resampling exists to expose — and the
    count is a computational knob, so the interval would get narrower the
    longer the workload was willing to run.
    """
    store, run_id = _run(tmp_path, [{"pr_auc": 0.30}, {"pr_auc": 0.90}, {"pr_auc": 0.60}])

    reading = extract_objective(
        store, run_id, ObjectiveSpec(metric="pr_auc", mode="max", aggregate="bootstrap")
    )

    assert reading.value is not None
    assert abs(reading.value - 0.60) < 1e-9
    assert reading.stderr is not None
    assert abs(reading.stderr - 0.3) < 1e-9


def test_more_replicates_do_not_narrow_the_interval(tmp_path):
    """A workload cannot buy confidence by resampling more.

    This is the guarantee that makes `bootstrap` safe to declare.
    """
    few, few_id = _run(tmp_path / "few", [{"m": 0.4}, {"m": 0.6}] * 5)
    many, many_id = _run(tmp_path / "many", [{"m": 0.4}, {"m": 0.6}] * 50)
    spec = ObjectiveSpec(metric="m", mode="max", aggregate="bootstrap")

    thin = extract_objective(few, few_id, spec)
    thick = extract_objective(many, many_id, spec)

    assert thin.stderr is not None
    assert thick.stderr is not None
    assert abs(thin.stderr - thick.stderr) < 0.01
    assert thick.n_observations == 100


def test_averaging_and_a_baseline_compose(tmp_path):
    """The mean of the per-fold lifts, which is what walk-forward measures."""
    store, run_id = _run(
        tmp_path,
        [
            {"pr_auc": 0.40, "baseline_pr_auc": 0.38},  # +0.02
            {"pr_auc": 0.80, "baseline_pr_auc": 0.45},  # +0.35
        ],
    )

    reading = extract_objective(
        store,
        run_id,
        ObjectiveSpec(
            metric="pr_auc",
            baseline_metric="baseline_pr_auc",
            mode="max",
            aggregate="mean",
        ),
    )

    assert reading.value is not None
    assert abs(reading.value - 0.185) < 1e-9


def test_taking_the_best_is_still_the_default(tmp_path):
    """Epoch-reporting workloads must not change behaviour."""
    store, run_id = _run(tmp_path, [{"loss": 0.9}, {"loss": 0.2}, {"loss": 0.5}])

    reading = extract_objective(store, run_id, ObjectiveSpec(metric="loss", mode="min"))

    assert reading.value == 0.2
    assert reading.stderr is None


# --- the campaign's own verdict -------------------------------------------


def _campaign(scored: list[tuple[str, float, float | None]], **objective_kwargs):
    """Build a finished campaign whose trials already carry their readings."""
    goal = GoalSpec(
        name="c",
        goal="find the best window",
        objective=ObjectiveSpec(metric="pr_auc", mode="max", **objective_kwargs),
        workload=WorkloadSpec(entrypoint="python train.py"),
        search_space={"window_days": {"type": "choice", "values": [30, 60]}},
    )
    state = CampaignState(
        campaign_id="c1",
        name="c",
        goal="goal.yaml",
        trials=[
            TrialRecord(
                trial_id=trial_id,
                params={},
                status="completed",
                objective_value=value,
                objective_stderr=stderr,
                objective_observations=5,
            )
            for trial_id, value, stderr in scored
        ],
    )
    return state, goal


def test_a_lead_inside_the_noise_is_not_a_winner():
    """Decide in code whether the winner's lead over the runner-up is real.

    The campaign's headline is the whole product of a run, and 24 trials of a noisy objective
    produce a max that beats the runner-up by less than a fold's worth of variation roughly always.
    Whether that gap is real is arithmetic, not judgement, so the harness does it.
    """
    state, goal = _campaign([("t1", 0.12, 0.07), ("t2", 0.10, 0.07)], aggregate="mean")

    summary = summarize_campaign(state, goal)

    assert summary.best is not None
    assert summary.best.trial_id == "t1"
    assert summary.verdict.separated is False
    assert summary.verdict.runner_up_trial_id == "t2"


def test_a_lead_outside_the_noise_is_a_winner():
    state, goal = _campaign([("t1", 0.90, 0.01), ("t2", 0.10, 0.01)], aggregate="mean")

    summary = summarize_campaign(state, goal)

    assert summary.verdict.separated is True
    assert summary.verdict.margin is not None
    assert abs(summary.verdict.margin - 0.80) < 1e-9


def test_separation_is_unknown_when_nobody_measured_it():
    """`best` objectives report no spread, so the answer is "not measured".

    That is a different thing from "the lead is not real".
    """
    state, goal = _campaign([("t1", 0.9, None), ("t2", 0.1, None)])

    summary = summarize_campaign(state, goal)

    assert summary.verdict.separated is None


def test_a_single_trial_has_no_runner_up_to_be_separated_from():
    state, goal = _campaign([("t1", 0.9, 0.01)], aggregate="mean")

    summary = summarize_campaign(state, goal)

    assert summary.verdict.runner_up_trial_id is None
    assert summary.verdict.separated is None


def test_a_lift_whose_interval_contains_zero_does_not_beat_the_baseline():
    """With a baseline the objective *is* the lift.

    "Better than doing nothing" is then the question the interval answers.
    """
    state, goal = _campaign(
        [("t1", 0.1189, 0.0737)], baseline_metric="baseline_pr_auc", aggregate="mean"
    )

    summary = summarize_campaign(state, goal)

    assert summary.verdict.beats_baseline is False
    assert summary.best is not None
    assert summary.best.ci95 is not None
    lo, hi = summary.best.ci95
    assert lo < 0 < hi


def test_a_lift_clear_of_zero_beats_the_baseline():
    state, goal = _campaign(
        [("t1", 0.30, 0.02)], baseline_metric="baseline_pr_auc", aggregate="mean"
    )

    summary = summarize_campaign(state, goal)

    assert summary.verdict.beats_baseline is True


def test_without_a_baseline_there_is_nothing_to_beat():
    state, goal = _campaign([("t1", 0.9, 0.01)], aggregate="mean")

    summary = summarize_campaign(state, goal)

    assert summary.verdict.beats_baseline is None


def test_the_summary_says_which_baseline_the_score_is_measured_against():
    """Name the scale, so a lift is not read as a raw PR-AUC.

    A reader who sees `pr_auc: 0.12` and does not see that it is a lift will compare it against
    published PR-AUCs and conclude the model is bad.
    """
    state, goal = _campaign([("t1", 0.12, 0.01)], baseline_metric="baseline_pr_auc")

    summary = summarize_campaign(state, goal)

    assert summary.objective.baseline_metric == "baseline_pr_auc"
    assert summary.objective.aggregate == "best"


# --- saying it out loud ---------------------------------------------------


def test_the_result_lines_refuse_to_announce_a_winner_inside_the_noise():
    """The line after the headline is the finding.

    A report that prints only `Best: t1 pr_auc=0.12` is how a max-of-24 becomes a headline.
    """
    state, goal = _campaign(
        [("t1", 0.12, 0.07), ("t2", 0.10, 0.07)],
        baseline_metric="baseline_pr_auc",
        aggregate="mean",
    )

    lines = result_lines(summarize_campaign(state, goal))

    assert any("±" in line for line in lines)
    assert any("not distinguishable from t2" in line for line in lines)
    assert any("does not clear the baseline" in line for line in lines)


def test_the_result_lines_say_so_when_the_lead_is_real():
    state, goal = _campaign(
        [("t1", 0.30, 0.02), ("t2", 0.10, 0.02)],
        baseline_metric="baseline_pr_auc",
        aggregate="mean",
    )

    lines = result_lines(summarize_campaign(state, goal))

    assert any("beats t2" in line for line in lines)
    assert any("clears the baseline" in line for line in lines)


def test_the_result_lines_stay_quiet_about_what_was_never_measured():
    """An epoch-scored campaign has no interval, so no verdict is invented for it.

    Inventing one would be the same overclaim in the other direction.
    """
    state, goal = _campaign([("t1", 0.9, None), ("t2", 0.1, None)])

    lines = result_lines(summarize_campaign(state, goal))

    assert any("t1" in line for line in lines)
    assert not any("distinguishable" in line for line in lines)
    assert not any("±" in line for line in lines)


def test_there_is_nothing_to_report_without_a_scored_trial():
    state, goal = _campaign([])

    assert result_lines(summarize_campaign(state, goal)) == []


# --- what counts as success, written down beforehand ----------------------


def _with_criteria(scored, criteria, **objective_kwargs):
    state, goal = _campaign(scored, **objective_kwargs)
    goal = goal.model_copy(update={"success_criteria": SuccessCriteria(**criteria)})
    return summarize_campaign(state, goal)


def test_a_campaign_that_declares_nothing_cannot_be_said_to_have_succeeded():
    """A goal that never said what working means gets neither "yes" nor "no".

    The honest answer to "did it work?" is that nobody wrote it down. Silence here is how a max-
    of-24 becomes a headline.
    """
    summary = _with_criteria([("t1", 0.9, 0.01)], {}, aggregate="mean")

    assert summary.success.declared is False
    assert summary.success.met is None


def test_the_score_has_to_clear_the_bar_that_was_set():
    summary = _with_criteria([("t1", 0.04, 0.001)], {"min_objective": 0.05}, aggregate="mean")

    assert summary.success.met is False
    assert any("0.05" in reason for reason in summary.success.unmet)


def test_a_score_over_the_bar_meets_the_criteria():
    summary = _with_criteria([("t1", 0.30, 0.001)], {"min_objective": 0.05}, aggregate="mean")

    assert summary.success.met is True
    assert summary.success.unmet == []


def test_a_bar_read_the_other_way_round_when_lower_is_better():
    state, goal = _campaign([("t1", 0.04, 0.001)], aggregate="mean")
    goal = goal.model_copy(
        update={
            "objective": goal.objective.model_copy(update={"mode": "min"}),
            "success_criteria": SuccessCriteria(min_objective=0.05),
        }
    )

    assert summarize_campaign(state, goal).success.met is True


def test_proof_that_was_never_collected_does_not_count_as_proof():
    """Demanding separation from a campaign that measured no spread fails.

    The criterion asks for evidence, and "not measured" is not evidence.
    """
    summary = _with_criteria([("t1", 0.9, None), ("t2", 0.1, None)], {"require_separation": True})

    assert summary.success.met is False
    assert any("never measured" in reason for reason in summary.success.unmet)


def test_a_winner_inside_the_noise_fails_the_separation_criterion():
    summary = _with_criteria(
        [("t1", 0.12, 0.07), ("t2", 0.10, 0.07)],
        {"require_separation": True},
        aggregate="mean",
    )

    assert summary.success.met is False


def test_a_lift_that_does_not_clear_the_baseline_fails_that_criterion():
    summary = _with_criteria(
        [("t1", 0.1189, 0.0737)],
        {"require_beats_baseline": True},
        baseline_metric="baseline_pr_auc",
        aggregate="mean",
    )

    assert summary.success.met is False
    assert any("baseline" in reason for reason in summary.success.unmet)


def test_one_lucky_fold_is_not_enough_observations():
    """A winner resting on a single evaluation has measured the evaluation, not the model."""
    state, goal = _campaign([("t1", 0.9, None)], aggregate="mean")
    state.trials[0].objective_observations = 1
    goal = goal.model_copy(update={"success_criteria": SuccessCriteria(min_observations=5)})

    summary = summarize_campaign(state, goal)

    assert summary.success.met is False
    assert any("1 observation" in reason for reason in summary.success.unmet)


def test_criteria_cannot_be_met_by_a_campaign_with_no_scored_trial():
    summary = _with_criteria([], {"min_objective": 0.05})

    assert summary.success.met is False
    assert any("no trial" in reason for reason in summary.success.unmet)


def test_the_result_lines_state_the_verdict_on_the_criteria():
    summary = _with_criteria([("t1", 0.04, 0.001)], {"min_objective": 0.05}, aggregate="mean")

    lines = result_lines(summary)

    assert any("success criteria NOT met" in line for line in lines)
