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
    """Raw PR-AUC is not comparable when a search space dimension moves the
    base rate: the trial with more positives starts higher without being
    more predictable. The score is the lift over the baseline the trial
    itself reported.
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
    """Two kinds of run report many observations, and they need opposite
    aggregations.

    Epochs are successive states of one model, so the best is the answer.
    Folds are independent evaluations of the *same* configuration, so the
    best is max-of-k: biased upward by exactly the noise the folds exist to
    measure. `aggregate: mean` is how a cross-validated workload says which
    kind it is.
    """
    store, run_id = _run(
        tmp_path, [{"pr_auc": 0.30}, {"pr_auc": 0.90}, {"pr_auc": 0.60}]
    )

    reading = extract_objective(
        store, run_id, ObjectiveSpec(metric="pr_auc", mode="max", aggregate="mean")
    )

    assert reading.value is not None
    assert abs(reading.value - 0.60) < 1e-9


def test_an_averaged_score_carries_how_much_the_folds_disagreed(tmp_path):
    """A mean without its spread is the number that hides a dead fold."""
    store, run_id = _run(
        tmp_path, [{"pr_auc": 0.30}, {"pr_auc": 0.90}, {"pr_auc": 0.60}]
    )

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
    """A finished campaign whose trials already carry their readings."""
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
    """The campaign's headline is the whole product of a run, and 24 trials
    of a noisy objective produce a max that beats the runner-up by less than
    a fold's worth of variation roughly always. Whether that gap is real is
    arithmetic, not judgement, so the harness does it.
    """
    state, goal = _campaign(
        [("t1", 0.12, 0.07), ("t2", 0.10, 0.07)], aggregate="mean"
    )

    summary = summarize_campaign(state, goal)

    assert summary["best"]["trial_id"] == "t1"
    assert summary["verdict"]["separated"] is False
    assert summary["verdict"]["runner_up_trial_id"] == "t2"


def test_a_lead_outside_the_noise_is_a_winner():
    state, goal = _campaign(
        [("t1", 0.90, 0.01), ("t2", 0.10, 0.01)], aggregate="mean"
    )

    summary = summarize_campaign(state, goal)

    assert summary["verdict"]["separated"] is True
    assert abs(summary["verdict"]["margin"] - 0.80) < 1e-9


def test_separation_is_unknown_when_nobody_measured_it():
    """`best` objectives report no spread, so the answer is "not measured",
    which is a different thing from "the lead is not real"."""
    state, goal = _campaign([("t1", 0.9, None), ("t2", 0.1, None)])

    summary = summarize_campaign(state, goal)

    assert summary["verdict"]["separated"] is None


def test_a_single_trial_has_no_runner_up_to_be_separated_from():
    state, goal = _campaign([("t1", 0.9, 0.01)], aggregate="mean")

    summary = summarize_campaign(state, goal)

    assert summary["verdict"]["runner_up_trial_id"] is None
    assert summary["verdict"]["separated"] is None


def test_a_lift_whose_interval_contains_zero_does_not_beat_the_baseline():
    """With a baseline the objective *is* the lift, so "better than doing
    nothing" is the question the interval answers."""
    state, goal = _campaign(
        [("t1", 0.1189, 0.0737)], baseline_metric="baseline_pr_auc", aggregate="mean"
    )

    summary = summarize_campaign(state, goal)

    assert summary["verdict"]["beats_baseline"] is False
    lo, hi = summary["best"]["ci95"]
    assert lo < 0 < hi


def test_a_lift_clear_of_zero_beats_the_baseline():
    state, goal = _campaign(
        [("t1", 0.30, 0.02)], baseline_metric="baseline_pr_auc", aggregate="mean"
    )

    summary = summarize_campaign(state, goal)

    assert summary["verdict"]["beats_baseline"] is True


def test_without_a_baseline_there_is_nothing_to_beat():
    state, goal = _campaign([("t1", 0.9, 0.01)], aggregate="mean")

    summary = summarize_campaign(state, goal)

    assert summary["verdict"]["beats_baseline"] is None


def test_the_summary_says_which_baseline_the_score_is_measured_against():
    """A reader who sees `pr_auc: 0.12` and does not see that it is a lift
    will compare it against published PR-AUCs and conclude the model is bad.
    """
    state, goal = _campaign([("t1", 0.12, 0.01)], baseline_metric="baseline_pr_auc")

    summary = summarize_campaign(state, goal)

    assert summary["objective"]["baseline_metric"] == "baseline_pr_auc"
    assert summary["objective"]["aggregate"] == "best"


# --- saying it out loud ---------------------------------------------------


def test_the_result_lines_refuse_to_announce_a_winner_inside_the_noise():
    """A report that prints only `Best: t1 pr_auc=0.12` is how a max-of-24
    becomes a headline. The line that follows it is the finding.
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
    """An epoch-scored campaign has no interval, and inventing a verdict for
    it would be the same overclaim in the other direction."""
    state, goal = _campaign([("t1", 0.9, None), ("t2", 0.1, None)])

    lines = result_lines(summarize_campaign(state, goal))

    assert any("t1" in line for line in lines)
    assert not any("distinguishable" in line for line in lines)
    assert not any("±" in line for line in lines)


def test_there_is_nothing_to_report_without_a_scored_trial():
    state, goal = _campaign([])

    assert result_lines(summarize_campaign(state, goal)) == []
