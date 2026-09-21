"""Reading a trial's score out of what the workload reported."""

from __future__ import annotations

from ai_experiments.planner.analysis import extract_objective
from ai_experiments.schemas import (
    ExperimentManifest,
    MetricPoint,
    ObjectiveSpec,
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
