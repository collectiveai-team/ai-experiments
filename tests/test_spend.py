"""What a campaign cost, on a machine with no GPUs.

`Spend: 0 gpu-hours` after eight minutes of saturated CPU is not a rounding
error, it is the report saying the campaign was free. Every budget decision
downstream — how many folds, how many trials, whether to run at all — is made
against that number.
"""

from __future__ import annotations

from datetime import timedelta

from ai_experiments.orchestrator import trial_wall_hours
from ai_experiments.planner.analysis import summarize_campaign
from ai_experiments.schemas import (
    CampaignState,
    GoalSpec,
    ObjectiveSpec,
    TrialRecord,
    WorkloadSpec,
    utc_now,
)


def test_wall_time_is_recorded_even_with_no_gpu():
    started = utc_now()

    hours = trial_wall_hours(started, started + timedelta(seconds=1800))

    assert abs(hours - 0.5) < 1e-9


def test_a_trial_that_never_started_has_no_wall_time():
    assert trial_wall_hours(None, utc_now()) is None


def test_the_campaign_reports_what_it_actually_spent():
    goal = GoalSpec(
        goal="g",
        name="n",
        objective=ObjectiveSpec(metric="loss", mode="min"),
        search_space={"lr": {"type": "uniform", "low": 0.0, "high": 1.0}},
        workload=WorkloadSpec(entrypoint="python train.py"),
    )
    state = CampaignState(
        campaign_id="c1",
        name="n",
        goal="goal.yaml",
        trials=[
            TrialRecord(trial_id="t1", params={}, status="completed", wall_hours=0.1),
            TrialRecord(trial_id="t2", params={}, status="failed", wall_hours=0.025),
        ],
    )

    summary = summarize_campaign(state, goal)

    assert abs(summary["wall_hours"] - 0.125) < 1e-9
    assert summary["gpu_hours"] == 0
