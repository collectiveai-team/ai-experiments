"""The python surface an agent imports.

These tests hold the API to the same promise the CLI makes: plain data in,
plain data out, and one error type with the CLI's own codes — never a
traceback from three modules down.
"""

from __future__ import annotations

import sys

import pytest
import yaml

from ai_experiments import api


def _goal_dict(tmp_path) -> dict:
    script = tmp_path / "train.py"
    script.write_text(
        "import json, os\n"
        "params = json.loads(os.environ['IAX_PARAMS'])\n"
        "loss = (params['x'] - 2.0) ** 2\n"
        "print('IAX_METRIC ' + json.dumps({'step': 1, 'loss': loss}), flush=True)\n"
    )
    return {
        "goal": "minimize (x-2)^2",
        "name": "api",
        "objective": {"metric": "loss", "mode": "min", "target": 0.25},
        "search_space": {"x": {"type": "uniform", "low": 1.0, "high": 3.0}},
        "workload": {
            "entrypoint": sys.executable,
            "args": [str(script)],
            "working_dir": str(tmp_path),
        },
        "budget": {"max_trials": 12, "max_parallel": 2},
        "strategy": {"name": "adaptive", "seed": 3},
        "monitoring": {"interval_seconds": 1, "stuck_after_minutes": 5},
    }


def test_a_goal_an_agent_invented_fails_with_the_cli_error(tmp_path):
    with pytest.raises(api.IaxError) as caught:
        api.goal_from_dict({"goal": "be better", "objective": {"metric": "loss"}})

    assert caught.value.code == "invalid_input"
    assert caught.value.exit_code == 2


def test_goal_from_yaml_reports_a_missing_file_as_not_found(tmp_path):
    with pytest.raises(api.IaxError) as caught:
        api.goal_from_yaml(tmp_path / "absent.yaml")

    assert caught.value.code == "not_found"


def test_goal_from_yaml_reads_a_goal_the_cli_would_accept(tmp_path):
    path = tmp_path / "goal.yaml"
    path.write_text(yaml.safe_dump(_goal_dict(tmp_path)))

    goal = api.goal_from_yaml(path)

    assert goal.objective.metric == "loss"


def test_start_then_advance_drives_the_same_campaign(tmp_path):
    runs = tmp_path / "runs"
    goal = api.goal_from_dict(_goal_dict(tmp_path))

    started = api.start_campaign(goal, runs_dir=runs)
    advanced = api.advance_campaign(started["campaign_id"], runs_dir=runs)

    assert advanced["campaign_id"] == started["campaign_id"]
    assert advanced["trials_total"] >= started["trials_total"]


def test_every_lookup_of_an_unknown_campaign_says_not_found(tmp_path):
    runs = tmp_path / "runs"
    calls = [
        lambda: api.campaign_report("cmp_nope", runs_dir=runs),
        lambda: api.advance_campaign("cmp_nope", runs_dir=runs),
        lambda: api.campaign_rounds("cmp_nope", runs_dir=runs),
        lambda: api.suggest_trial("cmp_nope", {"x": 1.0}, runs_dir=runs),
    ]

    for call in calls:
        with pytest.raises(api.IaxError) as caught:
            call()
        assert caught.value.code == "not_found"
        assert caught.value.exit_code == 1


def test_a_suggestion_outside_the_search_space_is_rejected(tmp_path):
    runs = tmp_path / "runs"
    goal = api.goal_from_dict(_goal_dict(tmp_path))
    started = api.start_campaign(goal, runs_dir=runs)

    with pytest.raises(api.IaxError) as caught:
        api.suggest_trial(started["campaign_id"], {"x": 99.0}, runs_dir=runs)

    assert caught.value.code == "invalid_input"


def test_an_accepted_suggestion_is_queued_for_the_next_round(tmp_path):
    runs = tmp_path / "runs"
    goal = api.goal_from_dict(_goal_dict(tmp_path))
    started = api.start_campaign(goal, runs_dir=runs)

    trial = api.suggest_trial(
        started["campaign_id"], {"x": 2.0}, note="the analytic optimum", runs_dir=runs
    )

    assert trial["source"] == "agent"
    assert trial["params"] == {"x": 2.0}


def test_campaign_rounds_explain_what_the_loop_believed(tmp_path):
    runs = tmp_path / "runs"
    goal = api.goal_from_dict(_goal_dict(tmp_path))
    started = api.start_campaign(goal, runs_dir=runs)

    records = api.campaign_rounds(started["campaign_id"], runs_dir=runs)

    assert records
    assert records[0]["stage"] == "propose"
    assert records[0]["trial_ids"]


def test_list_campaigns_reports_each_one(tmp_path):
    runs = tmp_path / "runs"
    goal = api.goal_from_dict(_goal_dict(tmp_path))
    first = api.start_campaign(goal, runs_dir=runs)
    second = api.start_campaign(goal, runs_dir=runs)

    ids = {report["campaign_id"] for report in api.list_campaigns(runs_dir=runs)}

    assert {first["campaign_id"], second["campaign_id"]} <= ids


def test_run_loop_reaches_the_target_from_python(tmp_path):
    goal = api.goal_from_dict(_goal_dict(tmp_path))

    report = api.run_loop(
        goal, runs_dir=tmp_path / "runs", interval_seconds=0, max_seconds=120
    )

    assert report.target_reached
    assert report.best["objective_value"] <= 0.25


def test_run_loop_resumes_a_campaign_it_bounded_earlier(tmp_path):
    runs = tmp_path / "runs"
    goal = api.goal_from_dict(_goal_dict(tmp_path))
    first = api.run_loop(goal, runs_dir=runs, interval_seconds=0, max_rounds=1)

    second = api.run_loop(
        goal,
        runs_dir=runs,
        campaign_id=first.campaign_id,
        interval_seconds=0,
        max_seconds=120,
    )

    assert second.campaign_id == first.campaign_id


def test_run_loop_refuses_to_resume_a_campaign_that_does_not_exist(tmp_path):
    goal = api.goal_from_dict(_goal_dict(tmp_path))

    with pytest.raises(api.IaxError) as caught:
        api.run_loop(goal, runs_dir=tmp_path / "runs", campaign_id="cmp_nope")

    assert caught.value.code == "not_found"
