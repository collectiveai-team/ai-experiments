"""A quiet daemon and a dead daemon must not look the same (#22).

A healthy tick prints nothing, so an unattended loop could sit stalled all
night while the terminal showed exactly what a working night shows. The daemon
now stamps every tick into the run store, and the commands a user runs while
wondering read that stamp back.
"""

from __future__ import annotations

import json
from datetime import timedelta

import pytest
import yaml
from typer.testing import CliRunner

from ai_experiments.cli import app
from ai_experiments.daemon import MonitorDaemon
from ai_experiments.heartbeat import (
    Heartbeat,
    daemon_warning,
    heartbeat_path,
    read_heartbeat,
    write_heartbeat,
)
from ai_experiments.schemas import GoalSpec, utc_now
from ai_experiments.store import FilesystemRunStore
from ai_experiments.store.campaign import CampaignStore

runner = CliRunner()

GOAL = {
    "goal": "minimize loss",
    "name": "beat",
    "objective": {"metric": "loss", "mode": "min", "target": 0.0},
    "search_space": {"lr": {"type": "choice", "values": [0.1, 0.2]}},
    "workload": {"entrypoint": "true"},
}


@pytest.fixture(autouse=True)
def _no_env_override(monkeypatch):
    monkeypatch.delenv("IAX_RUNS_DIR", raising=False)


def _store(tmp_path) -> FilesystemRunStore:
    return FilesystemRunStore(tmp_path / "runs")


def test_a_tick_that_did_nothing_still_leaves_proof_it_ran(tmp_path):
    store = _store(tmp_path)

    report = MonitorDaemon(store).tick()

    assert report.actions == [] and report.errors == []
    beat = read_heartbeat(store.root)
    assert beat is not None
    assert beat.ticks == 1
    assert not beat.is_stale()


def test_the_heartbeat_counts_ticks_and_records_the_interval(tmp_path):
    store = _store(tmp_path)
    daemon = MonitorDaemon(store)

    daemon.tick(interval_seconds=30)
    daemon.tick(interval_seconds=30)

    beat = read_heartbeat(store.root)
    assert beat is not None
    assert beat.ticks == 2
    assert beat.interval_seconds == 30


def test_the_heartbeat_does_not_become_a_run(tmp_path):
    """`_daemon` shares the store root with the runs, like `_campaigns`."""
    store = _store(tmp_path)

    MonitorDaemon(store).tick()

    assert list(store.list_runs()) == []
    assert heartbeat_path(store.root).exists()


def test_an_unreadable_heartbeat_reads_as_no_daemon(tmp_path):
    store = _store(tmp_path)
    path = heartbeat_path(store.root)
    path.parent.mkdir(parents=True)
    path.write_text("{ truncated")

    assert read_heartbeat(store.root) is None
    assert "No daemon has ticked" in (daemon_warning(store.root) or "")


def test_no_daemon_at_all_is_reported_as_such(tmp_path):
    assert "No daemon has ticked" in (daemon_warning(tmp_path) or "")


def test_a_stale_heartbeat_names_its_age_and_the_fix(tmp_path):
    write_heartbeat(
        tmp_path, Heartbeat(timestamp=utc_now() - timedelta(minutes=45), pid=4242)
    )

    warning = daemon_warning(tmp_path)

    assert warning is not None
    assert "45m" in warning
    assert "pid 4242" in warning
    assert "`iax daemon`" in warning


def test_a_fresh_heartbeat_says_nothing(tmp_path):
    write_heartbeat(tmp_path, Heartbeat())

    assert daemon_warning(tmp_path) is None


def test_an_hours_old_heartbeat_reads_in_hours(tmp_path):
    write_heartbeat(tmp_path, Heartbeat(timestamp=utc_now() - timedelta(hours=3)))

    assert "3h00m" in (daemon_warning(tmp_path) or "")


# -- what the user sees --------------------------------------------------------


def _goal_file(tmp_path):
    path = tmp_path / "goal.yaml"
    path.write_text(yaml.safe_dump(GOAL))
    return path


def test_starting_a_campaign_with_no_daemon_says_so_in_json_mode(tmp_path):
    """An agent that gets JSON never sees the human hint on stdout."""
    result = runner.invoke(
        app,
        [
            "campaign",
            "start",
            str(_goal_file(tmp_path)),
            "--runs-dir",
            str(tmp_path / "runs"),
            "--json",
        ],
    )

    assert result.exit_code == 0
    json.loads(result.stdout)  # stdout stays parseable
    assert "No daemon has ticked" in result.stderr


def test_campaign_list_warns_only_while_a_campaign_is_running(tmp_path):
    store = _store(tmp_path)
    campaign_store = CampaignStore(store.root)
    state = campaign_store.create_campaign(GoalSpec(**GOAL))

    listing = runner.invoke(app, ["campaign", "list", "--runs-dir", str(store.root)])
    assert "No daemon" in listing.stderr

    state.status = "completed"
    campaign_store.write_state(state)

    done = runner.invoke(app, ["campaign", "list", "--runs-dir", str(store.root)])
    assert "No daemon" not in done.stderr


def test_a_ticking_daemon_silences_the_warning(tmp_path):
    store = _store(tmp_path)
    CampaignStore(store.root).create_campaign(GoalSpec(**GOAL))
    MonitorDaemon(store).tick()

    listing = runner.invoke(app, ["campaign", "list", "--runs-dir", str(store.root)])

    assert "No daemon" not in listing.stderr


def test_a_quiet_daemon_prints_a_line_a_reader_can_find_tomorrow(tmp_path, capsys):
    store = _store(tmp_path)

    MonitorDaemon(store).run_forever(interval_seconds=0, max_ticks=1)

    printed = json.loads(capsys.readouterr().out.strip())
    assert printed["heartbeat"] is True
    assert printed["next_tick_seconds"] == 0
