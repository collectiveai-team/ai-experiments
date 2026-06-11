from __future__ import annotations

from datetime import timedelta
from unittest.mock import patch

from ai_experiments.daemon import MonitorDaemon
from ai_experiments.schemas import (
    EscalationPolicy,
    ExperimentManifest,
    MonitorPolicy,
    RunStatus,
    WorkloadSpec,
    utc_now,
)
from ai_experiments.store import FilesystemRunStore


def _store(tmp_path) -> FilesystemRunStore:
    return FilesystemRunStore(tmp_path / "runs")


def _running_run(
    store: FilesystemRunStore,
    monitoring: MonitorPolicy,
    **status_overrides: object,
) -> str:
    manifest = ExperimentManifest(
        experiment="daemon-test",
        backend="local",
        workload=WorkloadSpec(entrypoint="python train.py"),
        monitoring=monitoring,
    )
    run_id, run_dir = store.create_run(manifest)
    base = {
        "run_id": run_id,
        "backend": "local",
        "status": "running",
        "status_uri": str(store.status_path(run_id)),
        "run_dir": str(run_dir),
        "started_at": utc_now() - timedelta(minutes=10),
        "details": {"heartbeat_at": utc_now().isoformat()},
    }
    base.update(status_overrides)
    store.write_status(RunStatus(**base))
    return run_id


def test_daemon_auto_kills_timed_out_run(tmp_path):
    store = _store(tmp_path)
    run_id = _running_run(
        store, MonitorPolicy(timeout_seconds=60, auto_kill=True), pid=None
    )

    report = MonitorDaemon(store).tick()

    actions = {a.run_id: a for a in report.actions}
    assert actions[run_id].action == "auto_killed"
    assert "timeout_exceeded" in actions[run_id].reasons
    assert store.read_status(run_id).status == "cancelled"


def test_daemon_escalates_fatal_without_auto_kill(tmp_path):
    store = _store(tmp_path)
    run_id = _running_run(
        store, MonitorPolicy(timeout_seconds=60, auto_kill=False), pid=None
    )

    report = MonitorDaemon(store).tick()

    actions = {a.run_id: a for a in report.actions}
    assert actions[run_id].action == "escalated_fatal"
    assert (store.root / "_escalations" / f"{run_id}.json").exists()
    assert store.read_status(run_id).status == "running"  # not killed


def test_daemon_reaps_dead_worker(tmp_path):
    store = _store(tmp_path)
    run_id = _running_run(store, MonitorPolicy(), pid=99999)

    with patch(
        "ai_experiments.monitoring.rules._default_pid_alive", return_value=False
    ):
        report = MonitorDaemon(store).tick()

    actions = {a.run_id: a for a in report.actions}
    assert actions[run_id].action == "reaped_dead_process"
    assert store.read_status(run_id).status == "failed"


def test_daemon_escalates_suspicious_run_after_ladder_threshold(tmp_path):
    store = _store(tmp_path)
    stale_heartbeat = (utc_now() - timedelta(minutes=20)).isoformat()
    run_id = _running_run(
        store,
        MonitorPolicy(
            escalation=EscalationPolicy(after_suspicious_ticks=2, cooldown_minutes=0)
        ),
        pid=None,
        details={"heartbeat_at": stale_heartbeat},
    )
    daemon = MonitorDaemon(store)

    first = daemon.tick()
    second = daemon.tick()

    assert first.actions[0].action == "suspicion_recorded"
    assert second.actions[0].action == "escalated"
    assert (store.root / "_escalations" / f"{run_id}.json").exists()


def test_daemon_ignores_terminal_runs(tmp_path):
    store = _store(tmp_path)
    _running_run(store, MonitorPolicy(), status="completed", pid=None)

    report = MonitorDaemon(store).tick()

    assert report.runs_checked == 0
    assert report.actions == []
