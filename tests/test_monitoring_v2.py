from __future__ import annotations

from datetime import timedelta

from ai_experiments.monitoring.rules import diagnose_run
from ai_experiments.schemas import (
    ExperimentManifest,
    MetricPoint,
    MonitorPolicy,
    RunStatus,
    WorkloadSpec,
    utc_now,
)
from ai_experiments.store import FilesystemRunStore


def _make_run(
    tmp_path,
    monitoring: MonitorPolicy | None = None,
    **status_overrides: object,
) -> tuple[FilesystemRunStore, str]:
    store = FilesystemRunStore(tmp_path / "runs")
    manifest = ExperimentManifest(
        experiment="mon-v2",
        backend="local",
        workload=WorkloadSpec(entrypoint="python train.py"),
        monitoring=monitoring or MonitorPolicy(),
    )
    run_id, run_dir = store.create_run(manifest)
    base = {
        "run_id": run_id,
        "backend": "local",
        "status": "running",
        "status_uri": str(store.status_path(run_id)),
        "run_dir": str(run_dir),
        "details": {"heartbeat_at": utc_now().isoformat()},
    }
    base.update(status_overrides)
    store.write_status(RunStatus(**base))
    return store, run_id


def test_nan_objective_triggers_kill(tmp_path):
    store, run_id = _make_run(tmp_path)
    store.append_metric(run_id, MetricPoint(step=1, values={"loss": 0.5}))
    store.append_metric(run_id, MetricPoint(step=2, values={"loss": float("nan")}))

    report = diagnose_run(store, run_id, pid_alive=lambda pid: True)

    assert report.decision.decision == "kill"
    assert "non_finite_metric:loss" in report.decision.reasons


def test_timeout_triggers_kill(tmp_path):
    store, run_id = _make_run(
        tmp_path,
        monitoring=MonitorPolicy(timeout_seconds=60),
        started_at=utc_now() - timedelta(minutes=10),
    )
    store.append_metric(run_id, MetricPoint(step=1, values={"loss": 0.5}))

    report = diagnose_run(store, run_id, pid_alive=lambda pid: True)

    assert report.decision.decision == "kill"
    assert "timeout_exceeded" in report.decision.reasons


def test_dead_process_triggers_kill(tmp_path):
    store, run_id = _make_run(tmp_path, pid=99999)

    report = diagnose_run(store, run_id, pid_alive=lambda pid: False)

    assert report.decision.decision == "kill"
    assert "process_dead" in report.decision.reasons


def test_stale_heartbeat_is_suspicious(tmp_path):
    stale = (utc_now() - timedelta(minutes=10)).isoformat()
    store, run_id = _make_run(tmp_path, details={"heartbeat_at": stale})
    store.append_metric(run_id, MetricPoint(step=1, values={"loss": 0.5}))

    report = diagnose_run(store, run_id, pid_alive=lambda pid: True)

    assert report.decision.decision == "delegate_diagnosis"
    assert any(r.startswith("heartbeat_stale") for r in report.decision.reasons)


def test_metric_staleness_is_suspicious(tmp_path):
    store, run_id = _make_run(tmp_path, monitoring=MonitorPolicy(stuck_after_minutes=5))
    old = MetricPoint(step=1, values={"loss": 0.5})
    old.timestamp = utc_now() - timedelta(minutes=30)
    store.append_metric(run_id, old)

    report = diagnose_run(store, run_id, pid_alive=lambda pid: True)

    assert report.decision.decision == "delegate_diagnosis"
    assert any(r.startswith("no_metric_progress") for r in report.decision.reasons)


def test_plateau_is_suspicious(tmp_path):
    store, run_id = _make_run(
        tmp_path,
        monitoring=MonitorPolicy(objective_metric="loss", plateau_patience_points=3),
    )
    for step, loss in enumerate([1.0, 0.5, 0.6, 0.6, 0.6]):
        store.append_metric(run_id, MetricPoint(step=step, values={"loss": loss}))

    report = diagnose_run(store, run_id, pid_alive=lambda pid: True)

    assert report.decision.decision == "delegate_diagnosis"
    assert any(r.startswith("objective_plateau") for r in report.decision.reasons)


def test_improving_run_continues(tmp_path):
    store, run_id = _make_run(
        tmp_path,
        monitoring=MonitorPolicy(
            objective_metric="loss", plateau_patience_points=3, timeout_seconds=3600
        ),
        pid=1,
    )
    for step, loss in enumerate([1.0, 0.8, 0.6, 0.4, 0.2]):
        store.append_metric(run_id, MetricPoint(step=step, values={"loss": loss}))

    report = diagnose_run(store, run_id, pid_alive=lambda pid: True)

    assert report.decision.decision == "continue_waiting"
    assert report.decision.reasons == ["run_active"]
