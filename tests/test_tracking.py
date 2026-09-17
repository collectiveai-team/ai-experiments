"""MLflow integration, tested against a fake mlflow module.

The real package is optional and heavy. An end-to-end check against real mlflow lives
in the live smoke flow, not in the unit suite.
"""

from __future__ import annotations

from unittest.mock import patch

from ai_experiments.daemon import MonitorDaemon
from ai_experiments.schemas import (
    ExperimentManifest,
    MetricPoint,
    RunStatus,
    TrackingSpec,
    WorkloadSpec,
    utc_now,
)
from ai_experiments.store import FilesystemRunStore
from ai_experiments.tracking import (
    MlflowTracker,
    begin_tracking,
    finalize_tracking,
)


def _manifest(**overrides: object) -> ExperimentManifest:
    data: dict[str, object] = {
        "experiment": "camp/t000",
        "backend": "local",
        "workload": WorkloadSpec(entrypoint="python train.py"),
        "tracking": TrackingSpec(mlflow=True),
        "metadata": {"campaign": "camp", "trial_id": "t000", "params": {"lr": 0.01}},
    }
    data.update(overrides)
    return ExperimentManifest.model_validate(data)


def _seeded_run(tmp_path, manifest):
    store = FilesystemRunStore(tmp_path / "runs", capture_repro=False)
    run_id, run_dir = store.create_run(manifest)
    store.write_status(
        RunStatus(
            run_id=run_id,
            backend="local",
            status="running",
            status_uri=str(store.status_path(run_id)),
            run_dir=str(run_dir),
        )
    )
    return store, run_id


def test_start_run_logs_params_and_tags(tmp_path, fake_mlflow_module_factory):
    fake = fake_mlflow_module_factory()
    manifest = _manifest()
    store, run_id = _seeded_run(tmp_path, manifest)
    tracker = MlflowTracker(mlflow_module=fake)

    mlflow_run_id = tracker.start_run(store, run_id, manifest)

    client = fake.last_client
    assert client is not None, "start_run must have created a client"
    run = client.runs[mlflow_run_id]
    assert run["tags"]["iax.run_id"] == run_id
    assert run["tags"]["iax.campaign"] == "camp"
    assert run["params"]["lr"] == 0.01
    assert client.experiments == {"camp": "exp_0"}


def test_file_store_gets_allow_flag(tmp_path, monkeypatch, fake_mlflow_module_factory):
    monkeypatch.delenv("MLFLOW_ALLOW_FILE_STORE", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    fake = fake_mlflow_module_factory()
    manifest = _manifest(tracking=TrackingSpec(mlflow=True, tracking_uri="file:///tmp/mlruns"))
    store, run_id = _seeded_run(tmp_path, manifest)

    with patch("ai_experiments.tracking._load_mlflow", return_value=fake):
        env = begin_tracking(store, run_id, manifest)

    # MLflow 3.x gates the file store; iax opts out for the harness process
    # and the workload env when a file store is the explicit choice.
    assert env["MLFLOW_ALLOW_FILE_STORE"] == "true"
    assert __import__("os").environ["MLFLOW_ALLOW_FILE_STORE"] == "true"


def test_remote_store_does_not_get_allow_flag(tmp_path, monkeypatch, fake_mlflow_module_factory):
    monkeypatch.delenv("MLFLOW_ALLOW_FILE_STORE", raising=False)
    fake = fake_mlflow_module_factory()
    manifest = _manifest(tracking=TrackingSpec(mlflow=True, tracking_uri="http://mlflow:5000"))
    store, run_id = _seeded_run(tmp_path, manifest)

    with patch("ai_experiments.tracking._load_mlflow", return_value=fake):
        env = begin_tracking(store, run_id, manifest)

    assert "MLFLOW_ALLOW_FILE_STORE" not in env
    assert "MLFLOW_ALLOW_FILE_STORE" not in __import__("os").environ


def test_user_file_store_opt_out_is_respected(tmp_path, monkeypatch, fake_mlflow_module_factory):
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "false")
    fake = fake_mlflow_module_factory()
    manifest = _manifest(tracking=TrackingSpec(mlflow=True, tracking_uri="file:///tmp/mlruns"))
    store, run_id = _seeded_run(tmp_path, manifest)

    with patch("ai_experiments.tracking._load_mlflow", return_value=fake):
        env = begin_tracking(store, run_id, manifest)

    assert env["MLFLOW_ALLOW_FILE_STORE"] == "false"


def test_begin_tracking_records_details_and_env(tmp_path, fake_mlflow_module_factory):
    fake = fake_mlflow_module_factory()
    manifest = _manifest()
    store, run_id = _seeded_run(tmp_path, manifest)

    with patch("ai_experiments.tracking._load_mlflow", return_value=fake):
        env = begin_tracking(store, run_id, manifest)

    status = store.read_status(run_id)
    assert env["MLFLOW_RUN_ID"] == status.details["mlflow_run_id"]
    assert env["MLFLOW_TRACKING_URI"] == "file:///fake-mlruns"


def test_begin_tracking_disabled_is_noop(tmp_path):
    manifest = _manifest(tracking=TrackingSpec(mlflow=False))
    store, run_id = _seeded_run(tmp_path, manifest)

    assert begin_tracking(store, run_id, manifest) == {}
    assert "mlflow_run_id" not in store.read_status(run_id).details


def test_begin_tracking_survives_missing_mlflow(tmp_path):
    manifest = _manifest()
    store, run_id = _seeded_run(tmp_path, manifest)

    with patch(
        "ai_experiments.tracking._load_mlflow",
        side_effect=ImportError("no mlflow"),
    ):
        env = begin_tracking(store, run_id, manifest)

    assert env == {}
    events = store.read_events(run_id)
    assert any("mlflow tracking unavailable" in e.message for e in events)


def test_finalize_mirrors_metrics_artifacts_and_status(tmp_path, fake_mlflow_module_factory):
    fake = fake_mlflow_module_factory()
    manifest = _manifest()
    store, run_id = _seeded_run(tmp_path, manifest)
    with patch("ai_experiments.tracking._load_mlflow", return_value=fake):
        begin_tracking(store, run_id, manifest)
    client = fake.last_client
    assert client is not None, "begin_tracking must have created a client"

    store.append_metric(run_id, MetricPoint(step=1, values={"loss": 0.5}))
    store.append_metric(run_id, MetricPoint(step=2, values={"loss": 0.2}))
    (store.artifacts_dir(run_id) / "model.bin").write_bytes(b"w")
    store.update_status(run_id, status="completed", completed_at=utc_now())

    with patch("ai_experiments.tracking._load_mlflow", return_value=fake):
        synced = finalize_tracking(store, store.read_status(run_id))

    assert synced is True
    mlflow_run_id = store.read_status(run_id).details["mlflow_run_id"]
    run = client.runs[mlflow_run_id]
    assert ("loss", 0.5, 1) in run["metrics"]
    assert ("loss", 0.2, 2) in run["metrics"]
    assert run["artifacts"] == [str(store.artifacts_dir(run_id))]
    assert run["status"] == "FINISHED"
    assert store.read_status(run_id).details["mlflow_synced"] is True

    # second pass is a no-op
    with patch("ai_experiments.tracking._load_mlflow", return_value=fake):
        assert finalize_tracking(store, store.read_status(run_id)) is False


def test_daemon_finalizes_tracked_terminal_runs(tmp_path, fake_mlflow_module_factory):
    fake = fake_mlflow_module_factory()
    manifest = _manifest()
    store, run_id = _seeded_run(tmp_path, manifest)
    with patch("ai_experiments.tracking._load_mlflow", return_value=fake):
        begin_tracking(store, run_id, manifest)
    store.update_status(run_id, status="completed", completed_at=utc_now())

    with patch("ai_experiments.tracking._load_mlflow", return_value=fake):
        report = MonitorDaemon(store).tick()

    actions = {a.run_id: a.action for a in report.actions}
    assert actions[run_id] == "mlflow_synced"
    assert store.read_status(run_id).details["mlflow_synced"] is True
