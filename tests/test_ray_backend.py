from __future__ import annotations

from unittest.mock import patch

import pytest

from ai_experiments.backends.local import LocalBackend
from ai_experiments.backends.ray import RayBackend
from ai_experiments.schemas import ExperimentManifest, TrackingSpec, WorkloadSpec
from ai_experiments.store import FilesystemRunStore


class FakeRayClient:
    def __init__(self, *, status: str, message: str = "", logs: str = "") -> None:
        self.status = status
        self.message = message
        self.logs = logs
        self.stopped: list[str] = []

    def submit_job(self, *, entrypoint: str, runtime_env: dict) -> str:
        self.entrypoint = entrypoint
        self.runtime_env = runtime_env
        return "ray-job-1"

    def get_job_status(self, job_id: str) -> str:
        assert job_id == "ray-job-1"
        return self.status

    def get_job_info(self, job_id: str) -> dict:
        assert job_id == "ray-job-1"
        return {"status": self.status, "message": self.message}

    def get_job_logs(self, job_id: str) -> str:
        assert job_id == "ray-job-1"
        return self.logs

    def stop_job(self, job_id: str) -> None:
        self.stopped.append(job_id)


def _manifest(tmp_path) -> ExperimentManifest:
    return ExperimentManifest(
        experiment="ray-smoke",
        backend="ray",
        workload=WorkloadSpec(
            entrypoint="python train.py",
            working_dir=str(tmp_path),
        ),
    )


def test_ray_pending_without_resource_pressure_is_quiet(tmp_path):
    client = FakeRayClient(status="PENDING", message="Waiting for available worker.")
    backend = RayBackend(
        store=FilesystemRunStore(tmp_path / "runs"),
        client_factory=lambda _address: client,
    )
    handle = backend.submit(_manifest(tmp_path))

    status = backend.inspect(handle.run_id)
    assert status.status == "submitted"
    assert status.details["ray_status"] == "pending"
    assert status.details["ray_condition"] == "queued"

    report = backend.diagnose(handle.run_id)
    assert report.decision.decision == "continue_waiting"


def test_ray_resource_starvation_delegates_diagnosis(tmp_path):
    client = FakeRayClient(
        status="PENDING",
        message="The task cannot be scheduled because requested resources are not available.",
        logs="No available node types can fulfill resource request {'GPU': 1}",
    )
    backend = RayBackend(
        store=FilesystemRunStore(tmp_path / "runs"),
        client_factory=lambda _address: client,
    )
    handle = backend.submit(_manifest(tmp_path))

    status = backend.inspect(handle.run_id)
    assert status.status == "submitted"
    assert status.details["ray_condition"] == "resource_starved"
    assert "ray_log_tail" in status.details

    report = backend.diagnose(handle.run_id)
    assert report.decision.decision == "delegate_diagnosis"
    assert "ray_resource_starved" in report.decision.reasons


def test_ray_failed_status_records_error(tmp_path):
    client = FakeRayClient(status="FAILED", message="worker crashed")
    backend = RayBackend(
        store=FilesystemRunStore(tmp_path / "runs"),
        client_factory=lambda _address: client,
    )
    handle = backend.submit(_manifest(tmp_path))

    status = backend.inspect(handle.run_id)
    assert status.status == "failed"
    assert status.error == "worker crashed"

    report = backend.diagnose(handle.run_id)
    assert report.decision.decision == "training_failed"


@pytest.mark.parametrize(
    ("explicit_address", "env_address", "expected_address"),
    [
        (
            "https://manifest.example.com",
            "https://env.example.com",
            "https://manifest.example.com",
        ),
        (None, "https://env.example.com", "https://env.example.com"),
        (None, None, "http://127.0.0.1:8265"),
    ],
)
def test_ray_backend_resolves_address_precedence(
    tmp_path,
    monkeypatch,
    explicit_address,
    env_address,
    expected_address,
):
    if env_address is None:
        monkeypatch.delenv("RAY_ADDRESS", raising=False)
    else:
        monkeypatch.setenv("RAY_ADDRESS", env_address)

    captured_addresses: list[str] = []
    client = FakeRayClient(status="RUNNING")

    def client_factory(address: str) -> FakeRayClient:
        captured_addresses.append(address)
        return client

    backend = RayBackend(
        store=FilesystemRunStore(tmp_path / "runs"),
        address=explicit_address,
        client_factory=client_factory,
    )
    handle = backend.submit(_manifest(tmp_path))
    status = backend.inspect(handle.run_id)

    assert captured_addresses == [expected_address, expected_address]
    assert handle.dashboard_url == expected_address
    assert status.details["ray_address"] == expected_address


# -- the MLflow linkage must survive submit on every backend ------------------
#
# `begin_tracking` records details.mlflow_run_id, and the daemon's
# finalize_tracking keys on it. Any submit path that writes a fresh status
# afterwards silently destroys the linkage: runs are never mirrored and sit
# RUNNING in MLflow forever. Assert the invariant per backend rather than
# trusting the order of calls inside submit.


def _tracked_manifest(tmp_path, backend: str) -> ExperimentManifest:
    return ExperimentManifest(
        experiment="mlflow-linkage",
        backend=backend,
        workload=WorkloadSpec(entrypoint="python train.py", working_dir=str(tmp_path)),
        tracking=TrackingSpec(mlflow=True, tracking_uri=f"file://{tmp_path}/mlruns"),
    )


def test_ray_submit_preserves_the_mlflow_linkage(tmp_path):
    from test_tracking import FakeMlflowModule

    store = FilesystemRunStore(tmp_path / "runs")
    client = FakeRayClient(status="RUNNING")
    backend = RayBackend(store=store, client_factory=lambda _address: client)

    with patch("ai_experiments.tracking._load_mlflow", return_value=FakeMlflowModule()):
        handle = backend.submit(_tracked_manifest(tmp_path, "ray"))

    status = store.read_status(handle.run_id)
    assert status.details["mlflow_run_id"]
    assert status.details["mlflow_tracking_uri"]
    # The status must describe the run, not the store's fallback for a file it
    # could not find.
    assert status.backend == "ray"
    assert status.error is None
    assert status.external_id == "ray-job-1"


def test_local_submit_preserves_the_mlflow_linkage(tmp_path):
    from test_tracking import FakeMlflowModule

    store = FilesystemRunStore(tmp_path / "runs")
    backend = LocalBackend(store=store)

    with (
        patch("ai_experiments.tracking._load_mlflow", return_value=FakeMlflowModule()),
        patch("subprocess.Popen"),
    ):
        handle = backend.submit(_tracked_manifest(tmp_path, "local"))

    status = store.read_status(handle.run_id)
    assert status.details["mlflow_run_id"]
    assert status.backend == "local"
    assert status.error is None
