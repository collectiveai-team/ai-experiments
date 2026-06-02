from __future__ import annotations

from ai_experiments.backends.ray import RayBackend
from ai_experiments.schemas import ExperimentManifest, WorkloadSpec
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
