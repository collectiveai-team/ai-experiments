from __future__ import annotations

from unittest.mock import patch

import pytest

from ai_experiments.backends.local import LocalBackend
from ai_experiments.backends.ray import RayBackend
from ai_experiments.schemas import (
    DataSpec,
    ExperimentManifest,
    TrackingSpec,
    WorkloadSpec,
)
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
    # The prefix says where the failure came from; the tail says what it was.
    assert status.error == "Ray job failed: worker crashed"

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


# -- cancelling must not rewrite how a job actually ended ----------------------


def test_ray_cancel_does_not_rewrite_a_job_that_already_failed(tmp_path):
    """A Ray run has no local supervisor, so its stored status is only as
    fresh as the last inspect. Cancelling used to stamp `cancelled` over a job
    that had failed on its own -- and the MLflow mirror then reported KILLED,
    "someone stopped this on purpose", in the one place an operator looks to
    find out why training died."""
    store = FilesystemRunStore(tmp_path / "runs")
    client = FakeRayClient(status="RUNNING")
    backend = RayBackend(store=store, client_factory=lambda _address: client)
    handle = backend.submit(_manifest(tmp_path))
    assert store.read_status(handle.run_id).status == "submitted"

    # The job fails on the cluster. Nothing has inspected it since, so the
    # store still says the run is live.
    client.status = "FAILED"
    client.message = "the workload raised"

    backend.cancel(handle.run_id)

    status = store.read_status(handle.run_id)
    assert status.status == "failed"
    # The message is the job's own, trimmed to the part that explains it.
    assert "the workload raised" in str(status.error)
    assert client.stopped == []  # nothing to stop; nothing was signalled


def test_ray_cancel_still_stops_a_running_job(tmp_path):
    """The control: cancellation of a live job must keep working."""
    store = FilesystemRunStore(tmp_path / "runs")
    client = FakeRayClient(status="RUNNING")
    backend = RayBackend(store=store, client_factory=lambda _address: client)
    handle = backend.submit(_manifest(tmp_path))

    backend.cancel(handle.run_id)

    assert client.stopped == ["ray-job-1"]
    assert store.read_status(handle.run_id).status == "cancelled"


def test_an_argument_with_spaces_stays_one_argument(tmp_path):
    captured = {}

    class _Client:
        def submit_job(self, **kwargs):
            captured.update(kwargs)
            return "raysubmit_1"

    backend = RayBackend(
        address="http://fake:8265",
        store=FilesystemRunStore(tmp_path),
        client_factory=lambda _address: _Client(),
    )
    manifest = ExperimentManifest(
        experiment="e",
        backend="ray",
        workload=WorkloadSpec(
            entrypoint="python train.py",
            args=["--label", "hello world"],
            working_dir=str(tmp_path),
        ),
    )

    backend.submit(manifest)

    assert "--label 'hello world'" in captured["entrypoint"]


def test_a_two_phase_workload_chains_both_phases_remotely(tmp_path):
    captured = {}

    class _Client:
        def submit_job(self, **kwargs):
            captured.update(kwargs)
            return "raysubmit_2"

    backend = RayBackend(
        address="http://fake:8265",
        store=FilesystemRunStore(tmp_path),
        client_factory=lambda _address: _Client(),
    )
    manifest = ExperimentManifest(
        experiment="e",
        backend="ray",
        workload=WorkloadSpec(
            entrypoint="python train.py",
            train="python train.py",
            evaluate="python evaluate.py",
            working_dir=str(tmp_path),
            data=DataSpec(train="data/train", test="data/test"),
        ),
    )

    backend.submit(manifest)

    entrypoint = captured["entrypoint"]
    assert "python train.py" in entrypoint
    assert "python evaluate.py" in entrypoint
    assert entrypoint.index("train.py") < entrypoint.index("evaluate.py")
    assert "&&" in entrypoint


def test_each_phase_is_preceded_by_an_echoed_phase_marker(tmp_path):
    """`_sync_results_from_logs` has no supervisor to ask which phase is
    running, so the entrypoint has to echo it for the client-side log parser.
    The ingestion tests below hand-craft their own log text, so they would
    stay green even if this echo were dropped from `submit` -- this is the
    one test that would actually catch that regression.
    """
    captured = {}

    class _Client:
        def submit_job(self, **kwargs):
            captured.update(kwargs)
            return "raysubmit_4"

    backend = RayBackend(
        address="http://fake:8265",
        store=FilesystemRunStore(tmp_path),
        client_factory=lambda _address: _Client(),
    )
    manifest = ExperimentManifest(
        experiment="e",
        backend="ray",
        workload=WorkloadSpec(
            entrypoint="python train.py",
            train="python train.py",
            evaluate="python evaluate.py",
            working_dir=str(tmp_path),
            data=DataSpec(train="data/train", test="data/test"),
        ),
    )

    backend.submit(manifest)

    entrypoint = captured["entrypoint"]
    assert "echo IAX_PHASE=train" in entrypoint
    assert "echo IAX_PHASE=evaluate" in entrypoint
    assert (
        entrypoint.index("echo IAX_PHASE=train")
        < entrypoint.index("python train.py")
        < entrypoint.index("echo IAX_PHASE=evaluate")
        < entrypoint.index("python evaluate.py")
    )


def test_the_remote_run_carries_the_harness_variables(tmp_path):
    captured = {}

    class _Client:
        def submit_job(self, **kwargs):
            captured.update(kwargs)
            return "raysubmit_3"

    backend = RayBackend(
        address="http://fake:8265",
        store=FilesystemRunStore(tmp_path),
        client_factory=lambda _address: _Client(),
    )
    manifest = ExperimentManifest(
        experiment="e",
        backend="ray",
        workload=WorkloadSpec(entrypoint="python toy.py", working_dir=str(tmp_path)),
    )

    handle = backend.submit(manifest)

    assert captured["runtime_env"]["env_vars"]["IAX_RUN_ID"] == handle.run_id


def test_a_result_after_the_evaluate_marker_is_scored(tmp_path):
    client = FakeRayClient(
        status="RUNNING",
        logs='IAX_PHASE=evaluate\nIAX_RESULT {"acc": 0.9}\n',
    )
    backend = RayBackend(
        store=FilesystemRunStore(tmp_path), client_factory=lambda _address: client
    )
    handle = backend.submit(_manifest(tmp_path))

    backend.inspect(handle.run_id)

    results = backend.store.read_results(handle.run_id)
    assert [r.values for r in results] == [{"acc": 0.9}]


def test_a_result_after_the_train_marker_is_discarded_with_a_warning(tmp_path):
    client = FakeRayClient(
        status="RUNNING",
        logs='IAX_PHASE=train\nIAX_RESULT {"acc": 0.9}\n',
    )
    backend = RayBackend(
        store=FilesystemRunStore(tmp_path), client_factory=lambda _address: client
    )
    handle = backend.submit(_manifest(tmp_path))

    backend.inspect(handle.run_id)

    assert backend.store.read_results(handle.run_id) == []
    events = backend.store.read_events(handle.run_id)
    assert any(
        event.level == "warning"
        and event.message == "result reported from the train phase; discarded"
        for event in events
    )


def test_two_inspects_of_the_same_logs_do_not_duplicate_the_result(tmp_path):
    client = FakeRayClient(
        status="RUNNING",
        logs='IAX_PHASE=evaluate\nIAX_RESULT {"acc": 0.9}\n',
    )
    backend = RayBackend(
        store=FilesystemRunStore(tmp_path), client_factory=lambda _address: client
    )
    handle = backend.submit(_manifest(tmp_path))

    backend.inspect(handle.run_id)
    backend.inspect(handle.run_id)

    assert len(backend.store.read_results(handle.run_id)) == 1
