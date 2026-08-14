"""Ray + MLflow against real services.

Covers the chain the stubbed suite cannot: submit to a live Ray cluster, let
it run, harvest metrics from real job logs, and have the daemon mirror the run
into a real MLflow server.

Issue #6 lived exactly here. `RayBackend.submit` recorded
``details.mlflow_run_id`` via ``begin_tracking`` and then overwrote it with a
fresh ``RunStatus`` in ``write_handle``. ``finalize_tracking`` keys on that id,
so no Ray run was ever mirrored and every one sat RUNNING in MLflow forever.
Against ec0081e this module fails and the run is left RUNNING with no metrics.
"""

from __future__ import annotations

import time

import pytest

from ai_experiments.backends.ray import RayBackend
from ai_experiments.daemon import MonitorDaemon
from ai_experiments.schemas import ExperimentManifest, TrackingSpec, WorkloadSpec
from ai_experiments.store import FilesystemRunStore

pytestmark = pytest.mark.integration

WORKLOAD = """import time
for step in range(1, 4):
    print('IAX_METRIC {"step": %d, "val_loss": %.4f}' % (step, 1.0 / step), flush=True)
    time.sleep(1)
print("training finished", flush=True)
"""

TERMINAL = {"completed", "failed", "cancelled"}


@pytest.fixture(scope="module")
def completed_ray_run(tmp_path_factory, ray_address, mlflow_uri):
    """One real Ray job, submitted once and reused: a live submit plus job
    startup costs ~15s, and every assertion below inspects the same run."""
    work = tmp_path_factory.mktemp("ray_workload")
    (work / "train.py").write_text(WORKLOAD)
    store = FilesystemRunStore(work / "runs", capture_repro=False)

    manifest = ExperimentManifest(
        experiment="integration-ray-mlflow",
        backend="ray",
        backend_address=ray_address,
        workload=WorkloadSpec(entrypoint="python train.py", working_dir=str(work)),
        tracking=TrackingSpec(mlflow=True, tracking_uri=mlflow_uri),
    )

    backend = RayBackend(store=store, address=ray_address)
    handle = backend.submit(manifest)
    # Linkage is asserted from the status written at submit time, before any
    # later write can repair it.
    at_submit = store.read_status(handle.run_id)

    deadline = time.time() + 180
    status = at_submit
    while time.time() < deadline:
        status = backend.inspect(handle.run_id)
        if status.status in TERMINAL:
            break
        time.sleep(3)

    return {
        "store": store,
        "backend": backend,
        "handle": handle,
        "at_submit": at_submit,
        "final": status,
    }


def test_submit_preserves_the_mlflow_linkage(completed_ray_run):
    """The regression itself: the id must still be there after submit."""
    details = completed_ray_run["at_submit"].details
    assert details.get("mlflow_run_id"), (
        "details.mlflow_run_id was lost during submit; finalize_tracking will "
        "never mirror this run and it will sit RUNNING in MLflow forever"
    )
    assert details.get("mlflow_tracking_uri")


def test_submit_writes_a_ray_status_not_the_store_fallback(completed_ray_run):
    """begin_tracking used to create status.json through read_status's
    "file not found" fallback, briefly labelling a Ray run backend="local"."""
    at_submit = completed_ray_run["at_submit"]
    assert at_submit.backend == "ray"
    assert at_submit.error is None
    assert at_submit.external_id, "the ray job id should be recorded on the status"


def test_the_real_ray_job_completes(completed_ray_run):
    final = completed_ray_run["final"]
    assert final.status == "completed", (
        f"ray reported {final.details.get('ray_status')}"
    )


def test_metrics_are_harvested_from_real_ray_job_logs(completed_ray_run):
    metrics = completed_ray_run["store"].read_metrics(
        completed_ray_run["handle"].run_id
    )
    assert [p.step for p in metrics] == [1, 2, 3]
    assert metrics[-1].values["val_loss"] == pytest.approx(1 / 3, rel=1e-3)


def test_linkage_survives_repeated_inspects(completed_ray_run):
    store, handle = completed_ray_run["store"], completed_ray_run["handle"]
    expected = completed_ray_run["at_submit"].details["mlflow_run_id"]
    assert store.read_status(handle.run_id).details.get("mlflow_run_id") == expected


def test_daemon_mirrors_the_run_into_real_mlflow(completed_ray_run, mlflow_api):
    """End of the chain, asserted against the tracking server itself."""
    store = completed_ray_run["store"]
    handle = completed_ray_run["handle"]
    mlflow_run_id = completed_ray_run["at_submit"].details["mlflow_run_id"]

    report = MonitorDaemon(store).tick()

    assert report.errors == []
    assert "mlflow_synced" in [a.action for a in report.actions]

    run = mlflow_api("runs/get", run_id=mlflow_run_id)["run"]
    tags = {t["key"]: t["value"] for t in run["data"].get("tags", [])}
    metrics = {m["key"]: m["value"] for m in run["data"].get("metrics", [])}

    assert run["info"]["status"] == "FINISHED", "run left RUNNING in MLflow"
    assert tags.get("iax.run_id") == handle.run_id
    assert metrics.get("val_loss") == pytest.approx(1 / 3, rel=1e-3)
