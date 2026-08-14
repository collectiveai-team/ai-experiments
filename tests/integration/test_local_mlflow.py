"""Local backend against a real MLflow server.

`test_tracking.py` covers this chain with a fake mlflow module, so nothing has
ever checked it against a real tracking server. Two halves of tracking.py's
contract are only observable with one:

* **harness-side mirroring** -- the daemon replays stored metrics, uploads
  ``artifacts/`` and closes the run, mapping the iax terminal state onto
  MLflow's (``completed``/``failed``/``cancelled`` -> ``FINISHED``/``FAILED``/
  ``KILLED``). A fake accepts any status string; a real server validates it.
* **workload-side handoff** -- the worker injects ``MLFLOW_RUN_ID`` and
  ``MLFLOW_TRACKING_URI`` so a workload that imports mlflow attaches to the run
  the harness already created, instead of opening a second one. Only a real
  server can show the two writers landing on the same run.

Every run here spawns a genuine detached worker and a genuine workload process.
"""

from __future__ import annotations

import sys
import time

import pytest

from ai_experiments.backends.local import LocalBackend
from ai_experiments.daemon import MonitorDaemon
from ai_experiments.schemas import ExperimentManifest, TrackingSpec, WorkloadSpec
from ai_experiments.store import FilesystemRunStore

pytestmark = pytest.mark.integration

TERMINAL = {"completed", "failed", "cancelled"}

TRAINER = """import json, os, pathlib, time
for step in range(1, 4):
    print('IAX_METRIC {"step": %d, "val_loss": %.4f}' % (step, 1.0 / step), flush=True)
    time.sleep(0.2)
artifacts = pathlib.Path(os.environ["IAX_ARTIFACTS_DIR"])
(artifacts / "checkpoint.txt").write_text("weights")
print("training finished", flush=True)
"""

# Attaches to the run the harness created, via the injected MLFLOW_RUN_ID.
MLFLOW_AWARE_TRAINER = """import os
import mlflow
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
with mlflow.start_run(run_id=os.environ["MLFLOW_RUN_ID"]):
    mlflow.log_param("logged_by", "workload")
    mlflow.log_metric("workload_metric", 42.0)
print("workload logged to mlflow directly", flush=True)
"""

FAILING = "import sys; print('boom', flush=True); sys.exit(3)"
SLEEPER = "import time; print('sleeping', flush=True); time.sleep(300)"


def _run(tmp_path, mlflow_uri, script: str, name: str):
    """Submit a real detached local run and hand back its store + handle."""
    script_path = tmp_path / f"{name}.py"
    script_path.write_text(script)
    store = FilesystemRunStore(tmp_path / "runs", capture_repro=False)
    manifest = ExperimentManifest(
        experiment=f"integration-local-{name}",
        backend="local",
        workload=WorkloadSpec(
            entrypoint=f"{sys.executable} {script_path}",
            working_dir=str(tmp_path),
        ),
        tracking=TrackingSpec(mlflow=True, tracking_uri=mlflow_uri),
    )
    handle = LocalBackend(store=store).submit(manifest)
    return store, handle


def _wait_terminal(store, run_id, timeout: float = 60.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = store.read_status(run_id)
        if status.status in TERMINAL:
            return status
        time.sleep(0.2)
    raise AssertionError(f"{run_id} never reached a terminal state")


@pytest.fixture(scope="module")
def completed_local_run(tmp_path_factory, mlflow_uri):
    tmp_path = tmp_path_factory.mktemp("local_ok")
    store, handle = _run(tmp_path, mlflow_uri, TRAINER, "trainer")
    at_submit = store.read_status(handle.run_id)
    final = _wait_terminal(store, handle.run_id)
    report = MonitorDaemon(store).tick()
    return {
        "store": store,
        "handle": handle,
        "at_submit": at_submit,
        "final": final,
        "report": report,
    }


def test_submit_preserves_the_mlflow_linkage(completed_local_run):
    details = completed_local_run["at_submit"].details
    assert details.get("mlflow_run_id")
    assert details.get("mlflow_tracking_uri")


def test_a_real_detached_worker_runs_the_workload(completed_local_run):
    final = completed_local_run["final"]
    assert final.status == "completed"
    assert final.exit_code == 0
    assert final.error is None

    metrics = completed_local_run["store"].read_metrics(
        completed_local_run["handle"].run_id
    )
    assert [p.step for p in metrics] == [1, 2, 3]


def test_daemon_mirrors_a_completed_run_as_finished(completed_local_run, mlflow_api):
    report = completed_local_run["report"]
    assert report.errors == []
    assert "mlflow_synced" in [a.action for a in report.actions]

    mlflow_run_id = completed_local_run["at_submit"].details["mlflow_run_id"]
    run = mlflow_api("runs/get", run_id=mlflow_run_id)["run"]
    tags = {t["key"]: t["value"] for t in run["data"].get("tags", [])}
    metrics = {m["key"]: m["value"] for m in run["data"].get("metrics", [])}

    assert run["info"]["status"] == "FINISHED"
    assert tags.get("iax.run_id") == completed_local_run["handle"].run_id
    assert tags.get("iax.backend") == "local"
    assert metrics.get("val_loss") == pytest.approx(1 / 3, rel=1e-3)


def test_run_artifacts_are_uploaded_to_mlflow(completed_local_run, mlflow_api):
    """The documented answer to "artifacts stay on the cluster": route them
    through MLflow's artifact store."""
    mlflow_run_id = completed_local_run["at_submit"].details["mlflow_run_id"]

    listing = mlflow_api("artifacts/list", run_id=mlflow_run_id)

    names = {f["path"] for f in listing.get("files", [])}
    assert "checkpoint.txt" in names


def test_mlflow_is_synced_only_once(completed_local_run):
    """finalize_tracking is idempotent -- a second tick must not re-upload."""
    report = MonitorDaemon(completed_local_run["store"]).tick()
    assert "mlflow_synced" not in [a.action for a in report.actions]


def test_workload_attaches_to_the_harness_mlflow_run(tmp_path, mlflow_uri, mlflow_api):
    """MLFLOW_RUN_ID handoff: the workload's own writes must land on the run
    the harness created, not a second one."""
    store, handle = _run(tmp_path, mlflow_uri, MLFLOW_AWARE_TRAINER, "mlflow_aware")
    mlflow_run_id = store.read_status(handle.run_id).details["mlflow_run_id"]

    final = _wait_terminal(store, handle.run_id)
    assert final.status == "completed", f"worker failed: {final.error}"

    run = mlflow_api("runs/get", run_id=mlflow_run_id)["run"]
    params = {p["key"]: p["value"] for p in run["data"].get("params", [])}
    metrics = {m["key"]: m["value"] for m in run["data"].get("metrics", [])}

    assert params.get("logged_by") == "workload"
    assert metrics.get("workload_metric") == pytest.approx(42.0)


def test_daemon_mirrors_a_failed_run_as_failed(tmp_path, mlflow_uri, mlflow_api):
    store, handle = _run(tmp_path, mlflow_uri, FAILING, "boom")
    mlflow_run_id = store.read_status(handle.run_id).details["mlflow_run_id"]

    final = _wait_terminal(store, handle.run_id)
    assert final.status == "failed"
    assert final.exit_code == 3

    MonitorDaemon(store).tick()

    run = mlflow_api("runs/get", run_id=mlflow_run_id)["run"]
    assert run["info"]["status"] == "FAILED"


def test_daemon_mirrors_a_cancelled_run_as_killed(tmp_path, mlflow_uri, mlflow_api):
    store, handle = _run(tmp_path, mlflow_uri, SLEEPER, "sleeper")
    mlflow_run_id = store.read_status(handle.run_id).details["mlflow_run_id"]

    backend = LocalBackend(store=store)
    for _ in range(50):  # let the worker actually start the workload
        if store.read_status(handle.run_id).status == "running":
            break
        time.sleep(0.2)
    backend.cancel(handle.run_id)

    final = _wait_terminal(store, handle.run_id)
    assert final.status == "cancelled"

    MonitorDaemon(store).tick()

    run = mlflow_api("runs/get", run_id=mlflow_run_id)["run"]
    assert run["info"]["status"] == "KILLED"
