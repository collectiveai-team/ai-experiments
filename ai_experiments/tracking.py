"""MLflow integration (optional — install the ``mlflow`` extra).

Two complementary halves:

1. **Workload-side**: `start_run` creates the MLflow run at submit time and
   the backends inject ``MLFLOW_RUN_ID`` + ``MLFLOW_TRACKING_URI`` into the
   workload environment — including Ray ``runtime_env`` — so a workload on a
   remote cluster node can ``mlflow.log_artifact(...)`` checkpoints straight
   to the central tracking server. This is the answer to "artifacts stay on
   the cluster": route them through MLflow's artifact store.
2. **Harness-side mirroring**: `finalize_run` (called by the daemon when a
   run reaches a terminal state) logs every collected ``IAX_METRIC`` point
   with steps, uploads the local ``artifacts/`` dir, and closes the MLflow
   run — so runs appear fully in MLflow even when the workload never imports
   mlflow.

Everything here is best-effort: a missing mlflow package or an unreachable
tracking server records a warning event and never blocks a submit or a
daemon tick.
"""

from __future__ import annotations

import os
from typing import Any

from ai_experiments.repro import read_repro
from ai_experiments.schemas import ExperimentManifest, RunEvent, RunStatus
from ai_experiments.store import FilesystemRunStore

_TERMINAL_MLFLOW_STATUS = {
    "completed": "FINISHED",
    "failed": "FAILED",
    "cancelled": "KILLED",
}


class TrackingUnavailable(RuntimeError):
    pass


def _load_mlflow() -> Any:
    try:
        import mlflow
    except ImportError as exc:
        raise TrackingUnavailable(
            "mlflow is not installed; install ai-experiments[mlflow]"
        ) from exc
    return mlflow


def _is_file_store(tracking_uri: str | None) -> bool:
    """True when the URI resolves to MLflow's local filesystem store
    (``file:...``, a plain path, or nothing — mlflow defaults to ./mlruns)."""
    resolved = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", "")
    return resolved == "" or resolved.startswith("file:") or "://" not in resolved


def _file_store_optout(tracking_uri: str | None) -> dict[str, str]:
    """MLflow 3.x gates the filesystem store behind MLFLOW_ALLOW_FILE_STORE.

    Configuring a file store in iax is an explicit choice (and the only local
    option with mlflow-skinny, which has no SQL store), so opt out of the
    gate on the user's behalf — for this process and for the workload env.
    An explicit MLFLOW_ALLOW_FILE_STORE=false set by the user is respected.
    """
    if not _is_file_store(tracking_uri):
        return {}
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    return {"MLFLOW_ALLOW_FILE_STORE": os.environ["MLFLOW_ALLOW_FILE_STORE"]}


class MlflowTracker:
    def __init__(
        self,
        tracking_uri: str | None = None,
        experiment: str | None = None,
        mlflow_module: Any = None,
    ) -> None:
        self._mlflow = mlflow_module or _load_mlflow()
        self.extra_env = _file_store_optout(tracking_uri)
        self.client = self._mlflow.MlflowClient(tracking_uri=tracking_uri)
        self.tracking_uri = tracking_uri or self._mlflow.get_tracking_uri()
        self.experiment = experiment

    def _experiment_id(self, name: str) -> str:
        existing = self.client.get_experiment_by_name(name)
        if existing is not None:
            return existing.experiment_id
        return self.client.create_experiment(name)

    def start_run(
        self, store: FilesystemRunStore, run_id: str, manifest: ExperimentManifest
    ) -> str:
        """Create the MLflow run mirroring an iax run; returns its mlflow id."""
        experiment_name = self.experiment or manifest.metadata.get(
            "campaign", manifest.experiment.split("/")[0]
        )
        tags = {
            "iax.run_id": run_id,
            "iax.backend": manifest.backend,
            "mlflow.runName": manifest.experiment,
        }
        for key in ("campaign", "trial_id"):
            if manifest.metadata.get(key):
                tags[f"iax.{key}"] = str(manifest.metadata[key])
        repro = read_repro(store.run_dir(run_id)) or {}
        if repro.get("git_sha"):
            tags["mlflow.source.git.commit"] = repro["git_sha"]
            tags["iax.git_dirty"] = str(repro.get("git_dirty"))

        mlflow_run = self.client.create_run(
            experiment_id=self._experiment_id(str(experiment_name)),
            tags=tags,
        )
        mlflow_run_id = mlflow_run.info.run_id

        params: dict[str, Any] = dict(manifest.metadata.get("params") or {})
        params.setdefault("entrypoint", manifest.workload.entrypoint)
        for name, value in params.items():
            self.client.log_param(mlflow_run_id, str(name), value)
        return mlflow_run_id

    def finalize_run(
        self, store: FilesystemRunStore, run_id: str, mlflow_run_id: str
    ) -> None:
        """Mirror collected metrics + local artifacts, then close the run."""
        for index, point in enumerate(store.read_metrics(run_id)):
            step = point.step if point.step is not None else index
            timestamp = int(point.timestamp.timestamp() * 1000)
            for name, value in point.values.items():
                try:
                    self.client.log_metric(
                        mlflow_run_id, name, value, timestamp=timestamp, step=step
                    )
                except Exception:
                    continue  # non-finite values may be rejected by some stores

        artifacts = store.artifacts_dir(run_id)
        if artifacts.exists() and any(artifacts.iterdir()):
            self.client.log_artifacts(mlflow_run_id, str(artifacts))

        status = store.read_status(run_id)
        self.client.set_terminated(
            mlflow_run_id, status=_TERMINAL_MLFLOW_STATUS.get(status.status, "FINISHED")
        )


def tracker_for(manifest: ExperimentManifest) -> MlflowTracker | None:
    """Tracker for a manifest, or None when tracking is off."""
    if not manifest.tracking.mlflow:
        return None
    return MlflowTracker(
        tracking_uri=manifest.tracking.tracking_uri,
        experiment=manifest.tracking.experiment,
    )


def begin_tracking(
    store: FilesystemRunStore, run_id: str, manifest: ExperimentManifest
) -> dict[str, str]:
    """Submit-time hook used by the backends.

    Returns env vars for the workload ({} when tracking is off or broken) and
    records the mlflow run id in the run's status details for the daemon's
    finalization pass.
    """
    if not manifest.tracking.mlflow:
        return {}
    try:
        tracker = tracker_for(manifest)
        assert tracker is not None
        mlflow_run_id = tracker.start_run(store, run_id, manifest)
    except Exception as exc:
        store.append_event(
            run_id,
            RunEvent(
                level="warning",
                message="mlflow tracking unavailable",
                details={"error": str(exc)},
            ),
        )
        return {}
    store.update_status(
        run_id,
        details={
            "mlflow_run_id": mlflow_run_id,
            "mlflow_tracking_uri": tracker.tracking_uri,
        },
    )
    env = {
        "MLFLOW_RUN_ID": mlflow_run_id,
        "MLFLOW_TRACKING_URI": str(tracker.tracking_uri),
        **tracker.extra_env,
    }
    return env


#: How many ticks a failing sync is retried before the daemon gives up on it.
#: A transient MLflow outage recovers inside this; a misconfiguration does not,
#: and repeating its error every tick forever would only bury the next problem.
MAX_SYNC_ATTEMPTS = 3


class TrackingSyncError(RuntimeError):
    """A mirroring pass ran and failed. Not "nothing to mirror" (#26)."""


def finalize_tracking(store: FilesystemRunStore, status: RunStatus) -> bool:
    """Daemon hook: mirror + close the MLflow run once, at terminal state.

    Returns True when a sync happened, False when there was nothing to do, and
    raises :class:`TrackingSyncError` when a sync was attempted and failed --
    the daemon reports that as a tick error. Returning False for a failure made
    a dead mirror look like a clean tick, so metrics and artifacts silently
    never arrived (#26).
    """
    mlflow_run_id = status.details.get("mlflow_run_id")
    if (
        not mlflow_run_id
        or status.details.get("mlflow_synced")
        or status.details.get("mlflow_sync_failed")
        or status.status not in _TERMINAL_MLFLOW_STATUS
    ):
        return False
    manifest = store.read_manifest(status.run_id)
    if manifest is None:
        return False
    try:
        tracker = tracker_for(manifest)
        if tracker is None:
            return False
        tracker.finalize_run(store, status.run_id, str(mlflow_run_id))
    except Exception as exc:
        attempts = int(status.details.get("mlflow_sync_attempts", 0)) + 1
        give_up = attempts >= MAX_SYNC_ATTEMPTS
        store.update_status(
            status.run_id,
            details={
                "mlflow_sync_attempts": attempts,
                "mlflow_sync_error": str(exc),
                **({"mlflow_sync_failed": True} if give_up else {}),
            },
        )
        store.append_event(
            status.run_id,
            RunEvent(
                level="error" if give_up else "warning",
                message=(
                    "mlflow finalize failed; giving up"
                    if give_up
                    else "mlflow finalize failed"
                ),
                details={"error": str(exc), "attempt": attempts},
            ),
        )
        raise TrackingSyncError(
            f"mlflow finalize failed (attempt {attempts}/{MAX_SYNC_ATTEMPTS}): {exc}"
        ) from exc
    store.update_status(status.run_id, details={"mlflow_synced": True})
    store.append_event(
        status.run_id,
        RunEvent(
            message="mlflow run synced",
            details={"mlflow_run_id": mlflow_run_id},
        ),
    )
    return True
