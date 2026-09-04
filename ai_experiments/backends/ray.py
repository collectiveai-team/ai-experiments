from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

from ai_experiments.backends.base import ExperimentBackend
from ai_experiments.monitoring.ray_rules import classify_ray_condition
from ai_experiments.monitoring.rules import diagnose_run
from ai_experiments.report import parse_metric_line
from ai_experiments.schemas import (
    ACTIVE_RUN_STATES,
    DiagnosisReport,
    ExperimentManifest,
    MetricPoint,
    RunEvent,
    RunHandle,
    RunStatus,
    utc_now,
)
from ai_experiments.store import FilesystemRunStore


DEFAULT_RAY_ADDRESS = "http://127.0.0.1:8265"


def resolve_ray_address(address: str | None = None) -> str:
    if address is not None:
        stripped = address.strip()
        if not stripped:
            raise ValueError("Ray address must not be empty")
        return stripped
    env_address = os.environ.get("RAY_ADDRESS")
    if env_address and env_address.strip():
        return env_address.strip()
    return DEFAULT_RAY_ADDRESS


class RayBackend(ExperimentBackend):
    """Detached Ray Jobs backend.

    This adapter intentionally uses the Ray Jobs API when available so submit
    returns a run handle instead of waiting on Ray object refs.
    """

    def __init__(
        self,
        store: FilesystemRunStore | None = None,
        address: str | None = None,
        client_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self.store = store or FilesystemRunStore()
        self.address = resolve_ray_address(address)
        self._client_factory = client_factory

    def _client(self) -> Any:
        if self._client_factory is not None:
            return self._client_factory(self.address)
        try:
            from ray.job_submission import JobSubmissionClient
        except ImportError as exc:  # pragma: no cover - depends on optional ray extra
            raise RuntimeError(
                "Ray is not installed. Install ai-experiments[ray]."
            ) from exc
        return JobSubmissionClient(self.address)

    def submit(self, manifest: ExperimentManifest) -> RunHandle:
        run_id, run_dir = self.store.create_run(manifest)
        status_path = self.store.status_path(run_id)
        entrypoint = " ".join(
            [manifest.workload.entrypoint, *manifest.workload.args]
        ).strip()

        # Establish the real status *first*. `begin_tracking` below records the
        # MLflow linkage via update_status, and the Ray job id is only known
        # after submit_job -- so the status file has to exist before either.
        # Writing it last (as this once did) discarded the linkage, leaving
        # runs unmirrored and stuck RUNNING in MLflow forever.
        handle = RunHandle(
            run_id=run_id,
            backend="ray",
            status="submitted",
            status_uri=str(status_path),
            run_dir=str(run_dir),
            dashboard_url=self.address,
        )
        self.store.write_handle(handle)
        self.store.update_status(
            run_id,
            details={
                "stuck_after_minutes": manifest.monitoring.stuck_after_minutes,
                "timeout_seconds": manifest.monitoring.timeout_seconds,
                "experiment": manifest.experiment,
                "ray_address": self.address,
            },
        )

        try:
            client = self._client()
        except RuntimeError as exc:
            error = "Ray is not installed. Install ai-experiments[ray]."
            self.store.update_status(
                run_id, status="failed", error=error, completed_at=utc_now()
            )
            raise RuntimeError(error) from exc

        from ai_experiments.tracking import begin_tracking

        tracking_env = begin_tracking(self.store, run_id, manifest)
        runtime_env = {
            "working_dir": str(Path(manifest.workload.working_dir).resolve()),
            "env_vars": {**manifest.workload.env, **tracking_env},
        }
        external_id = client.submit_job(entrypoint=entrypoint, runtime_env=runtime_env)
        handle.external_id = external_id
        self.store.update_status(run_id, external_id=external_id)
        self.store.append_event(
            run_id,
            RunEvent(message="ray job submitted", details={"ray_job_id": external_id}),
        )
        return handle

    def inspect(self, run_id: str) -> RunStatus:
        status = self.store.read_status(run_id)
        if not status.external_id:
            return status
        try:
            client = self._client()
            ray_status = client.get_job_status(status.external_id)
            mapped = _map_ray_status(ray_status)
            details = self._ray_details(run_id, client, status.external_id, ray_status)
            if (
                mapped in {"completed", "failed", "cancelled"}
                and status.completed_at is None
            ):
                status = self.store.update_status(
                    run_id,
                    status=mapped,
                    completed_at=utc_now(),
                    details=details,
                )
            else:
                status = self.store.update_status(
                    run_id, status=mapped, details=details
                )
            if mapped == "failed" and not status.error:
                message = details.get("ray_message") or details.get("ray_error_type")
                status = self.store.update_status(
                    run_id, error=str(message or "Ray job failed")
                )
            return status
        except Exception as exc:  # pragma: no cover - depends on live Ray cluster
            return self.store.update_status(run_id, error=str(exc))

    def _ray_details(
        self, run_id: str, client: Any, external_id: str, ray_status: Any
    ) -> dict[str, Any]:
        details: dict[str, Any] = {
            "ray_status": _ray_status_text(ray_status),
            "ray_address": self.address,
        }
        job_info = _job_info_dict(_safe_call(client, "get_job_info", external_id))
        if job_info:
            details["ray_job_info"] = job_info
            message = job_info.get("message") or job_info.get("status_message")
            if message:
                details["ray_message"] = str(message)
            error_type = job_info.get("error_type")
            if error_type:
                details["ray_error_type"] = str(error_type)

        logs = _safe_call(client, "get_job_logs", external_id)
        if isinstance(logs, str) and logs:
            lines = logs.splitlines()
            details["ray_log_tail"] = "\n".join(lines[-50:])
            details["ray_log_line_count"] = len(lines)
            last_point = self._sync_metrics_from_logs(run_id, lines)
            if last_point is not None:
                details["last_metric_at"] = last_point.timestamp.isoformat()
                details["last_step"] = last_point.step
                details["last_metrics"] = last_point.values

        details["ray_condition"] = classify_ray_condition(details)
        return details

    def _sync_metrics_from_logs(
        self, run_id: str, lines: list[str]
    ) -> MetricPoint | None:
        """Append metric points newly seen in the job logs to the run store.

        Ray job logs carry no timestamps, so only points beyond the count
        already persisted are appended (stamped at observation time). This
        keeps metric-staleness checks honest across repeated inspects.
        """
        parsed = [m for m in (parse_metric_line(line) for line in lines) if m]
        existing = self.store.read_metrics(run_id)
        last: MetricPoint | None = existing[-1] if existing else None
        for metric in parsed[len(existing) :]:
            last = MetricPoint(step=metric["step"], values=metric["values"])
            self.store.append_metric(run_id, last)
        return last

    def logs(self, run_id: str, tail: int = 200) -> list[RunEvent]:
        return self.store.read_events(run_id, tail=tail)

    def cancel(self, run_id: str) -> None:
        # Refresh from the cluster first. A Ray run has no local supervisor
        # keeping its record current -- the stored status is only as fresh as
        # the last inspect -- so deciding from the store alone would happily
        # stamp "cancelled" onto a job that failed on its own hours ago, which
        # is the one thing a cancel must never do.
        status = self.inspect(run_id)
        if status.status not in ACTIVE_RUN_STATES:
            return
        if status.external_id:
            self._client().stop_job(status.external_id)
        self.store.update_status(run_id, status="cancelled", completed_at=utc_now())
        self.store.append_event(run_id, RunEvent(message="ray job cancelled"))

    def diagnose(self, run_id: str) -> DiagnosisReport:
        self.inspect(run_id)
        return diagnose_run(self.store, run_id)


def _ray_status_text(ray_status: Any) -> str:
    value = getattr(ray_status, "value", ray_status)
    name = getattr(value, "name", value)
    return str(name).split(".")[-1].lower()


def _map_ray_status(ray_status: Any) -> str:
    text = _ray_status_text(ray_status)
    return {
        "pending": "submitted",
        "running": "running",
        "succeeded": "completed",
        "failed": "failed",
        "stopped": "cancelled",
    }.get(text, "unknown")


def _safe_call(client: Any, method_name: str, *args: Any) -> Any:
    method = getattr(client, method_name, None)
    if method is None:
        return None
    try:
        return method(*args)
    except Exception:
        return None


def _job_info_dict(job_info: Any) -> dict[str, Any]:
    if job_info is None:
        return {}
    if isinstance(job_info, dict):
        return {str(k): _jsonable(v) for k, v in job_info.items()}
    if hasattr(job_info, "model_dump"):
        return {str(k): _jsonable(v) for k, v in job_info.model_dump().items()}
    if hasattr(job_info, "dict"):
        return {str(k): _jsonable(v) for k, v in job_info.dict().items()}

    result: dict[str, Any] = {}
    for attr in (
        "status",
        "entrypoint",
        "message",
        "status_message",
        "error_type",
        "start_time",
        "end_time",
        "metadata",
        "runtime_env",
        "driver_node_id",
    ):
        if hasattr(job_info, attr):
            result[attr] = _jsonable(getattr(job_info, attr))
    return result


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)
