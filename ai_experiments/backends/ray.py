from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

from ai_experiments.backends.base import ExperimentBackend
from ai_experiments.monitoring.ray_rules import classify_ray_condition
from ai_experiments.monitoring.rules import diagnose_run
from ai_experiments.schemas import (
    DiagnosisReport,
    ExperimentManifest,
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

        try:
            client = self._client()
        except RuntimeError as exc:
            status = RunStatus(
                run_id=run_id,
                backend="ray",
                status="failed",
                status_uri=str(status_path),
                run_dir=str(run_dir),
                error="Ray is not installed. Install ai-experiments[ray].",
                completed_at=utc_now(),
            )
            self.store.write_status(status)
            raise RuntimeError(status.error) from exc

        runtime_env = {
            "working_dir": str(Path(manifest.workload.working_dir).resolve()),
            "env_vars": manifest.workload.env,
        }
        external_id = client.submit_job(entrypoint=entrypoint, runtime_env=runtime_env)
        handle = RunHandle(
            run_id=run_id,
            backend="ray",
            status="submitted",
            status_uri=str(status_path),
            run_dir=str(run_dir),
            external_id=external_id,
            dashboard_url=self.address,
        )
        self.store.write_handle(handle)
        self.store.update_status(
            run_id,
            details={
                "stuck_after_minutes": manifest.monitoring.stuck_after_minutes,
                "timeout_seconds": manifest.monitoring.timeout_seconds,
                "experiment": manifest.experiment,
            },
        )
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
            details = self._ray_details(client, status.external_id, ray_status)
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
        self, client: Any, external_id: str, ray_status: Any
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

        details["ray_condition"] = classify_ray_condition(details)
        return details

    def logs(self, run_id: str, tail: int = 200) -> list[RunEvent]:
        return self.store.read_events(run_id, tail=tail)

    def cancel(self, run_id: str) -> None:
        status = self.store.read_status(run_id)
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
