from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


RunState = Literal[
    "submitted",
    "running",
    "completed",
    "failed",
    "cancelled",
    "unknown",
]

BackendName = Literal["local", "ray"]


class WorkloadSpec(BaseModel):
    """Executable workload for a training experiment."""

    entrypoint: str
    args: list[str] = Field(default_factory=list)
    working_dir: str = "."
    env: dict[str, str] = Field(default_factory=dict)


class ResourceSpec(BaseModel):
    cpus: float = 1
    gpus: float = 0
    memory_gb: float | None = None


class ArtifactSpec(BaseModel):
    output_dir: str = "outputs/training"
    status_path: str | None = None


class MonitorPolicy(BaseModel):
    interval_seconds: int = 300
    stuck_after_minutes: int = 30
    no_event_after_minutes: int | None = None
    timeout_seconds: int | None = None
    checks: list[str] = Field(
        default_factory=lambda: [
            "no_status_update",
            "no_log_progress",
            "process_exit",
        ]
    )


class ExperimentManifest(BaseModel):
    """Generic detached training experiment manifest."""

    experiment: str
    backend: BackendName = "local"
    workload: WorkloadSpec
    resources: ResourceSpec = Field(default_factory=ResourceSpec)
    artifacts: ArtifactSpec = Field(default_factory=ArtifactSpec)
    monitoring: MonitorPolicy = Field(default_factory=MonitorPolicy)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("experiment")
    @classmethod
    def experiment_not_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("experiment must not be empty")
        return value

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentManifest:
        with Path(path).open() as fh:
            return cls(**(yaml.safe_load(fh) or {}))

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False)


class RunHandle(BaseModel):
    run_id: str
    backend: BackendName
    status: RunState
    status_uri: str
    run_dir: str
    external_id: str | None = None
    dashboard_url: str | None = None
    submitted_at: datetime = Field(default_factory=utc_now)


class RunStatus(BaseModel):
    run_id: str
    backend: BackendName
    status: RunState = "unknown"
    status_uri: str
    run_dir: str
    external_id: str | None = None
    pid: int | None = None
    submitted_at: datetime | None = None
    started_at: datetime | None = None
    updated_at: datetime = Field(default_factory=utc_now)
    completed_at: datetime | None = None
    exit_code: int | None = None
    error: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class RunEvent(BaseModel):
    timestamp: datetime = Field(default_factory=utc_now)
    level: Literal["info", "warning", "error"] = "info"
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class MonitorDecision(BaseModel):
    run_id: str
    decision: Literal[
        "continue_waiting",
        "training_complete",
        "training_failed",
        "delegate_diagnosis",
        "unknown",
    ]
    severity: Literal["info", "warning", "error"] = "info"
    reasons: list[str] = Field(default_factory=list)


class DiagnosisReport(BaseModel):
    run_id: str
    status: RunStatus
    decision: MonitorDecision
    events: list[RunEvent] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
