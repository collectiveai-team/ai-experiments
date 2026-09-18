from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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


class EscalationPolicy(BaseModel):
    """Controls when a suspicious run is handed to an agent for diagnosis.

    Programmatic checks are free; agent checks cost tokens. The ladder only
    escalates after `after_suspicious_ticks` consecutive suspicious monitor
    ticks, then enforces a cooldown and a per-run budget of agent calls.
    """

    after_suspicious_ticks: int = 3
    cooldown_minutes: int = 30
    max_agent_calls: int = 5
    agent_command: str | None = None
    agent_timeout_seconds: int = 600


class MonitorPolicy(BaseModel):
    interval_seconds: int = 300
    stuck_after_minutes: int = 30
    no_event_after_minutes: int | None = None
    timeout_seconds: int | None = None
    auto_kill: bool = False
    fatal_on_nan: bool = True
    objective_metric: str | None = None
    plateau_patience_points: int | None = None
    escalation: EscalationPolicy = Field(default_factory=EscalationPolicy)
    checks: list[str] = Field(
        default_factory=lambda: [
            "no_status_update",
            "no_log_progress",
            "process_exit",
        ]
    )


class TrackingSpec(BaseModel):
    """Optional MLflow experiment tracking.

    When enabled, the harness creates the MLflow run at submit time, injects
    ``MLFLOW_RUN_ID``/``MLFLOW_TRACKING_URI`` into the workload env (so
    workloads on remote Ray nodes can log artifacts straight to the tracking
    server), and mirrors params, IAX_METRIC points, tags, terminal status,
    and local artifacts — so runs appear in MLflow even when the workload
    never imports mlflow.
    """

    mlflow: bool = False
    tracking_uri: str | None = None  # falls back to MLFLOW_TRACKING_URI env
    experiment: str | None = None  # defaults to the iax experiment name


class ExperimentManifest(BaseModel):
    """Generic detached training experiment manifest."""

    experiment: str
    backend: BackendName = "local"
    backend_address: str | None = None
    workload: WorkloadSpec
    resources: ResourceSpec = Field(default_factory=ResourceSpec)
    artifacts: ArtifactSpec = Field(default_factory=ArtifactSpec)
    monitoring: MonitorPolicy = Field(default_factory=MonitorPolicy)
    tracking: TrackingSpec = Field(default_factory=TrackingSpec)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("experiment")
    @classmethod
    def experiment_not_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("experiment must not be empty")
        return value

    @field_validator("backend_address")
    @classmethod
    def backend_address_urlish(cls, value: str | None) -> str | None:
        if value is None:
            return value
        stripped = value.strip()
        if not stripped:
            raise ValueError("backend_address must not be empty")
        if not stripped.startswith(("http://", "https://")):
            raise ValueError("backend_address must start with http:// or https://")
        return stripped

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


class MetricPoint(BaseModel):
    """One reported metrics observation from a workload."""

    timestamp: datetime = Field(default_factory=utc_now)
    step: int | None = None
    values: dict[str, float] = Field(default_factory=dict)


class MetricLine(BaseModel):
    """One parsed ``IAX_METRIC`` stdout line, before it is stamped into a MetricPoint.

    `step`/`values` are the fixed schema; `values` itself stays a raw
    ``{name: float}`` map because the workload's own metric names are not
    fixed. No timestamp here -- callers stamp that at observation time.
    """

    step: int | None = None
    values: dict[str, float] = Field(default_factory=dict)


class MonitorDecision(BaseModel):
    run_id: str
    decision: Literal[
        "continue_waiting",
        "training_complete",
        "training_failed",
        "delegate_diagnosis",
        "kill",
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


# --- Goal / campaign layer -------------------------------------------------


class ChoiceParam(BaseModel):
    type: Literal["choice"]
    values: list[Any]

    @field_validator("values")
    @classmethod
    def values_not_empty(cls, value: list[Any]) -> list[Any]:
        if not value:
            raise ValueError("choice param needs at least one value")
        return value


class UniformParam(BaseModel):
    type: Literal["uniform"]
    low: float
    high: float

    @model_validator(mode="after")
    def low_below_high(self) -> UniformParam:
        if self.low >= self.high:
            raise ValueError("uniform param requires low < high")
        return self


class LogUniformParam(BaseModel):
    type: Literal["loguniform"]
    low: float
    high: float

    @model_validator(mode="after")
    def positive_range(self) -> LogUniformParam:
        if self.low <= 0:
            raise ValueError("loguniform param requires low > 0")
        if self.low >= self.high:
            raise ValueError("loguniform param requires low < high")
        return self


class IntParam(BaseModel):
    type: Literal["int"]
    low: int
    high: int

    @model_validator(mode="after")
    def low_below_high(self) -> IntParam:
        if self.low > self.high:
            raise ValueError("int param requires low <= high")
        return self


ParamSpec = Annotated[
    ChoiceParam | UniformParam | LogUniformParam | IntParam,
    Field(discriminator="type"),
]


class ObjectiveSpec(BaseModel):
    metric: str
    mode: Literal["min", "max"] = "min"
    target: float | None = None


class BudgetSpec(BaseModel):
    max_trials: int = 10
    max_parallel: int = 1
    max_hours: float | None = None
    max_gpu_hours: float | None = None
    gpu_hour_rate: float | None = None  # currency per GPU-hour, for cost display

    @model_validator(mode="after")
    def positive_budget(self) -> BudgetSpec:
        if self.max_trials < 1:
            raise ValueError("max_trials must be >= 1")
        if self.max_parallel < 1:
            raise ValueError("max_parallel must be >= 1")
        return self


class StrategySpec(BaseModel):
    name: Literal["grid", "random", "adaptive"] = "adaptive"
    seed: int = 0
    batch_size: int | None = None
    grid_resolution: int = 4
    exploration: float = 0.3
    top_k: int = 3


class AnalysisSpec(BaseModel):
    agent_review: bool = False


class GoalSpec(BaseModel):
    """A research goal the harness pursues autonomously.

    The planner turns this into a campaign: batches of experiment manifests,
    submitted, monitored, analyzed, and iterated until the target is reached
    or the budget is exhausted.
    """

    goal: str
    name: str
    objective: ObjectiveSpec
    search_space: dict[str, ParamSpec]
    workload: WorkloadSpec
    budget: BudgetSpec = Field(default_factory=BudgetSpec)
    strategy: StrategySpec = Field(default_factory=StrategySpec)
    analysis: AnalysisSpec = Field(default_factory=AnalysisSpec)
    backend: BackendName = "local"
    backend_address: str | None = None
    cluster: str | None = None
    resources: ResourceSpec = Field(default_factory=ResourceSpec)
    monitoring: MonitorPolicy = Field(default_factory=MonitorPolicy)
    tracking: TrackingSpec = Field(default_factory=TrackingSpec)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("goal", "name")
    @classmethod
    def not_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be empty")
        return value

    @field_validator("search_space")
    @classmethod
    def search_space_not_empty(  # ast-grep-ignore: no-dict-return-annotation
        cls, value: dict[str, ParamSpec]
    ) -> dict[str, ParamSpec]:
        # A pydantic field validator's signature must return exactly the type
        # it validates -- search_space is itself a name -> ParamSpec mapping,
        # not a fixed schema.
        if not value:
            raise ValueError("search_space needs at least one parameter")
        return value

    @model_validator(mode="after")
    def objective_metric_monitored(self) -> GoalSpec:
        if self.monitoring.objective_metric is None:
            self.monitoring.objective_metric = self.objective.metric
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> GoalSpec:
        with Path(path).open() as fh:
            return cls(**(yaml.safe_load(fh) or {}))

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False)


TrialState = Literal[
    "planned",
    "submitted",
    "running",
    "completed",
    "failed",
    "cancelled",
]

CampaignStatus = Literal["running", "paused", "stopping", "completed", "stopped", "failed"]


class TrialRecord(BaseModel):
    trial_id: str
    params: dict[str, Any]
    source: Literal["strategy", "agent"] = "strategy"
    run_id: str | None = None
    status: TrialState = "planned"
    objective_value: float | None = None
    final_metrics: dict[str, float] = Field(default_factory=dict)
    gpu_hours: float | None = None
    created_at: datetime = Field(default_factory=utc_now)
    completed_at: datetime | None = None
    error: str | None = None


class CampaignState(BaseModel):
    campaign_id: str
    name: str
    goal: str
    status: CampaignStatus = "running"
    stop_reason: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    trials: list[TrialRecord] = Field(default_factory=list)
    best_trial_id: str | None = None
    rounds: int = 0


# --- Server / CLI response models ------------------------------------------


class HealthStatus(BaseModel):
    """The `/api/health` body."""

    model_config = ConfigDict(extra="forbid")

    status: str
    runs_root: str


class CancelAck(BaseModel):
    """The `/api/runs/{run_id}/cancel` body."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    cancelled: bool


class ReproContext(BaseModel):
    """Reproducibility bundle captured at submit time (`repro/context.json`), and nothing else.

    This is exactly what `capture_repro` writes and `read_repro` reads back — no presentation-only
    fields. `extra="ignore"` (not `"forbid"`) is deliberate: this model validates a file written by
    whatever version of `capture_repro` ran at submit time, which may be older or newer than the
    version of this code reading it back. Forbidding extras would turn a future field addition into
    a crash for every older reader that opens an existing run directory; ignoring extras keeps that
    read forward-compatible. All fields are optional, tolerating a hand-authored or partial bundle
    that predates a field being added.
    """

    model_config = ConfigDict(extra="ignore")

    captured_at: str | None = None
    git_sha: str | None = None
    git_branch: str | None = None
    git_dirty: bool | None = None
    python: str | None = None
    platform: str | None = None
    working_dir: str | None = None


class RunReproDetail(ReproContext):
    """The `/api/runs/{run_id}/repro` body: the persisted context plus whether a diff was captured.

    `has_diff` is presentation-only — `server/app.py` derives it from whether `diff.patch` exists
    on disk, it is never part of the persisted bundle — so it lives on this composed model, not on
    `ReproContext` itself. `extra="forbid"` is fine here: unlike `ReproContext`, this model is only
    ever constructed in code, never parsed back off disk.
    """

    model_config = ConfigDict(extra="forbid")

    has_diff: bool


class ReproBundleInfo(ReproContext):
    """`iax repro`'s output: the persisted context plus where the bundle lives on disk.

    `bundle_dir` is presentation-only — the CLI fills it in after `read_repro` loads the bundle
    back — so it lives on this composed model, not on `ReproContext` itself. `extra="forbid"` is
    fine here for the same reason as `RunReproDetail`: only ever constructed in code.
    """

    model_config = ConfigDict(extra="forbid")

    bundle_dir: str


class CampaignHistoryEntry(BaseModel):
    """One row of `CampaignSummary.history`: a completed trial's outcome."""

    model_config = ConfigDict(extra="forbid")

    trial_id: str
    objective_value: float | None
    params: dict[str, Any]


class BestTrialSummary(BaseModel):
    """`CampaignSummary.best`: the best-scoring trial so far, or absent if none has scored."""

    model_config = ConfigDict(extra="forbid")

    trial_id: str
    run_id: str | None
    objective_value: float | None
    params: dict[str, Any]


class BudgetSummary(BaseModel):
    """`CampaignSummary.budget`: the budget fields relevant to progress display."""

    model_config = ConfigDict(extra="forbid")

    max_trials: int
    max_gpu_hours: float | None
    gpu_hour_rate: float | None


class CampaignSummary(BaseModel):
    """`summarize_campaign`'s return: budget/objective snapshot, trial history, best trial."""

    model_config = ConfigDict(extra="forbid")

    campaign_id: str
    name: str
    goal: str
    status: CampaignStatus
    stop_reason: str | None
    gpu_hours: float
    estimated_cost: float | None
    budget: BudgetSummary
    objective: ObjectiveSpec
    rounds: int
    trials_by_status: dict[str, int]
    trials_total: int
    best: BestTrialSummary | None
    history: list[CampaignHistoryEntry]


class CampaignDetail(BaseModel):
    """The `/api/campaigns/{campaign_id}` body."""

    model_config = ConfigDict(extra="forbid")

    state: CampaignState
    summary: CampaignSummary


class ArtifactEntry(BaseModel):
    """One row of the `/api/runs/{run_id}/artifacts` listing.

    Mirrors `FilesystemRunStore.list_artifacts`'s fixed 3-key shape (`path`, `size_bytes`,
    `modified_at`) at the server boundary; the store method itself still returns
    `list[dict[str, object]]`, which is out of scope for this task.
    """

    model_config = ConfigDict(extra="forbid")

    path: str
    size_bytes: int
    modified_at: str


class LeaderboardRow(BaseModel):
    """One row of the `/api/leaderboard` body."""

    model_config = ConfigDict(extra="forbid")

    campaign_id: str
    name: str
    status: CampaignStatus
    metric: str
    mode: Literal["min", "max"]
    best_value: float
    best_params: dict[str, Any]
    best_run_id: str | None
    trials: int
    gpu_hours: float
    estimated_cost: float | None
    updated_at: str
