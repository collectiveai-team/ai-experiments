from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Literal, TypeVar, Union

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from ai_experiments.schema_errors import describe

#: Keys older versions of iax defined, defaulted, documented -- and never read.
#: They are rejected in a hand-written file and dropped from a stored one (#14).
REMOVED_MONITOR_KEYS = ("checks", "no_event_after_minutes")


class ConfigModel(BaseModel):
    """Base for every model a human or an agent writes by hand.

    An unknown key is an error, not a comment: silently dropping ``monitor:``
    for ``monitoring:`` leaves the author sure they configured something they
    did not. Models that only describe *stored* state stay permissive, so an
    older file keeps loading after a field is added.
    """

    model_config = ConfigDict(extra="forbid")


ConfigT = TypeVar("ConfigT", bound=ConfigModel)


def load_config(model: type[ConfigT], path: str | Path) -> ConfigT:
    """Load a hand-written YAML config, reporting a bad key by name."""
    with Path(path).open() as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a YAML mapping of fields")
    try:
        return model(**data)
    except ValidationError as exc:
        raise ValueError(describe(model, exc)) from exc


def load_stored(model: type[ConfigT], path: str | Path) -> ConfigT:
    """Load a config iax itself wrote, tolerating keys older versions emitted.

    A run directory written before a field was removed still has to replay,
    so the stored path drops those keys instead of refusing the file.
    """
    with Path(path).open() as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a YAML mapping of fields")
    monitoring = data.get("monitoring")
    if isinstance(monitoring, dict):
        for key in REMOVED_MONITOR_KEYS:
            monitoring.pop(key, None)
    try:
        return model(**data)
    except ValidationError as exc:
        raise ValueError(describe(model, exc)) from exc


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

#: States a run can still be acted on in. Cancelling anything else would
#: rewrite how the run actually ended, so both backends and the daemon gate on
#: this same set rather than on their own copies of it.
ACTIVE_RUN_STATES: frozenset[str] = frozenset({"submitted", "running"})

BackendName = Literal["local", "ray"]


#: How a search space key becomes a command-line flag. Keys are Python
#: identifiers, so a two-word parameter is ``label_source``; the CLI
#: convention every argument parser follows spells it ``--label-source``.
FlagStyle = Literal["hyphen", "underscore"]

#: How a run's observations collapse into one objective value.
Aggregate = Literal["best", "mean"]


class WorkloadSpec(ConfigModel):
    """Executable workload for a training experiment."""

    entrypoint: str
    args: list[str] = Field(default_factory=list)
    working_dir: str = "."
    env: dict[str, str] = Field(default_factory=dict)
    #: Exactly one spelling is sent. Emitting both "to be safe" is what
    #: breaks argparse, which rejects any long option it did not declare.
    flag_style: FlagStyle = "hyphen"


class ResourceSpec(ConfigModel):
    cpus: float = 1
    gpus: float = 0
    memory_gb: float | None = None


class ArtifactSpec(ConfigModel):
    output_dir: str = "outputs/training"
    status_path: str | None = None


class EscalationPolicy(ConfigModel):
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


class MonitorPolicy(ConfigModel):
    interval_seconds: int = 300
    stuck_after_minutes: int = 30
    timeout_seconds: int | None = None
    auto_kill: bool = False
    fatal_on_nan: bool = True
    objective_metric: str | None = None
    plateau_patience_points: int | None = None
    escalation: EscalationPolicy = Field(default_factory=EscalationPolicy)

    @model_validator(mode="before")
    @classmethod
    def reject_removed_keys(cls, data: Any) -> Any:
        if isinstance(data, dict):
            dead = [key for key in REMOVED_MONITOR_KEYS if key in data]
            if dead:
                raise ValueError(
                    f"{', '.join(dead)}: removed; iax never read this key. "
                    "Delete it. The monitoring rules are not configurable."
                )
        return data


class TrackingSpec(ConfigModel):
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


class ExperimentManifest(ConfigModel):
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
        return load_config(cls, path)

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


class ParamBase(ConfigModel):
    """What every search space dimension can say about itself."""

    #: This dimension changes the data or the labels a trial is evaluated
    #: on, not just how the model is fit. Trials that differ on it are
    #: scored on different problems, so their raw metrics are not
    #: comparable and the objective needs a ``baseline_metric`` to become
    #: a lift. ``window_days``, ``resample_freq`` and a choice of label
    #: source are all of this kind; a learning rate is not.
    changes_data: bool = False
    #: Draw this dimension only for trials where the named parameters take
    #: one of the listed values, as in ``{"model": ["hist_gb"]}``. A space
    #: without it hands every model every other model's knobs: those trials
    #: are duplicates the deduplicator cannot see, and their spread reads as
    #: evidence that the knobs do nothing. Conditions may only name
    #: unconditional keys — see :meth:`GoalSpec.conditions_resolve`.
    when: dict[str, list[Any]] = Field(default_factory=dict)


class ChoiceParam(ParamBase):
    type: Literal["choice"]
    values: list[Any]

    @field_validator("values")
    @classmethod
    def values_not_empty(cls, value: list[Any]) -> list[Any]:
        if not value:
            raise ValueError("choice param needs at least one value")
        return value


class UniformParam(ParamBase):
    type: Literal["uniform"]
    low: float
    high: float

    @model_validator(mode="after")
    def low_below_high(self) -> UniformParam:
        if self.low >= self.high:
            raise ValueError("uniform param requires low < high")
        return self


class LogUniformParam(ParamBase):
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


class IntParam(ParamBase):
    type: Literal["int"]
    low: int
    high: int

    @model_validator(mode="after")
    def low_below_high(self) -> IntParam:
        if self.low > self.high:
            raise ValueError("int param requires low <= high")
        return self


ParamSpec = Annotated[
    Union[ChoiceParam, UniformParam, LogUniformParam, IntParam],
    Field(discriminator="type"),
]


class ObjectiveSpec(ConfigModel):
    metric: str
    #: A metric that is a property of the trial's *data* rather than its
    #: model: a class base rate, a naive forecast, last release's number.
    #: When set, a trial scores ``metric - baseline_metric`` taken from the
    #: same observation, so trials evaluated on different slices stay
    #: comparable. Without it, a search space dimension that moves the
    #: baseline is rewarded for moving it.
    baseline_metric: str | None = None
    mode: Literal["min", "max"] = "min"
    target: float | None = None
    #: How a run's many observations become one score. Epochs are successive
    #: states of one model, so ``best`` is the answer. Folds are independent
    #: evaluations of the *same* configuration, and there ``best`` is
    #: max-of-k: biased upward by exactly the noise the folds exist to
    #: measure. ``mean`` averages them and reports the standard error, so the
    #: campaign can tell a real lead from a lucky fold.
    aggregate: Aggregate = "best"


class SuccessCriteria(ConfigModel):
    """What this campaign has to show before its result counts.

    ``objective.target`` is the value the campaign *stops* at; this is the bar
    the result has to clear to be believed, and it is written before anything
    runs so that afterwards the answer is arithmetic instead of an argument.
    A goal that declares none gets ``met: null`` — not a pass.
    """

    #: The objective value the best trial has to reach, read in the
    #: objective's own direction.
    min_objective: float | None = None
    #: How many observations that value has to be averaged over. A winner
    #: resting on one evaluation has measured the evaluation, not the model.
    min_observations: int | None = None
    #: The best trial must be distinguishable from the runner-up at 95%.
    #: Unmeasurable counts as unmet: the criterion asks for evidence.
    require_separation: bool = False
    #: The best trial's interval must clear its declared baseline.
    require_beats_baseline: bool = False

    @property
    def declared(self) -> bool:
        return (
            self.min_objective is not None
            or self.min_observations is not None
            or self.require_separation
            or self.require_beats_baseline
        )


class BudgetSpec(ConfigModel):
    max_trials: int = 10
    max_parallel: int = 1
    max_hours: float | None = None
    max_gpu_hours: float | None = None
    gpu_hour_rate: float | None = None  # currency per GPU-hour, for cost display
    #: Consecutive failed trials, with nothing ever scored, before the
    #: campaign gives up. A workload the harness cannot talk to fails
    #: identically every time, so the rest of the budget buys nothing.
    halt_after_failures: int = 8

    @model_validator(mode="after")
    def positive_budget(self) -> BudgetSpec:
        if self.max_trials < 1:
            raise ValueError("max_trials must be >= 1")
        if self.max_parallel < 1:
            raise ValueError("max_parallel must be >= 1")
        if self.halt_after_failures < 1:
            raise ValueError("halt_after_failures must be >= 1")
        return self


class StrategySpec(ConfigModel):
    name: Literal["grid", "random", "adaptive", "agent"] = "adaptive"
    seed: int = 0
    batch_size: int | None = None
    grid_resolution: int = 4
    exploration: float = 0.3
    top_k: int = 3
    #: Used when ``name == "agent"`` and the agent cannot deliver a usable
    #: round. A campaign must keep making progress without a working agent.
    fallback: Literal["grid", "random", "adaptive"] = "adaptive"


class AgentSpec(ConfigModel):
    """How the harness reaches the agent that plans and reviews rounds.

    The command is operator-supplied configuration. It receives the brief on
    stdin, never as an argument (CONVENTIONS.md §9).
    """

    command: str = "claude"
    timeout_seconds: int = 600
    #: Hard ceiling on agent invocations per campaign. An unattended loop that
    #: keeps asking is an unattended loop that keeps spending.
    max_calls: int = 20


class VariantSpec(ConfigModel):
    """Whether, and how far, the loop may change the workload's own code.

    Off by default: a loop that edits code without being asked is a surprise.
    """

    enabled: bool = False
    #: What gets copied for each variant. Defaults to ``workload.working_dir``.
    source_dir: str | None = None
    #: Glob allowlist for the files a variant may write. Empty means any path
    #: inside the copied workload.
    editable_paths: list[str] = Field(default_factory=list)
    #: Run inside the variant before any trial uses it. A non-zero exit means
    #: the variant is discarded instead of costing a whole round.
    smoke_command: list[str] = Field(default_factory=list)
    smoke_timeout_seconds: int = 120


class AnalysisSpec(ConfigModel):
    agent_review: bool = False
    #: Ask the agent for a verdict between rounds during ``iax loop``. The
    #: agent can end a campaign it judges hopeless instead of burning the
    #: whole budget on it.
    review_between_rounds: bool = False
    #: Let an accepted review widen the search space or the budget on its own.
    #: Off by default: a loop that rewrites its own goal unasked is a surprise.
    apply_agent_changes: bool = False


class GoalSpec(ConfigModel):
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
    #: The bar the result has to clear to count. Declaring none is
    #: allowed and is reported as such; it is not a pass.
    success_criteria: SuccessCriteria = Field(default_factory=SuccessCriteria)
    budget: BudgetSpec = Field(default_factory=BudgetSpec)
    strategy: StrategySpec = Field(default_factory=StrategySpec)
    agent: AgentSpec = Field(default_factory=AgentSpec)
    variants: VariantSpec = Field(default_factory=VariantSpec)
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
    def search_space_not_empty(
        cls, value: dict[str, ParamSpec]
    ) -> dict[str, ParamSpec]:
        if not value:
            raise ValueError("search_space needs at least one parameter")
        return value

    @model_validator(mode="after")
    def objective_metric_monitored(self) -> GoalSpec:
        if self.monitoring.objective_metric is None:
            self.monitoring.objective_metric = self.objective.metric
        return self

    @model_validator(mode="after")
    def conditions_resolve(self) -> GoalSpec:
        """A ``when`` has to name a key that exists and is drawn first.

        Restricting conditions to one level keeps the sampling order obvious
        — unconditional keys, then everything that depends on them — and
        costs nothing anyone has asked for. A typo'd condition would
        otherwise silently never hold, quietly deleting a dimension from the
        search.
        """
        for name, spec in self.search_space.items():
            for other in spec.when:
                if other not in self.search_space:
                    raise ValueError(
                        f"search space key '{name}' is conditional on "
                        f"'{other}', which the space does not define — typo?"
                    )
                if self.search_space[other].when:
                    raise ValueError(
                        f"search space key '{name}' is conditional on "
                        f"'{other}', which is itself conditional; conditions "
                        "may only name unconditional keys"
                    )
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> GoalSpec:
        return load_config(cls, path)

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

CampaignStatus = Literal[
    "running", "paused", "stopping", "completed", "stopped", "failed"
]


class TrialRecord(BaseModel):
    trial_id: str
    params: dict[str, Any]
    source: Literal["strategy", "agent"] = "strategy"
    #: The workload variant this trial ran against, if the loop changed code.
    variant_id: str | None = None
    run_id: str | None = None
    status: TrialState = "planned"
    objective_value: float | None = None
    #: Standard error of ``objective_value`` when the objective averages its
    #: observations, and ``None`` when nothing measured the spread. A score
    #: without it cannot be compared to another score honestly.
    objective_stderr: float | None = None
    #: How many observations the score was computed from.
    objective_observations: int = 0
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
    #: Agent invocations spent on this campaign, capped by ``GoalSpec.agent.max_calls``.
    agent_calls: int = 0
