from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal

import yaml
from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
)

from ai_experiments.config_loading import (
    REMOVED_MONITOR_KEYS,
    ConfigModel,
    load_config,
)

if TYPE_CHECKING:
    from pathlib import Path


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


class DataSpec(ConfigModel):
    """Where the workload's data lives, by reference.

    Guarantee: the train phase's environment never carries ``IAX_DATA_TEST``.
    ``env_for`` only ever adds keys, so ``worker.py`` scrubs ``ENV_KEYS`` from
    the *inherited* environment first -- otherwise a value already set on the
    parent process (``iax daemon`` runs with whatever environment the
    operator started it in) would pass through untouched. Not yet guaranteed:
    ``IAX_RUN_DIR`` is exported to every phase, and its ``manifest.yaml``
    names ``data.test`` literally, so a trainer that goes looking through its
    own run directory still reaches it (closed by the la-tesis plan, Task 4,
    not here).
    """

    #: Keys `env_for` may set. The scrub in `worker.py` reads this list
    #: rather than naming the three keys again, so a new data reference
    #: cannot drift out of sync with what gets scrubbed.
    ENV_KEYS: ClassVar[tuple[str, ...]] = (
        "IAX_DATA_TRAIN",
        "IAX_DATA_VAL",
        "IAX_DATA_TEST",
    )

    train: str | None = None
    val: str | None = None
    test: str | None = None

    def env_for(self, phase: str) -> dict[str, str]:  # ast-grep-ignore: no-dict-return-annotation
        """Environment for one phase: the test split is exposed only to `evaluate`."""
        env: dict[str, str] = {}
        if self.train:
            env["IAX_DATA_TRAIN"] = self.train
        if self.val:
            env["IAX_DATA_VAL"] = self.val
        if phase == "evaluate" and self.test:
            env["IAX_DATA_TEST"] = self.test
        return env


#: How a search space key becomes a command-line flag. Keys are Python
#: identifiers, so a two-word parameter is ``label_source``; the CLI
#: convention every argument parser follows spells it ``--label-source``.
FlagStyle = Literal["hyphen", "underscore"]

#: How a run's observations collapse into one objective value.
Aggregate = Literal["best", "mean", "bootstrap"]


class WorkloadSpec(ConfigModel):
    """Executable workload for a training experiment.

    A workload may declare one command (``entrypoint``) or two (``train`` and
    ``evaluate``). Two is what lets the harness protect the evaluator: only
    the evaluate phase may declare a result, and only it sees the test data.
    """

    entrypoint: str
    args: list[str] = Field(default_factory=list)
    working_dir: str = "."
    env: dict[str, str] = Field(default_factory=dict)
    train: str | None = None
    evaluate: str | None = None
    data: DataSpec = Field(default_factory=DataSpec)

    @model_validator(mode="after")
    def phases_are_coherent(self) -> WorkloadSpec:
        if self.train and not self.evaluate:
            raise ValueError(
                "declaring 'train' requires 'evaluate': a train phase alone "
                "would be ignored, and the entrypoint scored in its place"
            )
        if self.evaluate and not self.train:
            raise ValueError(
                "declaring 'evaluate' requires 'train': with only an evaluate "
                "phase there is nothing for it to score"
            )
        if self.data.test and not self.evaluate:
            raise ValueError(
                "data.test requires an 'evaluate' phase: without one there is "
                "no protected phase to receive the test reference"
            )
        return self

    def phases(self) -> list[tuple[str, str]]:
        # Re-checked here, not only in the validator: `model_copy(update=...)`
        # and attribute assignment skip "after" validators, and the planner
        # builds every trial manifest with model_copy. A half-declared
        # workload must fail loudly here rather than quietly run one phase.
        if self.train and self.evaluate:
            return [("train", self.train), ("evaluate", self.evaluate)]
        if self.train or self.evaluate:
            raise ValueError(
                "workload declares only one of 'train'/'evaluate'; both are "
                "required to run as two phases"
            )
        return [("evaluate", self.entrypoint)]

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


class ResultRecord(BaseModel):
    """The evaluation result of a run: the number that may be scored.

    Separate from :class:`MetricPoint` on purpose. A metric is a point on a
    curve and picking its best value is a biased estimator; a result is what
    the protected evaluator declared, and there is at most one that counts.
    """

    timestamp: datetime = Field(default_factory=utc_now)
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
    ChoiceParam | UniformParam | LogUniformParam | IntParam,
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
    #: campaign can tell a real lead from a lucky fold. ``bootstrap`` is for
    #: observations that are resamples of *one* evaluation: their spread is
    #: already the standard error of the statistic, so dividing it by the
    #: square root of their count would shrink the interval by exactly the
    #: factor the resampling exists to expose.
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
    #: Let an accepted review widen the search space on its own. Off by
    #: default: a loop that rewrites its own goal unasked is a surprise. The
    #: search space is the only section a review can change -- see
    #: ``loop.APPLICABLE_KEYS``. A review may redistribute effort within the
    #: budget, but raising the budget is the user's call: a ceiling the
    #: optimizer can move is not a ceiling.
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

    @model_validator(mode="after")
    def conditions_resolve(self) -> GoalSpec:
        """Require every ``when`` to name a key that exists and is drawn first.

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

CampaignStatus = Literal["running", "paused", "stopping", "completed", "stopped", "failed"]


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
    #: How long the trial actually occupied the machine. Recorded whether
    #: or not there are GPUs, because a CPU campaign that reports only
    #: `gpu_hours` reports that it cost nothing.
    wall_hours: float | None = None
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
