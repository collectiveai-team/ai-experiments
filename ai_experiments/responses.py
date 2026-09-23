"""Response models: what the server and the CLI hand back.

Every boundary that used to return a raw dict returns one of these instead
(CES-79), so a client -- the dashboard, `--json`, a test -- reads a shape that
is pinned here rather than inferred from whatever the handler built.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# Field types, not just annotations: pydantic resolves them when the class is built.
from ai_experiments.schemas import (  # noqa: TC001
    CampaignState,
    CampaignStatus,
    ObjectiveSpec,
    TrialState,
)


class HealthStatus(BaseModel):
    """The `/api/health` body.

    `mutations` tells a client whether this bind will accept cancel/stop/
    pause/resume at all, so a dashboard can grey the buttons out instead of
    discovering the 403 after the operator clicks (#24).
    """

    model_config = ConfigDict(extra="forbid")

    status: str
    runs_root: str
    mutations: Literal["allowed", "read-only"]


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

    Forward-compatible here means "does not crash", not "does not lose data": an unknown key is
    dropped, so a field written by a newer `capture_repro` is invisible to an older reader and to
    `GET /api/runs/{run_id}/repro`, which serialises this model. `capture_repro` writes exactly
    the fields declared below, so no shipped bundle is affected today.
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
    """One row of `CampaignSummary.history`: a trial an agent can learn from.

    A failure teaches as much as a score, so a trial qualifies on either a
    non-None `objective_value` or a non-None `error`; `status` and `error`
    are what tell the two apart.
    """

    model_config = ConfigDict(extra="forbid")

    trial_id: str
    status: TrialState
    objective_value: float | None
    params: dict[str, Any]
    error: str | None = None


class BestTrialSummary(BaseModel):
    """`CampaignSummary.best`: the best-scoring trial so far, or absent if none has scored."""

    model_config = ConfigDict(extra="forbid")

    trial_id: str
    run_id: str | None
    objective_value: float | None
    #: Standard error of `objective_value`; `None` when the objective takes the
    #: best observation and so carries no spread.
    stderr: float | None = None
    #: How many observations the score rests on.
    n_observations: int | None = None
    #: The 95% interval around the score, or `None` when nothing measured it.
    ci95: tuple[float, float] | None = None
    params: dict[str, Any]


class CampaignVerdict(BaseModel):
    """`CampaignSummary.verdict`: whether the campaign actually found anything, decided in code.

    ``None`` on `separated` or `beats_baseline` means *not measured*, which is a
    different claim from ``False``, and every reader must keep them apart.
    """

    model_config = ConfigDict(extra="forbid")

    best_trial_id: str | None
    runner_up_trial_id: str | None
    #: The best trial's lead over the runner-up, in the objective's direction.
    margin: float | None
    #: Whether that lead clears the noise at `confidence`.
    separated: bool | None
    #: Whether the best trial's interval clears its declared baseline.
    beats_baseline: bool | None
    confidence: float = 0.95


class SuccessReport(BaseModel):
    """`CampaignSummary.success`: whether the campaign cleared the bar its goal declared.

    ``met`` is ``None`` when the goal declared no criteria, which is not a pass.
    """

    model_config = ConfigDict(extra="forbid")

    declared: bool
    met: bool | None
    unmet: list[str] = Field(default_factory=list)


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
    created_at: str
    last_advanced_at: str
    gpu_hours: float
    #: What the campaign cost on a machine with no GPUs, which is every
    #: campaign this harness has actually run.
    wall_hours: float
    estimated_cost: float | None
    budget: BudgetSummary
    objective: ObjectiveSpec
    rounds: int
    agent_calls: int
    trials_by_status: dict[str, int]
    trials_total: int
    best: BestTrialSummary | None
    verdict: CampaignVerdict
    success: SuccessReport
    history: list[CampaignHistoryEntry]


class CampaignDetail(BaseModel):
    """The `/api/campaigns/{campaign_id}` body."""

    model_config = ConfigDict(extra="forbid")

    state: CampaignState
    summary: CampaignSummary


class ArtifactEntry(BaseModel):
    """One row of the `/api/runs/{run_id}/artifacts` listing.

    Returned directly by `FilesystemRunStore.list_artifacts`; the server
    handler passes the list through rather than re-validating raw dicts.
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
