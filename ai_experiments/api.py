"""The python surface an agent talks to.

A chat agent that runs experiments should not have to know about stores,
orchestrators, campaign directories, or which module a helper lives in. It has
one goal and a handful of questions: start it, push it forward, tell me where
it stands, tell me why it did that, and — most of the time — just run it.

Every function here takes plain data and returns plain data, so a reply can be
handed straight back into a conversation. Failures raise :class:`IaxError`
with the same codes the CLI exits on, so an agent that drives `iax` and an
agent that imports this module handle failure the same way.

```python
from ai_experiments import api

report = api.run_loop(api.goal_from_yaml("goal.yaml"))
if not report.target_reached:
    print(report.stop_reason, report.best)
```
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ai_experiments.cli_support import IaxError, invalid_input, not_found
from ai_experiments.improve.rounds import RoundLog
from ai_experiments.loop import LoopReport
from ai_experiments.loop import run_loop as _run_loop
from ai_experiments.orchestrator import CampaignOrchestrator
from ai_experiments.planner.analysis import summarize_campaign
from ai_experiments.schema_errors import describe
from ai_experiments.schemas import GoalSpec
from ai_experiments.store import FilesystemRunStore
from ai_experiments.store.campaign import CampaignStore

__all__ = [
    "IaxError",
    "LoopReport",
    "advance_campaign",
    "campaign_report",
    "campaign_rounds",
    "goal_from_dict",
    "goal_from_yaml",
    "list_campaigns",
    "run_loop",
    "start_campaign",
    "suggest_trial",
]

RunsDir = str | Path | None


# -- goals ---------------------------------------------------------------------


def goal_from_dict(data: dict[str, Any]) -> GoalSpec:
    """Build a goal from a dict, reporting a bad one the way the CLI does.

    An agent composes this dict field by field from a conversation, so the
    common case is not a typo in a file but a field it invented. It needs the
    validation error, not a traceback.
    """
    try:
        return GoalSpec(**data)
    except ValidationError as exc:
        invalid_input(f"invalid goal: {describe(GoalSpec, exc)}")
    except Exception as exc:  # TypeError, ...
        invalid_input(f"invalid goal: {exc}")


def goal_from_yaml(path: str | Path) -> GoalSpec:
    try:
        return GoalSpec.from_yaml(Path(path))
    except FileNotFoundError:
        not_found("goal file", str(path))
    except Exception as exc:
        invalid_input(f"invalid goal {path}: {exc}")


# -- campaigns -----------------------------------------------------------------


def start_campaign(goal: GoalSpec, *, runs_dir: RunsDir = None) -> dict[str, Any]:
    """Create a campaign and submit its first round. Returns its report."""
    orchestrator = _orchestrator(runs_dir)
    state = orchestrator.start(goal)
    return summarize_campaign(state, goal)


def advance_campaign(campaign_id: str, *, runs_dir: RunsDir = None) -> dict[str, Any]:
    """Run exactly one loop step: collect finished trials, then plan the next.

    This is the step `iax loop` repeats. Call it directly when the agent wants
    to think between rounds instead of delegating the whole loop.
    """
    orchestrator = _orchestrator(runs_dir)
    _require(orchestrator.campaign_store, campaign_id)
    state = orchestrator.advance(campaign_id)
    goal = orchestrator.campaign_store.read_goal(campaign_id)
    return summarize_campaign(state, goal)


def campaign_report(campaign_id: str, *, runs_dir: RunsDir = None) -> dict[str, Any]:
    """Where the campaign stands, without advancing it."""
    campaigns = CampaignStore(FilesystemRunStore(runs_dir).root)
    _require(campaigns, campaign_id)
    state = campaigns.read_state(campaign_id)
    return summarize_campaign(state, campaigns.read_goal(campaign_id))


def campaign_rounds(
    campaign_id: str, limit: int | None = None, *, runs_dir: RunsDir = None
) -> list[dict[str, Any]]:
    """Why each round tried what it tried, oldest first.

    A campaign report says what happened. This says what the loop believed at
    the time, which is what an agent needs to decide whether to trust it.
    """
    campaigns = CampaignStore(FilesystemRunStore(runs_dir).root)
    _require(campaigns, campaign_id)
    log = RoundLog(campaigns.campaign_dir(campaign_id))
    return [record.model_dump(mode="json") for record in log.read(limit)]


def list_campaigns(*, runs_dir: RunsDir = None) -> list[dict[str, Any]]:
    campaigns = CampaignStore(FilesystemRunStore(runs_dir).root)
    reports = []
    for campaign_id in campaigns.list_campaigns():
        state = campaigns.read_state(campaign_id)
        reports.append(summarize_campaign(state, campaigns.read_goal(campaign_id)))
    return reports


def suggest_trial(
    campaign_id: str,
    params: dict[str, Any],
    note: str = "",
    *,
    runs_dir: RunsDir = None,
) -> dict[str, Any]:
    """Queue one trial the agent chose itself; the next advance submits it.

    A suggestion is a proposal, not an override. Params outside the search
    space and a spent trial budget are rejected here, not silently absorbed.
    """
    orchestrator = _orchestrator(runs_dir)
    _require(orchestrator.campaign_store, campaign_id)
    try:
        trial = orchestrator.suggest(campaign_id, params, note=note)
    except ValueError as exc:
        invalid_input(f"suggestion rejected: {exc}")
    return trial.model_dump(mode="json")


# -- the whole loop ------------------------------------------------------------


def run_loop(
    goal: GoalSpec,
    *,
    runs_dir: RunsDir = None,
    campaign_id: str | None = None,
    max_rounds: int | None = None,
    max_seconds: float | None = None,
    interval_seconds: float = 5.0,
) -> LoopReport:
    """Drive one goal to its answer and return the report `iax loop` prints.

    ``campaign_id`` resumes instead of starting over, so a loop that hit
    ``max_rounds`` in one chat turn continues in the next one.
    """
    store = FilesystemRunStore(runs_dir)
    if campaign_id is not None:
        _require(CampaignStore(store.root), campaign_id)
    return _run_loop(
        goal,
        store,
        campaign_id=campaign_id,
        max_rounds=max_rounds,
        max_seconds=max_seconds,
        interval_seconds=interval_seconds,
    )


# -- internals -----------------------------------------------------------------


def _orchestrator(runs_dir: RunsDir) -> CampaignOrchestrator:
    store = FilesystemRunStore(runs_dir)
    return CampaignOrchestrator(store, CampaignStore(store.root))


def _require(campaigns: CampaignStore, campaign_id: str) -> None:
    """Turn a missing campaign into the CLI's `not_found`, not an OSError."""
    try:
        campaigns.read_state(campaign_id)
    except (FileNotFoundError, NotADirectoryError):
        not_found("campaign", campaign_id, "list them with `iax campaign list`")
