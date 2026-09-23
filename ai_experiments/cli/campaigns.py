from __future__ import annotations

import json
from pathlib import Path  # noqa: TC003  # Typer resolves this annotation at runtime
from typing import TYPE_CHECKING

import typer

from ai_experiments.cli import (
    IaxCommand,
    _echo_json,
    _preflight,
    _require_campaign,
    _warn_if_no_daemon,
    campaign_app,
)
from ai_experiments.cli_support import (
    IaxError,
    invalid_input,
)
from ai_experiments.schemas import GoalSpec
from ai_experiments.store import FilesystemRunStore

if TYPE_CHECKING:
    from ai_experiments.improve.rounds import RoundRecord


def _orchestrator(runs_dir: Path | None):
    from ai_experiments.orchestrator import CampaignOrchestrator

    store = FilesystemRunStore(runs_dir)
    return CampaignOrchestrator(store)


@campaign_app.command("validate", cls=IaxCommand)
def campaign_validate(
    config: Path = typer.Argument(..., help="Path to goal YAML"),
) -> None:
    try:
        goal = GoalSpec.from_yaml(config)
    except Exception as exc:
        invalid_input(f"invalid goal {config}: {exc}")
    typer.echo(f"Goal valid: {config}")
    typer.echo(f"  Goal:      {goal.goal}")
    typer.echo(
        f"  Objective: {goal.objective.mode} {goal.objective.metric}"
        + (f" (target {goal.objective.target})" if goal.objective.target is not None else "")
    )
    typer.echo(f"  Budget:    {goal.budget.max_trials} trials, {goal.budget.max_parallel} parallel")
    typer.echo(f"  Strategy:  {goal.strategy.name}")
    typer.echo(f"  Backend:   {goal.backend}")


@campaign_app.command("start", cls=IaxCommand)
def campaign_start(
    config: Path = typer.Argument(..., help="Path to goal YAML"),
    runs_dir: Path | None = typer.Option(None, "--runs-dir"),
    output_json: bool = typer.Option(False, "--json"),
    strict: bool = typer.Option(
        False, "--strict", help="Refuse to start a workload that looks unable to run"
    ),
) -> None:
    """Create a campaign from a goal and submit the first batch of trials.

    Keep `iax daemon` running so the campaign advances automatically.
    """
    try:
        goal = GoalSpec.from_yaml(config)
    except Exception as exc:
        invalid_input(f"invalid goal {config}: {exc}")
    # Every trial runs this one workload, so one check answers for all of them.
    _preflight(goal, strict, "workload has warnings and --strict is set")
    try:
        state = _orchestrator(runs_dir).start(goal)
    except Exception as exc:
        raise IaxError(f"campaign start failed: {exc}", code="backend_unavailable") from exc
    # A campaign nobody drives submits its first batch and then stops forever.
    _warn_if_no_daemon(FilesystemRunStore(runs_dir), True)
    if output_json:
        _echo_json(state)
    else:
        active = sum(1 for t in state.trials if t.status in {"submitted", "running"})
        typer.echo(f"Campaign {state.campaign_id} started ({active} trials submitted)")
        typer.echo("Run `iax daemon` to drive the experiment loop.")


@campaign_app.command("list", cls=IaxCommand)
def campaign_list(
    runs_dir: Path | None = typer.Option(None, "--runs-dir"),
    output_json: bool = typer.Option(False, "--json"),
) -> None:
    from ai_experiments.store.campaign import CampaignStore

    store = FilesystemRunStore(runs_dir)
    campaign_store = CampaignStore(store.root)
    states = [campaign_store.read_state(cid) for cid in campaign_store.list_campaigns()]
    if not states:
        typer.echo(f"No campaigns in {store.root}", err=True)
    _warn_if_no_daemon(store, any(state.status in {"running", "stopping"} for state in states))
    if output_json:
        _echo_json([state.model_dump(mode="json") for state in states])
        return
    for state in states:
        typer.echo(
            f"{state.campaign_id}  {state.status:<10} trials={len(state.trials)} {state.name}"
        )


@campaign_app.command("status", cls=IaxCommand)
def campaign_status(
    campaign_id: str = typer.Argument(...),
    runs_dir: Path | None = typer.Option(None, "--runs-dir"),
    output_json: bool = typer.Option(False, "--json"),
) -> None:
    from ai_experiments.planner.analysis import summarize_campaign
    from ai_experiments.store.campaign import CampaignStore

    store = FilesystemRunStore(runs_dir)
    _require_campaign(store, campaign_id)
    campaign_store = CampaignStore(store.root)
    state = campaign_store.read_state(campaign_id)
    goal = campaign_store.read_goal(campaign_id)
    summary = summarize_campaign(state, goal)
    _warn_if_no_daemon(store, state.status in {"running", "stopping"})
    if output_json:
        _echo_json(summary)
        return
    typer.echo(
        f"{state.campaign_id}: {state.status}"
        + (f" ({state.stop_reason})" if state.stop_reason else "")
    )
    typer.echo(f"  Goal:   {state.goal}")
    typer.echo(f"  Trials: {summary.trials_by_status}")
    typer.echo(f"  Loop:   round {summary.rounds}, last advanced {summary.last_advanced_at}")
    cost = summary.estimated_cost
    typer.echo(
        f"  Spend:  {summary.gpu_hours:g} gpu-hours"
        + (f" (~${cost})" if cost is not None else "")
        + (
            f" of {goal.budget.max_gpu_hours:g} budgeted"
            if goal.budget.max_gpu_hours is not None
            else ""
        )
    )
    best = summary.best
    if best and best.objective_value is not None:
        typer.echo(f"  Best:   {best.trial_id} {goal.objective.metric}={best.objective_value:.6g}")
        typer.echo(f"          params={best.params}")


@campaign_app.command("advance", cls=IaxCommand)
def campaign_advance(
    campaign_id: str = typer.Argument(...),
    runs_dir: Path | None = typer.Option(None, "--runs-dir"),
    output_json: bool = typer.Option(False, "--json"),
) -> None:
    """Run one orchestrator step now (what the daemon does every tick)."""
    _require_campaign(FilesystemRunStore(runs_dir), campaign_id)
    state = _orchestrator(runs_dir).advance(campaign_id)
    if output_json:
        _echo_json(state)
    else:
        typer.echo(f"{state.campaign_id}: {state.status} ({len(state.trials)} trials)")


@campaign_app.command("stop", cls=IaxCommand)
def campaign_stop(
    campaign_id: str = typer.Argument(...),
    runs_dir: Path | None = typer.Option(None, "--runs-dir"),
    output_json: bool = typer.Option(False, "--json"),
) -> None:
    _require_campaign(FilesystemRunStore(runs_dir), campaign_id)
    state = _orchestrator(runs_dir).stop(campaign_id)
    if output_json:
        _echo_json(state)
    else:
        typer.echo(f"Stopped {state.campaign_id}")


@campaign_app.command("pause", cls=IaxCommand)
def campaign_pause(
    campaign_id: str = typer.Argument(...),
    runs_dir: Path | None = typer.Option(None, "--runs-dir"),
    output_json: bool = typer.Option(False, "--json"),
) -> None:
    """Stop scheduling new trials (active ones keep running). Resume later."""
    _require_campaign(FilesystemRunStore(runs_dir), campaign_id)
    try:
        state = _orchestrator(runs_dir).pause(campaign_id)
    except ValueError as exc:
        invalid_input(str(exc))
    if output_json:
        _echo_json(state)
    else:
        typer.echo(
            f"Paused {campaign_id} — edit the goal with `iax campaign edit`, "
            "then `iax campaign resume`."
        )


@campaign_app.command("resume", cls=IaxCommand)
def campaign_resume(
    campaign_id: str = typer.Argument(...),
    runs_dir: Path | None = typer.Option(None, "--runs-dir"),
    output_json: bool = typer.Option(False, "--json"),
) -> None:
    _require_campaign(FilesystemRunStore(runs_dir), campaign_id)
    try:
        state = _orchestrator(runs_dir).resume(campaign_id)
    except ValueError as exc:
        invalid_input(str(exc))
    if output_json:
        _echo_json(state)
    else:
        typer.echo(f"Resumed {state.campaign_id} ({state.status})")


@campaign_app.command("edit", cls=IaxCommand)
def campaign_edit(
    campaign_id: str = typer.Argument(...),
    goal_file: Path = typer.Argument(..., help="New goal YAML to apply"),
    runs_dir: Path | None = typer.Option(None, "--runs-dir"),
    output_json: bool = typer.Option(False, "--json"),
) -> None:
    """Replace the campaign's goal mid-flight (search space, budget, strategy).

    Existing trial history is kept and feeds the strategy under the new goal.
    The objective metric cannot change. Typical flow: pause -> edit -> resume.
    """
    _require_campaign(FilesystemRunStore(runs_dir), campaign_id)
    try:
        new_goal = GoalSpec.from_yaml(goal_file)
    except Exception as exc:
        invalid_input(f"invalid goal {goal_file}: {exc}")
    try:
        _orchestrator(runs_dir).edit_goal(campaign_id, new_goal)
    except ValueError as exc:
        invalid_input(str(exc))
    if output_json:
        _echo_json({"campaign_id": campaign_id, "goal_updated": True})
    else:
        typer.echo(f"Updated goal for {campaign_id}.")


@campaign_app.command("suggest", cls=IaxCommand)
def campaign_suggest(
    campaign_id: str = typer.Argument(...),
    params: str = typer.Option(
        ..., "--params", help="Trial params as JSON, e.g. '{\"lr\": 0.001}'"
    ),
    note: str = typer.Option("", "--note", help="Why this trial is worth running"),
    runs_dir: Path | None = typer.Option(None, "--runs-dir"),
    output_json: bool = typer.Option(False, "--json"),
) -> None:
    """Queue an agent/human-suggested trial for the next planning round."""
    try:
        parsed = json.loads(params)
        if not isinstance(parsed, dict):
            raise ValueError("params must be a JSON object")
    except (json.JSONDecodeError, ValueError) as exc:
        invalid_input(f"invalid --params: {exc}")
    _require_campaign(FilesystemRunStore(runs_dir), campaign_id)
    try:
        trial = _orchestrator(runs_dir).suggest(campaign_id, parsed, note=note)
    except ValueError as exc:
        invalid_input(f"suggestion rejected: {exc}")
    if output_json:
        _echo_json(trial)
    else:
        typer.echo(f"Queued {trial.trial_id} with params {trial.params}")


@campaign_app.command("trials", cls=IaxCommand)
def campaign_trials(
    campaign_id: str = typer.Argument(...),
    runs_dir: Path | None = typer.Option(None, "--runs-dir"),
    output_json: bool = typer.Option(False, "--json"),
) -> None:
    """List every trial: status, objective value, run id, and error."""
    from ai_experiments.store.campaign import CampaignStore

    store = FilesystemRunStore(runs_dir)
    _require_campaign(store, campaign_id)
    state = CampaignStore(store.root).read_state(campaign_id)
    if output_json:
        _echo_json([t.model_dump(mode="json") for t in state.trials])
        return
    if not state.trials:
        typer.echo("No trials yet.")
        return
    # `source` says who chose the params: the planner, the reviewing agent, or
    # a person via `campaign suggest`. Without it a reader cannot tell whether
    # the agent's hypotheses are beating the search (#23).
    typer.echo(f"{'TRIAL':<8} {'SOURCE':<10} {'STATUS':<10} {'OBJECTIVE':<14} {'RUN':<24} PARAMS")
    for trial in state.trials:
        value = f"{trial.objective_value:.6g}" if trial.objective_value is not None else "-"
        typer.echo(
            f"{trial.trial_id:<8} {trial.source:<10} {trial.status:<10} {value:<14} "
            f"{trial.run_id or '-':<24} {trial.params}"
        )
        if trial.error:
            typer.echo(f"         error: {trial.error}")


@campaign_app.command("rounds", cls=IaxCommand)
def campaign_rounds(
    campaign_id: str = typer.Argument(...),
    tail: int | None = typer.Option(None, "--tail", help="Only the last N records"),
    runs_dir: Path | None = typer.Option(None, "--runs-dir"),
    output_json: bool = typer.Option(False, "--json"),
) -> None:
    """Replay the improvement loop: what each round tried, and what it measured."""
    from ai_experiments.improve.rounds import RoundLog
    from ai_experiments.store.campaign import CampaignStore

    store = FilesystemRunStore(runs_dir)
    _require_campaign(store, campaign_id)
    records = RoundLog(CampaignStore(store.root).campaign_dir(campaign_id)).read(limit=tail)
    if output_json:
        _echo_json([r.model_dump(mode="json") for r in records])
        return
    if not records:
        typer.echo("No rounds recorded yet.")
        return
    for record in records:
        _print_round(record)


def _print_round(record: RoundRecord) -> None:
    """Render one round: what it tried, why, and -- for an evaluate round -- what came back.

    Every optional line is omitted rather than printed empty: a round log read
    at a glance should carry only what that round actually recorded.
    """
    typer.echo(
        f"round {record.round} [{record.stage}] via {record.strategy}"
        + (" (fallback)" if record.used_fallback else "")
    )
    if record.hypothesis:
        typer.echo(f"  hypothesis: {record.hypothesis}")
    if record.rationale:
        typer.echo(f"  rationale:  {record.rationale}")
    if record.trial_ids:
        typer.echo(f"  trials:     {', '.join(record.trial_ids)}")
    if record.stage == "evaluate":
        _print_round_values(record)
    for rejection in record.rejected:
        typer.echo(f"  rejected:   {rejection.get('reason')}")


def _print_round_values(record: RoundRecord) -> None:
    """Per-trial outcomes of an evaluate round.

    A trial with no numeric value prints ``-`` rather than being skipped: a
    failure is part of what the round measured, and its ``error`` is the line
    that explains the gap.
    """
    for trial_id, result in (record.outcome.get("values") or {}).items():
        value = result.get("objective_value")
        shown = f"{value:.6g}" if isinstance(value, (int, float)) else "-"
        typer.echo(f"    {trial_id}: {result.get('status')} {shown}")
        if result.get("error"):
            typer.echo(f"      error: {result['error']}")
