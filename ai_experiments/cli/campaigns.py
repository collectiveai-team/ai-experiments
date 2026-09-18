from __future__ import annotations

import json
from pathlib import Path  # noqa: TC003  # Typer resolves this annotation at runtime

import typer

from ai_experiments.cli import _echo_json, campaign_app
from ai_experiments.schemas import GoalSpec
from ai_experiments.store import FilesystemRunStore

# --- campaign commands -------------------------------------------------------


def _orchestrator(runs_dir: Path | None):
    from ai_experiments.orchestrator import CampaignOrchestrator

    store = FilesystemRunStore(runs_dir)
    return CampaignOrchestrator(store)


@campaign_app.command("validate")
def campaign_validate(
    config: Path = typer.Argument(..., help="Path to goal YAML"),
) -> None:
    try:
        goal = GoalSpec.from_yaml(config)
    except Exception as exc:
        typer.echo(f"Error: invalid goal: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Goal valid: {config}")
    typer.echo(f"  Goal:      {goal.goal}")
    typer.echo(
        f"  Objective: {goal.objective.mode} {goal.objective.metric}"
        + (f" (target {goal.objective.target})" if goal.objective.target is not None else "")
    )
    typer.echo(f"  Budget:    {goal.budget.max_trials} trials, {goal.budget.max_parallel} parallel")
    typer.echo(f"  Strategy:  {goal.strategy.name}")
    typer.echo(f"  Backend:   {goal.backend}")


@campaign_app.command("start")
def campaign_start(
    config: Path = typer.Argument(..., help="Path to goal YAML"),
    runs_dir: Path | None = typer.Option(None, "--runs-dir"),
    output_json: bool = typer.Option(False, "--json"),
) -> None:
    """Create a campaign from a goal and submit the first batch of trials.

    Keep `iax daemon` running so the campaign advances automatically.
    """
    try:
        goal = GoalSpec.from_yaml(config)
        state = _orchestrator(runs_dir).start(goal)
    except Exception as exc:
        typer.echo(f"Error: campaign start failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if output_json:
        _echo_json(state)
    else:
        active = sum(1 for t in state.trials if t.status in {"submitted", "running"})
        typer.echo(f"Campaign {state.campaign_id} started ({active} trials submitted)")
        typer.echo("Run `iax daemon` to drive the experiment loop.")


@campaign_app.command("list")
def campaign_list(
    runs_dir: Path | None = typer.Option(None, "--runs-dir"),
    output_json: bool = typer.Option(False, "--json"),
) -> None:
    from ai_experiments.store.campaign import CampaignStore

    store = FilesystemRunStore(runs_dir)
    campaign_store = CampaignStore(store.root)
    states = [campaign_store.read_state(cid) for cid in campaign_store.list_campaigns()]
    if output_json:
        _echo_json([state.model_dump(mode="json") for state in states])
        return
    for state in states:
        typer.echo(
            f"{state.campaign_id}  {state.status:<10} trials={len(state.trials)} {state.name}"
        )


@campaign_app.command("status")
def campaign_status(
    campaign_id: str = typer.Argument(...),
    runs_dir: Path | None = typer.Option(None, "--runs-dir"),
    output_json: bool = typer.Option(False, "--json"),
) -> None:
    from ai_experiments.planner.analysis import summarize_campaign
    from ai_experiments.store.campaign import CampaignStore

    store = FilesystemRunStore(runs_dir)
    campaign_store = CampaignStore(store.root)
    state = campaign_store.read_state(campaign_id)
    goal = campaign_store.read_goal(campaign_id)
    summary = summarize_campaign(state, goal)
    if output_json:
        _echo_json(summary)
        return
    typer.echo(
        f"{state.campaign_id}: {state.status}"
        + (f" ({state.stop_reason})" if state.stop_reason else "")
    )
    typer.echo(f"  Goal:   {state.goal}")
    typer.echo(f"  Trials: {summary.trials_by_status}")
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
    if summary.best:
        best = summary.best
        typer.echo(f"  Best:   {best.trial_id} {goal.objective.metric}={best.objective_value:.6g}")
        typer.echo(f"          params={best.params}")


@campaign_app.command("advance")
def campaign_advance(
    campaign_id: str = typer.Argument(...),
    runs_dir: Path | None = typer.Option(None, "--runs-dir"),
    output_json: bool = typer.Option(False, "--json"),
) -> None:
    """Run one orchestrator step now (what the daemon does every tick)."""
    state = _orchestrator(runs_dir).advance(campaign_id)
    if output_json:
        _echo_json(state)
    else:
        typer.echo(f"{state.campaign_id}: {state.status} ({len(state.trials)} trials)")


@campaign_app.command("stop")
def campaign_stop(
    campaign_id: str = typer.Argument(...),
    runs_dir: Path | None = typer.Option(None, "--runs-dir"),
) -> None:
    state = _orchestrator(runs_dir).stop(campaign_id)
    typer.echo(f"Stopped {state.campaign_id}")


@campaign_app.command("pause")
def campaign_pause(
    campaign_id: str = typer.Argument(...),
    runs_dir: Path | None = typer.Option(None, "--runs-dir"),
) -> None:
    """Stop scheduling new trials (active ones keep running). Resume later."""
    try:
        _orchestrator(runs_dir).pause(campaign_id)
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        f"Paused {campaign_id} — edit the goal with `iax campaign edit`, "
        "then `iax campaign resume`."
    )


@campaign_app.command("resume")
def campaign_resume(
    campaign_id: str = typer.Argument(...),
    runs_dir: Path | None = typer.Option(None, "--runs-dir"),
) -> None:
    try:
        state = _orchestrator(runs_dir).resume(campaign_id)
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Resumed {state.campaign_id} ({state.status})")


@campaign_app.command("edit")
def campaign_edit(
    campaign_id: str = typer.Argument(...),
    goal_file: Path = typer.Argument(..., help="New goal YAML to apply"),
    runs_dir: Path | None = typer.Option(None, "--runs-dir"),
) -> None:
    """Replace the campaign's goal mid-flight (search space, budget, strategy).

    Existing trial history is kept and feeds the strategy under the new goal.
    The objective metric cannot change. Typical flow: pause -> edit -> resume.
    """
    try:
        new_goal = GoalSpec.from_yaml(goal_file)
    except Exception as exc:
        typer.echo(f"Error: invalid goal: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    try:
        _orchestrator(runs_dir).edit_goal(campaign_id, new_goal)
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Updated goal for {campaign_id}.")


@campaign_app.command("suggest")
def campaign_suggest(
    campaign_id: str = typer.Argument(...),
    params: str = typer.Option(
        ..., "--params", help="Trial params as JSON, e.g. '{\"lr\": 0.001}'"
    ),
    note: str = typer.Option("", "--note", help="Why this trial is worth running"),
    runs_dir: Path | None = typer.Option(None, "--runs-dir"),
) -> None:
    """Queue an agent/human-suggested trial for the next planning round."""
    try:
        parsed = json.loads(params)
        if not isinstance(parsed, dict):
            raise ValueError("params must be a JSON object")
    except (json.JSONDecodeError, ValueError) as exc:
        typer.echo(f"Error: invalid --params: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    trial = _orchestrator(runs_dir).suggest(campaign_id, parsed, note=note)
    typer.echo(f"Queued {trial.trial_id} with params {trial.params}")
