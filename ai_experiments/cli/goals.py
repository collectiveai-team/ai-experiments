from __future__ import annotations

from pathlib import Path  # noqa: TC003  # Typer resolves this annotation at runtime
from typing import TYPE_CHECKING

import typer

from ai_experiments.cli import (
    IaxCommand,
    _echo_json,
    _require_campaign,
    app,
)
from ai_experiments.cli_support import (
    EXIT_BACKEND_UNAVAILABLE,
    EXIT_GOAL_NOT_REACHED,
    invalid_input,
)
from ai_experiments.schemas import GoalSpec
from ai_experiments.store import FilesystemRunStore

if TYPE_CHECKING:
    from ai_experiments.daemon import MonitorDaemon
    from ai_experiments.schemas import CampaignState


@app.command("run", cls=IaxCommand)
def run_goal(
    config: Path = typer.Argument(..., help="Path to goal YAML"),
    interval: int = typer.Option(10, "--interval", help="Seconds between ticks"),
    port: int = typer.Option(8585, "--port", help="Dashboard port"),
    serve_dashboard: bool = typer.Option(
        True, "--serve/--no-serve", help="Also serve the web dashboard"
    ),
    open_browser: bool = typer.Option(False, "--open", help="Open the dashboard in a browser"),
    runs_dir: Path | None = typer.Option(None, "--runs-dir", help="Override run store root"),
) -> None:
    """Run a goal end to end in the foreground.

    Start the campaign, serve the dashboard, and drive the
    monitor/experiment loop until the campaign finishes.
    """
    from ai_experiments.daemon import MonitorDaemon

    try:
        goal = GoalSpec.from_yaml(config)
    except Exception as exc:
        invalid_input(f"invalid goal {config}: {exc}")

    store = FilesystemRunStore(runs_dir)
    monitor_daemon = MonitorDaemon(store)

    if serve_dashboard:
        url = _start_dashboard_thread(store, port)
        if url:
            typer.echo(f"Dashboard:  {url}")
            if open_browser:
                import webbrowser

                webbrowser.open(url)

    state = monitor_daemon.orchestrator.start(goal)
    active = sum(1 for t in state.trials if t.status in {"submitted", "running"})
    typer.echo(f"Campaign:   {state.campaign_id} ({active} trials submitted)")
    typer.echo(f"Goal:       {goal.goal}")
    typer.echo(f"Loop:       tick every {interval}s — Ctrl+C detaches, runs keep going")

    state = _tick_until_finished(monitor_daemon, state, interval)
    _print_outcome(state, goal, store)


def _start_dashboard_thread(store: FilesystemRunStore, port: int) -> str | None:
    """Serve the dashboard from a daemon thread.

    Returns its URL, or None when the server extra is missing -- the loop
    still works without it.
    """
    import threading

    try:
        import uvicorn

        from ai_experiments.server.app import create_app
    except ImportError:
        typer.echo(
            "Note: dashboard skipped — install 'ai-experiments[server]' to enable it.",
            err=True,
        )
        return None

    server = uvicorn.Server(
        uvicorn.Config(create_app(store), host="127.0.0.1", port=port, log_level="error")
    )
    threading.Thread(target=server.run, daemon=True).start()
    return f"http://127.0.0.1:{port}"


@app.command("loop", cls=IaxCommand)
def loop_goal(
    config: Path = typer.Argument(..., help="Path to goal YAML"),
    resume: str | None = typer.Option(
        None, "--resume", help="Continue an existing campaign instead of starting one"
    ),
    max_rounds: int | None = typer.Option(
        None, "--max-rounds", help="Stop after this many planning rounds"
    ),
    max_seconds: float | None = typer.Option(
        None, "--max-seconds", help="Stop after this much wall clock time"
    ),
    interval: float = typer.Option(5.0, "--interval", help="Seconds between loop iterations"),
    runs_dir: Path | None = typer.Option(None, "--runs-dir", help="Override run store root"),
    output_json: bool = typer.Option(False, "--json", help="Print the report as JSON"),
) -> None:
    """Run the whole improvement loop and report whether the goal was reached.

    This is the command an agent drives: it blocks until the campaign is
    finished (or a limit is hit), prints one report, and exits 0 only when the
    objective's target was actually reached. Exit 4 means the work ran and the
    target was missed — the report says what the best trial was. Exit 3 means
    no trial could start, because the backend refused every submit.
    """
    from ai_experiments.loop import run_loop

    try:
        goal = GoalSpec.from_yaml(config)
    except Exception as exc:
        invalid_input(f"invalid goal {config}: {exc}")

    store = FilesystemRunStore(runs_dir)
    if resume is not None:
        _require_campaign(store, resume)

    report = run_loop(
        goal,
        store,
        campaign_id=resume,
        max_rounds=max_rounds,
        max_seconds=max_seconds,
        interval_seconds=interval,
    )

    if output_json:
        _echo_json(report)
    else:
        _print_loop_report(report)
    if report.stop_reason == "backend_unavailable":
        # The work never ran, so this is not "ran and missed the target". The
        # report already names the submit errors; the code says start the
        # cluster, not widen the goal.
        raise typer.Exit(code=EXIT_BACKEND_UNAVAILABLE)
    if not report.target_reached:
        raise typer.Exit(code=EXIT_GOAL_NOT_REACHED)


def _print_loop_report(report) -> None:
    typer.echo(f"{report.campaign_id}: {report.status} ({report.stop_reason})")
    typer.echo(
        f"  Loop:    {report.rounds} rounds, {report.trials} trials, "
        f"{report.agent_calls} agent calls, {report.elapsed_seconds:g}s"
    )
    objective = report.objective
    metric = objective.metric if objective else "objective"
    if report.best and report.best.objective_value is not None:
        typer.echo(f"  Best:    {report.best.trial_id} {metric}={report.best.objective_value:.6g}")
        typer.echo(f"           params={report.best.params}")
    else:
        typer.echo("  Best:    no trial produced a usable objective value")
    target = objective.target if objective else None
    mode = objective.mode if objective else ""
    if report.target_reached:
        typer.echo(f"  Target:  reached ({metric} {mode} {target})")
    elif target is not None:
        typer.echo(f"  Target:  NOT reached (wanted {metric} {mode} {target})")
    else:
        typer.echo("  Target:  none set, so the loop ran to its budget")
    for review in report.reviews:
        if review.get("verdict"):
            typer.echo(f"  Review:  {review['verdict']} — {review.get('reason', '')}")
    if report.pending_trials:
        # The loop hit a limit while trials were still running. Saying nothing
        # here invites the reader to treat an unfinished campaign as an answer.
        typer.echo(
            f"  Pending: {len(report.pending_trials)} trial(s) still in flight "
            f"({', '.join(report.pending_trials)}); resume to collect them"
        )
    if report.change_request:
        # The loop is telling the reader to stop searching and start developing.
        typer.echo(f"  Blocked: {report.change_request['title']}")
        typer.echo("           no parameter fixes this; the ticket is in `iax escalations`")
    typer.echo(f"  Rounds:  iax campaign rounds {report.campaign_id}")


def _tick_until_finished(
    monitor_daemon: MonitorDaemon, state: CampaignState, interval: int
) -> CampaignState:
    """Drive the loop in the foreground until the campaign stops advancing.

    Ctrl+C exits 0, not an error code: the trials are detached processes that
    keep running, so interrupting the watcher is a detach, not a failure.
    """
    import time as _time

    try:
        while True:
            report = monitor_daemon.tick()
            for action in report.actions:
                typer.echo(f"  [{action.run_id}] {action.action}: {', '.join(action.reasons)}")
            for error in report.errors:
                typer.echo(f"  error: {error}", err=True)
            state = monitor_daemon.campaign_store.read_state(state.campaign_id)
            if state.status in {"completed", "stopped", "failed"}:
                return state
            _time.sleep(interval)
    except KeyboardInterrupt:
        typer.echo(
            f"\nDetached. Campaign {state.campaign_id} is still active — resume the "
            f"loop with `iax daemon` or check it with `iax campaign status "
            f"{state.campaign_id}`."
        )
        raise typer.Exit(code=0) from None


def _print_outcome(state: CampaignState, goal: GoalSpec, store: FilesystemRunStore) -> None:
    """Report how the campaign ended, and where to go back to it."""
    typer.echo(f"\nCampaign {state.campaign_id}: {state.status} ({state.stop_reason})")
    best = next((t for t in state.trials if t.trial_id == state.best_trial_id), None)
    if best is not None:
        typer.echo(f"  Best: {best.trial_id} {goal.objective.metric}={best.objective_value:.6g}")
        typer.echo(f"        params={best.params}")
    typer.echo(f"  Revisit any time: iax serve --runs-dir {store.root}")
