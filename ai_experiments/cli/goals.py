from __future__ import annotations

from pathlib import Path  # noqa: TC003  # Typer resolves this annotation at runtime
from typing import TYPE_CHECKING

import typer

from ai_experiments.cli import app
from ai_experiments.schemas import CampaignState, GoalSpec
from ai_experiments.store import FilesystemRunStore

if TYPE_CHECKING:
    from ai_experiments.daemon import TickReport


@app.command("run")
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
    """Everything in one command.

    Starts the campaign, serves the dashboard, and drives the monitor/experiment loop until
    the campaign finishes.
    """
    from ai_experiments.daemon import MonitorDaemon

    goal = _load_goal(config)
    store = FilesystemRunStore(runs_dir)
    monitor_daemon = MonitorDaemon(store)

    if serve_dashboard:
        _maybe_serve_dashboard(store, port, open_browser)

    state = monitor_daemon.orchestrator.start(goal)
    _print_campaign_started(state, goal, interval)

    try:
        state = monitor_daemon.orchestrator.run_to_completion(
            state, monitor_daemon.tick, _print_tick, interval
        )
    except KeyboardInterrupt:
        _print_detached(state)
        raise typer.Exit(code=0) from None

    _print_campaign_summary(state, goal, store)


def _load_goal(config: Path) -> GoalSpec:
    try:
        return GoalSpec.from_yaml(config)
    except Exception as exc:
        typer.echo(f"Error: invalid goal: {exc}", err=True)
        raise typer.Exit(code=1) from exc


def _maybe_serve_dashboard(store: FilesystemRunStore, port: int, open_browser: bool) -> None:
    url = _start_dashboard_thread(store, port)
    if not url:
        return
    typer.echo(f"Dashboard:  {url}")
    if open_browser:
        import webbrowser

        webbrowser.open(url)


def _print_campaign_started(state: CampaignState, goal: GoalSpec, interval: int) -> None:
    active = sum(1 for t in state.trials if t.status in {"submitted", "running"})
    typer.echo(f"Campaign:   {state.campaign_id} ({active} trials submitted)")
    typer.echo(f"Goal:       {goal.goal}")
    typer.echo(f"Loop:       tick every {interval}s — Ctrl+C detaches, runs keep going")


def _print_tick(report: TickReport) -> None:
    for action in report.actions:
        typer.echo(f"  [{action.run_id}] {action.action}: {', '.join(action.reasons)}")
    for error in report.errors:
        typer.echo(f"  error: {error}", err=True)


def _print_detached(state: CampaignState) -> None:
    typer.echo(
        f"\nDetached. Campaign {state.campaign_id} is still active — resume the "
        f"loop with `iax daemon` or check it with `iax campaign status "
        f"{state.campaign_id}`."
    )


def _print_campaign_summary(
    state: CampaignState, goal: GoalSpec, store: FilesystemRunStore
) -> None:
    typer.echo(f"\nCampaign {state.campaign_id}: {state.status} ({state.stop_reason})")
    best = next((t for t in state.trials if t.trial_id == state.best_trial_id), None)
    if best is not None:
        typer.echo(f"  Best: {best.trial_id} {goal.objective.metric}={best.objective_value:.6g}")
        typer.echo(f"        params={best.params}")
    typer.echo(f"  Revisit any time: iax serve --runs-dir {store.root}")


def _start_dashboard_thread(store: FilesystemRunStore, port: int) -> str | None:
    """Serve the dashboard from a daemon thread.

    Returns its URL, or None when the server extra is missing (the loop still works
    without it).
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
