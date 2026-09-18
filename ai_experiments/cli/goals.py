from __future__ import annotations

from pathlib import Path  # noqa: TC003  # Typer resolves this annotation at runtime

import typer

from ai_experiments.cli import app
from ai_experiments.schemas import GoalSpec
from ai_experiments.store import FilesystemRunStore


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

    try:
        goal = GoalSpec.from_yaml(config)
    except Exception as exc:
        typer.echo(f"Error: invalid goal: {exc}", err=True)
        raise typer.Exit(code=1) from exc

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
                break
            _time.sleep(interval)
    except KeyboardInterrupt:
        typer.echo(
            f"\nDetached. Campaign {state.campaign_id} is still active — resume the "
            f"loop with `iax daemon` or check it with `iax campaign status "
            f"{state.campaign_id}`."
        )
        raise typer.Exit(code=0) from None

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
