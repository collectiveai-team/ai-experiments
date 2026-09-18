from __future__ import annotations

from pathlib import Path  # noqa: TC003  # Typer resolves this annotation at runtime

import typer

from ai_experiments.cli import _echo_json, app
from ai_experiments.store import FilesystemRunStore


@app.command()
def daemon(
    interval: int = typer.Option(30, "--interval", help="Seconds between ticks"),
    once: bool = typer.Option(False, "--once", help="Run a single tick and exit"),
    notify_webhook: str | None = typer.Option(
        None, "--notify-webhook", help="Webhook URL (Slack-compatible) for alerts"
    ),
    notify_command: str | None = typer.Option(
        None, "--notify-command", help="Command run with the alert JSON on stdin"
    ),
    runs_dir: Path | None = typer.Option(None, "--runs-dir", help="Override run store root"),
) -> None:
    """Monitor daemon: check runs, kill/escalate stuck ones, advance campaigns."""
    from ai_experiments.daemon import MonitorDaemon
    from ai_experiments.notify import Notifier

    store = FilesystemRunStore(runs_dir)
    notifier = Notifier(store.root, webhook_url=notify_webhook, command=notify_command)
    monitor_daemon = MonitorDaemon(store, notifier=notifier)
    if once:
        _echo_json(monitor_daemon.tick())
        return
    typer.echo(f"iax daemon watching {store.root} every {interval}s", err=True)
    monitor_daemon.run_forever(interval_seconds=interval)


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8585, "--port"),
    runs_dir: Path | None = typer.Option(None, "--runs-dir", help="Override run store root"),
) -> None:
    """Web dashboard + REST API over the run and campaign stores."""
    try:
        import uvicorn

        from ai_experiments.server.app import create_app
    except ImportError as exc:
        typer.echo(
            "Error: the dashboard needs the server extra: pip install 'ai-experiments[server]'",
            err=True,
        )
        raise typer.Exit(code=1) from exc

    store = FilesystemRunStore(runs_dir)
    uvicorn.run(create_app(store), host=host, port=port, log_level="warning")
