from __future__ import annotations

from pathlib import Path  # noqa: TC003  # Typer resolves this annotation at runtime

import typer

from ai_experiments.cli import (
    IaxCommand,
    _echo_json,
    app,
)
from ai_experiments.cli_support import (
    IaxError,
)
from ai_experiments.store import FilesystemRunStore


@app.command(cls=IaxCommand)
def daemon(
    interval: int = typer.Option(30, "--interval", help="Seconds between ticks"),
    once: bool = typer.Option(False, "--once", help="Run a single tick and exit"),
    heartbeat: int = typer.Option(
        300, "--heartbeat", help="Seconds between 'still alive' lines on a quiet daemon"
    ),
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
    monitor_daemon.run_forever(interval_seconds=interval, heartbeat_seconds=heartbeat)


@app.command(cls=IaxCommand)
def serve(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8585, "--port"),
    allow_remote_mutations: bool = typer.Option(
        False,
        "--allow-remote-mutations",
        help="Serve cancel/stop/pause/resume to the network. No authentication.",
    ),
    runs_dir: Path | None = typer.Option(None, "--runs-dir", help="Override run store root"),
) -> None:
    """Web dashboard + REST API over the run and campaign stores.

    The dashboard has no authentication. Bound to anything but loopback it
    serves reads only, unless you pass --allow-remote-mutations.
    """
    try:
        import uvicorn

        from ai_experiments.server.app import create_app
    except ImportError as exc:
        raise IaxError(
            "the dashboard needs the server extra: pip install 'ai-experiments[server]'",
            code="invalid_input",
        ) from exc

    from ai_experiments.server.app import is_loopback

    store = FilesystemRunStore(runs_dir)
    if not is_loopback(host):
        if allow_remote_mutations:
            typer.echo(
                f"WARNING: {host}:{port} serves unauthenticated cancel/stop/pause "
                "to anyone who can reach it. Put it behind a proxy that "
                "authenticates, or bind 127.0.0.1 and use an SSH tunnel.",
                err=True,
            )
        else:
            typer.echo(
                f"Bound to {host}: serving reads only. "
                "Pass --allow-remote-mutations to allow cancel/stop/pause.",
                err=True,
            )
    uvicorn.run(
        create_app(store, host=host, allow_remote_mutations=allow_remote_mutations),
        host=host,
        port=port,
        log_level="warning",
    )
