from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import click

import typer
from pydantic import BaseModel
from typer.core import TyperCommand

from ai_experiments.backends.factory import backend_for_run, get_backend
from ai_experiments.cli_support import (
    EXIT_BACKEND_UNAVAILABLE,
    EXIT_GOAL_NOT_REACHED,
    IaxError,
    invalid_input,
    not_found,
    report,
)
from ai_experiments.handoff import HandoffResult
from ai_experiments.schemas import ExperimentManifest, GoalSpec
from ai_experiments.store import FilesystemRunStore

app = typer.Typer(
    name="iax",
    help="Detached experiment runtime for industrial AI training workloads.",
    no_args_is_help=True,
)

campaign_app = typer.Typer(
    name="campaign",
    help="Goal-driven campaigns: plan, submit, analyze, iterate.",
    no_args_is_help=True,
)
cluster_app = typer.Typer(
    name="cluster",
    help="Named Ray cluster profiles (local, aws, gcp, azure).",
    no_args_is_help=True,
)
new_app = typer.Typer(
    name="new",
    help="Scaffold a manifest, a goal, or an instrumented workload.",
    no_args_is_help=True,
)
app.add_typer(campaign_app)
app.add_typer(cluster_app)
app.add_typer(new_app)


class IaxCommand(TyperCommand):
    """Turns an :class:`IaxError` into the CLI's error contract.

    Without this every unknown id surfaced as a python traceback (or, worse,
    as a silent exit 0), which an agent driving the CLI cannot branch on.
    """

    def invoke(self, ctx: click.Context) -> Any:
        try:
            return super().invoke(ctx)
        except IaxError as exc:
            report(exc, json_mode=bool(ctx.params.get("output_json")))


def _echo_json(payload: object) -> None:
    if isinstance(payload, BaseModel):
        typer.echo(json.dumps(payload.model_dump(mode="json"), indent=2))
    else:
        typer.echo(json.dumps(payload, indent=2))


def _warn_if_no_daemon(store: FilesystemRunStore, work_is_waiting: bool) -> None:
    """Say so when something needs a daemon and no daemon is ticking (#22).

    On stderr: this is context for the person reading, not part of the data a
    caller parses from stdout.
    """
    if not work_is_waiting:
        return
    from ai_experiments.heartbeat import daemon_warning

    warning = daemon_warning(store.root)
    if warning:
        typer.echo(warning, err=True)


def _require_run(store: FilesystemRunStore, run_id: str) -> None:
    if not store.run_dir(run_id).exists():
        not_found(
            "run",
            run_id,
            hint=f"run store is {store.root}; list them with `iax runs`",
        )


def _require_campaign(store: FilesystemRunStore, campaign_id: str) -> None:
    from ai_experiments.store.campaign import CampaignStore

    if not (
        CampaignStore(store.root).campaign_dir(campaign_id) / "state.json"
    ).exists():
        not_found(
            "campaign",
            campaign_id,
            hint=(f"run store is {store.root}; list them with `iax campaign list`"),
        )


def _backend_for_run(run_id: str, store: FilesystemRunStore):
    _require_run(store, run_id)
    return backend_for_run(store, run_id)


@app.command(cls=IaxCommand)
def validate(
    config: Path = typer.Argument(..., help="Path to experiment manifest YAML"),
) -> None:
    try:
        manifest = ExperimentManifest.from_yaml(config)
    except Exception as exc:
        invalid_input(f"invalid manifest {config}: {exc}")
    typer.echo(f"Manifest valid: {config}")
    typer.echo(f"  Experiment: {manifest.experiment}")
    typer.echo(f"  Backend:    {manifest.backend}")
    typer.echo(f"  Workload:   {manifest.workload.entrypoint}")


@app.command(cls=IaxCommand)
def submit(
    config: Path = typer.Argument(..., help="Path to experiment manifest YAML"),
    runs_dir: Optional[Path] = typer.Option(
        None, "--runs-dir", help="Override run store root"
    ),
    output_json: bool = typer.Option(False, "--json", help="Print JSON output"),
) -> None:
    try:
        manifest = ExperimentManifest.from_yaml(config)
        store = FilesystemRunStore(runs_dir)
        handle = get_backend(
            manifest.backend,
            store=store,
            address=manifest.backend_address,
        ).submit(manifest)
    except FileNotFoundError as exc:
        not_found("manifest", str(config), hint=str(exc))
    except ValueError as exc:
        invalid_input(f"submit failed: {exc}")
    except Exception as exc:
        raise IaxError(f"submit failed: {exc}", code="backend_unavailable") from exc

    if output_json:
        _echo_json(handle)
    else:
        typer.echo(f"Submitted {handle.run_id} ({handle.backend})")
        typer.echo(f"  Status: {handle.status_uri}")


@app.command(cls=IaxCommand)
def status(
    run_id: str = typer.Argument(...),
    runs_dir: Optional[Path] = typer.Option(
        None, "--runs-dir", help="Override run store root"
    ),
    output_json: bool = typer.Option(False, "--json", help="Print JSON output"),
) -> None:
    store = FilesystemRunStore(runs_dir)
    run_status = _backend_for_run(run_id, store).inspect(run_id)
    if output_json:
        _echo_json(run_status)
    else:
        typer.echo(f"{run_status.run_id}: {run_status.status}")
        if run_status.error:
            typer.echo(f"  Error: {run_status.error}")


@app.command(cls=IaxCommand)
def logs(
    run_id: str = typer.Argument(...),
    tail: int = typer.Option(200, "--tail", help="Number of recent events"),
    runs_dir: Optional[Path] = typer.Option(
        None, "--runs-dir", help="Override run store root"
    ),
    output_json: bool = typer.Option(False, "--json", help="Print JSON output"),
) -> None:
    store = FilesystemRunStore(runs_dir)
    events = _backend_for_run(run_id, store).logs(run_id, tail=tail)
    if output_json:
        _echo_json([event.model_dump(mode="json") for event in events])
    else:
        for event in events:
            typer.echo(
                f"[{event.timestamp.isoformat()}] {event.level}: {event.message}"
            )


@app.command(cls=IaxCommand)
def diagnose(
    run_id: str = typer.Argument(...),
    runs_dir: Optional[Path] = typer.Option(
        None, "--runs-dir", help="Override run store root"
    ),
    output_json: bool = typer.Option(False, "--json", help="Print JSON output"),
) -> None:
    store = FilesystemRunStore(runs_dir)
    report = _backend_for_run(run_id, store).diagnose(run_id)
    if output_json:
        _echo_json(report)
    else:
        typer.echo(f"{report.run_id}: {report.decision.decision}")
        for reason in report.decision.reasons:
            typer.echo(f"  - {reason}")


@app.command(cls=IaxCommand)
def monitor(
    run_id: str = typer.Argument(...),
    runs_dir: Optional[Path] = typer.Option(
        None, "--runs-dir", help="Override run store root"
    ),
    output_json: bool = typer.Option(False, "--json", help="Print JSON output"),
    quiet_when_waiting: bool = typer.Option(
        False,
        "--quiet-when-waiting",
        help="Print nothing when the run should continue waiting",
    ),
) -> None:
    """Scheduler-friendly diagnosis that can stay quiet while a run is healthy."""
    store = FilesystemRunStore(runs_dir)
    report = _backend_for_run(run_id, store).diagnose(run_id)
    if quiet_when_waiting and report.decision.decision == "continue_waiting":
        return
    if output_json:
        _echo_json(report)
    else:
        typer.echo(f"{report.run_id}: {report.decision.decision}")
        for reason in report.decision.reasons:
            typer.echo(f"  - {reason}")
        for recommendation in report.recommendations:
            typer.echo(f"recommendation: {recommendation}")


@app.command(cls=IaxCommand)
def cancel(
    run_id: str = typer.Argument(...),
    runs_dir: Optional[Path] = typer.Option(
        None, "--runs-dir", help="Override run store root"
    ),
    output_json: bool = typer.Option(False, "--json", help="Print JSON output"),
) -> None:
    store = FilesystemRunStore(runs_dir)
    _backend_for_run(run_id, store).cancel(run_id)
    if output_json:
        _echo_json({"run_id": run_id, "cancelled": True})
    else:
        typer.echo(f"Cancelled {run_id}")


@app.command(cls=IaxCommand)
def runs(
    runs_dir: Optional[Path] = typer.Option(
        None, "--runs-dir", help="Override run store root"
    ),
    output_json: bool = typer.Option(False, "--json", help="Print JSON output"),
) -> None:
    """List all runs in the run store."""
    store = FilesystemRunStore(runs_dir)
    statuses = [store.read_status(run_id) for run_id in sorted(store.list_runs())]
    if not statuses:
        # On stderr: an empty list is the moment the caller needs to know
        # which store was read, and stdout may be JSON someone is parsing.
        typer.echo(f"No runs in {store.root}", err=True)
    _warn_if_no_daemon(
        store, any(status.status in {"submitted", "running"} for status in statuses)
    )
    if output_json:
        _echo_json([status.model_dump(mode="json") for status in statuses])
        return
    for status in statuses:
        experiment = status.details.get("experiment", "")
        typer.echo(
            f"{status.run_id}  {status.status:<10} {status.backend:<6} {experiment}"
        )


@app.command(cls=IaxCommand)
def metrics(
    run_id: str = typer.Argument(...),
    tail: int = typer.Option(50, "--tail", help="Number of recent points"),
    runs_dir: Optional[Path] = typer.Option(
        None, "--runs-dir", help="Override run store root"
    ),
    output_json: bool = typer.Option(False, "--json", help="Print JSON output"),
) -> None:
    """Show metrics reported by a run's workload."""
    store = FilesystemRunStore(runs_dir)
    _require_run(store, run_id)
    points = store.read_metrics(run_id, tail=tail)
    if output_json:
        _echo_json([point.model_dump(mode="json") for point in points])
        return
    for point in points:
        values = " ".join(f"{k}={v:.6g}" for k, v in point.values.items())
        typer.echo(f"[{point.timestamp.isoformat()}] step={point.step} {values}")


@app.command(cls=IaxCommand)
def escalations(
    runs_dir: Optional[Path] = typer.Option(
        None, "--runs-dir", help="Override run store root"
    ),
) -> None:
    """List pending escalations awaiting agent diagnosis (always JSON)."""
    from ai_experiments.monitoring.escalation import list_escalations

    store = FilesystemRunStore(runs_dir)
    _echo_json([item.model_dump(mode="json") for item in list_escalations(store)])


@app.command(cls=IaxCommand)
def handoff(
    campaign_id: Optional[str] = typer.Argument(
        None, help="Blocked campaign to hand off; default every pending one"
    ),
    repo: Path = typer.Option(
        Path("."), "--repo", help="Git repository the code change belongs to"
    ),
    base: str = typer.Option(
        "HEAD", "--base", help="Commit the experimentation branch starts from"
    ),
    worktree_root: Optional[Path] = typer.Option(
        None, "--worktree-root", help="Where to check the branch out"
    ),
    command: Optional[str] = typer.Option(
        None,
        "--command",
        help="Development flow to call; {issue} and {branch} are substituted",
    ),
    run_flow: bool = typer.Option(
        False, "--run", help="Let the flow execute the work, not only plan it"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show what would happen and change nothing"
    ),
    force: bool = typer.Option(
        False, "--force", help="Hand off again even if this ticket was handled"
    ),
    runs_dir: Optional[Path] = typer.Option(
        None, "--runs-dir", help="Override run store root"
    ),
    output_json: bool = typer.Option(False, "--json", help="Print JSON output"),
) -> None:
    """Turn a blocked campaign into development work on an experimentation branch.

    A campaign that stopped with `blocked_on_change` needs code, not
    parameters. This creates `exp/<campaign>-<digest>` in its own worktree and
    gives the ticket to a development flow. Exit 3 means the flow failed.
    """
    import shlex

    from ai_experiments.handoff import DEFAULT_COMMAND, hand_off
    from ai_experiments.monitoring.escalation import list_change_requests

    store = FilesystemRunStore(runs_dir)
    pending = list_change_requests(store)
    if campaign_id:
        pending = [r for r in pending if r.campaign_id == campaign_id]
        if not pending:
            not_found("blocked campaign", campaign_id, "see `iax escalations`")

    flow = shlex.split(command) if command else list(DEFAULT_COMMAND)
    if run_flow and not command:
        flow = [part for part in flow if part != "--no-run"]

    results = [
        hand_off(
            store,
            request,
            repo=repo,
            base=base,
            worktree_root=worktree_root,
            command=flow,
            dry_run=dry_run,
            force=force,
        )
        for request in pending
    ]

    if output_json:
        _echo_json([r.model_dump(mode="json") for r in results])
    elif not results:
        typer.echo("No campaign is blocked on a code change.")
    else:
        for result in results:
            _print_handoff(result)

    if any(r.status == "failed" for r in results):
        raise typer.Exit(code=EXIT_BACKEND_UNAVAILABLE)


def _print_handoff(result: HandoffResult) -> None:
    typer.echo(f"{result.plan.campaign_id}: {result.status}")
    typer.echo(f"  Branch:   {result.plan.branch}")
    typer.echo(f"  Worktree: {result.plan.worktree}")
    typer.echo(f"  Issue:    {result.plan.issue_path}")
    if result.status == "already_handled":
        typer.echo("  Handed off before; pass --force to do it again")
    if result.error:
        typer.echo(f"  Error:    {result.error}")
    if result.output:
        typer.echo(f"  Output:   {result.output.strip().splitlines()[-1]}")
    typer.echo(f"  Next:     {result.next_step}")


@app.command(cls=IaxCommand)
def artifacts(
    run_id: str = typer.Argument(...),
    runs_dir: Optional[Path] = typer.Option(
        None, "--runs-dir", help="Override run store root"
    ),
    output_json: bool = typer.Option(False, "--json", help="Print JSON output"),
) -> None:
    """List files a run's workload wrote to $IAX_ARTIFACTS_DIR."""
    store = FilesystemRunStore(runs_dir)
    _require_run(store, run_id)
    entries = store.list_artifacts(run_id)
    if output_json:
        _echo_json(entries)
        return
    if not entries:
        typer.echo(f"No artifacts for {run_id} ({store.artifacts_dir(run_id)})")
        return
    typer.echo(f"Artifacts in {store.artifacts_dir(run_id)}:")
    for entry in entries:
        typer.echo(f"  {entry['path']}  ({entry['size_bytes']} bytes)")


@app.command(cls=IaxCommand)
def repro(
    run_id: str = typer.Argument(...),
    runs_dir: Optional[Path] = typer.Option(
        None, "--runs-dir", help="Override run store root"
    ),
    output_json: bool = typer.Option(
        True, "--json", help="Accepted for uniformity; repro always prints JSON"
    ),
) -> None:
    """Show the reproducibility bundle captured at submit time (always JSON)."""
    from ai_experiments.repro import read_repro

    store = FilesystemRunStore(runs_dir)
    context = read_repro(store.run_dir(run_id))
    if context is None:
        _require_run(store, run_id)
        not_found("repro bundle for run", run_id)
    context["bundle_dir"] = str(store.run_dir(run_id) / "repro")
    _echo_json(context)


@app.command(cls=IaxCommand)
def rerun(
    run_id: str = typer.Argument(..., help="Run to repeat exactly"),
    runs_dir: Optional[Path] = typer.Option(
        None, "--runs-dir", help="Override run store root"
    ),
    output_json: bool = typer.Option(False, "--json", help="Print JSON output"),
) -> None:
    """Resubmit a run's persisted manifest (params are baked in), warning when
    the current git state differs from the one recorded at submit time."""
    from ai_experiments.repro import current_git_sha, read_repro

    store = FilesystemRunStore(runs_dir)
    manifest = store.read_manifest(run_id)
    if manifest is None:
        _require_run(store, run_id)
        not_found("persisted manifest for run", run_id)

    recorded = read_repro(store.run_dir(run_id)) or {}
    recorded_sha = recorded.get("git_sha")
    now_sha = current_git_sha(manifest.workload.working_dir)
    if recorded_sha and now_sha and recorded_sha != now_sha:
        typer.echo(
            f"Warning: working dir is at {now_sha[:12]} but the run was submitted "
            f"from {recorded_sha[:12]} — check out that commit for an exact rerun "
            f"(diff of uncommitted changes, if any: "
            f"{store.run_dir(run_id) / 'repro' / 'diff.patch'})",
            err=True,
        )
    if recorded.get("git_dirty"):
        typer.echo(
            "Warning: the original submit had uncommitted changes "
            f"(see {store.run_dir(run_id) / 'repro' / 'diff.patch'})",
            err=True,
        )

    handle = get_backend(
        manifest.backend, store=store, address=manifest.backend_address
    ).submit(manifest)
    if output_json:
        _echo_json(handle)
    else:
        typer.echo(f"Resubmitted as {handle.run_id} (from {run_id})")


@app.command(cls=IaxCommand)
def leaderboard(
    runs_dir: Optional[Path] = typer.Option(
        None, "--runs-dir", help="Override run store root"
    ),
    output_json: bool = typer.Option(False, "--json", help="Print JSON output"),
) -> None:
    """Campaigns ranked by their best objective value."""
    from ai_experiments.planner.analysis import summarize_campaign
    from ai_experiments.store.campaign import CampaignStore

    store = FilesystemRunStore(runs_dir)
    campaign_store = CampaignStore(store.root)
    rows = []
    for campaign_id in campaign_store.list_campaigns():
        state = campaign_store.read_state(campaign_id)
        goal = campaign_store.read_goal(campaign_id)
        summary = summarize_campaign(state, goal)
        if summary["best"] is None:
            continue
        rows.append(
            {
                "campaign_id": campaign_id,
                "name": state.name,
                "metric": goal.objective.metric,
                "mode": goal.objective.mode,
                "best_value": summary["best"]["objective_value"],
                "best_params": summary["best"]["params"],
                "trials": len(state.trials),
                "gpu_hours": summary["gpu_hours"],
                "estimated_cost": summary["estimated_cost"],
            }
        )
    rows.sort(
        key=lambda r: (
            r["metric"],
            -r["best_value"] if r["mode"] == "max" else r["best_value"],
        )
    )
    if output_json:
        _echo_json(rows)
        return
    for row in rows:
        cost = (
            f" ~${row['estimated_cost']}" if row["estimated_cost"] is not None else ""
        )
        typer.echo(
            f"{row['mode']} {row['metric']}={row['best_value']:.6g}  "
            f"{row['name']} ({row['campaign_id']}, {row['trials']} trials, "
            f"{row['gpu_hours']:g} gpu-h{cost})  params={row['best_params']}"
        )


@app.command(cls=IaxCommand)
def daemon(
    interval: int = typer.Option(30, "--interval", help="Seconds between ticks"),
    once: bool = typer.Option(False, "--once", help="Run a single tick and exit"),
    heartbeat: int = typer.Option(
        300, "--heartbeat", help="Seconds between 'still alive' lines on a quiet daemon"
    ),
    notify_webhook: Optional[str] = typer.Option(
        None, "--notify-webhook", help="Webhook URL (Slack-compatible) for alerts"
    ),
    notify_command: Optional[str] = typer.Option(
        None, "--notify-command", help="Command run with the alert JSON on stdin"
    ),
    runs_dir: Optional[Path] = typer.Option(
        None, "--runs-dir", help="Override run store root"
    ),
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
    runs_dir: Optional[Path] = typer.Option(
        None, "--runs-dir", help="Override run store root"
    ),
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
            "the dashboard needs the server extra: "
            "pip install 'ai-experiments[server]'",
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


@app.command("run", cls=IaxCommand)
def run_goal(
    config: Path = typer.Argument(..., help="Path to goal YAML"),
    interval: int = typer.Option(10, "--interval", help="Seconds between ticks"),
    port: int = typer.Option(8585, "--port", help="Dashboard port"),
    serve_dashboard: bool = typer.Option(
        True, "--serve/--no-serve", help="Also serve the web dashboard"
    ),
    open_browser: bool = typer.Option(
        False, "--open", help="Open the dashboard in a browser"
    ),
    runs_dir: Optional[Path] = typer.Option(
        None, "--runs-dir", help="Override run store root"
    ),
) -> None:
    """Everything in one command: start the campaign, serve the dashboard,
    and drive the monitor/experiment loop until the campaign finishes."""
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

    import time as _time

    try:
        while True:
            report = monitor_daemon.tick()
            for action in report.actions:
                typer.echo(
                    f"  [{action.run_id}] {action.action}: {', '.join(action.reasons)}"
                )
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
        raise typer.Exit(code=0)

    typer.echo(f"\nCampaign {state.campaign_id}: {state.status} ({state.stop_reason})")
    best = next((t for t in state.trials if t.trial_id == state.best_trial_id), None)
    if best is not None:
        typer.echo(
            f"  Best: {best.trial_id} {goal.objective.metric}={best.objective_value:.6g}"
        )
        typer.echo(f"        params={best.params}")
    typer.echo(f"  Revisit any time: iax serve --runs-dir {store.root}")


def _start_dashboard_thread(store: FilesystemRunStore, port: int) -> str | None:
    """Serve the dashboard from a daemon thread; returns its URL, or None when
    the server extra is missing (the loop still works without it)."""
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
        uvicorn.Config(
            create_app(store), host="127.0.0.1", port=port, log_level="error"
        )
    )
    threading.Thread(target=server.run, daemon=True).start()
    return f"http://127.0.0.1:{port}"


# --- campaign commands -------------------------------------------------------


def _orchestrator(runs_dir: Optional[Path]):
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
        + (
            f" (target {goal.objective.target})"
            if goal.objective.target is not None
            else ""
        )
    )
    typer.echo(
        f"  Budget:    {goal.budget.max_trials} trials, {goal.budget.max_parallel} parallel"
    )
    typer.echo(f"  Strategy:  {goal.strategy.name}")
    typer.echo(f"  Backend:   {goal.backend}")


@campaign_app.command("start", cls=IaxCommand)
def campaign_start(
    config: Path = typer.Argument(..., help="Path to goal YAML"),
    runs_dir: Optional[Path] = typer.Option(None, "--runs-dir"),
    output_json: bool = typer.Option(False, "--json"),
) -> None:
    """Create a campaign from a goal and submit the first batch of trials.

    Keep `iax daemon` running so the campaign advances automatically.
    """
    try:
        goal = GoalSpec.from_yaml(config)
    except Exception as exc:
        invalid_input(f"invalid goal {config}: {exc}")
    try:
        state = _orchestrator(runs_dir).start(goal)
    except Exception as exc:
        raise IaxError(
            f"campaign start failed: {exc}", code="backend_unavailable"
        ) from exc
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
    runs_dir: Optional[Path] = typer.Option(None, "--runs-dir"),
    output_json: bool = typer.Option(False, "--json"),
) -> None:
    from ai_experiments.store.campaign import CampaignStore

    store = FilesystemRunStore(runs_dir)
    campaign_store = CampaignStore(store.root)
    states = [campaign_store.read_state(cid) for cid in campaign_store.list_campaigns()]
    if not states:
        typer.echo(f"No campaigns in {store.root}", err=True)
    _warn_if_no_daemon(
        store, any(state.status in {"running", "stopping"} for state in states)
    )
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
    runs_dir: Optional[Path] = typer.Option(None, "--runs-dir"),
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
    typer.echo(f"  Trials: {summary['trials_by_status']}")
    typer.echo(
        f"  Loop:   round {summary['rounds']}, "
        f"last advanced {summary['last_advanced_at']}"
    )
    cost = summary["estimated_cost"]
    typer.echo(
        f"  Spend:  {summary['gpu_hours']:g} gpu-hours"
        + (f" (~${cost})" if cost is not None else "")
        + (
            f" of {goal.budget.max_gpu_hours:g} budgeted"
            if goal.budget.max_gpu_hours is not None
            else ""
        )
    )
    if summary["best"]:
        best = summary["best"]
        typer.echo(
            f"  Best:   {best['trial_id']} {goal.objective.metric}={best['objective_value']:.6g}"
        )
        typer.echo(f"          params={best['params']}")


@campaign_app.command("advance", cls=IaxCommand)
def campaign_advance(
    campaign_id: str = typer.Argument(...),
    runs_dir: Optional[Path] = typer.Option(None, "--runs-dir"),
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
    runs_dir: Optional[Path] = typer.Option(None, "--runs-dir"),
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
    runs_dir: Optional[Path] = typer.Option(None, "--runs-dir"),
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
    runs_dir: Optional[Path] = typer.Option(None, "--runs-dir"),
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
    runs_dir: Optional[Path] = typer.Option(None, "--runs-dir"),
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
    runs_dir: Optional[Path] = typer.Option(None, "--runs-dir"),
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


# --- cluster commands ---------------------------------------------------------


@cluster_app.command("list", cls=IaxCommand)
def cluster_list(
    config: Optional[Path] = typer.Option(None, "--config", help="clusters.yaml path"),
) -> None:
    from ai_experiments.clusters import load_clusters

    profiles = load_clusters(config)
    if not profiles:
        typer.echo("No clusters configured (create clusters.yaml).")
        return
    for profile in profiles.values():
        typer.echo(f"{profile.name:<16} {profile.provider:<6} {profile.address or '-'}")


@cluster_app.command("status", cls=IaxCommand)
def cluster_status_cmd(
    name: str = typer.Argument(...),
    config: Optional[Path] = typer.Option(None, "--config", help="clusters.yaml path"),
) -> None:
    from ai_experiments.clusters import cluster_status, get_cluster

    _echo_json(cluster_status(get_cluster(name, config)))


@cluster_app.command("up", cls=IaxCommand)
def cluster_up_cmd(
    name: str = typer.Argument(...),
    config: Optional[Path] = typer.Option(None, "--config", help="clusters.yaml path"),
) -> None:
    """Provision a cloud cluster via Ray's cluster launcher (`ray up`)."""
    from ai_experiments.clusters import cluster_up, get_cluster

    result = cluster_up(get_cluster(name, config))
    typer.echo(result.stdout)
    if result.returncode != 0:
        typer.echo(result.stderr, err=True)
        raise typer.Exit(code=result.returncode)


@cluster_app.command("down", cls=IaxCommand)
def cluster_down_cmd(
    name: str = typer.Argument(...),
    config: Optional[Path] = typer.Option(None, "--config", help="clusters.yaml path"),
) -> None:
    """Tear down a cloud cluster via `ray down`."""
    from ai_experiments.clusters import cluster_down, get_cluster

    result = cluster_down(get_cluster(name, config))
    typer.echo(result.stdout)
    if result.returncode != 0:
        typer.echo(result.stderr, err=True)
        raise typer.Exit(code=result.returncode)


@new_app.command("manifest", cls=IaxCommand)
def new_manifest(
    path: Path = typer.Argument(
        Path("experiment.yaml"), help="File (or directory) to write"
    ),
    from_run: Optional[str] = typer.Option(
        None, "--from-run", help="Rebuild the manifest of an existing run instead"
    ),
    runs_dir: Optional[Path] = typer.Option(
        None, "--runs-dir", help="Override run store root"
    ),
    force: bool = typer.Option(False, "--force", help="Overwrite an existing file"),
) -> None:
    """Write a commented experiment manifest you can submit as-is."""
    from ai_experiments import scaffold

    if from_run is None:
        _scaffold_write("manifest", path, force)
        return
    store = FilesystemRunStore(runs_dir)
    _require_run(store, from_run)
    try:
        text = scaffold.manifest_from_run(store, from_run)
    except scaffold.ScaffoldError as exc:
        not_found("manifest for run", from_run, hint=str(exc))
    target = scaffold.resolve_target("manifest", path)
    if target.exists() and not force:
        invalid_input(f"{target} already exists; pass --force to overwrite")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)
    typer.echo(f"Wrote {target} from run {from_run}")
    typer.echo(f"Next: iax submit {target} --json")


@new_app.command("goal", cls=IaxCommand)
def new_goal(
    path: Path = typer.Argument(Path("goal.yaml"), help="File (or directory) to write"),
    force: bool = typer.Option(False, "--force", help="Overwrite an existing file"),
) -> None:
    """Write a commented goal manifest for a campaign."""
    _scaffold_write("goal", path, force)


@new_app.command("workload", cls=IaxCommand)
def new_workload(
    path: Path = typer.Argument(Path("train.py"), help="File (or directory) to write"),
    force: bool = typer.Option(False, "--force", help="Overwrite an existing file"),
) -> None:
    """Write a minimal workload that already reports metrics to the harness."""
    _scaffold_write("workload", path, force)


_NEXT_STEP = {
    "manifest": "iax validate {path} && iax submit {path} --json",
    "goal": "iax campaign validate {path} && iax loop {path}",
    "workload": "point a manifest's workload.args at {path}",
}


def _scaffold_write(kind: str, path: Path, force: bool) -> None:
    from ai_experiments import scaffold

    try:
        target = scaffold.write(kind, path, force=force)  # type: ignore[arg-type]
    except scaffold.ScaffoldError as exc:
        invalid_input(str(exc))
    typer.echo(f"Wrote {target}")
    typer.echo(f"Next: {_NEXT_STEP[kind].format(path=target)}")


@campaign_app.command("trials", cls=IaxCommand)
def campaign_trials(
    campaign_id: str = typer.Argument(...),
    runs_dir: Optional[Path] = typer.Option(None, "--runs-dir"),
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
    typer.echo(
        f"{'TRIAL':<8} {'SOURCE':<10} {'STATUS':<10} "
        f"{'OBJECTIVE':<14} {'RUN':<24} PARAMS"
    )
    for trial in state.trials:
        value = (
            f"{trial.objective_value:.6g}" if trial.objective_value is not None else "-"
        )
        typer.echo(
            f"{trial.trial_id:<8} {trial.source:<10} {trial.status:<10} {value:<14} "
            f"{trial.run_id or '-':<24} {trial.params}"
        )
        if trial.error:
            typer.echo(f"         error: {trial.error}")


@campaign_app.command("rounds", cls=IaxCommand)
def campaign_rounds(
    campaign_id: str = typer.Argument(...),
    tail: Optional[int] = typer.Option(None, "--tail", help="Only the last N records"),
    runs_dir: Optional[Path] = typer.Option(None, "--runs-dir"),
    output_json: bool = typer.Option(False, "--json"),
) -> None:
    """Replay the improvement loop: what each round tried, and what it measured."""
    from ai_experiments.improve.rounds import RoundLog
    from ai_experiments.store.campaign import CampaignStore

    store = FilesystemRunStore(runs_dir)
    _require_campaign(store, campaign_id)
    records = RoundLog(CampaignStore(store.root).campaign_dir(campaign_id)).read(
        limit=tail
    )
    if output_json:
        _echo_json([r.model_dump(mode="json") for r in records])
        return
    if not records:
        typer.echo("No rounds recorded yet.")
        return
    for record in records:
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
            for trial_id, result in (record.outcome.get("values") or {}).items():
                value = result.get("objective_value")
                shown = f"{value:.6g}" if isinstance(value, (int, float)) else "-"
                typer.echo(f"    {trial_id}: {result.get('status')} {shown}")
                if result.get("error"):
                    typer.echo(f"      error: {result['error']}")
        for rejection in record.rejected:
            typer.echo(f"  rejected:   {rejection.get('reason')}")


@app.command("loop", cls=IaxCommand)
def loop_goal(
    config: Path = typer.Argument(..., help="Path to goal YAML"),
    resume: Optional[str] = typer.Option(
        None, "--resume", help="Continue an existing campaign instead of starting one"
    ),
    max_rounds: Optional[int] = typer.Option(
        None, "--max-rounds", help="Stop after this many planning rounds"
    ),
    max_seconds: Optional[float] = typer.Option(
        None, "--max-seconds", help="Stop after this much wall clock time"
    ),
    interval: float = typer.Option(
        5.0, "--interval", help="Seconds between loop iterations"
    ),
    runs_dir: Optional[Path] = typer.Option(
        None, "--runs-dir", help="Override run store root"
    ),
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
    metric = report.objective.get("metric", "objective")
    if report.best:
        typer.echo(
            f"  Best:    {report.best['trial_id']} {metric}={report.best['objective_value']:.6g}"
        )
        typer.echo(f"           params={report.best['params']}")
    else:
        typer.echo("  Best:    no trial produced a usable objective value")
    target = report.objective.get("target")
    if report.target_reached:
        typer.echo(
            f"  Target:  reached ({metric} {report.objective.get('mode')} {target})"
        )
    elif target is not None:
        typer.echo(
            f"  Target:  NOT reached (wanted {metric} {report.objective.get('mode')} {target})"
        )
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
        typer.echo(
            "           no parameter fixes this; the ticket is in `iax escalations`"
        )
    typer.echo(f"  Rounds:  iax campaign rounds {report.campaign_id}")
