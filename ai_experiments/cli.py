from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from pydantic import BaseModel

from ai_experiments.backends.factory import backend_for_run, get_backend
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
app.add_typer(campaign_app)
app.add_typer(cluster_app)


def _echo_json(payload: object) -> None:
    if isinstance(payload, BaseModel):
        typer.echo(json.dumps(payload.model_dump(mode="json"), indent=2))
    else:
        typer.echo(json.dumps(payload, indent=2))


def _backend_for_run(run_id: str, store: FilesystemRunStore):
    return backend_for_run(store, run_id)


def _worker_log(
    store: FilesystemRunStore, run_id: str, tail: int, output_json: bool
) -> None:
    """The supervisor's own stdout/stderr.

    Anything that kills a supervisor before it can report leaves its traceback
    here and nowhere else, so this file has to be reachable without knowing
    the run store's layout.
    """
    recorded = store.read_status(run_id).details.get("log_path")
    path = Path(str(recorded)) if recorded else store.run_dir(run_id) / "worker.log"
    if not path.exists():
        typer.echo(f"Error: no worker log for {run_id} ({path})", err=True)
        raise typer.Exit(code=1)
    lines = path.read_text(errors="replace").splitlines()[-tail:]
    if output_json:
        _echo_json({"path": str(path), "lines": lines})
    else:
        for line in lines:
            typer.echo(line)


@app.command()
def validate(
    config: Path = typer.Argument(..., help="Path to experiment manifest YAML"),
    strict: bool = typer.Option(
        False, "--strict", help="Fail on warnings, not just on invalid manifests"
    ),
) -> None:
    from ai_experiments.preflight import workload_warnings

    try:
        manifest = ExperimentManifest.from_yaml(config)
    except Exception as exc:
        typer.echo(f"Error: invalid manifest: {exc}", err=True)
        raise typer.Exit(code=1)
    typer.echo(f"Manifest valid: {config}")
    typer.echo(f"  Experiment: {manifest.experiment}")
    typer.echo(f"  Backend:    {manifest.backend}")
    typer.echo(f"  Workload:   {manifest.workload.entrypoint}")

    warnings = workload_warnings(manifest)
    for warning in warnings:
        typer.echo(f"Warning: {warning}", err=True)
    if warnings and strict:
        typer.echo("Error: manifest has warnings and --strict is set", err=True)
        raise typer.Exit(code=1)


@app.command()
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
    except Exception as exc:
        typer.echo(f"Error: submit failed: {exc}", err=True)
        raise typer.Exit(code=1)

    if output_json:
        _echo_json(handle)
    else:
        typer.echo(f"Submitted {handle.run_id} ({handle.backend})")
        typer.echo(f"  Status: {handle.status_uri}")


@app.command()
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


@app.command()
def logs(
    run_id: str = typer.Argument(...),
    tail: int = typer.Option(200, "--tail", help="Number of recent events"),
    runs_dir: Optional[Path] = typer.Option(
        None, "--runs-dir", help="Override run store root"
    ),
    worker: bool = typer.Option(
        False,
        "--worker",
        help="Show the supervisor's own log instead of the run's events",
    ),
    output_json: bool = typer.Option(False, "--json", help="Print JSON output"),
) -> None:
    store = FilesystemRunStore(runs_dir)
    if worker:
        _worker_log(store, run_id, tail=tail, output_json=output_json)
        return
    events = _backend_for_run(run_id, store).logs(run_id, tail=tail)
    if output_json:
        _echo_json([event.model_dump(mode="json") for event in events])
    else:
        for event in events:
            typer.echo(
                f"[{event.timestamp.isoformat()}] {event.level}: {event.message}"
            )


@app.command()
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


@app.command()
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


@app.command()
def cancel(
    run_id: str = typer.Argument(...),
    runs_dir: Optional[Path] = typer.Option(
        None, "--runs-dir", help="Override run store root"
    ),
) -> None:
    store = FilesystemRunStore(runs_dir)
    _backend_for_run(run_id, store).cancel(run_id)
    # Cancelling a run that already ended leaves it alone, so report what the
    # run actually is rather than what was asked for.
    final = store.read_status(run_id).status
    if final == "cancelled":
        typer.echo(f"Cancelled {run_id}")
    else:
        typer.echo(f"{run_id} was not cancelled: it is already {final}")


@app.command()
def runs(
    runs_dir: Optional[Path] = typer.Option(
        None, "--runs-dir", help="Override run store root"
    ),
    output_json: bool = typer.Option(False, "--json", help="Print JSON output"),
) -> None:
    """List all runs in the run store."""
    store = FilesystemRunStore(runs_dir)
    statuses = [store.read_status(run_id) for run_id in sorted(store.list_runs())]
    if output_json:
        _echo_json([status.model_dump(mode="json") for status in statuses])
        return
    for status in statuses:
        experiment = status.details.get("experiment", "")
        typer.echo(
            f"{status.run_id}  {status.status:<10} {status.backend:<6} {experiment}"
        )


@app.command()
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
    points = store.read_metrics(run_id, tail=tail)
    if output_json:
        _echo_json([point.model_dump(mode="json") for point in points])
        return
    for point in points:
        values = " ".join(f"{k}={v:.6g}" for k, v in point.values.items())
        typer.echo(f"[{point.timestamp.isoformat()}] step={point.step} {values}")


@app.command()
def escalations(
    runs_dir: Optional[Path] = typer.Option(
        None, "--runs-dir", help="Override run store root"
    ),
) -> None:
    """List pending escalations awaiting agent diagnosis (always JSON)."""
    from ai_experiments.monitoring.escalation import list_escalations

    store = FilesystemRunStore(runs_dir)
    _echo_json([request.model_dump(mode="json") for request in list_escalations(store)])


@app.command()
def artifacts(
    run_id: str = typer.Argument(...),
    runs_dir: Optional[Path] = typer.Option(
        None, "--runs-dir", help="Override run store root"
    ),
    output_json: bool = typer.Option(False, "--json", help="Print JSON output"),
) -> None:
    """List files a run's workload wrote to $IAX_ARTIFACTS_DIR."""
    store = FilesystemRunStore(runs_dir)
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


@app.command()
def repro(
    run_id: str = typer.Argument(...),
    runs_dir: Optional[Path] = typer.Option(
        None, "--runs-dir", help="Override run store root"
    ),
) -> None:
    """Show the reproducibility bundle captured at submit time (always JSON)."""
    from ai_experiments.repro import read_repro

    store = FilesystemRunStore(runs_dir)
    context = read_repro(store.run_dir(run_id))
    if context is None:
        typer.echo(f"Error: no repro bundle for {run_id}", err=True)
        raise typer.Exit(code=1)
    context["bundle_dir"] = str(store.run_dir(run_id) / "repro")
    _echo_json(context)


@app.command()
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
        typer.echo(f"Error: no persisted manifest for {run_id}", err=True)
        raise typer.Exit(code=1)

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


@app.command()
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


@app.command()
def daemon(
    interval: int = typer.Option(30, "--interval", help="Seconds between ticks"),
    once: bool = typer.Option(False, "--once", help="Run a single tick and exit"),
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
    monitor_daemon.run_forever(interval_seconds=interval)


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8585, "--port"),
    runs_dir: Optional[Path] = typer.Option(
        None, "--runs-dir", help="Override run store root"
    ),
) -> None:
    """Web dashboard + REST API over the run and campaign stores."""
    try:
        import uvicorn

        from ai_experiments.server.app import create_app
    except ImportError as exc:
        typer.echo(
            "Error: the dashboard needs the server extra: "
            "pip install 'ai-experiments[server]'",
            err=True,
        )
        raise typer.Exit(code=1) from exc

    store = FilesystemRunStore(runs_dir)
    uvicorn.run(create_app(store), host=host, port=port, log_level="warning")


@app.command("run")
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
        typer.echo(f"Error: invalid goal: {exc}", err=True)
        raise typer.Exit(code=1)

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


@campaign_app.command("validate")
def campaign_validate(
    config: Path = typer.Argument(..., help="Path to goal YAML"),
) -> None:
    try:
        goal = GoalSpec.from_yaml(config)
    except Exception as exc:
        typer.echo(f"Error: invalid goal: {exc}", err=True)
        raise typer.Exit(code=1)
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


@campaign_app.command("start")
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
        state = _orchestrator(runs_dir).start(goal)
    except Exception as exc:
        typer.echo(f"Error: campaign start failed: {exc}", err=True)
        raise typer.Exit(code=1)
    if output_json:
        _echo_json(state)
    else:
        active = sum(1 for t in state.trials if t.status in {"submitted", "running"})
        typer.echo(f"Campaign {state.campaign_id} started ({active} trials submitted)")
        typer.echo("Run `iax daemon` to drive the experiment loop.")


@campaign_app.command("list")
def campaign_list(
    runs_dir: Optional[Path] = typer.Option(None, "--runs-dir"),
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
    runs_dir: Optional[Path] = typer.Option(None, "--runs-dir"),
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
    typer.echo(f"  Trials: {summary['trials_by_status']}")
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


@campaign_app.command("advance")
def campaign_advance(
    campaign_id: str = typer.Argument(...),
    runs_dir: Optional[Path] = typer.Option(None, "--runs-dir"),
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
    runs_dir: Optional[Path] = typer.Option(None, "--runs-dir"),
) -> None:
    state = _orchestrator(runs_dir).stop(campaign_id)
    typer.echo(f"Stopped {state.campaign_id}")


@campaign_app.command("pause")
def campaign_pause(
    campaign_id: str = typer.Argument(...),
    runs_dir: Optional[Path] = typer.Option(None, "--runs-dir"),
) -> None:
    """Stop scheduling new trials (active ones keep running). Resume later."""
    try:
        _orchestrator(runs_dir).pause(campaign_id)
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)
    typer.echo(
        f"Paused {campaign_id} — edit the goal with `iax campaign edit`, "
        "then `iax campaign resume`."
    )


@campaign_app.command("resume")
def campaign_resume(
    campaign_id: str = typer.Argument(...),
    runs_dir: Optional[Path] = typer.Option(None, "--runs-dir"),
) -> None:
    try:
        state = _orchestrator(runs_dir).resume(campaign_id)
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)
    typer.echo(f"Resumed {state.campaign_id} ({state.status})")


@campaign_app.command("edit")
def campaign_edit(
    campaign_id: str = typer.Argument(...),
    goal_file: Path = typer.Argument(..., help="New goal YAML to apply"),
    runs_dir: Optional[Path] = typer.Option(None, "--runs-dir"),
) -> None:
    """Replace the campaign's goal mid-flight (search space, budget, strategy).

    Existing trial history is kept and feeds the strategy under the new goal.
    The objective metric cannot change. Typical flow: pause -> edit -> resume.
    """
    try:
        new_goal = GoalSpec.from_yaml(goal_file)
    except Exception as exc:
        typer.echo(f"Error: invalid goal: {exc}", err=True)
        raise typer.Exit(code=1)
    try:
        _orchestrator(runs_dir).edit_goal(campaign_id, new_goal)
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)
    typer.echo(f"Updated goal for {campaign_id}.")


@campaign_app.command("suggest")
def campaign_suggest(
    campaign_id: str = typer.Argument(...),
    params: str = typer.Option(
        ..., "--params", help="Trial params as JSON, e.g. '{\"lr\": 0.001}'"
    ),
    note: str = typer.Option("", "--note", help="Why this trial is worth running"),
    runs_dir: Optional[Path] = typer.Option(None, "--runs-dir"),
) -> None:
    """Queue an agent/human-suggested trial for the next planning round."""
    try:
        parsed = json.loads(params)
        if not isinstance(parsed, dict):
            raise ValueError("params must be a JSON object")
    except (json.JSONDecodeError, ValueError) as exc:
        typer.echo(f"Error: invalid --params: {exc}", err=True)
        raise typer.Exit(code=1)
    trial = _orchestrator(runs_dir).suggest(campaign_id, parsed, note=note)
    typer.echo(f"Queued {trial.trial_id} with params {trial.params}")


# --- cluster commands ---------------------------------------------------------


@cluster_app.command("list")
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


@cluster_app.command("status")
def cluster_status_cmd(
    name: str = typer.Argument(...),
    config: Optional[Path] = typer.Option(None, "--config", help="clusters.yaml path"),
) -> None:
    from ai_experiments.clusters import cluster_status, get_cluster

    _echo_json(cluster_status(get_cluster(name, config)))


@cluster_app.command("up")
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


@cluster_app.command("down")
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
