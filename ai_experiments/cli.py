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


@app.command()
def validate(
    config: Path = typer.Argument(..., help="Path to experiment manifest YAML"),
) -> None:
    try:
        manifest = ExperimentManifest.from_yaml(config)
    except Exception as exc:
        typer.echo(f"Error: invalid manifest: {exc}", err=True)
        raise typer.Exit(code=1)
    typer.echo(f"Manifest valid: {config}")
    typer.echo(f"  Experiment: {manifest.experiment}")
    typer.echo(f"  Backend:    {manifest.backend}")
    typer.echo(f"  Workload:   {manifest.workload.entrypoint}")


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
    typer.echo(f"Cancelled {run_id}")


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
        typer.echo(f"{status.run_id}  {status.status:<10} {status.backend:<6} {experiment}")


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
def daemon(
    interval: int = typer.Option(30, "--interval", help="Seconds between ticks"),
    once: bool = typer.Option(False, "--once", help="Run a single tick and exit"),
    runs_dir: Optional[Path] = typer.Option(
        None, "--runs-dir", help="Override run store root"
    ),
) -> None:
    """Monitor daemon: check runs, kill/escalate stuck ones, advance campaigns."""
    from ai_experiments.daemon import MonitorDaemon

    store = FilesystemRunStore(runs_dir)
    monitor_daemon = MonitorDaemon(store)
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
    typer.echo(f"  Objective: {goal.objective.mode} {goal.objective.metric}"
               + (f" (target {goal.objective.target})" if goal.objective.target is not None else ""))
    typer.echo(f"  Budget:    {goal.budget.max_trials} trials, {goal.budget.max_parallel} parallel")
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
    typer.echo(f"{state.campaign_id}: {state.status}"
               + (f" ({state.stop_reason})" if state.stop_reason else ""))
    typer.echo(f"  Goal:   {state.goal}")
    typer.echo(f"  Trials: {summary['trials_by_status']}")
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


@campaign_app.command("suggest")
def campaign_suggest(
    campaign_id: str = typer.Argument(...),
    params: str = typer.Option(..., "--params", help='Trial params as JSON, e.g. \'{"lr": 0.001}\''),
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
        typer.echo(
            f"{profile.name:<16} {profile.provider:<6} {profile.address or '-'}"
        )


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
