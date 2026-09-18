from __future__ import annotations

from pathlib import Path  # noqa: TC003  # Typer resolves this annotation at runtime

import typer

from ai_experiments.backends.factory import get_backend
from ai_experiments.cli import _backend_for_run, _echo_json, app
from ai_experiments.schemas import ExperimentManifest, ReproBundleInfo
from ai_experiments.store import FilesystemRunStore


@app.command()
def validate(
    config: Path = typer.Argument(..., help="Path to experiment manifest YAML"),
) -> None:
    try:
        manifest = ExperimentManifest.from_yaml(config)
    except Exception as exc:
        typer.echo(f"Error: invalid manifest: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Manifest valid: {config}")
    typer.echo(f"  Experiment: {manifest.experiment}")
    typer.echo(f"  Backend:    {manifest.backend}")
    typer.echo(f"  Workload:   {manifest.workload.entrypoint}")


@app.command()
def submit(
    config: Path = typer.Argument(..., help="Path to experiment manifest YAML"),
    runs_dir: Path | None = typer.Option(None, "--runs-dir", help="Override run store root"),
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
        raise typer.Exit(code=1) from exc

    if output_json:
        _echo_json(handle)
    else:
        typer.echo(f"Submitted {handle.run_id} ({handle.backend})")
        typer.echo(f"  Status: {handle.status_uri}")


@app.command()
def status(
    run_id: str = typer.Argument(...),
    runs_dir: Path | None = typer.Option(None, "--runs-dir", help="Override run store root"),
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
    runs_dir: Path | None = typer.Option(None, "--runs-dir", help="Override run store root"),
    output_json: bool = typer.Option(False, "--json", help="Print JSON output"),
) -> None:
    store = FilesystemRunStore(runs_dir)
    events = _backend_for_run(run_id, store).logs(run_id, tail=tail)
    if output_json:
        _echo_json([event.model_dump(mode="json") for event in events])
    else:
        for event in events:
            typer.echo(f"[{event.timestamp.isoformat()}] {event.level}: {event.message}")


@app.command()
def diagnose(
    run_id: str = typer.Argument(...),
    runs_dir: Path | None = typer.Option(None, "--runs-dir", help="Override run store root"),
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
    runs_dir: Path | None = typer.Option(None, "--runs-dir", help="Override run store root"),
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
    runs_dir: Path | None = typer.Option(None, "--runs-dir", help="Override run store root"),
) -> None:
    store = FilesystemRunStore(runs_dir)
    _backend_for_run(run_id, store).cancel(run_id)
    typer.echo(f"Cancelled {run_id}")


@app.command()
def runs(
    runs_dir: Path | None = typer.Option(None, "--runs-dir", help="Override run store root"),
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
    runs_dir: Path | None = typer.Option(None, "--runs-dir", help="Override run store root"),
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
    runs_dir: Path | None = typer.Option(None, "--runs-dir", help="Override run store root"),
) -> None:
    """List pending escalations awaiting agent diagnosis (always JSON)."""
    from ai_experiments.monitoring.escalation import list_escalations

    store = FilesystemRunStore(runs_dir)
    _echo_json([request.model_dump(mode="json") for request in list_escalations(store)])


@app.command()
def artifacts(
    run_id: str = typer.Argument(...),
    runs_dir: Path | None = typer.Option(None, "--runs-dir", help="Override run store root"),
    output_json: bool = typer.Option(False, "--json", help="Print JSON output"),
) -> None:
    """List files a run's workload wrote to $IAX_ARTIFACTS_DIR."""
    store = FilesystemRunStore(runs_dir)
    entries = store.list_artifacts(run_id)
    if output_json:
        _echo_json([entry.model_dump(mode="json") for entry in entries])
        return
    if not entries:
        typer.echo(f"No artifacts for {run_id} ({store.artifacts_dir(run_id)})")
        return
    typer.echo(f"Artifacts in {store.artifacts_dir(run_id)}:")
    for entry in entries:
        typer.echo(f"  {entry.path}  ({entry.size_bytes} bytes)")


@app.command()
def repro(
    run_id: str = typer.Argument(...),
    runs_dir: Path | None = typer.Option(None, "--runs-dir", help="Override run store root"),
) -> None:
    """Show the reproducibility bundle captured at submit time (always JSON)."""
    from ai_experiments.repro import read_repro

    store = FilesystemRunStore(runs_dir)
    context = read_repro(store.run_dir(run_id))
    if context is None:
        typer.echo(f"Error: no repro bundle for {run_id}", err=True)
        raise typer.Exit(code=1)
    bundle_info = ReproBundleInfo(
        **context.model_dump(), bundle_dir=str(store.run_dir(run_id) / "repro")
    )
    _echo_json(bundle_info)


@app.command()
def rerun(
    run_id: str = typer.Argument(..., help="Run to repeat exactly"),
    runs_dir: Path | None = typer.Option(None, "--runs-dir", help="Override run store root"),
    output_json: bool = typer.Option(False, "--json", help="Print JSON output"),
) -> None:
    """Resubmit a run's persisted manifest (params are baked in).

    Warns when the current git state differs from the one recorded at submit time.
    """
    from ai_experiments.repro import current_git_sha, read_repro

    store = FilesystemRunStore(runs_dir)
    manifest = store.read_manifest(run_id)
    if manifest is None:
        typer.echo(f"Error: no persisted manifest for {run_id}", err=True)
        raise typer.Exit(code=1)

    recorded = read_repro(store.run_dir(run_id))
    recorded_sha = recorded.git_sha if recorded is not None else None
    now_sha = current_git_sha(manifest.workload.working_dir)
    if recorded_sha and now_sha and recorded_sha != now_sha:
        typer.echo(
            f"Warning: working dir is at {now_sha[:12]} but the run was submitted "
            f"from {recorded_sha[:12]} — check out that commit for an exact rerun "
            f"(diff of uncommitted changes, if any: "
            f"{store.run_dir(run_id) / 'repro' / 'diff.patch'})",
            err=True,
        )
    if recorded is not None and recorded.git_dirty:
        typer.echo(
            "Warning: the original submit had uncommitted changes "
            f"(see {store.run_dir(run_id) / 'repro' / 'diff.patch'})",
            err=True,
        )

    handle = get_backend(manifest.backend, store=store, address=manifest.backend_address).submit(
        manifest
    )
    if output_json:
        _echo_json(handle)
    else:
        typer.echo(f"Resubmitted as {handle.run_id} (from {run_id})")


@app.command()
def leaderboard(
    runs_dir: Path | None = typer.Option(None, "--runs-dir", help="Override run store root"),
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
        if summary.best is None:
            continue
        rows.append(
            {
                "campaign_id": campaign_id,
                "name": state.name,
                "metric": goal.objective.metric,
                "mode": goal.objective.mode,
                "best_value": summary.best.objective_value,
                "best_params": summary.best.params,
                "trials": len(state.trials),
                "gpu_hours": summary.gpu_hours,
                "estimated_cost": summary.estimated_cost,
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
        cost = f" ~${row['estimated_cost']}" if row["estimated_cost"] is not None else ""
        typer.echo(
            f"{row['mode']} {row['metric']}={row['best_value']:.6g}  "
            f"{row['name']} ({row['campaign_id']}, {row['trials']} trials, "
            f"{row['gpu_hours']:g} gpu-h{cost})  params={row['best_params']}"
        )
