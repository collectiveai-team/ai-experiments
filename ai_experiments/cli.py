from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from pydantic import BaseModel

from ai_experiments.backends.factory import get_backend
from ai_experiments.schemas import BackendName, ExperimentManifest
from ai_experiments.store import FilesystemRunStore

app = typer.Typer(
    name="iax",
    help="Detached experiment runtime for industrial AI training workloads.",
    no_args_is_help=True,
)


def _echo_json(payload: object) -> None:
    if isinstance(payload, BaseModel):
        typer.echo(json.dumps(payload.model_dump(mode="json"), indent=2))
    else:
        typer.echo(json.dumps(payload, indent=2))


def _backend_for_run(run_id: str, store: FilesystemRunStore):
    status = store.read_status(run_id)
    return get_backend(
        status.backend,
        store=store,
        address=_backend_address_for_run(status.backend, store, run_id),
    )


def _backend_address_for_run(
    backend: BackendName,
    store: FilesystemRunStore,
    run_id: str,
) -> str | None:
    if backend != "ray":
        return None
    manifest_path = store.run_dir(run_id) / "manifest.yaml"
    if not manifest_path.exists():
        return None
    return ExperimentManifest.from_yaml(manifest_path).backend_address


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
