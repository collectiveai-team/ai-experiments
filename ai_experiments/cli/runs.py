from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import typer

from ai_experiments.backends.factory import get_backend
from ai_experiments.cli import (
    IaxCommand,
    _backend_for_run,
    _echo_json,
    _preflight,
    _require_run,
    _warn_if_no_daemon,
    _worker_log,
    app,
)
from ai_experiments.cli_support import (
    EXIT_BACKEND_UNAVAILABLE,
    IaxError,
    invalid_input,
    not_found,
)
from ai_experiments.schemas import ExperimentManifest, ReproBundleInfo
from ai_experiments.store import FilesystemRunStore

if TYPE_CHECKING:
    from ai_experiments.handoff import HandoffResult
    from ai_experiments.monitoring.escalation import ChangeRequest


@app.command(cls=IaxCommand)
def validate(
    config: Path = typer.Argument(..., help="Path to experiment manifest YAML"),
    strict: bool = typer.Option(
        False, "--strict", help="Fail on warnings, not just on invalid manifests"
    ),
) -> None:
    try:
        manifest = ExperimentManifest.from_yaml(config)
    except Exception as exc:
        invalid_input(f"invalid manifest {config}: {exc}")
    typer.echo(f"Manifest valid: {config}")
    typer.echo(f"  Experiment: {manifest.experiment}")
    typer.echo(f"  Backend:    {manifest.backend}")
    typer.echo(f"  Workload:   {manifest.workload.entrypoint}")

    _preflight(manifest, strict, "manifest has warnings and --strict is set")


@app.command(cls=IaxCommand)
def submit(
    config: Path = typer.Argument(..., help="Path to experiment manifest YAML"),
    runs_dir: Path | None = typer.Option(None, "--runs-dir", help="Override run store root"),
    output_json: bool = typer.Option(False, "--json", help="Print JSON output"),
    strict: bool = typer.Option(
        False, "--strict", help="Refuse to submit a workload that looks unable to start"
    ),
) -> None:
    try:
        manifest = ExperimentManifest.from_yaml(config)
    except Exception as exc:
        typer.echo(f"Error: submit failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    # Before the run exists: a refused submit must leave no run behind.
    _preflight(manifest, strict, "workload has warnings and --strict is set")
    try:
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


@app.command(cls=IaxCommand)
def logs(
    run_id: str = typer.Argument(...),
    tail: int = typer.Option(200, "--tail", help="Number of recent events"),
    runs_dir: Path | None = typer.Option(None, "--runs-dir", help="Override run store root"),
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
            typer.echo(f"[{event.timestamp.isoformat()}] {event.level}: {event.message}")


@app.command(cls=IaxCommand)
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


@app.command(cls=IaxCommand)
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


@app.command(cls=IaxCommand)
def cancel(
    run_id: str = typer.Argument(...),
    runs_dir: Path | None = typer.Option(None, "--runs-dir", help="Override run store root"),
    output_json: bool = typer.Option(False, "--json", help="Print JSON output"),
) -> None:
    store = FilesystemRunStore(runs_dir)
    _backend_for_run(run_id, store).cancel(run_id)
    # Cancelling a run that already ended leaves it alone, so report what the
    # run actually is rather than what was asked for.
    final = store.read_status(run_id).status
    if output_json:
        _echo_json({"run_id": run_id, "cancelled": final == "cancelled", "status": final})
    elif final == "cancelled":
        typer.echo(f"Cancelled {run_id}")
    else:
        typer.echo(f"{run_id} was not cancelled: it is already {final}")


@app.command(cls=IaxCommand)
def runs(
    runs_dir: Path | None = typer.Option(None, "--runs-dir", help="Override run store root"),
    output_json: bool = typer.Option(False, "--json", help="Print JSON output"),
) -> None:
    """List all runs in the run store."""
    store = FilesystemRunStore(runs_dir)
    statuses = [store.read_status(run_id) for run_id in sorted(store.list_runs())]
    if not statuses:
        # On stderr: an empty list is the moment the caller needs to know
        # which store was read, and stdout may be JSON someone is parsing.
        typer.echo(f"No runs in {store.root}", err=True)
    _warn_if_no_daemon(store, any(status.status in {"submitted", "running"} for status in statuses))
    if output_json:
        _echo_json([status.model_dump(mode="json") for status in statuses])
        return
    for status in statuses:
        experiment = status.details.get("experiment", "")
        typer.echo(f"{status.run_id}  {status.status:<10} {status.backend:<6} {experiment}")


@app.command(cls=IaxCommand)
def metrics(
    run_id: str = typer.Argument(...),
    tail: int = typer.Option(50, "--tail", help="Number of recent points"),
    runs_dir: Path | None = typer.Option(None, "--runs-dir", help="Override run store root"),
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
    runs_dir: Path | None = typer.Option(None, "--runs-dir", help="Override run store root"),
) -> None:
    """List pending escalations awaiting agent diagnosis (always JSON)."""
    from ai_experiments.monitoring.escalation import list_escalations

    store = FilesystemRunStore(runs_dir)
    _echo_json([item.model_dump(mode="json") for item in list_escalations(store)])


@app.command(cls=IaxCommand)
def handoff(
    campaign_id: str | None = typer.Argument(
        None, help="Blocked campaign to hand off; default every pending one"
    ),
    repo: Path = typer.Option(Path(), "--repo", help="Git repository the code change belongs to"),
    base: str = typer.Option(
        "HEAD", "--base", help="Commit the experimentation branch starts from"
    ),
    worktree_root: Path | None = typer.Option(
        None, "--worktree-root", help="Where to check the branch out"
    ),
    command: str | None = typer.Option(
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
    runs_dir: Path | None = typer.Option(None, "--runs-dir", help="Override run store root"),
    output_json: bool = typer.Option(False, "--json", help="Print JSON output"),
) -> None:
    """Turn a blocked campaign into development work on an experimentation branch.

    A campaign that stopped with `blocked_on_change` needs code, not
    parameters. This creates `exp/<campaign>-<digest>` in its own worktree and
    gives the ticket to a development flow. Exit 3 means the flow failed.
    """
    from ai_experiments.handoff import hand_off

    store = FilesystemRunStore(runs_dir)
    flow = _flow_command(command, run_flow)
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
        for request in _blocked_campaigns(store, campaign_id)
    ]
    _report_handoffs(results, output_json=output_json)

    if any(r.status == "failed" for r in results):
        raise typer.Exit(code=EXIT_BACKEND_UNAVAILABLE)


def _blocked_campaigns(store: FilesystemRunStore, campaign_id: str | None) -> list[ChangeRequest]:
    """Return the change requests this invocation will hand off.

    Naming a campaign that is not blocked is a mistake worth reporting; asking
    for all of them and finding none is just an empty afternoon.
    """
    from ai_experiments.monitoring.escalation import list_change_requests

    pending = list_change_requests(store)
    if not campaign_id:
        return pending
    matching = [r for r in pending if r.campaign_id == campaign_id]
    if not matching:
        not_found("blocked campaign", campaign_id, "see `iax escalations`")
    return matching


def _flow_command(command: str | None, run_flow: bool) -> list[str]:
    """Build the development flow's argv.

    `--run` only speaks to the default command, whose `--no-run` it drops. A
    caller who wrote their own `--command` already said what they wanted.
    """
    import shlex

    from ai_experiments.handoff import DEFAULT_COMMAND

    if command:
        return shlex.split(command)
    flow = list(DEFAULT_COMMAND)
    return [part for part in flow if part != "--no-run"] if run_flow else flow


def _report_handoffs(results: list[HandoffResult], *, output_json: bool) -> None:
    if output_json:
        _echo_json(results)
    elif not results:
        typer.echo("No campaign is blocked on a code change.")
    else:
        for result in results:
            _print_handoff(result)


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
    runs_dir: Path | None = typer.Option(None, "--runs-dir", help="Override run store root"),
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
        typer.echo(f"  {entry.path}  ({entry.size_bytes} bytes)")


@app.command(cls=IaxCommand)
def repro(
    run_id: str = typer.Argument(...),
    runs_dir: Path | None = typer.Option(None, "--runs-dir", help="Override run store root"),
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
    # Typed, not the dict `read_repro` used to hand back: `bundle_dir` is
    # presentation-only, so it belongs on the composed model rather than being
    # poked into the persisted bundle (CES-79).
    _echo_json(
        ReproBundleInfo(**context.model_dump(), bundle_dir=str(store.run_dir(run_id) / "repro"))
    )


@app.command(cls=IaxCommand)
def rerun(
    run_id: str = typer.Argument(..., help="Run to repeat exactly"),
    runs_dir: Path | None = typer.Option(None, "--runs-dir", help="Override run store root"),
    portable: bool = typer.Option(
        False,
        "--portable",
        help="Resubmit the manifest as it was authored, re-resolving a "
        "relative working_dir against the current directory, instead of "
        "repeating the exact paths recorded at submit time",
    ),
    output_json: bool = typer.Option(False, "--json", help="Print JSON output"),
) -> None:
    """Resubmit a run's persisted manifest, with its params baked in.

    Warns when the current git state differs from the one recorded at
    submit time.

    By default this repeats the run exactly, on the paths it actually used.
    ``--portable`` is for the other machine: it takes the manifest as
    submitted, whose relative ``working_dir`` is what makes it movable.
    """
    from ai_experiments.repro import current_git_sha, read_repro

    store = FilesystemRunStore(runs_dir)
    manifest = store.read_manifest(run_id, source=portable)
    if manifest is None:
        _require_run(store, run_id)
        not_found("persisted manifest for run", run_id)

    recorded = read_repro(store.run_dir(run_id))
    recorded_sha = recorded.git_sha if recorded else None
    now_sha = current_git_sha(manifest.workload.working_dir)
    if recorded_sha and now_sha and recorded_sha != now_sha:
        typer.echo(
            f"Warning: working dir is at {now_sha[:12]} but the run was submitted "
            f"from {recorded_sha[:12]} — check out that commit for an exact rerun "
            f"(diff of uncommitted changes, if any: "
            f"{store.run_dir(run_id) / 'repro' / 'diff.patch'})",
            err=True,
        )
    if recorded and recorded.git_dirty:
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


@app.command(cls=IaxCommand)
def leaderboard(
    runs_dir: Path | None = typer.Option(None, "--runs-dir", help="Override run store root"),
    output_json: bool = typer.Option(False, "--json", help="Print JSON output"),
) -> None:
    """Campaigns ranked by their best objective value."""
    from ai_experiments.leaderboard import leaderboard_rows
    from ai_experiments.store.campaign import CampaignStore

    store = FilesystemRunStore(runs_dir)
    rows = leaderboard_rows(CampaignStore(store.root))
    if output_json:
        _echo_json(rows)
        return
    for row in rows:
        cost = f" ~${row.estimated_cost}" if row.estimated_cost is not None else ""
        typer.echo(
            f"{row.mode} {row.metric}={row.best_value:.6g}  "
            f"{row.name} ({row.campaign_id}, {row.trials} trials, "
            f"{row.gpu_hours:g} gpu-h{cost})  params={row.best_params}"
        )
