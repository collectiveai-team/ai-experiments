"""The `iax` command line, split by resource.

One module per command group; `__init__` owns the Typer apps, the error-contract
command class, and the helpers more than one group needs. Splitting is what keeps
each module under the size and complexity ceilings the single `cli.py` had blown
through at 1397 lines.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import typer
from pydantic import BaseModel
from typer.core import TyperCommand

from ai_experiments.backends.factory import backend_for_run
from ai_experiments.cli_support import (
    IaxError,
    invalid_input,
    not_found,
    report,
)
from ai_experiments.schemas import GoalSpec

if TYPE_CHECKING:
    from ai_experiments.schemas import ExperimentManifest
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

    # `typer.Context` is a subclass of the vendored click context `TyperCommand`
    # declares, so this reads as a narrowing. Naming the parent's class means
    # importing `typer._click.core`, a private path; the public name is worth more
    # than the variance, since typer only ever passes its own Context.
    # pyrefly: ignore[bad-override]
    def invoke(self, ctx: typer.Context) -> Any:
        try:
            return super().invoke(ctx)
        except IaxError as exc:
            report(exc, json_mode=bool(ctx.params.get("output_json")))


def _echo_json(payload: object) -> None:
    typer.echo(json.dumps(_jsonable(payload), indent=2))


def _jsonable(payload: object) -> Any:
    """Reduce models to the plain structures `json.dumps` understands.

    Lists are walked, not just the top level: most stores hand back a list of
    models, and `json.dumps` refuses one as readily as it refuses a single model.
    """
    if isinstance(payload, BaseModel):
        return payload.model_dump(mode="json")
    if isinstance(payload, (list, tuple)):
        return [_jsonable(item) for item in payload]
    return payload


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

    if not (CampaignStore(store.root).campaign_dir(campaign_id) / "state.json").exists():
        not_found(
            "campaign",
            campaign_id,
            hint=(f"run store is {store.root}; list them with `iax campaign list`"),
        )


def _backend_for_run(run_id: str, store: FilesystemRunStore):
    _require_run(store, run_id)
    return backend_for_run(store, run_id)


def _worker_log(store: FilesystemRunStore, run_id: str, tail: int, output_json: bool) -> None:
    """Print the supervisor's own stdout/stderr.

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


def _preflight(source: ExperimentManifest | GoalSpec, strict: bool, refusal: str) -> None:
    """Warn on a workload that cannot start, and stop when asked to.

    Warnings go to stderr so `--json` stdout stays parseable, and they stay
    warnings by default: a Ray workload resolves its entrypoint on the
    cluster, so a binary missing here can still be right (#32).
    """
    from ai_experiments.preflight import WARNING_PREFIX, goal_warnings, workload_warnings

    warnings = workload_warnings(source)
    if isinstance(source, GoalSpec):
        # A goal carries a second kind of defect the manifest cannot have:
        # one that makes the campaign run fine and prove nothing.
        warnings = warnings + goal_warnings(source)
    for warning in warnings:
        typer.echo(f"{WARNING_PREFIX}{warning}", err=True)
    if warnings and strict:
        invalid_input(refusal, details={"warnings": warnings})


# Registration imports, deliberately last: each command module decorates one of the
# four Typer apps above and imports the shared helpers from here, so every name above
# must already be bound. Moving this line up raises ImportError from a partially
# initialized module at interpreter start.
from ai_experiments.cli import (  # noqa: E402,F401
    campaigns,
    clusters,
    goals,
    runs,
    scaffold,
    serving,
    variants,
)
