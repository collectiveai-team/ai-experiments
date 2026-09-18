from __future__ import annotations

import json

import typer
from pydantic import BaseModel

from ai_experiments.backends.factory import backend_for_run
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


# Registration imports, deliberately last: each command module decorates one of the
# three Typer apps above and imports _echo_json/_backend_for_run from here, so every
# name above must already be bound. Moving this line up raises ImportError from a
# partially initialized module at interpreter start -- loud, and it takes the whole
# suite's collection with it, so this ordering cannot regress silently.
from ai_experiments.cli import campaigns, clusters, goals, runs, serving  # noqa: E402,F401
