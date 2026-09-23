from __future__ import annotations

from pathlib import Path

import typer

from ai_experiments.cli import (
    IaxCommand,
    _require_run,
    new_app,
)
from ai_experiments.cli_support import (
    invalid_input,
    not_found,
)
from ai_experiments.store import FilesystemRunStore

_NEXT_STEP = {
    "manifest": "iax validate {path} && iax submit {path} --json",
    "goal": "iax campaign validate {path} && iax loop {path}",
    "workload": "point a manifest's workload.args at {path}",
}


@new_app.command("manifest", cls=IaxCommand)
def new_manifest(
    path: Path = typer.Argument(Path("experiment.yaml"), help="File (or directory) to write"),
    from_run: str | None = typer.Option(
        None, "--from-run", help="Rebuild the manifest of an existing run instead"
    ),
    runs_dir: Path | None = typer.Option(None, "--runs-dir", help="Override run store root"),
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


def _scaffold_write(kind: str, path: Path, force: bool) -> None:
    from ai_experiments import scaffold

    try:
        target = scaffold.write(kind, path, force=force)  # type: ignore[arg-type]
    except scaffold.ScaffoldError as exc:
        invalid_input(str(exc))
    typer.echo(f"Wrote {target}")
    typer.echo(f"Next: {_NEXT_STEP[kind].format(path=target)}")
