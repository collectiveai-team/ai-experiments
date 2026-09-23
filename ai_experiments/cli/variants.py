"""`iax campaign variant` / `variants`: the way an agent's code reaches a campaign.

Whole files on disk, not text on a command line: what runs is exactly what was
written, and it stays readable afterwards.
"""

from __future__ import annotations

from pathlib import Path  # Typer resolves this annotation at runtime

import typer

from ai_experiments.cli import IaxCommand, _echo_json, _require_campaign, campaign_app
from ai_experiments.cli.campaigns import _orchestrator
from ai_experiments.cli_support import invalid_input
from ai_experiments.store import FilesystemRunStore


@campaign_app.command("variant", cls=IaxCommand)
def campaign_variant(
    campaign_id: str = typer.Argument(...),
    edit: list[str] = typer.Option(
        ...,
        "--edit",
        help="<path-in-workload>=<local file whose contents to write>; repeatable",
    ),
    hypothesis: str = typer.Option(
        "", "--hypothesis", help="What this change is supposed to improve"
    ),
    rationale: str = typer.Option("", "--rationale", help="Why it should work"),
    parent: str | None = typer.Option(None, "--parent", help="The variant this one builds on"),
    runs_dir: Path | None = typer.Option(None, "--runs-dir"),
    output_json: bool = typer.Option(False, "--json"),
) -> None:
    """Copy the workload, apply whole-file edits, and smoke-check the result.

    The exit code is the smoke check's verdict — 2 means the variant does not
    start and was discarded.
    """
    from ai_experiments.improve.variants import VariantEdit

    edits: list[VariantEdit] = []
    for item in edit:
        target, _, source = item.partition("=")
        if not target or not source:
            invalid_input(f"--edit expects <path>=<file>, got {item!r}")
        try:
            content = Path(source).read_text()
        except OSError as exc:
            invalid_input(f"cannot read the contents for {target}: {exc}")
        edits.append(VariantEdit(path=target, content=content))

    _require_campaign(FilesystemRunStore(runs_dir), campaign_id)
    try:
        record = _orchestrator(runs_dir).add_variant(
            campaign_id,
            edits,
            hypothesis=hypothesis,
            rationale=rationale,
            parent=parent,
        )
    except ValueError as exc:
        invalid_input(f"variant rejected: {exc}")

    if record.smoke_ok is False:
        fate = (
            f"and its directory could not be removed ({record.discard_error}); "
            f"delete {record.root} by hand"
            if record.discard_error
            else "and was discarded"
        )
        invalid_input(
            f"variant {record.variant_id} failed its smoke check {fate}:\n{record.smoke_output}",
            details={"variant_id": record.variant_id},
        )
    if output_json:
        _echo_json(record)
    else:
        verdict = "smoke ok" if record.smoke_ok else "not smoke-checked"
        typer.echo(f"{record.variant_id}  {verdict}  edited {', '.join(record.edited_paths)}")
        typer.echo(
            f"  iax campaign suggest {campaign_id} --params '{{...}}' --variant {record.variant_id}"
        )


@campaign_app.command("variants", cls=IaxCommand)
def campaign_variants(
    campaign_id: str = typer.Argument(...),
    runs_dir: Path | None = typer.Option(None, "--runs-dir"),
    output_json: bool = typer.Option(False, "--json"),
) -> None:
    """Every code variant this campaign was offered, accepted or rejected."""
    from ai_experiments.store.campaign import CampaignStore

    store = FilesystemRunStore(runs_dir)
    _require_campaign(store, campaign_id)
    records = CampaignStore(store.root).read_variants(campaign_id)
    if output_json:
        _echo_json([record.model_dump(mode="json") for record in records])
        return
    if not records:
        typer.echo("No variants proposed to this campaign.")
        return
    for record in records:
        verdict = {True: "smoke ok", False: "rejected", None: "unchecked"}[record.smoke_ok]
        if record.discard_error:
            verdict = "left on disk"
        typer.echo(
            f"{record.variant_id}  {verdict:<9} {', '.join(record.edited_paths)}"
            + (f"  — {record.hypothesis}" if record.hypothesis else "")
        )
