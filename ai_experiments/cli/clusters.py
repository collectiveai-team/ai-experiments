from __future__ import annotations

from pathlib import Path  # noqa: TC003  # Typer resolves this annotation at runtime

import typer

from ai_experiments.cli import _echo_json, cluster_app

# --- cluster commands ---------------------------------------------------------


@cluster_app.command("list")
def cluster_list(
    config: Path | None = typer.Option(None, "--config", help="clusters.yaml path"),
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
    config: Path | None = typer.Option(None, "--config", help="clusters.yaml path"),
) -> None:
    from ai_experiments.clusters import cluster_status, get_cluster

    _echo_json(cluster_status(get_cluster(name, config)))


@cluster_app.command("up")
def cluster_up_cmd(
    name: str = typer.Argument(...),
    config: Path | None = typer.Option(None, "--config", help="clusters.yaml path"),
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
    config: Path | None = typer.Option(None, "--config", help="clusters.yaml path"),
) -> None:
    """Tear down a cloud cluster via `ray down`."""
    from ai_experiments.clusters import cluster_down, get_cluster

    result = cluster_down(get_cluster(name, config))
    typer.echo(result.stdout)
    if result.returncode != 0:
        typer.echo(result.stderr, err=True)
        raise typer.Exit(code=result.returncode)
