"""Artifact and repro-bundle routes for a run: listing, download, repro detail."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from ai_experiments.responses import ArtifactEntry, RunReproDetail
from ai_experiments.server.runs import ensure_run

if TYPE_CHECKING:
    from ai_experiments.store import FilesystemRunStore


def build_artifacts_router(store: FilesystemRunStore) -> APIRouter:
    """Build the artifact/repro router, closing over the run store it reads."""
    router = APIRouter()

    @router.get("/api/runs/{run_id}/artifacts")
    def run_artifacts(run_id: str) -> list[ArtifactEntry]:
        ensure_run(store, run_id)
        return store.list_artifacts(run_id)

    @router.get("/api/runs/{run_id}/artifacts/{artifact_path:path}")
    def run_artifact_download(run_id: str, artifact_path: str) -> FileResponse:
        ensure_run(store, run_id)
        root = store.artifacts_dir(run_id).resolve()
        target = (root / artifact_path).resolve()
        if not target.is_relative_to(root) or not target.is_file():
            raise HTTPException(status_code=404, detail="unknown artifact")
        return FileResponse(target, filename=target.name)

    @router.get("/api/runs/{run_id}/repro")
    def run_repro(run_id: str) -> RunReproDetail:
        from ai_experiments.repro import read_repro

        ensure_run(store, run_id)
        context = read_repro(store.run_dir(run_id))
        if context is None:
            raise HTTPException(status_code=404, detail="no repro bundle")
        has_diff = (store.run_dir(run_id) / "repro" / "diff.patch").exists()
        return RunReproDetail(**context.model_dump(), has_diff=has_diff)

    return router
