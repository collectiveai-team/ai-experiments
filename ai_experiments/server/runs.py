"""Run-scoped routes: status, events, metrics, diagnosis, cancel.

Artifact listing/download and the repro bundle are close cousins of these
routes but live in `artifacts.py` — folding them in here would push this
router's handler count back over the cyclomatic-complexity ceiling, which is
exactly the failure mode this split exists to avoid.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, HTTPException

from ai_experiments.backends.factory import backend_for_run
from ai_experiments.monitoring.rules import diagnose_run
from ai_experiments.schemas import CancelAck, DiagnosisReport, RunEvent, RunStatus

if TYPE_CHECKING:
    from ai_experiments.store import FilesystemRunStore


def build_runs_router(store: FilesystemRunStore) -> APIRouter:
    """Build the `/api/runs` router, closing over the run store it reads."""
    router = APIRouter()

    @router.get("/api/runs")
    def runs() -> list[RunStatus]:
        return [store.read_status(run_id) for run_id in sorted(store.list_runs())]

    @router.get("/api/runs/{run_id}")
    def run_detail(run_id: str) -> RunStatus:
        ensure_run(store, run_id)
        return store.read_status(run_id)

    @router.get("/api/runs/{run_id}/events")
    def run_events(run_id: str, tail: int = 200) -> list[RunEvent]:
        ensure_run(store, run_id)
        return store.read_events(run_id, tail=tail)

    @router.get("/api/runs/{run_id}/metrics")
    def run_metrics(run_id: str, tail: int = 500) -> list[dict[str, Any]]:
        ensure_run(store, run_id)
        return [point.model_dump(mode="json") for point in store.read_metrics(run_id, tail=tail)]

    @router.get("/api/runs/{run_id}/diagnosis")
    def run_diagnosis(run_id: str) -> DiagnosisReport:
        ensure_run(store, run_id)
        return diagnose_run(store, run_id)

    @router.post("/api/runs/{run_id}/cancel")
    def run_cancel(run_id: str) -> CancelAck:
        ensure_run(store, run_id)
        backend_for_run(store, run_id).cancel(run_id)
        return CancelAck(run_id=run_id, cancelled=True)

    return router


def ensure_run(store: FilesystemRunStore, run_id: str) -> None:
    """Raise 404 unless `run_id` exists in `store`.

    Shared with `artifacts.py`, so it is not underscore-prefixed even though
    it is only ever reached through the routers, not part of the HTTP
    surface itself.
    """
    if not store.run_dir(run_id).exists():
        raise HTTPException(status_code=404, detail=f"unknown run: {run_id}")
