"""REST API + dashboard over the run and campaign stores.

The server is read-mostly: it reads the same filesystem stores the daemon and
CLI write, so it can run on any machine that sees the run store. Mutations are
limited to cancelling runs and stopping campaigns.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from ai_experiments.backends.factory import backend_for_run
from ai_experiments.monitoring.escalation import list_escalations
from ai_experiments.monitoring.rules import diagnose_run
from ai_experiments.orchestrator import CampaignOrchestrator
from ai_experiments.planner.analysis import summarize_campaign
from ai_experiments.store import FilesystemRunStore
from ai_experiments.store.campaign import CampaignStore

STATIC_DIR = Path(__file__).parent / "static"


def create_app(store: FilesystemRunStore | None = None) -> FastAPI:
    run_store = store or FilesystemRunStore()
    campaign_store = CampaignStore(run_store.root)

    app = FastAPI(title="iax dashboard", version="1.0")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (STATIC_DIR / "index.html").read_text()

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "runs_root": str(run_store.root)}

    @app.get("/api/runs")
    def runs() -> list[dict[str, Any]]:
        return [
            run_store.read_status(run_id).model_dump(mode="json")
            for run_id in sorted(run_store.list_runs())
        ]

    @app.get("/api/runs/{run_id}")
    def run_detail(run_id: str) -> dict[str, Any]:
        _ensure_run(run_store, run_id)
        return run_store.read_status(run_id).model_dump(mode="json")

    @app.get("/api/runs/{run_id}/events")
    def run_events(run_id: str, tail: int = 200) -> list[dict[str, Any]]:
        _ensure_run(run_store, run_id)
        return [
            event.model_dump(mode="json")
            for event in run_store.read_events(run_id, tail=tail)
        ]

    @app.get("/api/runs/{run_id}/metrics")
    def run_metrics(run_id: str, tail: int = 500) -> list[dict[str, Any]]:
        _ensure_run(run_store, run_id)
        return [
            point.model_dump(mode="json")
            for point in run_store.read_metrics(run_id, tail=tail)
        ]

    @app.get("/api/runs/{run_id}/diagnosis")
    def run_diagnosis(run_id: str) -> dict[str, Any]:
        _ensure_run(run_store, run_id)
        return diagnose_run(run_store, run_id).model_dump(mode="json")

    @app.post("/api/runs/{run_id}/cancel")
    def run_cancel(run_id: str) -> dict[str, Any]:
        _ensure_run(run_store, run_id)
        backend_for_run(run_store, run_id).cancel(run_id)
        return {"run_id": run_id, "cancelled": True}

    @app.get("/api/campaigns")
    def campaigns() -> list[dict[str, Any]]:
        return [
            campaign_store.read_state(campaign_id).model_dump(mode="json")
            for campaign_id in campaign_store.list_campaigns()
        ]

    @app.get("/api/campaigns/{campaign_id}")
    def campaign_detail(campaign_id: str) -> dict[str, Any]:
        _ensure_campaign(campaign_store, campaign_id)
        state = campaign_store.read_state(campaign_id)
        goal = campaign_store.read_goal(campaign_id)
        return {
            "state": state.model_dump(mode="json"),
            "summary": summarize_campaign(state, goal),
        }

    @app.post("/api/campaigns/{campaign_id}/stop")
    def campaign_stop(campaign_id: str) -> dict[str, Any]:
        _ensure_campaign(campaign_store, campaign_id)
        orchestrator = CampaignOrchestrator(run_store, campaign_store)
        state = orchestrator.stop(campaign_id)
        return state.model_dump(mode="json")

    @app.get("/api/escalations")
    def escalations() -> list[dict[str, Any]]:
        return [
            request.model_dump(mode="json") for request in list_escalations(run_store)
        ]

    return app


def _ensure_run(store: FilesystemRunStore, run_id: str) -> None:
    if not store.run_dir(run_id).exists():
        raise HTTPException(status_code=404, detail=f"unknown run: {run_id}")


def _ensure_campaign(store: CampaignStore, campaign_id: str) -> None:
    if not (store.campaign_dir(campaign_id) / "state.json").exists():
        raise HTTPException(status_code=404, detail=f"unknown campaign: {campaign_id}")
