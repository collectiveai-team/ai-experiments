"""REST API + dashboard over the run and campaign stores.

The server is read-mostly: it reads the same filesystem stores the daemon and
CLI write, so it can run on any machine that sees the run store. Mutations are
limited to cancelling runs and stopping campaigns.

Routes are grouped into one `APIRouter` per resource (runs, artifacts,
campaigns, leaderboard, escalations, clusters) in sibling modules, built by
factory functions that take whichever store the resource needs. `create_app`
wires the stores once and includes each router.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from ai_experiments.schemas import HealthStatus
from ai_experiments.server.artifacts import build_artifacts_router
from ai_experiments.server.campaigns import build_campaigns_router
from ai_experiments.server.clusters import build_clusters_router
from ai_experiments.server.escalations import build_escalations_router
from ai_experiments.server.leaderboard import build_leaderboard_router
from ai_experiments.server.runs import build_runs_router
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
    def health() -> HealthStatus:
        return HealthStatus(status="ok", runs_root=str(run_store.root))

    app.include_router(build_runs_router(run_store))
    app.include_router(build_artifacts_router(run_store))
    app.include_router(build_campaigns_router(run_store, campaign_store))
    app.include_router(build_leaderboard_router(campaign_store))
    app.include_router(build_escalations_router(run_store))
    app.include_router(build_clusters_router())

    return app
