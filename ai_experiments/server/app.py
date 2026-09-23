"""REST API + dashboard over the run and campaign stores.

The server is read-mostly: it reads the same filesystem stores the daemon and
CLI write, so it can run on any machine that sees the run store. Mutations are
limited to cancelling runs and stopping campaigns.

Routes are grouped into one `APIRouter` per resource (runs, artifacts,
campaigns, leaderboard, escalations, clusters) in sibling modules, built by
factory functions that take whichever store the resource needs. `create_app`
wires the stores once and includes each router.

There is no authentication. On the default loopback bind that is fine -- the
only caller is the person at the keyboard. Bound to a reachable address it is
not: anyone who can route to the port could cancel a week of training. So a
non-loopback bind refuses mutations unless the operator asks for them by name
(#24). The guard lives here and is handed to the routers that mutate, so the
bind address is decided once rather than re-derived per resource.
"""

from __future__ import annotations

import ipaddress
from pathlib import Path

from fastapi import FastAPI, HTTPException
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

#: What the operator has to pass to `iax serve` to mutate over the network.
ALLOW_REMOTE_FLAG = "--allow-remote-mutations"


def is_loopback(host: str) -> bool:
    """Report whether this bind address is reachable only from this machine.

    A name that is not an address cannot be checked here, so it counts as
    remote: refusing a mutation is recoverable, allowing a stranger's is not.
    """
    if host in {"localhost", ""}:
        return host == "localhost"
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def create_app(
    store: FilesystemRunStore | None = None,
    host: str = "127.0.0.1",
    allow_remote_mutations: bool = False,
) -> FastAPI:
    run_store = store or FilesystemRunStore()
    campaign_store = CampaignStore(run_store.root)
    mutations_allowed = allow_remote_mutations or is_loopback(host)

    app = FastAPI(title="iax dashboard", version="1.0")

    def _require_mutations() -> None:
        if mutations_allowed:
            return
        raise HTTPException(
            status_code=403,
            detail=(
                f"this dashboard is bound to {host} and has no authentication, "
                f"so it serves reads only; restart with {ALLOW_REMOTE_FLAG} to "
                "allow cancel/stop/pause/resume from the network"
            ),
        )

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (STATIC_DIR / "index.html").read_text()

    @app.get("/api/health")
    def health() -> HealthStatus:
        return HealthStatus(
            status="ok",
            runs_root=str(run_store.root),
            mutations="allowed" if mutations_allowed else "read-only",
        )

    app.include_router(build_runs_router(run_store, _require_mutations))
    app.include_router(build_artifacts_router(run_store))
    app.include_router(build_campaigns_router(run_store, campaign_store, _require_mutations))
    app.include_router(build_leaderboard_router(campaign_store))
    app.include_router(build_escalations_router(run_store))
    app.include_router(build_clusters_router())

    return app
