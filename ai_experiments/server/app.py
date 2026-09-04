"""REST API + dashboard over the run and campaign stores.

The server is read-mostly: it reads the same filesystem stores the daemon and
CLI write, so it can run on any machine that sees the run store. Mutations are
limited to cancelling runs and stopping campaigns.

There is no authentication. On the default loopback bind that is fine -- the
only caller is the person at the keyboard. Bound to a reachable address it is
not: anyone who can route to the port could cancel a week of training. So a
non-loopback bind refuses mutations unless the operator asks for them by name
(#24).
"""

from __future__ import annotations

import ipaddress
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse

from ai_experiments.backends.factory import backend_for_run
from ai_experiments.monitoring.escalation import list_escalations
from ai_experiments.monitoring.rules import diagnose_run
from ai_experiments.orchestrator import CampaignOrchestrator
from ai_experiments.planner.analysis import summarize_campaign
from ai_experiments.store import FilesystemRunStore
from ai_experiments.store.campaign import CampaignStore

STATIC_DIR = Path(__file__).parent / "static"

#: What the operator has to pass to `iax serve` to mutate over the network.
ALLOW_REMOTE_FLAG = "--allow-remote-mutations"


def is_loopback(host: str) -> bool:
    """Is this bind address reachable only from this machine?

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
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "runs_root": str(run_store.root),
            "mutations": "allowed" if mutations_allowed else "read-only",
        }

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
        _require_mutations()
        _ensure_run(run_store, run_id)
        backend_for_run(run_store, run_id).cancel(run_id)
        return {"run_id": run_id, "cancelled": True}

    @app.get("/api/runs/{run_id}/artifacts")
    def run_artifacts(run_id: str) -> list[dict[str, Any]]:
        _ensure_run(run_store, run_id)
        return run_store.list_artifacts(run_id)

    @app.get("/api/runs/{run_id}/artifacts/{artifact_path:path}")
    def run_artifact_download(run_id: str, artifact_path: str) -> FileResponse:
        _ensure_run(run_store, run_id)
        root = run_store.artifacts_dir(run_id).resolve()
        target = (root / artifact_path).resolve()
        if not target.is_relative_to(root) or not target.is_file():
            raise HTTPException(status_code=404, detail="unknown artifact")
        return FileResponse(target, filename=target.name)

    @app.get("/api/runs/{run_id}/repro")
    def run_repro(run_id: str) -> dict[str, Any]:
        from ai_experiments.repro import read_repro

        _ensure_run(run_store, run_id)
        context = read_repro(run_store.run_dir(run_id))
        if context is None:
            raise HTTPException(status_code=404, detail="no repro bundle")
        context["has_diff"] = (
            run_store.run_dir(run_id) / "repro" / "diff.patch"
        ).exists()
        return context

    @app.get("/api/leaderboard")
    def leaderboard() -> list[dict[str, Any]]:
        """Campaigns ranked by their best objective value, grouped per metric
        client-side (each row carries metric + mode)."""
        rows: list[dict[str, Any]] = []
        for campaign_id in campaign_store.list_campaigns():
            state = campaign_store.read_state(campaign_id)
            goal = campaign_store.read_goal(campaign_id)
            summary = summarize_campaign(state, goal)
            if summary["best"] is None:
                continue
            rows.append(
                {
                    "campaign_id": campaign_id,
                    "name": state.name,
                    "status": state.status,
                    "metric": goal.objective.metric,
                    "mode": goal.objective.mode,
                    "best_value": summary["best"]["objective_value"],
                    "best_params": summary["best"]["params"],
                    "best_run_id": summary["best"]["run_id"],
                    "trials": len(state.trials),
                    "gpu_hours": summary["gpu_hours"],
                    "estimated_cost": summary["estimated_cost"],
                    "updated_at": state.updated_at.isoformat(),
                }
            )
        rows.sort(
            key=lambda r: (
                r["metric"],
                -r["best_value"] if r["mode"] == "max" else r["best_value"],
            )
        )
        return rows

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
        _require_mutations()
        _ensure_campaign(campaign_store, campaign_id)
        orchestrator = CampaignOrchestrator(run_store, campaign_store)
        state = orchestrator.stop(campaign_id)
        return state.model_dump(mode="json")

    @app.post("/api/campaigns/{campaign_id}/pause")
    def campaign_pause(campaign_id: str) -> dict[str, Any]:
        _require_mutations()
        _ensure_campaign(campaign_store, campaign_id)
        orchestrator = CampaignOrchestrator(run_store, campaign_store)
        try:
            return orchestrator.pause(campaign_id).model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc))

    @app.post("/api/campaigns/{campaign_id}/resume")
    def campaign_resume(campaign_id: str) -> dict[str, Any]:
        _require_mutations()
        _ensure_campaign(campaign_store, campaign_id)
        orchestrator = CampaignOrchestrator(run_store, campaign_store)
        try:
            return orchestrator.resume(campaign_id).model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc))

    @app.get("/api/escalations")
    def escalations() -> list[dict[str, Any]]:
        return [item.model_dump(mode="json") for item in list_escalations(run_store)]

    @app.get("/api/clusters")
    def clusters() -> list[dict[str, Any]]:
        """Configured Ray cluster profiles with live reachability."""
        from ai_experiments.clusters import (
            ClusterConfigError,
            cluster_status,
            load_clusters,
        )

        try:
            profiles = load_clusters()
        except ClusterConfigError as exc:
            return [{"name": "(config error)", "reachable": False, "error": str(exc)}]
        return [
            {
                **cluster_status(profile, timeout=2.0),
                "provider": profile.provider,
                "description": profile.description,
            }
            for profile in profiles.values()
        ]

    return app


def _ensure_run(store: FilesystemRunStore, run_id: str) -> None:
    if not store.run_dir(run_id).exists():
        raise HTTPException(status_code=404, detail=f"unknown run: {run_id}")


def _ensure_campaign(store: CampaignStore, campaign_id: str) -> None:
    if not (store.campaign_dir(campaign_id) / "state.json").exists():
        raise HTTPException(status_code=404, detail=f"unknown campaign: {campaign_id}")
