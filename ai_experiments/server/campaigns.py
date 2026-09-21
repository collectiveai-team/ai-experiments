"""Campaign-scoped routes: listing, detail, and lifecycle actions.

The leaderboard is a campaign-adjacent read (it ranks across all campaigns)
but lives in `leaderboard.py` — folding it in here would push this router's
handler count back over the cyclomatic-complexity ceiling, which is exactly
the failure mode this split exists to avoid.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException

from ai_experiments.orchestrator import CampaignOrchestrator
from ai_experiments.planner.analysis import summarize_campaign
from ai_experiments.schemas import CampaignDetail, CampaignState

if TYPE_CHECKING:
    from collections.abc import Callable

    from ai_experiments.store import FilesystemRunStore
    from ai_experiments.store.campaign import CampaignStore


def build_campaigns_router(
    run_store: FilesystemRunStore,
    campaign_store: CampaignStore,
    require_mutations: Callable[[], None],
) -> APIRouter:
    """Build the `/api/campaigns` router.

    Closes over both stores: the campaign store for campaign state, and the
    run store the orchestrator needs to act on a campaign's runs.
    `require_mutations` raises 403 when the server is bound somewhere a
    stranger can reach it (#24).
    """
    router = APIRouter()

    @router.get("/api/campaigns")
    def campaigns() -> list[CampaignState]:
        return [
            campaign_store.read_state(campaign_id)
            for campaign_id in campaign_store.list_campaigns()
        ]

    @router.get("/api/campaigns/{campaign_id}")
    def campaign_detail(campaign_id: str) -> CampaignDetail:
        _ensure_campaign(campaign_store, campaign_id)
        state = campaign_store.read_state(campaign_id)
        goal = campaign_store.read_goal(campaign_id)
        return CampaignDetail(state=state, summary=summarize_campaign(state, goal))

    @router.post("/api/campaigns/{campaign_id}/stop")
    def campaign_stop(campaign_id: str) -> CampaignState:
        require_mutations()
        _ensure_campaign(campaign_store, campaign_id)
        orchestrator = CampaignOrchestrator(run_store, campaign_store)
        return orchestrator.stop(campaign_id)

    @router.post("/api/campaigns/{campaign_id}/pause")
    def campaign_pause(campaign_id: str) -> CampaignState:
        require_mutations()
        _ensure_campaign(campaign_store, campaign_id)
        orchestrator = CampaignOrchestrator(run_store, campaign_store)
        try:
            return orchestrator.pause(campaign_id)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.post("/api/campaigns/{campaign_id}/resume")
    def campaign_resume(campaign_id: str) -> CampaignState:
        require_mutations()
        _ensure_campaign(campaign_store, campaign_id)
        orchestrator = CampaignOrchestrator(run_store, campaign_store)
        try:
            return orchestrator.resume(campaign_id)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    return router


def _ensure_campaign(store: CampaignStore, campaign_id: str) -> None:
    if not (store.campaign_dir(campaign_id) / "state.json").exists():
        raise HTTPException(status_code=404, detail=f"unknown campaign: {campaign_id}")
