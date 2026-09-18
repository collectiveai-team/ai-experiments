"""Leaderboard route: campaigns ranked across the store by best objective value."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from fastapi import APIRouter

from ai_experiments.planner.analysis import summarize_campaign
from ai_experiments.schemas import LeaderboardRow

if TYPE_CHECKING:
    from ai_experiments.store.campaign import CampaignStore


def build_leaderboard_router(campaign_store: CampaignStore) -> APIRouter:
    """Build the `/api/leaderboard` router, closing over the campaign store."""
    router = APIRouter()

    @router.get("/api/leaderboard")
    def leaderboard() -> list[LeaderboardRow]:
        """Campaigns ranked by their best objective value.

        Grouped per metric client-side (each row carries metric + mode).
        """
        rows: list[LeaderboardRow] = []
        for campaign_id in campaign_store.list_campaigns():
            state = campaign_store.read_state(campaign_id)
            goal = campaign_store.read_goal(campaign_id)
            summary = summarize_campaign(state, goal)
            if summary.best is None:
                continue
            rows.append(
                LeaderboardRow(
                    campaign_id=campaign_id,
                    name=state.name,
                    status=state.status,
                    metric=goal.objective.metric,
                    mode=goal.objective.mode,
                    # best_trial() only sets `best` from trials with a non-None,
                    # finite objective_value; cast makes that invariant visible here.
                    best_value=cast("float", summary.best.objective_value),
                    best_params=summary.best.params,
                    best_run_id=summary.best.run_id,
                    trials=len(state.trials),
                    gpu_hours=summary.gpu_hours,
                    estimated_cost=summary.estimated_cost,
                    updated_at=state.updated_at.isoformat(),
                )
            )
        rows.sort(
            key=lambda r: (
                r.metric,
                -r.best_value if r.mode == "max" else r.best_value,
            )
        )
        return rows

    return router
