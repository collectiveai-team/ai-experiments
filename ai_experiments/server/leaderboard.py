"""Leaderboard route: campaigns ranked across the store by best objective value."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter

from ai_experiments.leaderboard import leaderboard_rows

# FastAPI resolves the response annotation at runtime to build the response
# model, so this one cannot move under TYPE_CHECKING.
from ai_experiments.schemas import LeaderboardRow  # noqa: TC001

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
        return leaderboard_rows(campaign_store)

    return router
