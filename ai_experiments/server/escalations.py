"""Escalation-listing route: runs the monitoring loop has flagged for a human."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import APIRouter

from ai_experiments.monitoring.escalation import list_escalations

if TYPE_CHECKING:
    from ai_experiments.store import FilesystemRunStore


def build_escalations_router(store: FilesystemRunStore) -> APIRouter:
    """Build the `/api/escalations` router, closing over the run store it reads."""
    router = APIRouter()

    @router.get("/api/escalations")
    def escalations() -> list[dict[str, Any]]:
        return [request.model_dump(mode="json") for request in list_escalations(store)]

    return router
