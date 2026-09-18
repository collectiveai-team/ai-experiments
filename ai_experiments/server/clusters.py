"""Cluster-profile route: configured Ray cluster profiles with live reachability.

Unlike the other resource routers, this one needs no store — cluster profiles
are read from config, not the run store — so `build_clusters_router` takes no
arguments. It keeps the same factory shape as its siblings for consistency.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter


def build_clusters_router() -> APIRouter:
    router = APIRouter()

    @router.get("/api/clusters")
    def clusters() -> list[dict[str, Any]]:
        """Return configured Ray cluster profiles with live reachability."""
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
                **cluster_status(profile, timeout=2.0).model_dump(mode="json"),
                "provider": profile.provider,
                "description": profile.description,
            }
            for profile in profiles.values()
        ]

    return router
