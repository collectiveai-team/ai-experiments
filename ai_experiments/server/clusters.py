"""Cluster-profile route: configured Ray cluster profiles with live reachability.

Unlike the other resource routers, this one needs no store — cluster profiles
are read from config, not the run store — so `build_clusters_router` takes no
arguments. It keeps the same factory shape as its siblings for consistency.
"""

from __future__ import annotations

from fastapi import APIRouter

from ai_experiments.clusters import ClusterSummary


def build_clusters_router() -> APIRouter:
    router = APIRouter()

    @router.get("/api/clusters")
    def clusters() -> list[ClusterSummary]:
        """Return configured Ray cluster profiles with live reachability."""
        from ai_experiments.clusters import ClusterConfigError, cluster_status, load_clusters

        try:
            profiles = load_clusters()
        except ClusterConfigError as exc:
            return [ClusterSummary(name="(config error)", reachable=False, error=str(exc))]
        return [
            ClusterSummary(
                **cluster_status(profile, timeout=2.0).model_dump(),
                provider=profile.provider,
                description=profile.description,
            )
            for profile in profiles.values()
        ]

    return router
