"""Named Ray cluster profiles for local and cloud (AWS/GCP/Azure) clusters.

Ray is the common execution substrate; clouds differ only in how the cluster
comes up. Provisioning delegates to Ray's own cluster launcher (``ray up``
with a provider-specific cluster YAML) rather than reimplementing cloud APIs.
A profile names a cluster, its dashboard address, and optionally the launcher
config used by ``iax cluster up/down``.

Config file (first match wins): ``$IAX_CLUSTERS``, ``./clusters.yaml``,
``~/.config/iax/clusters.yaml``::

    clusters:
      vader:
        provider: local
        address: http://vader:8265
      aws-gpu:
        provider: aws
        cluster_config: infra/ray-aws.yaml
        address: http://10.0.0.5:8265
"""

from __future__ import annotations

import json
import os
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import ValidationError

from ai_experiments.schema_errors import describe
from ai_experiments.schemas import ConfigModel

ClusterProvider = Literal["local", "aws", "gcp", "azure"]


class ClusterProfile(ConfigModel):
    name: str
    provider: ClusterProvider = "local"
    address: str | None = None
    cluster_config: str | None = None
    description: str = ""


class ClusterConfigError(RuntimeError):
    pass


def clusters_config_path(explicit: str | Path | None = None) -> Path | None:
    if explicit is not None:
        return Path(explicit)
    env = os.environ.get("IAX_CLUSTERS")
    if env:
        return Path(env)
    for candidate in (
        Path("clusters.yaml"),
        Path.home() / ".config" / "iax" / "clusters.yaml",
    ):
        if candidate.exists():
            return candidate
    return None


def load_clusters(path: str | Path | None = None) -> dict[str, ClusterProfile]:
    config_path = clusters_config_path(path)
    if config_path is None:
        return {}
    if not config_path.exists():
        raise ClusterConfigError(f"clusters config not found: {config_path}")
    with config_path.open() as fh:
        raw = yaml.safe_load(fh) or {}
    entries = raw.get("clusters", raw)
    if not isinstance(entries, dict):
        raise ClusterConfigError(
            f"{config_path}: expected a `clusters:` mapping of name -> profile"
        )
    profiles: dict[str, ClusterProfile] = {}
    for name, body in entries.items():
        try:
            profiles[str(name)] = ClusterProfile(name=str(name), **(body or {}))
        except ValidationError as exc:
            raise ClusterConfigError(
                f"{config_path}: cluster '{name}': {describe(ClusterProfile, exc)}"
            ) from exc
    return profiles


def get_cluster(name: str, path: str | Path | None = None) -> ClusterProfile:
    profiles = load_clusters(path)
    if name not in profiles:
        known = ", ".join(sorted(profiles)) or "(none configured)"
        raise ClusterConfigError(f"unknown cluster '{name}'; known: {known}")
    return profiles[name]


def resolve_cluster_address(name: str, path: str | Path | None = None) -> str:
    profile = get_cluster(name, path)
    if not profile.address:
        raise ClusterConfigError(
            f"cluster '{name}' has no address configured; "
            "set `address:` in clusters.yaml (the Ray dashboard URL)"
        )
    return profile.address


def cluster_status(profile: ClusterProfile, timeout: float = 5.0) -> dict[str, Any]:
    """Ping the Ray dashboard. Network-only; no Ray dependency needed."""
    if not profile.address:
        return {
            "name": profile.name,
            "reachable": False,
            "error": "no address configured",
        }
    url = profile.address.rstrip("/") + "/api/version"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310
            payload = json.loads(response.read().decode())
        return {
            "name": profile.name,
            "reachable": True,
            "address": profile.address,
            "ray_version": payload.get("ray_version"),
        }
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return {
            "name": profile.name,
            "reachable": False,
            "address": profile.address,
            "error": str(exc),
        }


def cluster_up(profile: ClusterProfile) -> subprocess.CompletedProcess[str]:
    return _ray_launcher(profile, "up")


def cluster_down(profile: ClusterProfile) -> subprocess.CompletedProcess[str]:
    return _ray_launcher(profile, "down")


def _ray_launcher(
    profile: ClusterProfile, action: Literal["up", "down"]
) -> subprocess.CompletedProcess[str]:
    if profile.provider == "local":
        raise ClusterConfigError(
            f"cluster '{profile.name}' is local; start it with `ray start --head "
            "--dashboard-host 0.0.0.0` on the target machine"
        )
    if not profile.cluster_config:
        raise ClusterConfigError(
            f"cluster '{profile.name}' has no cluster_config; point it at a Ray "
            f"cluster launcher YAML for {profile.provider}"
        )
    config = Path(profile.cluster_config)
    if not config.exists():
        raise ClusterConfigError(f"cluster_config not found: {config}")
    return subprocess.run(
        ["ray", action, str(config), "-y"],
        capture_output=True,
        text=True,
        check=False,
    )
