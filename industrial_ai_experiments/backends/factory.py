from __future__ import annotations

from industrial_ai_experiments.backends.base import ExperimentBackend
from industrial_ai_experiments.backends.local import LocalBackend
from industrial_ai_experiments.backends.ray import RayBackend
from industrial_ai_experiments.schemas import BackendName
from industrial_ai_experiments.store import FilesystemRunStore


def get_backend(name: BackendName, store: FilesystemRunStore | None = None) -> ExperimentBackend:
    if name == "local":
        return LocalBackend(store=store)
    if name == "ray":
        return RayBackend(store=store)
    raise ValueError(f"Unsupported experiment backend: {name}")
