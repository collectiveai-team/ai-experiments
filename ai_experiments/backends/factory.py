from __future__ import annotations

from ai_experiments.backends.base import ExperimentBackend
from ai_experiments.backends.local import LocalBackend
from ai_experiments.backends.ray import RayBackend
from ai_experiments.schemas import BackendName
from ai_experiments.store import FilesystemRunStore


def get_backend(
    name: BackendName,
    store: FilesystemRunStore | None = None,
    address: str | None = None,
) -> ExperimentBackend:
    if name == "local":
        return LocalBackend(store=store)
    if name == "ray":
        return RayBackend(store=store, address=address)
    raise ValueError(f"Unsupported experiment backend: {name}")
