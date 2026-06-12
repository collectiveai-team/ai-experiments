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


def backend_for_run(store: FilesystemRunStore, run_id: str) -> ExperimentBackend:
    """Backend for an existing run, recovering the Ray address it was
    submitted with (manifest `backend_address` → RAY_ADDRESS env → default)."""
    status = store.read_status(run_id)
    address: str | None = None
    if status.backend == "ray":
        manifest = store.read_manifest(run_id)
        if manifest is not None:
            address = manifest.backend_address
    return get_backend(status.backend, store=store, address=address)
