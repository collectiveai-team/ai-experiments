from __future__ import annotations

import pytest
from pydantic import ValidationError

from ai_experiments.schemas import ExperimentManifest, WorkloadSpec
from ai_experiments.store import FilesystemRunStore


def _manifest(**overrides: object) -> ExperimentManifest:
    data = {
        "experiment": "smoke",
        "backend": "ray",
        "workload": WorkloadSpec(entrypoint="python train.py"),
    }
    data.update(overrides)
    return ExperimentManifest(**data)


def test_backend_address_round_trips_yaml(tmp_path):
    manifest = _manifest(backend_address=" https://ray.example.com ")
    path = tmp_path / "manifest.yaml"
    path.write_text(manifest.to_yaml())

    restored = ExperimentManifest.from_yaml(path)

    assert restored.backend_address == "https://ray.example.com"


def test_backend_address_is_persisted_in_run_manifest(tmp_path):
    store = FilesystemRunStore(tmp_path / "runs")
    manifest = _manifest(backend_address="https://ray.example.com")

    run_id, _run_dir = store.create_run(manifest)

    persisted = store.read_manifest(run_id)
    assert persisted is not None
    assert persisted.backend_address == "https://ray.example.com"


def test_missing_run_manifest_falls_back_to_environment_resolution(tmp_path):
    store = FilesystemRunStore(tmp_path / "runs")

    assert store.read_manifest("run_missing") is None


def test_list_runs_skips_internal_directories(tmp_path):
    store = FilesystemRunStore(tmp_path / "runs")
    run_id, _ = store.create_run(_manifest(backend="local", backend_address=None))
    (store.root / "_campaigns").mkdir()
    (store.root / "_escalations").mkdir()

    assert list(store.list_runs()) == [run_id]


@pytest.mark.parametrize("backend_address", ["", "   ", "ray://cluster"])
def test_backend_address_rejects_malformed_values(backend_address):
    with pytest.raises(ValidationError):
        _manifest(backend_address=backend_address)


def test_local_manifest_does_not_need_backend_address():
    manifest = _manifest(backend="local", backend_address=None)

    assert manifest.backend == "local"
    assert manifest.backend_address is None
