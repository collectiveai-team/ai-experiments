from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from ai_experiments.schemas import (
    ExperimentManifest,
    MetricPoint,
    RunStatus,
    WorkloadSpec,
)
from ai_experiments.server.app import create_app
from ai_experiments.store import FilesystemRunStore


@pytest.fixture()
def store(tmp_path) -> FilesystemRunStore:
    return FilesystemRunStore(tmp_path / "runs")


@pytest.fixture()
def client(store) -> TestClient:
    return TestClient(create_app(store))


def _seed_run(store: FilesystemRunStore) -> str:
    manifest = ExperimentManifest(
        experiment="api-test",
        workload=WorkloadSpec(entrypoint="python train.py"),
    )
    run_id, run_dir = store.create_run(manifest)
    store.write_status(
        RunStatus(
            run_id=run_id,
            backend="local",
            status="running",
            status_uri=str(store.status_path(run_id)),
            run_dir=str(run_dir),
        )
    )
    store.append_metric(run_id, MetricPoint(step=1, values={"loss": 0.4}))
    return run_id


def test_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_dashboard_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "iax" in response.text


def test_runs_listing_and_detail(client, store):
    run_id = _seed_run(store)

    listing = client.get("/api/runs").json()
    assert [r["run_id"] for r in listing] == [run_id]

    detail = client.get(f"/api/runs/{run_id}").json()
    assert detail["status"] == "running"

    metrics = client.get(f"/api/runs/{run_id}/metrics").json()
    assert metrics[0]["values"] == {"loss": 0.4}

    diagnosis = client.get(f"/api/runs/{run_id}/diagnosis").json()
    assert diagnosis["decision"]["decision"] in {
        "continue_waiting",
        "delegate_diagnosis",
    }


def test_unknown_run_is_404(client):
    assert client.get("/api/runs/run_nope").status_code == 404
    assert client.post("/api/runs/run_nope/cancel").status_code == 404


def test_cancel_run(client, store):
    run_id = _seed_run(store)

    response = client.post(f"/api/runs/{run_id}/cancel")

    assert response.status_code == 200
    assert store.read_status(run_id).status == "cancelled"


def test_campaigns_empty(client):
    assert client.get("/api/campaigns").json() == []
    assert client.get("/api/campaigns/cmp_nope").status_code == 404


def test_escalations_empty(client):
    assert client.get("/api/escalations").json() == []


def test_clusters_endpoint(client, tmp_path, monkeypatch):
    config = tmp_path / "clusters.yaml"
    config.write_text(
        "clusters:\n"
        "  dead:\n"
        "    provider: aws\n"
        "    address: http://127.0.0.1:1\n"  # connection refused, fast
    )
    monkeypatch.setenv("IAX_CLUSTERS", str(config))

    payload = client.get("/api/clusters").json()

    assert len(payload) == 1
    assert payload[0]["name"] == "dead"
    assert payload[0]["provider"] == "aws"
    assert payload[0]["reachable"] is False


def test_clusters_endpoint_without_config(client, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # no ./clusters.yaml
    monkeypatch.delenv("IAX_CLUSTERS", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))  # no ~/.config/iax/clusters.yaml

    assert client.get("/api/clusters").json() == []
