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
    return FilesystemRunStore(tmp_path / "runs", capture_repro=False)


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


def test_artifacts_listing_and_download(client, store):
    run_id = _seed_run(store)
    artifacts = store.artifacts_dir(run_id)
    artifacts.mkdir(exist_ok=True)
    (artifacts / "model.bin").write_bytes(b"weights")

    listing = client.get(f"/api/runs/{run_id}/artifacts").json()
    assert [a["path"] for a in listing] == ["model.bin"]

    download = client.get(f"/api/runs/{run_id}/artifacts/model.bin")
    assert download.status_code == 200
    assert download.content == b"weights"


def test_artifact_path_traversal_blocked(client, store):
    run_id = _seed_run(store)

    response = client.get(f"/api/runs/{run_id}/artifacts/../status.json")

    assert response.status_code == 404


def test_repro_endpoint(client, store):
    run_id = _seed_run(store)
    repro_dir = store.run_dir(run_id) / "repro"
    repro_dir.mkdir(exist_ok=True)
    (repro_dir / "context.json").write_text('{"git_sha": "abc123", "git_dirty": false}')

    payload = client.get(f"/api/runs/{run_id}/repro").json()
    assert payload["git_sha"] == "abc123"
    assert payload["has_diff"] is False

    missing = _seed_run(store)
    import shutil

    shutil.rmtree(store.run_dir(missing) / "repro", ignore_errors=True)
    assert client.get(f"/api/runs/{missing}/repro").status_code == 404


def test_leaderboard_ranks_campaigns(client, store):
    import json

    from ai_experiments.schemas import (
        BudgetSpec,
        GoalSpec,
        ObjectiveSpec,
        TrialRecord,
        WorkloadSpec,
    )
    from ai_experiments.store.campaign import CampaignStore

    campaign_store = CampaignStore(store.root)
    for name, best in [("worse", 0.9), ("better", 0.1)]:
        goal = GoalSpec(
            goal=f"campaign {name}",
            name=name,
            objective=ObjectiveSpec(metric="loss", mode="min"),
            search_space={"x": {"type": "uniform", "low": 0.0, "high": 1.0}},
            workload=WorkloadSpec(entrypoint="python t.py"),
            budget=BudgetSpec(max_trials=1, gpu_hour_rate=2.5),
        )
        state = campaign_store.create_campaign(goal)
        state.trials.append(
            TrialRecord(
                trial_id="t000",
                params={"x": best},
                status="completed",
                objective_value=best,
                gpu_hours=1.0,
            )
        )
        state.best_trial_id = "t000"
        state.status = "completed"
        campaign_store.write_state(state)

    rows = client.get("/api/leaderboard").json()
    assert [r["name"] for r in rows] == ["better", "worse"]
    assert rows[0]["best_value"] == 0.1
    assert rows[0]["estimated_cost"] == 2.5
    assert json.loads(json.dumps(rows[0]["best_params"])) == {"x": 0.1}


def test_campaign_pause_resume_endpoints(client, store):
    from ai_experiments.schemas import (
        BudgetSpec,
        GoalSpec,
        ObjectiveSpec,
        WorkloadSpec,
    )
    from ai_experiments.store.campaign import CampaignStore

    campaign_store = CampaignStore(store.root)
    goal = GoalSpec(
        goal="pausable",
        name="pausable",
        objective=ObjectiveSpec(metric="loss", mode="min"),
        search_space={"x": {"type": "uniform", "low": 0.0, "high": 1.0}},
        workload=WorkloadSpec(entrypoint="python t.py"),
        budget=BudgetSpec(max_trials=1),
    )
    state = campaign_store.create_campaign(goal)

    paused = client.post(f"/api/campaigns/{state.campaign_id}/pause")
    assert paused.status_code == 200
    assert paused.json()["status"] == "paused"

    again = client.post(f"/api/campaigns/{state.campaign_id}/pause")
    assert again.status_code == 409

    resumed = client.post(f"/api/campaigns/{state.campaign_id}/resume")
    assert resumed.status_code == 200


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


# -- unauthenticated mutations over the network (#24) ---------------------------


@pytest.mark.parametrize(
    ("host", "loopback"),
    [
        ("127.0.0.1", True),
        ("::1", True),
        ("localhost", True),
        ("0.0.0.0", False),
        ("192.168.1.40", False),
        ("::", False),
        ("dashboard.internal", False),
    ],
)
def test_what_counts_as_reachable_only_from_this_machine(host, loopback):
    from ai_experiments.server.app import is_loopback

    assert is_loopback(host) is loopback


def _mutations(store, run_id: str, campaign_id: str) -> list[str]:
    return [
        f"/api/runs/{run_id}/cancel",
        f"/api/campaigns/{campaign_id}/stop",
        f"/api/campaigns/{campaign_id}/pause",
        f"/api/campaigns/{campaign_id}/resume",
    ]


def _seed_campaign(store) -> str:
    from ai_experiments.schemas import GoalSpec
    from ai_experiments.store.campaign import CampaignStore

    goal = GoalSpec(
        goal="minimize loss",
        name="served",
        objective={"metric": "loss"},
        search_space={"lr": {"type": "choice", "values": [0.1]}},
        workload={"entrypoint": "true"},
    )
    return CampaignStore(store.root).create_campaign(goal).campaign_id


def test_a_networked_dashboard_refuses_every_mutation(store):
    """Anyone who can route to the port could otherwise cancel a week of work."""
    run_id = _seed_run(store)
    campaign_id = _seed_campaign(store)
    client = TestClient(create_app(store, host="0.0.0.0"))

    for path in _mutations(store, run_id, campaign_id):
        response = client.post(path)
        assert response.status_code == 403, path
        assert "--allow-remote-mutations" in response.json()["detail"]

    assert store.read_status(run_id).status == "running"


def test_a_networked_dashboard_still_serves_every_read(store):
    run_id = _seed_run(store)
    client = TestClient(create_app(store, host="0.0.0.0"))

    assert client.get("/api/runs").status_code == 200
    assert client.get(f"/api/runs/{run_id}").status_code == 200
    assert client.get(f"/api/runs/{run_id}/metrics").status_code == 200
    assert client.get("/api/health").json()["mutations"] == "read-only"


def test_the_operator_can_ask_for_networked_mutations_by_name(store):
    run_id = _seed_run(store)
    client = TestClient(create_app(store, host="0.0.0.0", allow_remote_mutations=True))

    assert client.post(f"/api/runs/{run_id}/cancel").status_code == 200
    assert client.get("/api/health").json()["mutations"] == "allowed"


def test_the_default_loopback_dashboard_is_unchanged(client, store):
    run_id = _seed_run(store)

    assert client.post(f"/api/runs/{run_id}/cancel").status_code == 200
    assert client.get("/api/health").json()["mutations"] == "allowed"
