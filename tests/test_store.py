"""Run-store durability: no reader ever observes a partial or fabricated status.

Writers span four processes (worker main thread + heartbeat thread, daemon,
CLI/backends), so a reader can land mid-write and a crash can leave a
truncated file. These tests pin the two guarantees that make that survivable:
writes are indivisible, and an unreadable status is quarantined rather than
raised or overwritten.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from ai_experiments.schemas import (
    CampaignState,
    ExperimentManifest,
    GoalSpec,
    RunHandle,
    WorkloadSpec,
)
from ai_experiments.store import FilesystemRunStore
from ai_experiments.store.campaign import CampaignStore

TORN_STATUS = '{"run_id": "run_torn0000", "backend": "l'


def _store(tmp_path) -> FilesystemRunStore:
    return FilesystemRunStore(tmp_path / "runs", capture_repro=False)


def _manifest() -> ExperimentManifest:
    return ExperimentManifest(
        experiment="store-test",
        backend="local",
        workload=WorkloadSpec(entrypoint="python train.py"),
    )


def _submitted_run(store: FilesystemRunStore) -> str:
    run_id, run_dir = store.create_run(_manifest())
    store.write_handle(
        RunHandle(
            run_id=run_id,
            backend="local",
            status="submitted",
            status_uri=str(store.status_path(run_id)),
            run_dir=str(run_dir),
        )
    )
    return run_id


def _goal() -> GoalSpec:
    return GoalSpec(
        goal="store test",
        name="store-test",
        objective={"metric": "loss", "mode": "min"},
        search_space={"x": {"type": "uniform", "low": 0.0, "high": 1.0}},
        workload=WorkloadSpec(entrypoint="python train.py"),
    )


# -- reading a status that cannot be parsed ----------------------------------


def test_read_status_quarantines_a_torn_file_instead_of_raising(tmp_path):
    store = _store(tmp_path)
    run_id = _submitted_run(store)
    store.status_path(run_id).write_text(TORN_STATUS)

    status = store.read_status(run_id)

    assert status.status == "unknown"
    assert "corrupt" in (status.error or "")


def test_read_status_still_reports_a_missing_file(tmp_path):
    store = _store(tmp_path)
    run_id, _ = store.create_run(_manifest())

    status = store.read_status(run_id)

    assert status.status == "unknown"
    assert status.error == "status file not found"


def test_listing_runs_survives_one_corrupt_run(tmp_path):
    """The failure that took down `iax runs`: one bad file, every read dies."""
    store = _store(tmp_path)
    healthy = _submitted_run(store)
    corrupt = _submitted_run(store)
    store.status_path(corrupt).write_text(TORN_STATUS)

    statuses = {run_id: store.read_status(run_id) for run_id in store.list_runs()}

    assert statuses[healthy].status == "submitted"
    assert statuses[corrupt].status == "unknown"


# -- refusing to persist a status the store never actually read --------------


def test_update_status_refuses_to_overwrite_a_corrupt_file(tmp_path):
    """Merging onto a synthetic status would fabricate history and destroy the
    evidence of what went wrong."""
    store = _store(tmp_path)
    run_id = _submitted_run(store)
    store.status_path(run_id).write_text(TORN_STATUS)

    with pytest.raises(RuntimeError, match="corrupt"):
        store.update_status(run_id, status="running")

    assert store.status_path(run_id).read_text() == TORN_STATUS


def test_update_status_refuses_when_there_is_no_status_yet(tmp_path):
    store = _store(tmp_path)
    run_id, _ = store.create_run(_manifest())

    with pytest.raises(RuntimeError, match="not found"):
        store.update_status(run_id, status="running")

    assert not store.status_path(run_id).exists()


# -- establishing a status is a create, never an update ----------------------


def test_write_handle_refuses_to_clobber_an_existing_status(tmp_path):
    """A handle carries no details, so overwriting would silently drop them --
    this is how the Ray path lost its MLflow linkage."""
    store = _store(tmp_path)
    run_id = _submitted_run(store)
    store.update_status(run_id, details={"mlflow_run_id": "mlf_1"})

    with pytest.raises(RuntimeError, match="write_handle"):
        store.write_handle(
            RunHandle(
                run_id=run_id,
                backend="local",
                status="submitted",
                status_uri=str(store.status_path(run_id)),
                run_dir=str(store.run_dir(run_id)),
            )
        )

    assert store.read_status(run_id).details["mlflow_run_id"] == "mlf_1"


# -- indivisible writes ------------------------------------------------------


def test_write_status_leaves_no_temp_files_behind(tmp_path):
    store = _store(tmp_path)
    run_id = _submitted_run(store)
    store.update_status(run_id, status="running")

    leftovers = list(store.run_dir(run_id).glob("*.tmp"))

    assert leftovers == []


def test_a_failed_write_leaves_the_previous_status_intact(tmp_path):
    """A crash mid-write must leave the old document readable, never a
    truncated one -- and must not litter the run dir with temp files."""
    store = _store(tmp_path)
    run_id = _submitted_run(store)
    before = store.status_path(run_id).read_text()

    with patch("ai_experiments.store.filesystem.os.replace", side_effect=OSError("no")):
        with pytest.raises(OSError):
            store.update_status(run_id, status="running")

    assert store.status_path(run_id).read_text() == before
    assert list(store.run_dir(run_id).glob("*.tmp")) == []


def test_campaign_state_is_written_atomically(tmp_path):
    """Same non-atomic write pattern wedged campaigns permanently."""
    campaigns = CampaignStore(tmp_path / "runs")
    state = campaigns.create_campaign(_goal())

    campaigns.write_state(state)

    assert list(campaigns.campaign_dir(state.campaign_id).glob("*.tmp")) == []
    reloaded = campaigns.read_state(state.campaign_id)
    assert isinstance(reloaded, CampaignState)
    assert (
        json.loads(
            (campaigns.campaign_dir(state.campaign_id) / "state.json").read_text()
        )["campaign_id"]
        == state.campaign_id
    )
