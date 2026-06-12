from __future__ import annotations

import pytest

from ai_experiments.monitoring.rules import diagnose_run
from ai_experiments.schemas import ExperimentManifest, RunState, RunStatus, WorkloadSpec
from ai_experiments.store import FilesystemRunStore


def _store_with_status(
    tmp_path, status: RunState, details: dict[str, object] | None = None
) -> tuple[FilesystemRunStore, str]:
    store = FilesystemRunStore(tmp_path / "runs")
    manifest = ExperimentManifest(
        experiment="rules-smoke",
        backend="local",
        workload=WorkloadSpec(entrypoint="python train.py"),
    )
    run_id, run_dir = store.create_run(manifest)
    store.write_status(
        RunStatus(
            run_id=run_id,
            backend="local",
            status=status,
            status_uri=str(store.status_path(run_id)),
            run_dir=str(run_dir),
            details=details or {},
        )
    )
    return store, run_id


@pytest.mark.parametrize(
    ("status", "expected_reason"),
    [
        ("completed", "run_completed"),
        ("failed", "failed"),
        ("cancelled", "cancelled"),
    ],
)
def test_diagnose_run_terminal_reasons(tmp_path, status, expected_reason):
    store, run_id = _store_with_status(tmp_path, status)

    report = diagnose_run(store, run_id)

    assert expected_reason in report.decision.reasons


def test_diagnose_run_active_reason(tmp_path):
    store, run_id = _store_with_status(
        tmp_path, "running", details={"ray_condition": "running"}
    )

    report = diagnose_run(store, run_id)

    assert report.decision.decision == "continue_waiting"
    assert report.decision.reasons == ["run_active"]
