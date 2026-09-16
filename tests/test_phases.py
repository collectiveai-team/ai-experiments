# tests/test_phases.py
"""Two phases, run in order, with the evaluator's result the only score."""

from __future__ import annotations

import subprocess
import sys
import textwrap

from ai_experiments.phases import run_phases
from ai_experiments.schemas import ExperimentManifest, RunHandle, WorkloadSpec
from ai_experiments.store import FilesystemRunStore


def _store_with(tmp_path, train_body: str, evaluate_body: str):
    (tmp_path / "train.py").write_text(textwrap.dedent(train_body))
    (tmp_path / "evaluate.py").write_text(textwrap.dedent(evaluate_body))
    # NOTE: the bodies below hand off through IAX_WORK_DIR, never the cwd.
    store = FilesystemRunStore(tmp_path / "runs")
    manifest = ExperimentManifest(
        experiment="e",
        workload=WorkloadSpec(
            entrypoint=f"{sys.executable} train.py",
            train=f"{sys.executable} train.py",
            evaluate=f"{sys.executable} evaluate.py",
            working_dir=str(tmp_path),
        ),
    )
    run_id, run_dir = store.create_run(manifest)
    # update_status refuses to fabricate a status for a run that was never
    # handed one -- every real caller goes through a backend's submit(),
    # which writes this first (tests/test_worker_phases.py's helper does the
    # same for the same reason).
    store.write_handle(
        RunHandle(
            run_id=run_id,
            backend="local",
            status="submitted",
            status_uri=str(store.status_path(run_id)),
            run_dir=str(run_dir),
        )
    )
    return store, run_id


def test_the_evaluator_result_is_the_one_that_counts(tmp_path):
    store, run_id = _store_with(
        tmp_path,
        """
        print('IAX_METRIC {"step": 0, "loss": 0.1}')
        print('IAX_RESULT {"test_acc": 0.99}')
        """,
        """
        print('IAX_RESULT {"test_acc": 0.61}')
        """,
    )

    assert run_phases(store, run_id) == 0
    assert [r.values for r in store.read_results(run_id)] == [{"test_acc": 0.61}]


def test_a_failing_train_phase_stops_the_evaluation(tmp_path):
    store, run_id = _store_with(
        tmp_path,
        """
        import sys
        sys.exit(3)
        """,
        """
        print('IAX_RESULT {"test_acc": 0.61}')
        """,
    )

    assert run_phases(store, run_id) == 3
    assert store.read_results(run_id) == []


def test_both_phases_share_one_handoff_directory(tmp_path):
    store, run_id = _store_with(
        tmp_path,
        """
        import os, pathlib
        pathlib.Path(os.environ["IAX_WORK_DIR"], "model.txt").write_text("7")
        """,
        """
        import os, pathlib
        x = pathlib.Path(os.environ["IAX_WORK_DIR"], "model.txt").read_text()
        print('IAX_RESULT {"loss": %s}' % x)
        """,
    )

    assert run_phases(store, run_id) == 0
    assert [r.values for r in store.read_results(run_id)] == [{"loss": 7.0}]


def test_a_single_entrypoint_workload_still_runs(tmp_path):
    store = FilesystemRunStore(tmp_path / "runs")
    script = tmp_path / "toy.py"
    script.write_text("print('IAX_RESULT {\"loss\": 0.25}')\n")
    manifest = ExperimentManifest(
        experiment="e",
        workload=WorkloadSpec(
            entrypoint=f"{sys.executable} toy.py", working_dir=str(tmp_path)
        ),
    )
    run_id, run_dir = store.create_run(manifest)
    # See _store_with: update_status needs a status a real submit() would
    # already have written.
    store.write_handle(
        RunHandle(
            run_id=run_id,
            backend="local",
            status="submitted",
            status_uri=str(store.status_path(run_id)),
            run_dir=str(run_dir),
        )
    )

    assert run_phases(store, run_id) == 0
    assert [r.values for r in store.read_results(run_id)] == [{"loss": 0.25}]


def test_supervisor_failure_before_spawn_is_reported(tmp_path):
    """Amendment 2/3: production now launches `-m ai_experiments.phases`, so
    the failure contract that turns a crashed supervisor into a `failed`
    status has to be proven against that entry point, not just against
    `ai_experiments.worker` (tests/test_worker_lifecycle.py:135-165), which
    production no longer runs.
    """
    store = FilesystemRunStore(tmp_path / "runs")
    manifest = ExperimentManifest(
        experiment="e",
        workload=WorkloadSpec(
            entrypoint=f"{sys.executable} -c pass", working_dir=str(tmp_path)
        ),
    )
    run_id, run_dir = store.create_run(manifest)
    # update_status refuses to fabricate a status for a run that was never
    # handed one -- every real caller goes through a backend's submit(),
    # which writes this first (tests/test_worker_phases.py's helper does the
    # same for the same reason).
    store.write_handle(
        RunHandle(
            run_id=run_id,
            backend="local",
            status="running",
            status_uri=str(store.status_path(run_id)),
            run_dir=str(run_dir),
        )
    )

    # Corrupt the persisted manifest so the launcher's own read of it raises.
    (run_dir / "manifest.yaml").write_text("{not: [valid")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ai_experiments.phases",
            "--run-id",
            run_id,
            "--runs-dir",
            str(store.root),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    status = store.read_status(run_id)
    assert status.status == "failed"
    assert "supervisor failed" in (status.error or "")
