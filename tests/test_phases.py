# tests/test_phases.py
"""Two phases, run in order, with the evaluator's result the only score."""

from __future__ import annotations

import itertools
import subprocess
import sys
import textwrap
from datetime import timedelta
from typing import TYPE_CHECKING

import ai_experiments.phases as phases_module
import ai_experiments.worker as worker_module
from ai_experiments.phases import run_phases
from ai_experiments.schemas import (
    ExperimentManifest,
    RunHandle,
    WorkloadSpec,
    utc_now,
)
from ai_experiments.store import FilesystemRunStore

if TYPE_CHECKING:
    from pathlib import Path


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


def test_completed_is_not_written_until_the_final_phase(tmp_path, monkeypatch):
    """`completed` is written only by the final phase.

    `_Supervisor.run` only writes it when `self.final` is True (worker.py's `if self.final:` guard):
    a train phase that exits 0 hands off to evaluate, it does not finish the run. Checking only the
    end state (already covered by test_the_evaluator_result_is_the_one_that_counts) would miss a
    mutant that dropped the guard -- the final status is `completed` either way, only the sequence
    differs.
    """
    store, run_id = _store_with(
        tmp_path,
        """
        print('IAX_RESULT {"test_acc": 0.99}')
        """,
        """
        print('IAX_RESULT {"test_acc": 0.61}')
        """,
    )
    seen_statuses = []
    original_update_status = store.update_status

    def _recording_update_status(run_id, **updates):
        result = original_update_status(run_id, **updates)
        seen_statuses.append(result.status)
        return result

    monkeypatch.setattr(store, "update_status", _recording_update_status)

    assert run_phases(store, run_id) == 0

    assert "completed" not in seen_statuses[:-1]
    assert seen_statuses[-1] == "completed"


def test_started_at_is_not_moved_by_the_second_phase(tmp_path, monkeypatch):
    """`started_at` measures the *run's* start, not each phase's.

    It is for monitoring's timeout -- `_Supervisor.run` reads the stored value and only falls back
    to `utc_now()` when there isn't one yet (worker.py). Real wall-clock time is too coarse to trust
    here (both phases can start in the same tick), so `utc_now` is patched to something that visibly
    advances on every call: if the second phase wrote its own `started_at`, this would catch it even
    though real time might not have.
    """
    ticks = itertools.count()
    monkeypatch.setattr(worker_module, "utc_now", lambda: utc_now() + timedelta(hours=next(ticks)))
    store, run_id = _store_with(
        tmp_path,
        """
        print('IAX_RESULT {"test_acc": 0.99}')
        """,
        """
        print('IAX_RESULT {"test_acc": 0.61}')
        """,
    )
    started_at_writes = []
    original_update_status = store.update_status

    def _recording_update_status(run_id, **updates):
        if "started_at" in updates:
            started_at_writes.append(updates["started_at"])
        return original_update_status(run_id, **updates)

    monkeypatch.setattr(store, "update_status", _recording_update_status)

    assert run_phases(store, run_id) == 0

    assert len(started_at_writes) == 2, "expected one started_at write per phase"
    assert started_at_writes[0] == started_at_writes[1]


_SENTINEL = """
    import pathlib
    pathlib.Path(%r).write_text("ran")
"""


def test_a_cancellation_before_the_first_phase_runs_nothing(tmp_path):
    """The window `store.cancel_requested` uniquely covers, half one.

    A cancel can land between submit and the first phase -- no workload has
    started, so there is no pid to signal and no terminal status for the
    launcher's other guard (phases.py's `ACTIVE_RUN_STATES` check, which only
    runs *after* a phase) to notice. Only the check at the top of the loop
    stands between that request and a trainer that was never meant to run,
    and it has to stand there on the first iteration: weakening it to
    `index > 0` leaves exactly this window open.
    """
    store, run_id = _store_with(
        tmp_path,
        _SENTINEL % str(tmp_path / "train.ran"),
        _SENTINEL % str(tmp_path / "evaluate.ran"),
    )
    store.request_cancel(run_id)

    assert run_phases(store, run_id) == 1

    assert not (tmp_path / "train.ran").exists(), (
        "the train phase ran although cancellation was requested before it"
    )
    assert not (tmp_path / "evaluate.ran").exists()
    skipped = [
        event
        for event in store.read_events(run_id)
        if event.message == "remaining phases skipped: cancellation requested"
    ]
    assert len(skipped) == 1, "a cancellation that skipped every phase left no trace in the run log"
    assert skipped[0].level == "warning"
    assert skipped[0].details["phase"] == "train"


def test_a_cancellation_between_the_phases_stops_the_evaluator(tmp_path, monkeypatch):
    """The window `store.cancel_requested` uniquely covers, half two.

    Here the trainer has already exited, so the supervisor's own SIGTERM
    handler and its `_cancel_requested()` check are both behind us and there
    is no process left for anyone to signal. The status is still `running`
    (a non-final phase does not write `completed`), so the launcher's
    terminal-status guard sees nothing wrong either. The request is delivered
    by wrapping `_Supervisor` rather than from inside the workload on
    purpose: a marker the trainer writes itself is seen by that trainer's own
    supervisor, which ends the run as `cancelled` and hands the *other* guard
    the job -- which is precisely why deleting this one went unnoticed.
    """
    store, run_id = _store_with(
        tmp_path,
        _SENTINEL % str(tmp_path / "train.ran"),
        _SENTINEL % str(tmp_path / "evaluate.ran"),
    )

    class _CancelledAfterTheTrainer(phases_module._Supervisor):
        def run(self, command: str | None = None, work_dir: Path | None = None) -> int:
            exit_code = super().run(command=command, work_dir=work_dir)
            if self.phase == "train":
                store.request_cancel(run_id)
            return exit_code

    monkeypatch.setattr(phases_module, "_Supervisor", _CancelledAfterTheTrainer)

    assert run_phases(store, run_id) == 1

    assert (tmp_path / "train.ran").exists(), (
        "the fixture must let the trainer run, or the gap it tests never opens"
    )
    assert not (tmp_path / "evaluate.ran").exists(), (
        "the evaluator ran although cancellation was requested before it started"
    )
    assert store.read_status(run_id).status in {"running", "cancelled"}
    skipped = [
        event
        for event in store.read_events(run_id)
        if event.message == "remaining phases skipped: cancellation requested"
    ]
    assert len(skipped) == 1
    assert skipped[0].level == "warning"
    assert skipped[0].details["phase"] == "evaluate"


def test_a_second_declared_result_is_another_observation(tmp_path):
    """Each `IAX_RESULT` line is one observation; a fold loop declares several.

    Whether a run declared more than its objective can use is judged where
    the goal is known (`trial_sync`), so the store keeps every record, in
    order, and says nothing here.
    """
    store, run_id = _store_with(
        tmp_path,
        "pass",
        """
        print('IAX_RESULT {"test_acc": 0.10}')
        print('IAX_RESULT {"test_acc": 0.99}')
        """,
    )

    assert run_phases(store, run_id) == 0

    assert [r.values for r in store.read_results(run_id)] == [
        {"test_acc": 0.10},
        {"test_acc": 0.99},
    ]
    assert not [e for e in store.read_events(run_id) if e.level == "warning"]


def test_a_single_entrypoint_workload_still_runs(tmp_path):
    store = FilesystemRunStore(tmp_path / "runs")
    script = tmp_path / "toy.py"
    script.write_text("print('IAX_RESULT {\"loss\": 0.25}')\n")
    manifest = ExperimentManifest(
        experiment="e",
        workload=WorkloadSpec(entrypoint=f"{sys.executable} toy.py", working_dir=str(tmp_path)),
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
    """The failure contract is proven against `-m ai_experiments.phases`, the real entry point.

    Production now launches that module, so turning a crashed supervisor into a `failed` status has
    to be proven there, not just against `ai_experiments.worker` (tests/test_worker_lifecycle.py),
    which production no longer runs.
    """
    store = FilesystemRunStore(tmp_path / "runs")
    manifest = ExperimentManifest(
        experiment="e",
        workload=WorkloadSpec(entrypoint=f"{sys.executable} -c pass", working_dir=str(tmp_path)),
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

    result = subprocess.run(  # noqa: S603  # fixed argv: the test's own script
        [
            sys.executable,
            "-m",
            "ai_experiments.phases",
            "--run-id",
            run_id,
            "--runs-dir",
            str(store.root),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    status = store.read_status(run_id)
    assert status.status == "failed"
    assert "supervisor failed" in (status.error or "")
