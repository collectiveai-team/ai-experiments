"""The two-phase launcher under the real backend entry point, and its cancellation.

Split from `test_worker_lifecycle.py`, whose helpers it imports.
"""

from __future__ import annotations

from ai_experiments.backends.local import LocalBackend
from ai_experiments.schemas import (
    DataSpec,
    ExperimentManifest,
    WorkloadSpec,
)
from tests.test_worker_lifecycle import (
    _manifest,
    _script,
    _store,
    _wait_for_event,
    _wait_for_settled_terminal,
    _wait_for_terminal,
)

# -- C2: the production entry point is the two-phase launcher ---------------


def test_two_phase_workload_runs_through_the_real_backend_entry_point(tmp_path):
    """`LocalBackend.submit` spawns `-m ai_experiments.phases`, not `-m ai_experiments.worker`.

    Reverting that one string leaves the whole suite green (454 passed, 0 failed) unless something
    drives a two-phase workload through this exact path, not `run_phases` called in-process.
    """
    store = _store(tmp_path)
    train = _script(
        tmp_path / "train.py",
        "import os, pathlib\n"
        "seen_test = 'IAX_DATA_TEST' in os.environ\n"
        "pathlib.Path(os.environ['IAX_WORK_DIR'], 'saw_test.txt')"
        ".write_text(str(seen_test))\n"
        "print('IAX_RESULT {\"acc\": 0.99}', flush=True)\n",
    )
    evaluate = _script(
        tmp_path / "evaluate.py",
        "print('IAX_RESULT {\"acc\": 0.42}', flush=True)\n",
    )
    manifest = ExperimentManifest(
        experiment="lifecycle",
        backend="local",
        workload=WorkloadSpec(
            entrypoint=train,
            train=train,
            evaluate=evaluate,
            working_dir=str(tmp_path),
            data=DataSpec(test="s3://bucket/held-out.parquet"),
        ),
    )

    handle = LocalBackend(store=store).submit(manifest)
    status = _wait_for_terminal(store, handle.run_id)

    assert status.status == "completed"
    assert [r.values for r in store.read_results(handle.run_id)] == [{"acc": 0.42}]
    saw_test = (store.run_dir(handle.run_id) / "work" / "saw_test.txt").read_text()
    assert saw_test == "False"


# -- C1: cancellation must not be reversible by the second phase ------------


def test_iax_cancel_stops_the_evaluate_phase_too(tmp_path):
    """A requested cancellation is not undone by the train phase exiting 0.

    Observed before the fix: `backend.cancel` writes `cancelled`, then the evaluate phase runs to
    completion and the final status is `completed` with a score -- cancellation was reversible. The
    train phase here ignores SIGTERM outright and exits 0 on its own a little later, so nothing here
    depends on how the signal race between `killpg` and the supervisor's own handler happens to
    resolve: the only way the run can end up `cancelled`, with evaluate never started, is
    `store.request_cancel` (written before the signal is even sent) being consulted after the exit.
    """
    store = _store(tmp_path)
    train = _script(
        tmp_path / "train.py",
        "import signal, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "print('training', flush=True)\n"
        "time.sleep(0.5)\n",
    )
    evaluate = _script(
        tmp_path / "evaluate.py",
        "print('IAX_RESULT {\"acc\": 0.61}', flush=True)\n",
    )
    manifest = ExperimentManifest(
        experiment="lifecycle",
        backend="local",
        workload=WorkloadSpec(
            entrypoint=train, train=train, evaluate=evaluate, working_dir=str(tmp_path)
        ),
    )
    backend = LocalBackend(store=store)

    handle = backend.submit(manifest)
    _wait_for_event(store, handle.run_id, "training")
    backend.cancel(handle.run_id)
    # Not `_wait_for_terminal`: the bug this pins is a status that looks
    # terminal (`cancelled`, written by `cancel` itself) and then changes
    # again once the un-stopped evaluate phase finishes.
    status = _wait_for_settled_terminal(store, handle.run_id)

    assert status.status == "cancelled"
    assert store.read_results(handle.run_id) == []
    assert not any(
        event.details.get("phase") == "evaluate" and event.message == "phase started"
        for event in store.read_events(handle.run_id)
    )


def test_bare_sigterm_to_the_supervisor_stops_the_evaluate_phase_too(tmp_path):
    """The same bug as above, reached through the other door.

    `run_phases` decides whether to keep going by asking `store.cancel_requested`, the marker
    `LocalBackend.cancel` writes -- but `_Supervisor._cancel_requested` is `self._cancelled or
    store.cancel_requested(...)`, and `self._cancelled` is set by *any* SIGTERM the supervisor
    receives, marker or not. The train phase here signals its own parent (the supervisor process
    itself, not the backend) directly and exits 0 on its own; nobody ever calls `backend.cancel` or
    writes the cancel marker. Round 2's fix correctly records `train`'s own exit as `cancelled`; the
    bug is `run_phases` seeing no marker afterwards and starting `evaluate` anyway.
    """
    store = _store(tmp_path)
    train = _script(
        tmp_path / "train.py",
        "import os, signal, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "print('training', flush=True)\n"
        "os.kill(os.getppid(), signal.SIGTERM)\n"
        "time.sleep(0.5)\n",
    )
    evaluate = _script(
        tmp_path / "evaluate.py",
        "print('IAX_RESULT {\"acc\": 0.61}', flush=True)\n",
    )
    manifest = ExperimentManifest(
        experiment="lifecycle",
        backend="local",
        workload=WorkloadSpec(
            entrypoint=train, train=train, evaluate=evaluate, working_dir=str(tmp_path)
        ),
    )
    backend = LocalBackend(store=store)

    handle = backend.submit(manifest)
    # Not `_wait_for_terminal`: the bug this pins is a status that looks
    # terminal (`cancelled`, written when `train` exits) and then changes
    # again once the un-stopped evaluate phase finishes.
    status = _wait_for_settled_terminal(store, handle.run_id)

    assert status.status == "cancelled"
    assert store.read_results(handle.run_id) == []
    assert not any(
        event.details.get("phase") == "evaluate" and event.message == "phase started"
        for event in store.read_events(handle.run_id)
    )


def test_single_phase_workload_that_traps_sigterm_is_still_cancelled(tmp_path):
    """C1 gap 2 on the single-phase path.

    `test_iax_cancel_stops_the_evaluate_ phase_too` only pins "exit 0 after a requested cancellation
    is still `cancelled`" for the two-phase launcher; a single-entrypoint workload that traps
    SIGTERM and shuts down cleanly after `iax cancel` needs the same guarantee, and nothing here was
    exercising that path.
    """
    store = _store(tmp_path)
    entrypoint = _script(
        tmp_path / "traps_sigterm.py",
        "import signal, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "print('training', flush=True)\n"
        "time.sleep(0.5)\n",
    )
    backend = LocalBackend(store=store)

    handle = backend.submit(_manifest(tmp_path, entrypoint, working_dir=str(tmp_path)))
    _wait_for_event(store, handle.run_id, "training")
    backend.cancel(handle.run_id)
    status = _wait_for_settled_terminal(store, handle.run_id)

    assert status.status == "cancelled"
