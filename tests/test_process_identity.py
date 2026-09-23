"""Process identity: stable on Linux, and what the reaper does without it (#31).

Split from `test_worker_lifecycle.py`, whose helpers it imports.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from ai_experiments import procs
from ai_experiments.backends.local import LocalBackend
from ai_experiments.procs import process_identity
from ai_experiments.schemas import (
    RunHandle,
    utc_now,
)
from tests.test_worker_lifecycle import (
    _manifest,
    _store,
)

# -- process identity --------------------------------------------------------


@pytest.mark.skipif(not Path("/proc").is_dir(), reason="needs /proc")
def test_process_identity_is_stable_and_absent_for_dead_pids():
    assert process_identity(os.getpid()) == process_identity(os.getpid())
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    identity = process_identity(child.pid)
    child.wait()  # now a zombie until Popen reaps it
    assert process_identity(child.pid) is None
    assert identity != process_identity(os.getpid())


# -- identity off Linux (#31) ------------------------------------------------


def _no_proc(monkeypatch):
    """Pretend this machine has no /proc, as macOS and Windows do not."""
    monkeypatch.setattr(procs, "_proc_available", lambda: False)


def test_this_machine_can_identify_a_process():
    """Linux always can; the tests below turn that off deliberately."""
    assert procs.identity_supported()


def test_without_proc_or_psutil_nothing_can_be_identified(monkeypatch):
    _no_proc(monkeypatch)
    monkeypatch.setattr(procs, "_psutil", lambda: None)

    assert not procs.identity_supported()
    assert procs.process_identity(os.getpid()) is None
    # The liveness fallback stays usable, it just cannot see a zombie.
    assert procs.pid_alive(os.getpid())


def test_an_unidentifiable_workload_is_reported_and_never_signalled(monkeypatch):
    """The safe refusal is the one guarantee that must hold everywhere."""
    _no_proc(monkeypatch)
    monkeypatch.setattr(procs, "_psutil", lambda: None)
    bystander = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(30)"])
    try:
        report = procs.terminate_workload(bystander.pid, None)

        assert report["outcome"] == "identity_unverifiable"
        assert bystander.poll() is None
    finally:
        bystander.kill()
        bystander.wait()


def test_psutil_identifies_a_process_where_proc_cannot(monkeypatch):
    """The optional extra restores the guarantee off Linux."""
    pytest.importorskip("psutil")
    _no_proc(monkeypatch)

    assert procs.identity_supported()
    identity = procs.process_identity(os.getpid())
    assert identity
    assert identity == procs.process_identity(os.getpid())

    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child_identity = procs.process_identity(child.pid)
    assert child_identity != identity
    child.wait()  # a zombie until Popen reaps it: not alive
    assert procs.process_identity(child.pid) is None
    assert not procs.pid_alive(child.pid)


def test_psutil_says_nothing_about_a_pid_that_does_not_exist(monkeypatch):
    pytest.importorskip("psutil")
    _no_proc(monkeypatch)

    assert procs.process_identity(999999) is None


def test_psutil_lets_the_reaper_kill_an_orphan_off_linux(monkeypatch, tmp_path):
    pytest.importorskip("psutil")
    _no_proc(monkeypatch)
    store = _store(tmp_path)
    manifest = _manifest(tmp_path, "python train.py", working_dir=str(tmp_path))
    run_id, run_dir = store.create_run(manifest)
    orphan = subprocess.Popen(
        [sys.executable, "-c", "import time;time.sleep(30)"], start_new_session=True
    )
    try:
        store.write_handle(
            RunHandle(
                run_id=run_id,
                backend="local",
                status="running",
                status_uri=str(store.status_path(run_id)),
                run_dir=str(run_dir),
            )
        )
        store.update_status(
            run_id,
            pid=999999,
            started_at=utc_now(),
            details={
                "workload_pid": orphan.pid,
                "workload_identity": procs.process_identity(orphan.pid),
            },
        )

        report = LocalBackend(store=store).reap(run_id)

        assert report["outcome"] in {"terminated", "killed"}
        assert orphan.poll() is not None
    finally:
        if orphan.poll() is None:
            orphan.kill()
        orphan.wait()


def test_the_refusal_names_the_fix_when_the_platform_is_the_reason(monkeypatch, tmp_path):
    """`identity_unverifiable` alone tells an operator nothing to do."""
    _no_proc(monkeypatch)
    monkeypatch.setattr(procs, "_psutil", lambda: None)
    store = _store(tmp_path)
    manifest = _manifest(tmp_path, "python train.py", working_dir=str(tmp_path))
    run_id, run_dir = store.create_run(manifest)
    bystander = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(30)"])
    try:
        store.write_handle(
            RunHandle(
                run_id=run_id,
                backend="local",
                status="running",
                status_uri=str(store.status_path(run_id)),
                run_dir=str(run_dir),
            )
        )
        store.update_status(
            run_id,
            pid=999999,
            started_at=utc_now(),
            details={"workload_pid": bystander.pid, "workload_identity": None},
        )

        report = LocalBackend(store=store).reap(run_id)

        assert report["outcome"] == "identity_unverifiable"
        assert "psutil" in str(report["hint"])
        assert bystander.poll() is None
    finally:
        bystander.kill()
        bystander.wait()
