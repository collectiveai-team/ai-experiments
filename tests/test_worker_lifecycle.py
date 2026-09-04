"""End-to-end lifecycle behaviour of the local supervisor.

These drive real detached workers and real signals rather than stubs: every
bug they pin (issues #7, #8, #9, #10) is a bug about what actually happens to
a process, and each was invisible to the in-process unit suite.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from ai_experiments.backends.local import LocalBackend
from ai_experiments.daemon import MonitorDaemon
from ai_experiments import procs
from ai_experiments.procs import process_identity
from ai_experiments.schemas import (
    ExperimentManifest,
    RunHandle,
    RunStatus,
    WorkloadSpec,
    utc_now,
)
from ai_experiments.store import FilesystemRunStore

TERMINAL = {"completed", "failed", "cancelled"}


def _store(tmp_path: Path) -> FilesystemRunStore:
    return FilesystemRunStore(tmp_path / "runs", capture_repro=False)


def _manifest(tmp_path: Path, entrypoint: str, working_dir: str) -> ExperimentManifest:
    return ExperimentManifest(
        experiment="lifecycle",
        backend="local",
        workload=WorkloadSpec(entrypoint=entrypoint, working_dir=working_dir),
    )


def _script(path: Path, body: str) -> str:
    path.write_text(body)
    return f"{sys.executable} {path}"


def _wait_for(predicate, timeout: float = 30.0, interval: float = 0.05):
    deadline = time.monotonic() + timeout
    while True:
        value = predicate()
        if value:
            return value
        if time.monotonic() >= deadline:
            return value
        time.sleep(interval)


def _wait_for_terminal(store: FilesystemRunStore, run_id: str) -> RunStatus:
    _wait_for(lambda: store.read_status(run_id).status in TERMINAL)
    return store.read_status(run_id)


def _wait_for_workload_pid(store: FilesystemRunStore, run_id: str) -> int:
    pid = _wait_for(lambda: store.read_status(run_id).details.get("workload_pid"))
    assert pid, "the supervisor never recorded a workload pid"
    return int(pid)


def _wait_for_event(store: FilesystemRunStore, run_id: str, text: str) -> None:
    """Wait until the workload has produced ``text`` on stdout.

    The pid is recorded when the process is *spawned*, which is before the
    interpreter has run a line of it. Waiting for output instead pins the
    workload past its last write, so killing the supervisor after this point
    tests the orphan case and not "the workload died of a broken pipe".
    """
    found = _wait_for(
        lambda: any(text in event.message for event in store.read_events(run_id))
    )
    assert found, f"workload never logged {text!r}"


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _kill_supervisor(store: FilesystemRunStore, run_id: str, reap: bool) -> int:
    """SIGKILL a run's supervisor, optionally reaping the corpse.

    The supervisor is a child of *this* process (the backend spawned it
    in-process), so unless it is waited for it stays a zombie -- which is a
    real production state too, whenever the submitting process outlives the
    supervisor it started.
    """
    supervisor_pid = store.read_status(run_id).pid
    assert supervisor_pid
    os.kill(supervisor_pid, signal.SIGKILL)
    if reap:
        os.waitpid(supervisor_pid, 0)
        _wait_for(lambda: not _alive(supervisor_pid))
    else:
        _wait_for(lambda: process_identity(supervisor_pid) is None)
    return supervisor_pid


# -- #9: a workload that cannot start ----------------------------------------


def test_unstartable_workload_is_reported_as_failed(tmp_path):
    """Issue #9: this used to sit in `running` forever, with the traceback
    only in worker.log."""
    store = _store(tmp_path)
    manifest = _manifest(
        tmp_path, "definitely-not-a-binary-xyz12", working_dir=str(tmp_path)
    )

    handle = LocalBackend(store=store).submit(manifest)
    status = _wait_for_terminal(store, handle.run_id)

    assert status.status == "failed"
    assert "definitely-not-a-binary-xyz12" in (status.error or "")


def test_supervisor_failure_before_spawn_is_reported(tmp_path):
    """The failure contract has to cover every way run() can raise, not just
    the Popen call: here the manifest itself no longer parses."""
    store = _store(tmp_path)
    handle = LocalBackend(store=store).submit(
        _manifest(tmp_path, f"{sys.executable} -c pass", working_dir=str(tmp_path))
    )
    _wait_for_terminal(store, handle.run_id)

    # Re-run the supervisor by hand against a manifest that cannot be read.
    (store.run_dir(handle.run_id) / "manifest.yaml").write_text("{not: [valid")
    store.update_status(handle.run_id, status="running", error=None)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ai_experiments.worker",
            "--run-id",
            handle.run_id,
            "--runs-dir",
            str(store.root),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    status = store.read_status(handle.run_id)
    assert status.status == "failed"
    assert "supervisor failed" in (status.error or "")
    assert "Traceback" in result.stderr  # evidence still reaches worker.log
    messages = [event.message for event in store.read_events(handle.run_id)]
    assert any("supervisor failed" in message for message in messages)


def test_failure_contract_never_rewrites_a_terminal_status(tmp_path):
    """A crash *after* the workload finished must not turn a real result into
    a failure -- it is reported as an event instead."""
    store = _store(tmp_path)
    handle = LocalBackend(store=store).submit(
        _manifest(tmp_path, f"{sys.executable} -c pass", working_dir=str(tmp_path))
    )
    _wait_for_terminal(store, handle.run_id)
    assert store.read_status(handle.run_id).status == "completed"

    from ai_experiments.worker import report_supervisor_failure

    report_supervisor_failure(store, handle.run_id, RuntimeError("late boom"))

    assert store.read_status(handle.run_id).status == "completed"
    messages = [event.message for event in store.read_events(handle.run_id)]
    assert any("late boom" in message for message in messages)


# -- #10: relative working_dir ------------------------------------------------


def test_relative_working_dir_runs_in_that_directory(tmp_path, monkeypatch):
    """Issue #10: `working_dir: sub` executed in `sub/sub`."""
    project = tmp_path / "project"
    (project / "sub").mkdir(parents=True)
    (project / "sub" / "train.py").write_text("print('it trained', flush=True)\n")
    monkeypatch.chdir(project)
    store = _store(tmp_path)

    handle = LocalBackend(store=store).submit(
        _manifest(tmp_path, f"{sys.executable} train.py", working_dir="sub")
    )
    status = _wait_for_terminal(store, handle.run_id)

    assert status.status == "completed", status.error
    messages = [event.message for event in store.read_events(handle.run_id)]
    assert "it trained" in messages


def test_persisted_manifest_resolves_working_dir_once(tmp_path, monkeypatch):
    """The stored manifest is what every later process reads, so the relative
    path has to be gone by then -- for `iax rerun` as much as for the worker."""
    project = tmp_path / "project"
    (project / "sub").mkdir(parents=True)
    monkeypatch.chdir(project)
    store = _store(tmp_path)

    run_id, _ = store.create_run(
        _manifest(tmp_path, "python train.py", working_dir="sub")
    )

    persisted = store.read_manifest(run_id)
    assert persisted is not None
    assert persisted.workload.working_dir == str(project / "sub")


def test_the_manifest_as_authored_is_kept_alongside_the_resolved_one(
    tmp_path, monkeypatch
):
    """Resolving is what makes a run executable; the relative path is what
    makes the manifest portable. Keeping only one of the two throws away a
    kind of reproducibility."""
    project = tmp_path / "project"
    (project / "sub").mkdir(parents=True)
    monkeypatch.chdir(project)
    store = _store(tmp_path)

    run_id, _ = store.create_run(
        _manifest(tmp_path, "python train.py", working_dir="sub")
    )

    executed = store.read_manifest(run_id)
    authored = store.read_manifest(run_id, source=True)
    assert executed is not None and authored is not None
    assert executed.workload.working_dir == str(project / "sub")
    assert authored.workload.working_dir == "sub"


def test_no_source_manifest_when_nothing_was_resolved(tmp_path):
    """An absolute working_dir is already portable-or-not on its own terms;
    a byte-identical second copy would only be noise."""
    store = _store(tmp_path)

    run_id, _ = store.create_run(
        _manifest(tmp_path, "python train.py", working_dir=str(tmp_path))
    )

    assert not store.source_manifest_path(run_id).exists()
    assert store.read_manifest(run_id, source=True) == store.read_manifest(run_id)


def test_rerun_portable_re_resolves_on_the_machine_it_runs_from(tmp_path, monkeypatch):
    """The point of keeping the authored manifest: the run can be repeated
    somewhere else. Plain rerun still repeats the exact original paths."""
    from typer.testing import CliRunner

    from ai_experiments.cli import app

    here, elsewhere = tmp_path / "here", tmp_path / "elsewhere"
    for project in (here, elsewhere):
        (project / "sub").mkdir(parents=True)
        (project / "sub" / "train.py").write_text("print('trained', flush=True)\n")
    store = _store(tmp_path)
    monkeypatch.chdir(here)
    original = LocalBackend(store=store).submit(
        _manifest(tmp_path, f"{sys.executable} train.py", working_dir="sub")
    )
    _wait_for_terminal(store, original.run_id)

    monkeypatch.chdir(elsewhere)  # the run store moved with us; the paths did not
    runner = CliRunner()
    portable = runner.invoke(
        app,
        ["rerun", original.run_id, "--runs-dir", str(store.root), "--portable"],
    )
    exact = runner.invoke(
        app, ["rerun", original.run_id, "--runs-dir", str(store.root)]
    )

    assert portable.exit_code == 0, portable.output
    assert exact.exit_code == 0, exact.output
    portable_id = portable.stdout.split()[2]
    exact_id = exact.stdout.split()[2]
    assert store.read_manifest(portable_id).workload.working_dir == str(
        elsewhere / "sub"
    )
    assert store.read_manifest(exact_id).workload.working_dir == str(here / "sub")


def test_worker_reports_a_manifest_it_cannot_trust(tmp_path):
    """A relative working_dir in a persisted manifest means the run directory
    was not written by this version -- resolving it again is what caused #10,
    so it is refused loudly instead."""
    store = _store(tmp_path)
    handle = LocalBackend(store=store).submit(
        _manifest(tmp_path, f"{sys.executable} -c pass", working_dir=str(tmp_path))
    )
    _wait_for_terminal(store, handle.run_id)

    manifest_path = store.run_dir(handle.run_id) / "manifest.yaml"
    manifest_path.write_text(
        manifest_path.read_text().replace(f"working_dir: {tmp_path}", "working_dir: .")
    )
    store.update_status(handle.run_id, status="running", error=None)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "ai_experiments.worker",
            "--run-id",
            handle.run_id,
            "--runs-dir",
            str(store.root),
        ],
        capture_output=True,
        text=True,
    )

    status = store.read_status(handle.run_id)
    assert status.status == "failed"
    assert "must be absolute" in (status.error or "")


# -- #7: SIGKILL is not a cancellation ---------------------------------------


def test_sigkilled_workload_is_reported_as_failed(tmp_path):
    """Issue #7: an OOM-killed run reported as a user cancellation."""
    store = _store(tmp_path)
    entrypoint = _script(
        tmp_path / "oom.py",
        "import os, signal\nprint('training', flush=True)\n"
        "os.kill(os.getpid(), signal.SIGKILL)\n",
    )

    handle = LocalBackend(store=store).submit(
        _manifest(tmp_path, entrypoint, working_dir=str(tmp_path))
    )
    status = _wait_for_terminal(store, handle.run_id)

    assert status.status == "failed"
    assert "signal 9" in (status.error or "")
    assert "OOM" in (status.error or "")


def test_cancelled_workload_is_still_reported_as_cancelled(tmp_path):
    """The other half of #7: a real cancellation must not become a failure."""
    store = _store(tmp_path)
    entrypoint = _script(
        tmp_path / "sleeper.py",
        "import time\nprint('training', flush=True)\ntime.sleep(300)\n",
    )
    backend = LocalBackend(store=store)

    handle = backend.submit(_manifest(tmp_path, entrypoint, working_dir=str(tmp_path)))
    workload_pid = _wait_for_workload_pid(store, handle.run_id)
    _wait_for_event(store, handle.run_id, "training")
    backend.cancel(handle.run_id)
    status = _wait_for_terminal(store, handle.run_id)

    assert status.status == "cancelled"
    _wait_for(lambda: not _alive(workload_pid))
    assert not _alive(workload_pid)


def test_cancel_does_not_rewrite_a_finished_run(tmp_path):
    """Cancelling a run that already ended used to overwrite how it ended --
    including a failure it is meant to help diagnose."""
    store = _store(tmp_path)
    handle = LocalBackend(store=store).submit(
        _manifest(tmp_path, f"{sys.executable} -c pass", working_dir=str(tmp_path))
    )
    _wait_for_terminal(store, handle.run_id)

    LocalBackend(store=store).cancel(handle.run_id)

    assert store.read_status(handle.run_id).status == "completed"
    assert not store.cancel_requested(handle.run_id)


def test_cli_cancel_reports_that_a_finished_run_was_not_cancelled(tmp_path):
    from typer.testing import CliRunner

    from ai_experiments.cli import app

    store = _store(tmp_path)
    handle = LocalBackend(store=store).submit(
        _manifest(tmp_path, f"{sys.executable} -c pass", working_dir=str(tmp_path))
    )
    _wait_for_terminal(store, handle.run_id)

    result = CliRunner().invoke(
        app, ["cancel", handle.run_id, "--runs-dir", str(store.root)]
    )

    assert result.exit_code == 0
    assert "already completed" in result.stdout


# -- #8: an orphaned workload -------------------------------------------------


def test_reaper_kills_a_workload_whose_supervisor_died(tmp_path):
    """Issue #8: the daemon reported `reaped_dead_process` while the training
    process kept running, unsupervised and invisible."""
    store = _store(tmp_path)
    entrypoint = _script(
        tmp_path / "sleeper.py",
        "import time\nprint('training', flush=True)\ntime.sleep(300)\n",
    )
    handle = LocalBackend(store=store).submit(
        _manifest(tmp_path, entrypoint, working_dir=str(tmp_path))
    )
    workload_pid = _wait_for_workload_pid(store, handle.run_id)
    _wait_for_event(store, handle.run_id, "training")
    supervisor_pid = _kill_supervisor(store, handle.run_id, reap=True)
    assert not _alive(supervisor_pid)

    try:
        report = MonitorDaemon(store).tick()

        # The load-bearing assertion: nothing of this run may still be
        # running once the daemon has declared it dead.
        _wait_for(lambda: not _alive(workload_pid))
        assert not _alive(workload_pid), "the reaper left the workload running"
        status = store.read_status(handle.run_id)
        assert status.status == "failed"
        assert "orphaned workload" in (status.error or "")
        actions = {action.action: action for action in report.actions}
        assert "reaped_dead_process" in actions, (
            f"{report}\nstatus={status}\n"
            f"supervisor_identity={process_identity(supervisor_pid)}"
        )
        reap = actions["reaped_dead_process"].details["workload_reap"]
        assert reap["outcome"] in {"terminated", "killed"}
        assert reap["pid"] == workload_pid
    finally:
        if _alive(workload_pid):  # pragma: no cover - only on failure
            os.kill(workload_pid, signal.SIGKILL)


def test_dead_but_unreaped_supervisor_is_not_reported_healthy(tmp_path):
    """A supervisor whose parent outlives it (the campaign daemon, `iax
    serve`) stays in the process table as a zombie, and `kill(pid, 0)` still
    succeeds for one -- so liveness built on that alone leaves a dead
    supervisor's run `running` forever, supervised by nobody."""
    store = _store(tmp_path)
    entrypoint = _script(
        tmp_path / "sleeper.py",
        "import time\nprint('training', flush=True)\ntime.sleep(300)\n",
    )
    handle = LocalBackend(store=store).submit(
        _manifest(tmp_path, entrypoint, working_dir=str(tmp_path))
    )
    workload_pid = _wait_for_workload_pid(store, handle.run_id)
    _wait_for_event(store, handle.run_id, "training")
    supervisor_pid = _kill_supervisor(store, handle.run_id, reap=False)
    assert _alive(supervisor_pid)  # a zombie: dead, but still in the table

    try:
        report = MonitorDaemon(store).tick()

        actions = [action.action for action in report.actions]
        assert "reaped_dead_process" in actions, report
        assert store.read_status(handle.run_id).status == "failed"
    finally:
        os.waitpid(supervisor_pid, 0)
        if _alive(workload_pid):  # pragma: no cover - only on failure
            os.kill(workload_pid, signal.SIGKILL)


def test_reaper_refuses_to_signal_a_reused_pid(tmp_path):
    """The recorded pid is only a handle if it still names the same process.
    A pid the kernel has handed to something else must never be signalled."""
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
            details={
                "workload_pid": bystander.pid,
                "workload_identity": "1",  # not the bystander's start time
            },
        )

        report = LocalBackend(store=store).reap(run_id)

        assert report["outcome"] == "identity_mismatch"
        assert bystander.poll() is None  # untouched
    finally:
        bystander.kill()
        bystander.wait()


def test_reaper_reports_a_workload_it_cannot_identify(tmp_path):
    """Runs from before identities were recorded still get an honest report:
    a live pid nobody may signal is worse hidden than announced."""
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
            details={"workload_pid": bystander.pid},
        )

        report = LocalBackend(store=store).reap(run_id)

        assert report["outcome"] == "identity_unverifiable"
        assert bystander.poll() is None
        messages = [event.message for event in store.read_events(run_id)]
        assert any("left running" in message for message in messages)
    finally:
        bystander.kill()
        bystander.wait()


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


def test_the_refusal_names_the_fix_when_the_platform_is_the_reason(
    monkeypatch, tmp_path
):
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
