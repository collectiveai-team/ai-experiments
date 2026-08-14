"""Process identity and termination for locally supervised runs.

A bare pid is not a durable handle. The kernel reuses pids, so a pid recorded
in ``status.json`` minutes ago may name an unrelated process by the time
anything acts on it -- and signalling that process would be a far worse bug
than the orphan it was meant to clean up.

Every decision here is therefore gated on an *identity*: the process start
time the kernel reports in ``/proc/<pid>/stat``, which is fixed for the life
of a process and cannot be shared by a later process that reuses its pid. A
pid whose identity cannot be confirmed is never signalled.
"""

from __future__ import annotations

import os
import signal
import time
from pathlib import Path

#: How long a workload gets to exit on SIGTERM before it is SIGKILLed.
TERMINATE_GRACE_SECONDS = 5.0


def pid_alive(pid: int) -> bool:
    """Whether ``pid`` names a live process.

    A zombie is not alive. This matters for supervision: when the process that
    submitted a run outlives the supervisor it spawned (the campaign daemon,
    ``iax serve``), a crashed supervisor stays in the process table as an
    unreaped child of that process. ``kill(pid, 0)`` still succeeds for it, so
    a liveness check built on that alone reports a dead supervisor as healthy
    and the run is supervised by nobody, forever.
    """
    if _proc_available():
        return process_identity(pid) is not None
    # No /proc (macOS, Windows): fall back to signal 0, which cannot see the
    # difference. Better a missed zombie than reaping every healthy run.
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _proc_available() -> bool:
    return Path("/proc/self/stat").exists()


def process_identity(pid: int) -> str | None:
    """Kernel start time of ``pid``, or ``None`` when there is no such process.

    The value is opaque: callers only ever compare it to one recorded earlier
    for the same pid. ``None`` means "not a running process" -- the pid is
    gone, is a zombie awaiting reaping, or the platform does not expose
    ``/proc`` (macOS, Windows), in which case identity cannot be established
    at all and callers must refuse to signal.
    """
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    # Field 2 (comm) is parenthesised and may itself contain spaces and
    # parentheses, so the only safe split point is after its closing paren.
    try:
        fields = stat[stat.rindex(")") + 1 :].split()
        state, start_ticks = fields[0], fields[19]
    except (ValueError, IndexError):
        return None
    if state == "Z":  # exited, still in the table until its parent reaps it
        return None
    return start_ticks


def terminate_workload(
    pid: object,
    identity: object,
    grace_seconds: float = TERMINATE_GRACE_SECONDS,
    poll_seconds: float = 0.1,
) -> dict[str, object]:
    """Terminate a workload whose supervisor is gone, if it can be identified.

    Returns a report whose ``outcome`` is one of:

    ``not_recorded``
        No workload pid was ever recorded (the supervisor died before spawn).
    ``already_exited``
        Nothing to do.
    ``identity_unverifiable``
        A live pid, but no identity to check it against -- refused, because
        the pid may since have been reused. Reported so an operator can act.
    ``identity_mismatch``
        The pid has been reused by an unrelated process -- refused.
    ``terminated`` / ``killed``
        Gone, after SIGTERM / after escalating to SIGKILL.
    ``kill_failed``
        Still alive after SIGKILL, or not ours to signal.
    """
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return {"outcome": "not_recorded"}
    report: dict[str, object] = {"pid": pid}
    current = process_identity(pid)
    if current is None:
        return {**report, "outcome": "already_exited"}
    if not isinstance(identity, str) or not identity:
        return {**report, "outcome": "identity_unverifiable"}
    if current != identity:
        return {**report, "outcome": "identity_mismatch"}

    # The workload's process group holds every process the run spawned (the
    # supervisor was started with start_new_session, so the group is the run's
    # own). Taking the pgid from a pid whose identity is confirmed is safe:
    # the kernel cannot reuse a pid while it is still in use as a group id.
    try:
        pgid: int | None = os.getpgid(pid)
    except OSError:
        pgid = None
    if pgid is not None:
        report["pgid"] = pgid

    if not _signal(pid, pgid, signal.SIGTERM):
        return {**report, "outcome": "kill_failed", "error": "SIGTERM refused"}
    if _wait_for_exit(pid, identity, grace_seconds, poll_seconds):
        return {**report, "outcome": "terminated"}

    if not _signal(pid, pgid, signal.SIGKILL):
        return {**report, "outcome": "kill_failed", "error": "SIGKILL refused"}
    if _wait_for_exit(pid, identity, grace_seconds, poll_seconds):
        return {**report, "outcome": "killed"}
    return {**report, "outcome": "kill_failed", "error": "alive after SIGKILL"}


def _signal(pid: int, pgid: int | None, sig: int) -> bool:
    """Signal the workload's process group, falling back to the bare pid."""
    try:
        if pgid is not None:
            os.killpg(pgid, sig)
        else:
            os.kill(pid, sig)
    except ProcessLookupError:
        return True  # already gone; that is the outcome we wanted
    except PermissionError:
        return False
    return True


def _wait_for_exit(
    pid: int, identity: str, grace_seconds: float, poll_seconds: float
) -> bool:
    deadline = time.monotonic() + grace_seconds
    while True:
        if process_identity(pid) != identity:
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(poll_seconds)
