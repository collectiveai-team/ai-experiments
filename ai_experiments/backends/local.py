from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path

from ai_experiments.backends.base import ExperimentBackend
from ai_experiments.monitoring.rules import diagnose_run
from ai_experiments.procs import terminate_workload
from ai_experiments.schemas import (
    DiagnosisReport,
    ExperimentManifest,
    RunEvent,
    RunHandle,
    RunStatus,
    utc_now,
)
from ai_experiments.store import FilesystemRunStore

#: Run states a cancellation can still act on.
ACTIVE_RUN_STATES = {"submitted", "running"}


class LocalBackend(ExperimentBackend):
    """Detached local-process backend for arbitrary training workloads."""

    def __init__(self, store: FilesystemRunStore | None = None) -> None:
        self.store = store or FilesystemRunStore()

    def submit(self, manifest: ExperimentManifest) -> RunHandle:
        run_id, run_dir = self.store.create_run(manifest)
        # Take the working dir from the persisted manifest rather than
        # resolving it again here: the store resolved it once, against this
        # process's CWD, and the supervisor will read that same file. Two
        # resolutions of one relative path is exactly how `sub` became
        # `sub/sub`.
        executed = self.store.read_manifest(run_id) or manifest
        working_dir = executed.workload.working_dir
        status_path = self.store.status_path(run_id)
        handle = RunHandle(
            run_id=run_id,
            backend="local",
            status="submitted",
            status_uri=str(status_path),
            run_dir=str(run_dir),
        )
        self.store.write_handle(handle)
        self.store.update_status(
            run_id,
            details={
                "stuck_after_minutes": manifest.monitoring.stuck_after_minutes,
                "timeout_seconds": manifest.monitoring.timeout_seconds,
                "experiment": manifest.experiment,
            },
        )
        self.store.append_event(run_id, RunEvent(message="local run submitted"))

        from ai_experiments.tracking import begin_tracking

        begin_tracking(self.store, run_id, manifest)

        cmd = [
            sys.executable,
            "-m",
            "ai_experiments.worker",
            "--run-id",
            run_id,
            "--runs-dir",
            str(self.store.root),
        ]
        log_path = run_dir / "worker.log"
        log_file = log_path.open("a")
        env = os.environ.copy()
        package_root = Path(__file__).resolve().parents[2]
        env["PYTHONPATH"] = f"{package_root}:{env.get('PYTHONPATH', '')}"
        process = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=working_dir,
            env=env,
            start_new_session=True,
        )
        self.store.update_status(
            run_id, pid=process.pid, details={"log_path": str(log_path)}
        )
        self.store.append_event(
            run_id,
            RunEvent(message="local supervisor started", details={"pid": process.pid}),
        )
        return handle

    def inspect(self, run_id: str) -> RunStatus:
        return self.store.read_status(run_id)

    def logs(self, run_id: str, tail: int = 200) -> list[RunEvent]:
        return self.store.read_events(run_id, tail=tail)

    def cancel(self, run_id: str) -> None:
        status = self.store.read_status(run_id)
        if status.status not in ACTIVE_RUN_STATES:
            # Already finished. Rewriting it as cancelled would erase how the
            # run actually ended -- including a workload the kernel OOM-killed
            # a moment ago, which is exactly the confusion cancel/kill
            # reporting is supposed to remove.
            return
        # Record the intent *before* signalling: the supervisor reads this to
        # tell "iax stopped it" from "something else killed it", and the
        # signal may beat any bookkeeping that follows it.
        self.store.request_cancel(run_id)
        if status.pid:
            try:
                os.killpg(status.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass  # supervisor already gone; the status write below stands
            self.store.append_event(run_id, RunEvent(message="local run cancelled"))
        self.store.update_status(run_id, status="cancelled", completed_at=utc_now())

    def reap(self, run_id: str) -> dict[str, object]:
        """Terminate a workload whose supervisor died, if it is still running.

        Nothing else can: the supervisor is the workload's parent and the only
        process that would have noticed. Its recorded pid is the sole handle
        on an orphan that is still holding a GPU.
        """
        status = self.store.read_status(run_id)
        report = terminate_workload(
            status.details.get("workload_pid"),
            status.details.get("workload_identity"),
        )
        outcome = report.get("outcome")
        if outcome in {"terminated", "killed"}:
            self.store.append_event(
                run_id,
                RunEvent(
                    level="warning",
                    message="orphaned workload terminated",
                    details=report,
                ),
            )
        elif outcome in {"identity_unverifiable", "identity_mismatch", "kill_failed"}:
            self.store.append_event(
                run_id,
                RunEvent(
                    level="error",
                    message=f"orphaned workload left running ({outcome})",
                    details=report,
                ),
            )
        return report

    def diagnose(self, run_id: str) -> DiagnosisReport:
        return diagnose_run(self.store, run_id)
