from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path

from ai_experiments.backends.base import ExperimentBackend
from ai_experiments.monitoring.rules import diagnose_run
from ai_experiments.schemas import (
    DiagnosisReport,
    ExperimentManifest,
    RunEvent,
    RunHandle,
    RunStatus,
    utc_now,
)
from ai_experiments.store import FilesystemRunStore


class LocalBackend(ExperimentBackend):
    """Detached local-process backend for arbitrary training workloads."""

    def __init__(self, store: FilesystemRunStore | None = None) -> None:
        self.store = store or FilesystemRunStore()

    def submit(self, manifest: ExperimentManifest) -> RunHandle:
        run_id, run_dir = self.store.create_run(manifest)
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
            cwd=Path(manifest.workload.working_dir).resolve(),
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
        if status.pid and status.status in {"submitted", "running"}:
            os.killpg(status.pid, signal.SIGTERM)
            self.store.append_event(run_id, RunEvent(message="local run cancelled"))
        self.store.update_status(run_id, status="cancelled", completed_at=utc_now())

    def diagnose(self, run_id: str) -> DiagnosisReport:
        return diagnose_run(self.store, run_id)
