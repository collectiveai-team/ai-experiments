from __future__ import annotations

import os
import shlex
import signal
import subprocess
import threading
from pathlib import Path
from typing import TYPE_CHECKING

import typer

from ai_experiments.monitoring.rules import event_from_log_line
from ai_experiments.report import parse_metric_line
from ai_experiments.schemas import ExperimentManifest, MetricPoint, RunEvent, utc_now
from ai_experiments.store import FilesystemRunStore

if TYPE_CHECKING:
    from types import FrameType

HEARTBEAT_SECONDS = 15

app = typer.Typer(add_completion=False)


class _Supervisor:
    """Runs one workload process, streaming logs/metrics into the run store."""

    def __init__(self, store: FilesystemRunStore, run_id: str) -> None:
        self.store = store
        self.run_id = run_id
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.process: subprocess.Popen[str] | None = None

    def _update_status(self, **updates: object) -> None:
        with self._lock:
            self.store.update_status(self.run_id, **updates)

    def _heartbeat_loop(self) -> None:
        while not self._stop.wait(HEARTBEAT_SECONDS):
            self._update_status(details={"heartbeat_at": utc_now().isoformat()})

    def _handle_sigterm(self, signum: int, frame: FrameType | None) -> None:
        if self.process and self.process.poll() is None:
            self.process.terminate()

    def run(self) -> None:
        run_dir = self.store.run_dir(self.run_id)
        manifest = ExperimentManifest.from_yaml(run_dir / "manifest.yaml")

        command = [*shlex.split(manifest.workload.entrypoint), *manifest.workload.args]
        env = os.environ.copy()
        env.update(manifest.workload.env)
        env["IAX_RUN_ID"] = self.run_id
        env["IAX_RUN_DIR"] = str(run_dir)
        artifacts_dir = run_dir / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)
        env["IAX_ARTIFACTS_DIR"] = str(artifacts_dir)

        # MLflow handoff: workloads that import mlflow attach to the run the
        # harness created at submit time.
        details = self.store.read_status(self.run_id).details
        if details.get("mlflow_run_id"):
            env["MLFLOW_RUN_ID"] = str(details["mlflow_run_id"])
            env["MLFLOW_TRACKING_URI"] = str(details.get("mlflow_tracking_uri", ""))

        self._update_status(
            status="running",
            started_at=utc_now(),
            details={"heartbeat_at": utc_now().isoformat()},
        )
        self.store.append_event(
            self.run_id,
            RunEvent(message="workload started", details={"command": command}),
        )

        signal.signal(signal.SIGTERM, self._handle_sigterm)
        # The command is the user's own workload entrypoint: launching it is what this
        # worker exists to do, so there is no untrusted input to validate away.
        self.process = subprocess.Popen(  # noqa: S603  # user-supplied workload entrypoint
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=Path(manifest.workload.working_dir).resolve(),
            env=env,
        )
        self._update_status(details={"workload_pid": self.process.pid})

        heartbeat = threading.Thread(target=self._heartbeat_loop, daemon=True)
        heartbeat.start()

        assert self.process.stdout is not None  # noqa: S101  # type narrowing, not a runtime check
        for line in self.process.stdout:
            metric = parse_metric_line(line)
            if metric is not None:
                point = MetricPoint(step=metric["step"], values=metric["values"])
                self.store.append_metric(self.run_id, point)
                self._update_status(
                    details={
                        "last_metric_at": point.timestamp.isoformat(),
                        "last_step": point.step,
                        "last_metrics": point.values,
                    }
                )
            else:
                self.store.append_event(self.run_id, event_from_log_line(line))

        exit_code = self.process.wait()
        self._stop.set()
        if exit_code == 0:
            self._update_status(status="completed", exit_code=exit_code, completed_at=utc_now())
            self.store.append_event(self.run_id, RunEvent(message="workload completed"))
        elif exit_code in (-signal.SIGTERM, -signal.SIGKILL):
            self._update_status(
                status="cancelled",
                exit_code=exit_code,
                completed_at=utc_now(),
                error=f"workload terminated by signal {-exit_code}",
            )
            self.store.append_event(
                self.run_id,
                RunEvent(level="warning", message="workload terminated"),
            )
        else:
            self._update_status(
                status="failed",
                exit_code=exit_code,
                completed_at=utc_now(),
                error=f"workload exited with code {exit_code}",
            )
            self.store.append_event(
                self.run_id,
                RunEvent(
                    level="error",
                    message="workload failed",
                    details={"exit_code": exit_code},
                ),
            )


def _require_nonempty_runs_dir(value: str) -> str:
    """Reject an empty ``--runs-dir`` instead of silently redefining it.

    ``argparse`` delivered ``""`` as a falsy ``str``, so ``FilesystemRunStore("")``
    fell back to its default runs directory. A ``Path``-typed option instead
    turns ``""`` into ``Path(".")``, which is truthy and points at the
    process's cwd -- a different, equally silent behavior. Neither is
    defensible, so an empty value is now a loud usage error. The sole caller
    (`backends/local.py`) always passes ``str(self.store.root)``, never
    empty, so this cannot affect it.
    """
    if not value.strip():
        raise typer.BadParameter("must not be empty")
    return value


@app.command()
def main(
    run_id: str = typer.Option(..., "--run-id", help="Run id to supervise."),
    runs_dir: str = typer.Option(
        ...,
        "--runs-dir",
        help="Run store root.",
        callback=_require_nonempty_runs_dir,
    ),
) -> None:
    """Supervise one run: spawn its workload and stream metrics into the store."""
    store = FilesystemRunStore(Path(runs_dir))
    _Supervisor(store, run_id).run()


if __name__ == "__main__":
    app()
