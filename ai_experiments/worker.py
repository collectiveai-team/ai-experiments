from __future__ import annotations

import io
import os
import shlex
import signal
import subprocess
import sys
import threading
import traceback
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING

import typer

from ai_experiments.config_loading import load_stored
from ai_experiments.failures import ERROR_TAIL_LINES, failure_message
from ai_experiments.monitoring.rules import event_from_log_line
from ai_experiments.procs import (
    IDENTITY_UNAVAILABLE_HINT,
    identity_supported,
    process_identity,
)
from ai_experiments.report import parse_metric_line
from ai_experiments.schemas import (
    ExperimentManifest,
    MetricPoint,
    RunEvent,
    utc_now,
)
from ai_experiments.store import FilesystemRunStore

if TYPE_CHECKING:
    from collections.abc import Mapping
    from types import FrameType

HEARTBEAT_SECONDS = 15

app = typer.Typer(add_completion=False)

#: States the supervisor may still write a final status over. Anything else is
#: already terminal and must not be rewritten by the failure contract.
NON_TERMINAL_STATES = {"submitted", "running", "unknown"}


#: Variables that describe *the harness's* environment, not the workload's.
#: `uv`, `poetry` and `conda` all read VIRTUAL_ENV, and a workload that
#: manages its own environment either warns and ignores it or, worse,
#: resolves against the wrong interpreter. A workload that genuinely wants
#: one sets it in ``workload.env``.
_HARNESS_ONLY_VARS = ("VIRTUAL_ENV",)


def workload_env(  # ast-grep-ignore: no-dict-return-annotation
    base: Mapping[str, str], manifest: ExperimentManifest
) -> dict[str, str]:
    """Build the environment the workload runs in: ours, minus what is ours alone."""
    env = {k: v for k, v in base.items() if k not in _HARNESS_ONLY_VARS}
    env.update(manifest.workload.env)
    return env


class _Supervisor:
    """Runs one workload process, streaming logs/metrics into the run store."""

    def __init__(self, store: FilesystemRunStore, run_id: str) -> None:
        self.store = store
        self.run_id = run_id
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._cancelled = False
        self.process: subprocess.Popen[bytes] | None = None

    def _update_status(self, **updates: object) -> None:
        with self._lock:
            self.store.update_status(self.run_id, **updates)

    def _heartbeat_loop(self) -> None:
        while not self._stop.wait(HEARTBEAT_SECONDS):
            self._update_status(details={"heartbeat_at": utc_now().isoformat()})

    def _handle_sigterm(self, signum: int, frame: FrameType | None) -> None:
        # Being asked to stop *is* the cancellation: whoever sent this signal
        # (iax cancel killpg's the whole group, so it lands here too) wanted
        # the run stopped, which is what distinguishes it from a kernel OOM
        # kill of the workload alone.
        self._cancelled = True
        if self.process and self.process.poll() is None:
            self.process.terminate()

    def _cancel_requested(self) -> bool:
        return self._cancelled or self.store.cancel_requested(self.run_id)

    def _working_dir(self, manifest: ExperimentManifest) -> Path:
        """Return the directory to run the workload in, taken as given.

        The supervisor's own CWD is already the working dir (the backend
        starts it there), so resolving a relative path here would resolve it a
        second time -- ``working_dir: sub`` became ``sub/sub``. The store
        resolves it once at submit time, so a persisted manifest is always
        absolute; anything else is a corrupted run directory and is reported
        rather than guessed at.
        """
        working_dir = Path(manifest.workload.working_dir)
        if not working_dir.is_absolute():
            raise RuntimeError(
                f"working_dir must be absolute in a persisted manifest, got "
                f"{manifest.workload.working_dir!r}: "
                f"{self.store.run_dir(self.run_id) / 'manifest.yaml'} was not "
                "written by this version of iax"
            )
        if not working_dir.is_dir():
            raise RuntimeError(f"working_dir does not exist: {working_dir}")
        return working_dir

    def run(self) -> None:
        run_dir = self.store.run_dir(self.run_id)
        manifest = load_stored(ExperimentManifest, run_dir / "manifest.yaml")

        command = [*shlex.split(manifest.workload.entrypoint), *manifest.workload.args]
        working_dir = self._working_dir(manifest)
        env = workload_env(os.environ, manifest)
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
            cwd=working_dir,
            env=env,
        )
        # workload_identity pins which process that pid is, so a later reaper
        # cannot signal an unrelated process that inherited the number.
        identity = process_identity(self.process.pid)
        self._update_status(
            details={
                "workload_pid": self.process.pid,
                "workload_identity": identity,
            }
        )
        if identity is None and not identity_supported():
            # Say it once, here, where the run's own log keeps it. Otherwise
            # the guarantee is off and only a return value nobody reads says
            # so (#31).
            self.store.append_event(
                self.run_id,
                RunEvent(
                    level="warning",
                    message="orphan reaping is disabled on this machine",
                    details={"reason": IDENTITY_UNAVAILABLE_HINT},
                ),
            )

        heartbeat = threading.Thread(target=self._heartbeat_loop, daemon=True)
        heartbeat.start()

        assert self.process.stdout is not None  # noqa: S101  # type narrowing, not a runtime check
        recent: deque[str] = deque(maxlen=ERROR_TAIL_LINES)
        # A binary pipe wrapped with newline="\n". In text mode python splits
        # on "\r" too, so every refresh of a progress bar became its own line
        # -- and then its own event. errors="replace" because workload output
        # is untrusted, and one undecodable byte must not kill the supervisor.
        stream = io.TextIOWrapper(
            self.process.stdout, encoding="utf-8", errors="replace", newline="\n"
        )
        self._stream_output(stream, recent)

        exit_code = self.process.wait()
        self._stop.set()
        self._record_exit(exit_code, recent)

    def _stream_output(self, stream: io.TextIOWrapper, recent: deque[str]) -> None:
        """Route each line of workload output to the metric series or the event log."""
        for raw in stream:
            line = _overwrite(raw)
            if not line:
                continue
            metric = parse_metric_line(line)
            if metric is not None:
                point = MetricPoint(step=metric.step, values=metric.values)
                self.store.append_metric(self.run_id, point)
                self._update_status(
                    details={
                        "last_metric_at": point.timestamp.isoformat(),
                        "last_step": point.step,
                        "last_metrics": point.values,
                    }
                )
            else:
                recent.append(line)
                self.store.append_event(self.run_id, event_from_log_line(line))

    def _record_exit(self, exit_code: int, recent: deque[str]) -> None:
        """Write the final status and event the workload's exit code implies.

        Which of the four endings this was is the whole point: a signal nobody
        asked for reads very differently from a cancellation, and both read
        differently from a workload that simply returned non-zero.
        """
        if exit_code == 0:
            self._update_status(status="completed", exit_code=exit_code, completed_at=utc_now())
            self.store.append_event(self.run_id, RunEvent(message="workload completed"))
        elif exit_code < 0 and self._cancel_requested():
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
        elif exit_code < 0:
            # Nobody asked for this. SIGKILL in particular is what the kernel
            # OOM killer sends, and reporting that as "cancelled" sends the
            # operator looking for a person instead of for memory.
            signal_name = _signal_name(-exit_code)
            error = f"workload killed by signal {-exit_code} ({signal_name})"
            if -exit_code == signal.SIGKILL:
                error += "; no cancellation was requested, so this is most "
                error += "likely the kernel OOM killer"
            self._update_status(
                status="failed",
                exit_code=exit_code,
                completed_at=utc_now(),
                error=error,
            )
            self.store.append_event(
                self.run_id,
                RunEvent(
                    level="error",
                    message="workload killed",
                    details={"signal": -exit_code, "signal_name": signal_name},
                ),
            )
        else:
            self._update_status(
                status="failed",
                exit_code=exit_code,
                completed_at=utc_now(),
                error=failure_message(f"workload exited with code {exit_code}", list(recent)),
            )
            self.store.append_event(
                self.run_id,
                RunEvent(
                    level="error",
                    message="workload failed",
                    details={"exit_code": exit_code},
                ),
            )


def _overwrite(raw: str) -> str:
    r"""Collapse a carriage-return sequence the way a terminal displays it.

    `tqdm` and every other progress bar rewrites one line with "\r". Only the
    final state of that line carries information; the refreshes before it are
    the animation, and storing each one turned a single bar into thousands of
    events in a file that nothing bounds.
    """
    return raw.rsplit("\r", 1)[-1].rstrip("\n").rstrip()


def _signal_name(signum: int) -> str:
    try:
        return signal.Signals(signum).name
    except ValueError:
        return "unknown signal"


def report_supervisor_failure(store: FilesystemRunStore, run_id: str, exc: BaseException) -> None:
    """Turn a crashed supervisor into an honest ``failed`` status.

    Without this the run keeps whatever status it had -- ``running``, usually
    -- forever, and the only evidence is a traceback in ``worker.log``, a file
    no command reads. Writing the failure here rather than at each fallible
    call site covers every way ``run()`` can raise at once: a manifest that no
    longer parses, an unsplittable entrypoint, an unwritable artifacts dir, an
    unreadable status, a workload that cannot spawn, a decoding error mid-log.
    """
    error = f"supervisor failed: {type(exc).__name__}: {exc}"
    try:
        if store.read_status(run_id).status in NON_TERMINAL_STATES:
            store.update_status(
                run_id,
                status="failed",
                completed_at=utc_now(),
                error=error,
            )
        store.append_event(
            run_id,
            RunEvent(
                level="error",
                message=error,
                details={"traceback": traceback.format_exc()},
            ),
        )
    except Exception:  # pragma: no cover - the store itself is unusable
        # Reporting is best-effort: the traceback still reaches worker.log
        # below, and exiting non-zero still tells the parent something broke.
        traceback.print_exc(file=sys.stderr)


def _require_nonempty_runs_dir(value: str) -> str:
    """Reject an empty ``--runs-dir`` instead of silently redefining it.

    An empty value is not a missing one. ``FilesystemRunStore("")`` resolves
    ``Path("")`` to the process's cwd, so an empty ``--runs-dir`` silently
    supervises a run in a different store than the caller meant -- the same
    class of bug as #21, arriving through the flag instead of the search. The
    sole caller (`backends/local.py`) always passes ``str(self.store.root)``,
    never empty, so rejecting it cannot affect a real invocation.
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
    try:
        _Supervisor(store, run_id).run()
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)  # keep the evidence in worker.log
        report_supervisor_failure(store, run_id, exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    app()
