from __future__ import annotations

import secrets
import shlex
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ai_experiments.backends.base import ExperimentBackend
from ai_experiments.failures import failure_message
from ai_experiments.monitoring.ray_rules import classify_ray_condition
from ai_experiments.monitoring.rules import diagnose_run
from ai_experiments.report import parse_metric_line, parse_result_line
from ai_experiments.schemas import (
    ACTIVE_RUN_STATES,
    DataSpec,
    DiagnosisReport,
    ExperimentManifest,
    MetricPoint,
    ResultRecord,
    RunEvent,
    RunHandle,
    RunStatus,
    utc_now,
)
from ai_experiments.settings import get_settings
from ai_experiments.store import FilesystemRunStore

if TYPE_CHECKING:
    from collections.abc import Callable

DEFAULT_RAY_ADDRESS = "http://127.0.0.1:8265"

#: The marker's wire format, written once so the emitter (`submit`) and the
#: parser (`_sync_results_from_logs`) cannot drift apart (review finding 9).
PHASE_MARKER_PREFIX = "IAX_PHASE="


def _phase_marker(phase: str, token: str) -> str:
    """Build the line `submit` echoes ahead of a phase and the parser looks for.

    What the token buys is visibility and cost, not impossibility (commit
    9b89e17). A workload sharing the job's stdout can print a line starting
    `IAX_PHASE=evaluate` with one `print()`, and the token means that line is
    discarded rather than believed. It is not in the workload's environment --
    but it *is* in the job's entrypoint shell string, which Ray runs as the
    workload's parent, so `/proc/$PPID/cmdline` or `ps` hands it to a workload
    that goes looking, and Ray stores the entrypoint in the job's metadata
    besides. So the marker separates a noisy workload from the entrypoint
    reliably, and a determined one only by making the forgery deliberate
    rather than accidental. See `_sync_results_from_logs` for what this does
    and does not close, and prefer the local backend where the split must hold
    against workload code nobody read.
    """
    return f"{PHASE_MARKER_PREFIX}{phase} {token}"


def _parse_phase_marker(line: str, token: str) -> str | None:
    """Return the phase name if `line` is a genuine marker, else None.

    Matched by searching within the line -- like `report.parse_result_line`,
    not `startswith` -- because Ray prefixes log lines (``(pid=123) ...``) in
    some configurations, and a prefix must not break attribution while the
    result on the next line still parses (finding 7). A line that merely
    starts with ``IAX_PHASE=`` but carries the wrong token, no token, or a
    tampered phase name is not a marker at all: it is workload noise (or a
    forgery attempt) and the caller ignores it rather than trusting it.
    """
    stripped = line.strip()
    idx = stripped.find(PHASE_MARKER_PREFIX)
    if idx == -1:
        return None
    rest = stripped[idx + len(PHASE_MARKER_PREFIX) :]
    phase, _, candidate_token = rest.partition(" ")
    if not token or candidate_token != token:
        return None
    return phase


def resolve_ray_address(address: str | None = None) -> str:
    if address is not None:
        stripped = address.strip()
        if not stripped:
            raise ValueError("Ray address must not be empty")
        return stripped
    env_address = get_settings().ray_address
    if env_address and env_address.strip():
        return env_address.strip()
    return DEFAULT_RAY_ADDRESS


class RayBackend(ExperimentBackend):
    """Detached Ray Jobs backend.

    This adapter intentionally uses the Ray Jobs API when available so submit
    returns a run handle instead of waiting on Ray object refs.
    """

    def __init__(
        self,
        store: FilesystemRunStore | None = None,
        address: str | None = None,
        client_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self.store = store or FilesystemRunStore()
        self.address = resolve_ray_address(address)
        self._client_factory = client_factory

    def _client(self) -> Any:
        if self._client_factory is not None:
            return self._client_factory(self.address)
        try:
            from ray.job_submission import JobSubmissionClient
        except ImportError as exc:  # pragma: no cover - depends on optional ray extra
            raise RuntimeError("Ray is not installed. Install ai-experiments[ray].") from exc
        return JobSubmissionClient(self.address)

    def submit(self, manifest: ExperimentManifest) -> RunHandle:
        run_id, run_dir = self.store.create_run(manifest)
        status_path = self.store.status_path(run_id)
        # Ray takes a shell string, so every argument has to survive the
        # shell's own word splitting. `" ".join` did not: an argument with a
        # space arrived at the workload as two.
        args = shlex.join(manifest.workload.args)
        # Minted once per run and never exported to the workload: the
        # workload shares the job's stdout, so a plain `IAX_PHASE=evaluate`
        # marker is forgeable with one `print()`. The token lets the parser
        # tell "the entrypoint said so" from "the workload said so" on the
        # same stream. It is not a secret the workload cannot obtain -- it is
        # interpolated into the entrypoint below, and that shell is the
        # workload's own parent process -- so this raises the cost of a
        # forgery and makes one visible; it does not make one impossible.
        phase_token = secrets.token_hex(16)
        # The handoff directory is created once, inside the job's working dir,
        # and both phases see the same absolute path.
        commands = ["mkdir -p iax_work", "export IAX_WORK_DIR=$PWD/iax_work"]
        for phase, command in manifest.workload.phases():
            data_env = manifest.workload.data.env_for(phase)
            prefix = " ".join(
                f"{name}={shlex.quote(value)}"
                for name, value in sorted({**data_env, "IAX_PHASE": phase}.items())
            )
            # `_sync_results_from_logs` has no supervisor process to ask which
            # phase is running -- Ray shares no filesystem with the client, so
            # this echoed marker is the only phase signal that survives the
            # trip through `get_job_logs`. The env prefix on the next line is
            # what the workload itself reads; the two are read by different
            # sides of the same rule.
            commands.append(f"echo {_phase_marker(phase, phase_token)}")
            # A fresh `unset` ahead of every phase, not just the train one:
            # `data_env` above only *adds* keys, so a value already sitting in
            # the job's environment (`runtime_env["env_vars"]`, or whatever
            # the cluster node itself exports) would otherwise reach a phase
            # that never asked for it -- the same hole `worker.py` closes by
            # scrubbing `DataSpec.ENV_KEYS` before applying `env_for`. This has
            # to be its own `&&` link: `NAME=value cmd` prefix scoping only
            # covers the one command it decorates, so it cannot remove
            # anything from the shell that runs the phase command itself.
            commands.append("unset " + " ".join(DataSpec.ENV_KEYS))
            commands.append(f"{prefix} {command} {args}".strip())
        # `&&` and not `;`: a training phase that failed must not be followed
        # by an evaluation that would score whatever was left behind. (Note
        # for anyone tightening this further: `args` land after `command`'s
        # raw text, so on Ray -- unlike the local backend's argv list -- a
        # command containing a shell pipe or redirect puts them on the wrong
        # side of it. Pre-existing, not addressed here.)
        entrypoint = " && ".join(commands)

        # Establish the real status *first*. `begin_tracking` below records the
        # MLflow linkage via update_status, and the Ray job id is only known
        # after submit_job -- so the status file has to exist before either.
        # Writing it last (as this once did) discarded the linkage, leaving
        # runs unmirrored and stuck RUNNING in MLflow forever.
        handle = RunHandle(
            run_id=run_id,
            backend="ray",
            status="submitted",
            status_uri=str(status_path),
            run_dir=str(run_dir),
            dashboard_url=self.address,
        )
        self.store.write_handle(handle)
        self.store.update_status(
            run_id,
            details={
                "stuck_after_minutes": manifest.monitoring.stuck_after_minutes,
                "timeout_seconds": manifest.monitoring.timeout_seconds,
                "experiment": manifest.experiment,
                "ray_address": self.address,
                "ray_phase_token": phase_token,
            },
        )

        try:
            client = self._client()
        except RuntimeError as exc:
            error = "Ray is not installed. Install ai-experiments[ray]."
            self.store.update_status(run_id, status="failed", error=error, completed_at=utc_now())
            raise RuntimeError(error) from exc

        from ai_experiments.tracking import begin_tracking

        tracking_env = begin_tracking(self.store, run_id, manifest)
        # IAX_ARTIFACTS_DIR is left out on purpose: on Ray artifacts live in
        # cluster storage, and that transport is a later phase of the spec.
        # IAX_RUN_ID is the one variable the local worker injects that a
        # phase command cannot set for itself (it names the run, not the
        # phase), so it goes on the job rather than into each phase's prefix.
        runtime_env = {
            "working_dir": str(Path(manifest.workload.working_dir).resolve()),
            "env_vars": {
                **manifest.workload.env,
                **tracking_env,
                "IAX_RUN_ID": run_id,
            },
        }
        external_id = client.submit_job(entrypoint=entrypoint, runtime_env=runtime_env)
        handle.external_id = external_id
        self.store.update_status(run_id, external_id=external_id)
        self.store.append_event(
            run_id,
            RunEvent(message="ray job submitted", details={"ray_job_id": external_id}),
        )
        return handle

    def inspect(self, run_id: str) -> RunStatus:
        status = self.store.read_status(run_id)
        if not status.external_id:
            return status
        try:
            client = self._client()
            ray_status = client.get_job_status(status.external_id)
            mapped = _map_ray_status(ray_status)
            details = self._ray_details(run_id, client, status.external_id, ray_status)
            if mapped in {"completed", "failed", "cancelled"} and status.completed_at is None:
                status = self.store.update_status(
                    run_id,
                    status=mapped,
                    completed_at=utc_now(),
                    details=details,
                )
            else:
                status = self.store.update_status(run_id, status=mapped, details=details)
            if mapped == "failed" and not status.error:
                # Ray's message can carry 20,000 characters of job log. It ends
                # up in the planner's evidence block, so it gets the same tail
                # treatment as a local workload's output.
                message = details.get("ray_message") or details.get("ray_error_type")
                status = self.store.update_status(
                    run_id,
                    error=failure_message("Ray job failed", str(message or "")),
                )
            return status
        except Exception as exc:  # pragma: no cover - depends on live Ray cluster
            return self.store.update_status(run_id, error=str(exc))

    def _ray_details(  # ast-grep-ignore: no-dict-return-annotation
        self, run_id: str, client: Any, external_id: str, ray_status: Any
    ) -> dict[str, Any]:
        # An iax-invented vocabulary (ray_status/ray_address/ray_job_info/...),
        # not Ray's own keys -- those live only inside the nested ray_job_info
        # value (see _job_info_dict). This dict is consumed in-process, across
        # a module boundary, by this function's own last line:
        # classify_ray_condition (monitoring/ray_rules.py) reads ray_status,
        # ray_message, ray_error_type, ray_log_tail and ray_job_info by string
        # literal. The destination, RunStatus.details, is a genuinely
        # free-form dict[str, Any] written by several unrelated producers
        # (worker.py, tracking.py, backends/local.py), so typing one half of
        # that hop buys little.
        details: dict[str, Any] = {
            "ray_status": _ray_status_text(ray_status),
            "ray_address": self.address,
        }
        job_info = _job_info_dict(_safe_call(client, "get_job_info", external_id))
        if job_info:
            details["ray_job_info"] = job_info
            message = job_info.get("message") or job_info.get("status_message")
            if message:
                details["ray_message"] = str(message)
            error_type = job_info.get("error_type")
            if error_type:
                details["ray_error_type"] = str(error_type)

        logs = self._job_logs(run_id, client, external_id)
        if isinstance(logs, str) and logs:
            lines = logs.splitlines()
            details["ray_log_tail"] = "\n".join(lines[-50:])
            details["ray_log_line_count"] = len(lines)
            last_point = self._sync_metrics_from_logs(run_id, lines)
            if last_point is not None:
                details["last_metric_at"] = last_point.timestamp.isoformat()
                details["last_step"] = last_point.step
                details["last_metrics"] = last_point.values
            self._sync_results_from_logs(run_id, lines)

        details["ray_condition"] = classify_ray_condition(details)
        return details

    def _job_logs(self, run_id: str, client: Any, external_id: str) -> Any:
        """Read the job's logs, recording a failure instead of swallowing it.

        `_safe_call` is right for `get_job_info`, where a failure costs
        diagnostics. It is not right here: on Ray the job log is the *only*
        channel an ``IAX_RESULT`` travels on, so a transient dashboard error
        turns a trial that declared a number into ``no_result`` and the
        campaign records a miss that never happened. The fallback is
        unchanged -- a failed read must still not crash the poll -- but it
        now leaves the miss and its cause in the same run log.
        """
        method = getattr(client, "get_job_logs", None)
        if method is None:
            return None
        try:
            return method(external_id)
        except Exception as exc:
            self.store.append_event(
                run_id,
                RunEvent(
                    level="error",
                    message=(
                        "ray job logs could not be read; any result this poll "
                        "would have carried is missing, not absent"
                    ),
                    details={
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                        "external_id": external_id,
                    },
                ),
            )
            return None

    def _sync_metrics_from_logs(self, run_id: str, lines: list[str]) -> MetricPoint | None:
        """Append metric points newly seen in the job logs to the run store.

        Ray job logs carry no timestamps, so only points beyond the count
        already persisted are appended (stamped at observation time). This
        keeps metric-staleness checks honest across repeated inspects.
        """
        parsed = [m for m in (parse_metric_line(line) for line in lines) if m]
        existing = self.store.read_metrics(run_id)
        last: MetricPoint | None = existing[-1] if existing else None
        for metric in parsed[len(existing) :]:
            last = MetricPoint(step=metric.step, values=metric.values)
            self.store.append_metric(run_id, last)
        return last

    def _sync_results_from_logs(self, run_id: str, lines: list[str]) -> None:
        """Ingest IAX_RESULT lines from Ray job logs, enforcing the phase rule.

        Ray has no supervisor process in the loop, so the phase a given log
        line belongs to has to be reconstructed from the log text itself: the
        entrypoint echoes a token-bearing ``IAX_PHASE=<phase> <token>`` marker
        ahead of each phase (`_phase_marker`), and this walks the log in
        order, tracking the most recent *valid* marker as it goes -- one whose
        token matches `ray_phase_token` on this run's status. Fail-closed: the
        tracked phase starts at None, not "evaluate", and a result is scored
        only while it is exactly "evaluate". Everything else -- no marker yet,
        an unrecognised phase name, a forged marker with no or the wrong
        token -- is discarded, matching the amendment's "discard every
        IAX_RESULT seen outside evaluate" rather than only the ones seen
        during a phase literally named "train".

        What this guarantees and what it does not: the token stops a
        workload from *scoring* a forged result by printing to the shared
        stdout stream, which was the one-`print()` hole this fix closes. It
        does not make the channel authenticated end-to-end -- the token is
        itself part of the entrypoint string Ray hands to the job's shell, so
        a workload willing to read its own parent process (`/proc/$PPID/cmdline`)
        or call the Jobs API's `get_job_info().entrypoint` can still recover
        it and forge a marker that passes this check. Closing that requires
        running each phase as its own Ray job, so attribution comes from
        *which job's log this is* rather than text inside one shared log;
        that is out of scope here and tracked separately (la-tesis plan).

        Idempotent like `_sync_metrics_from_logs`, but counted differently: a
        discarded result never reaches `store.read_results`, so
        `len(existing)` alone would under-count and the same line would be
        reprocessed -- and re-warned -- on every inspect. A status detail
        tracks how many IAX_RESULT lines (accepted or discarded) have already
        been handled instead. That counter is written *after* the appends
        below, on purpose: a crash in between would re-process (and
        re-append) already-handled lines on the next inspect, but moving the
        write earlier would instead let a crash silently drop a result that
        was already scored -- worse, for a number nobody re-derives by hand.
        """
        status = self.store.read_status(run_id)
        already_seen = int(status.details.get("ray_results_seen", 0))
        token = str(status.details.get("ray_phase_token", ""))
        phase: str | None = None  # unattributed until a valid marker arrives
        seen = 0
        for line in lines:
            marker_phase = _parse_phase_marker(line, token)
            if marker_phase is not None:
                phase = marker_phase
                continue
            result = parse_result_line(line)
            if result is None:
                continue
            seen += 1
            if seen <= already_seen:
                continue
            if phase == "evaluate":
                self.store.append_result(run_id, ResultRecord(values=result))
                self.store.update_status(run_id, details={"result": result})
            elif phase == "train":
                # Byte-identical to worker.py's own message: an operator
                # scanning events for this phrase should find it on either
                # backend.
                self.store.append_event(
                    run_id,
                    RunEvent(
                        level="warning",
                        message="result reported from the train phase; discarded",
                        details={"values": result},
                    ),
                )
            else:
                # A different failure from the one above: nobody declared
                # this from train, the harness simply never saw a valid
                # marker for it (no marker yet, an unrecognised phase name,
                # or a forged marker with a bad token).
                self.store.append_event(
                    run_id,
                    RunEvent(
                        level="warning",
                        message="result reported outside a recognized phase; discarded",
                        details={"values": result},
                    ),
                )
        if seen > already_seen:
            self.store.update_status(run_id, details={"ray_results_seen": seen})

    def logs(self, run_id: str, tail: int = 200) -> list[RunEvent]:
        return self.store.read_events(run_id, tail=tail)

    def cancel(self, run_id: str) -> None:
        # Refresh from the cluster first. A Ray run has no local supervisor
        # keeping its record current -- the stored status is only as fresh as
        # the last inspect -- so deciding from the store alone would happily
        # stamp "cancelled" onto a job that failed on its own hours ago, which
        # is the one thing a cancel must never do.
        status = self.inspect(run_id)
        if status.status not in ACTIVE_RUN_STATES:
            return
        if status.external_id:
            self._client().stop_job(status.external_id)
        self.store.update_status(run_id, status="cancelled", completed_at=utc_now())
        self.store.append_event(run_id, RunEvent(message="ray job cancelled"))

    def diagnose(self, run_id: str) -> DiagnosisReport:
        self.inspect(run_id)
        return diagnose_run(self.store, run_id)


def _ray_status_text(ray_status: Any) -> str:
    value = getattr(ray_status, "value", ray_status)
    name = getattr(value, "name", value)
    return str(name).split(".")[-1].lower()


def _map_ray_status(ray_status: Any) -> str:
    text = _ray_status_text(ray_status)
    return {
        "pending": "submitted",
        "running": "running",
        "succeeded": "completed",
        "failed": "failed",
        "stopped": "cancelled",
    }.get(text, "unknown")


def _safe_call(client: Any, method_name: str, *args: Any) -> Any:
    method = getattr(client, method_name, None)
    if method is None:
        return None
    try:
        return method(*args)
    except Exception:
        return None


def _job_info_dict(job_info: Any) -> dict[str, Any]:  # ast-grep-ignore: no-dict-return-annotation
    # Passes Ray's own job-info keys through into the free-form
    # RunStatus.details blob; not a fixed schema iax defines.
    if job_info is None:
        return {}  # ast-grep-ignore: no-dict-literal-return  # empty job-info, same shape as above
    if isinstance(job_info, dict):
        return {str(k): _jsonable(v) for k, v in job_info.items()}
    if hasattr(job_info, "model_dump"):
        return {str(k): _jsonable(v) for k, v in job_info.model_dump().items()}
    if hasattr(job_info, "dict"):
        return {str(k): _jsonable(v) for k, v in job_info.dict().items()}

    result: dict[str, Any] = {}
    for attr in (
        "status",
        "entrypoint",
        "message",
        "status_message",
        "error_type",
        "start_time",
        "end_time",
        "metadata",
        "runtime_env",
        "driver_node_id",
    ):
        if hasattr(job_info, attr):
            result[attr] = _jsonable(getattr(job_info, attr))
    return result


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)
