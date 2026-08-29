from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import Callable

from ai_experiments.procs import pid_alive
from ai_experiments.schemas import (
    DiagnosisReport,
    MetricPoint,
    MonitorDecision,
    MonitorPolicy,
    RunEvent,
    RunStatus,
)
from ai_experiments.store import FilesystemRunStore

HEARTBEAT_STALE_MINUTES = 3.0


def _age_minutes(value: datetime | None) -> float | None:
    if value is None:
        return None
    now = datetime.now(timezone.utc)
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return (now - value).total_seconds() / 60


def _parse_iso(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _default_pid_alive(pid: int) -> bool:
    return pid_alive(pid)


def _objective_series(
    metrics: list[MetricPoint], objective: str | None
) -> tuple[str | None, list[float]]:
    """Pick the objective series; fall back to a loss-like metric when unset."""
    if not metrics:
        return objective, []
    name = objective
    if name is None:
        names = [n for point in metrics for n in point.values]
        name = next((n for n in names if "loss" in n.lower()), None)
    if name is None:
        return None, []
    return name, [p.values[name] for p in metrics if name in p.values]


def _fatal_reasons(
    status: RunStatus,
    policy: MonitorPolicy,
    metrics: list[MetricPoint],
    pid_alive: Callable[[int], bool],
) -> list[str]:
    reasons: list[str] = []

    if policy.fatal_on_nan and metrics:
        recent = metrics[-1].values
        for name, value in recent.items():
            if not math.isfinite(value):
                reasons.append(f"non_finite_metric:{name}")

    timeout = policy.timeout_seconds or status.details.get("timeout_seconds")
    started_age = _age_minutes(status.started_at or status.submitted_at)
    if timeout and started_age is not None and started_age * 60 > float(timeout):
        reasons.append("timeout_exceeded")

    if status.backend == "local" and status.pid and not pid_alive(status.pid):
        reasons.append("process_dead")

    return reasons


def _suspicious_reasons(
    status: RunStatus,
    policy: MonitorPolicy,
    metrics: list[MetricPoint],
    events: list[RunEvent],
) -> list[str]:
    reasons: list[str] = []

    if status.error:
        reasons.append("status_error_present")

    ray_condition = status.details.get("ray_condition")
    if ray_condition == "resource_starved":
        reasons.append("ray_resource_starved")
    elif ray_condition == "stuck_suspected":
        reasons.append("ray_stuck_suspected")

    threshold = float(
        policy.stuck_after_minutes or status.details.get("stuck_after_minutes", 30)
    )

    heartbeat_at = _parse_iso(status.details.get("heartbeat_at"))
    if heartbeat_at is not None:
        heartbeat_age = _age_minutes(heartbeat_at)
        if heartbeat_age is not None and heartbeat_age > HEARTBEAT_STALE_MINUTES:
            reasons.append(f"heartbeat_stale_for_{int(heartbeat_age)}m")

    if metrics:
        metric_age = _age_minutes(metrics[-1].timestamp)
        if metric_age is not None and metric_age > threshold:
            reasons.append(f"no_metric_progress_for_{int(metric_age)}m")
        plateau = _plateau_reason(policy, metrics)
        if plateau:
            reasons.append(plateau)
    else:
        updated_age = _age_minutes(status.updated_at)
        if updated_age is not None and updated_age > threshold:
            reasons.append(f"no_status_update_for_{int(updated_age)}m")
        if events:
            event_age = _age_minutes(events[-1].timestamp)
            if event_age is not None and event_age > threshold:
                reasons.append(f"no_event_progress_for_{int(event_age)}m")
        elif status.status in {"submitted", "running"} and ray_condition not in {
            "queued",
            "running",
        }:
            reasons.append("no_run_events")

    return reasons


def _plateau_reason(policy: MonitorPolicy, metrics: list[MetricPoint]) -> str | None:
    patience = policy.plateau_patience_points
    if not patience:
        return None
    name, series = _objective_series(metrics, policy.objective_metric)
    finite = [v for v in series if math.isfinite(v)]
    if name is None or len(finite) <= patience:
        return None
    head, tail = finite[:-patience], finite[-patience:]
    # Direction-agnostic: a plateau means the recent window never beat the
    # best value seen before it, whether the objective is min or max.
    improved = min(tail) < min(head) or max(tail) > max(head)
    if improved:
        return None
    return f"objective_plateau:{name}:{patience}_points"


def diagnose_run(
    store: FilesystemRunStore,
    run_id: str,
    pid_alive: Callable[[int], bool] | None = None,
) -> DiagnosisReport:
    status = store.read_status(run_id)
    events = store.read_events(run_id, tail=50)
    metrics = store.read_metrics(run_id, tail=200)
    manifest = store.read_manifest(run_id)
    policy = manifest.monitoring if manifest else MonitorPolicy()
    pid_check = pid_alive or _default_pid_alive

    if status.status == "completed":
        return DiagnosisReport(
            run_id=run_id,
            status=status,
            decision=MonitorDecision(
                run_id=run_id,
                decision="training_complete",
                severity="info",
                reasons=["run_completed"],
            ),
            events=events,
            recommendations=["Summarize results and stop the monitor job."],
        )

    if status.status in {"failed", "cancelled"}:
        return DiagnosisReport(
            run_id=run_id,
            status=status,
            decision=MonitorDecision(
                run_id=run_id,
                decision="training_failed",
                severity="error",
                reasons=[status.status],
            ),
            events=events,
            recommendations=[
                "Delegate to a training monitor agent for failure diagnosis."
            ],
        )

    fatal = _fatal_reasons(status, policy, metrics, pid_check)
    if fatal:
        return DiagnosisReport(
            run_id=run_id,
            status=status,
            decision=MonitorDecision(
                run_id=run_id,
                decision="kill",
                severity="error",
                reasons=fatal,
            ),
            events=events,
            recommendations=[
                "Terminate the run; the condition cannot recover on its own."
            ],
        )

    reasons = _suspicious_reasons(status, policy, metrics, events)
    if reasons:
        return DiagnosisReport(
            run_id=run_id,
            status=status,
            decision=MonitorDecision(
                run_id=run_id,
                decision="delegate_diagnosis",
                severity="warning",
                reasons=reasons,
            ),
            events=events,
            recommendations=[
                "Ask the training monitor agent to inspect logs and backend state."
            ],
        )

    return DiagnosisReport(
        run_id=run_id,
        status=status,
        decision=MonitorDecision(
            run_id=run_id,
            decision="continue_waiting",
            severity="info",
            reasons=["run_active"],
        ),
        events=events,
        recommendations=["Keep the scheduler monitor active."],
    )


#: A failure word standing on its own. The word boundary keeps `train_error`,
#: `error_rate` and `reconstruction_error` -- ordinary metric vocabulary --
#: out of it, and the lookahead drops `error=0.02`.
_FAILURE_WORD = re.compile(
    r"\b(error|traceback|fatal|exception)\b(?!\s*[:=]\s*[-+]?\d)", re.IGNORECASE
)
#: A python exception name: `RuntimeError`, `ValueError`, `KeyError`. There is
#: no word boundary before "Error" in those, so the rule above cannot see them.
_EXCEPTION_NAME = re.compile(r"\b[A-Z]\w*(Error|Exception)\b")


def event_from_log_line(line: str) -> RunEvent:
    """Classify one line of workload output.

    `level` is what `iax logs` filters on and what the monitoring rules read,
    so a healthy metric line must not arrive as a failure.
    """
    message = line.rstrip()
    failed = bool(_FAILURE_WORD.search(message) or _EXCEPTION_NAME.search(message))
    return RunEvent(level="error" if failed else "info", message=message)
