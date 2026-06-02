from __future__ import annotations

from datetime import datetime, timezone

from industrial_ai_experiments.schemas import (
    DiagnosisReport,
    MonitorDecision,
    RunEvent,
)
from industrial_ai_experiments.store import FilesystemRunStore


def _age_minutes(value: datetime | None) -> float | None:
    if value is None:
        return None
    now = datetime.now(timezone.utc)
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return (now - value).total_seconds() / 60


def diagnose_run(store: FilesystemRunStore, run_id: str) -> DiagnosisReport:
    status = store.read_status(run_id)
    events = store.read_events(run_id, tail=50)
    reasons: list[str] = []
    recommendations: list[str] = []

    if status.status == "completed":
        decision = MonitorDecision(
            run_id=run_id,
            decision="training_complete",
            severity="info",
            reasons=["run_completed"],
        )
        return DiagnosisReport(
            run_id=run_id,
            status=status,
            decision=decision,
            events=events,
            recommendations=["Summarize results and stop the monitor job."],
        )

    if status.status in {"failed", "cancelled"}:
        decision = MonitorDecision(
            run_id=run_id,
            decision="training_failed",
            severity="error",
            reasons=[status.status],
        )
        return DiagnosisReport(
            run_id=run_id,
            status=status,
            decision=decision,
            events=events,
            recommendations=["Delegate to a training monitor agent for failure diagnosis."],
        )

    if status.error:
        reasons.append("status_error_present")

    ray_condition = status.details.get("ray_condition")
    if ray_condition == "resource_starved":
        reasons.append("ray_resource_starved")
    elif ray_condition == "stuck_suspected":
        reasons.append("ray_stuck_suspected")

    updated_age = _age_minutes(status.updated_at)
    threshold = status.details.get("stuck_after_minutes", 30)
    if updated_age is not None and updated_age > threshold:
        reasons.append(f"no_status_update_for_{int(updated_age)}m")

    if events:
        event_age = _age_minutes(events[-1].timestamp)
        if event_age is not None and event_age > threshold:
            reasons.append(f"no_event_progress_for_{int(event_age)}m")
    elif status.status in {"submitted", "running"} and ray_condition not in {"queued", "running"}:
        reasons.append("no_run_events")

    if reasons:
        decision = MonitorDecision(
            run_id=run_id,
            decision="delegate_diagnosis",
            severity="warning",
            reasons=reasons,
        )
        recommendations.append("Ask the training monitor agent to inspect logs and backend state.")
    else:
        decision = MonitorDecision(
            run_id=run_id,
            decision="continue_waiting",
            severity="info",
            reasons=["run_active"],
        )
        recommendations.append("Keep the scheduler monitor active.")

    return DiagnosisReport(
        run_id=run_id,
        status=status,
        decision=decision,
        events=events,
        recommendations=recommendations,
    )


def event_from_log_line(line: str) -> RunEvent:
    level = "error" if "error" in line.lower() or "traceback" in line.lower() else "info"
    return RunEvent(level=level, message=line.rstrip())
