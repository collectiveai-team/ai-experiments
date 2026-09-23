"""The per-run supervision pass, shared by `MonitorDaemon.tick` and `run_loop`.

Both paths diagnose, escalate and kill by the same rules.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from ai_experiments.backends.factory import backend_for_run
from ai_experiments.monitoring.escalation import (
    EscalationLadder,
    clear_escalation,
    escalate,
)
from ai_experiments.schemas import (
    ACTIVE_RUN_STATES,
    MonitorPolicy,
    RunEvent,
    utc_now,
)
from ai_experiments.store.filesystem import SYNTHETIC_STATUS_KEY

if TYPE_CHECKING:
    from collections.abc import Iterable

    from ai_experiments.notify import Notifier
    from ai_experiments.store import FilesystemRunStore

#: How each reap outcome reads in a run's ``error``. Outcomes that mean the
#: workload was already gone say nothing -- there is nothing to report.
_ORPHAN_SUMMARY = {
    "terminated": "terminated",
    "killed": "killed (it ignored SIGTERM)",
    "identity_unverifiable": "left running: its pid could not be verified",
    "identity_mismatch": "already gone (its pid has been reused)",
    "kill_failed": "could not be killed and may still be running",
    "reap_failed": "could not be reaped",
}

NOTIFY_ACTIONS = {
    "auto_killed",
    "escalated",
    "escalated_fatal",
    "killed_by_agent_verdict",
    "reaped_dead_process",
}


class RunAction(BaseModel):
    run_id: str
    decision: str
    action: str
    reasons: list[str] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)


class SupervisionReport(BaseModel):
    """What one supervision pass did, and what it could not do.

    `errors` is not decoration: a pass that diagnosed nothing because every
    backend raised must not be indistinguishable from a pass that found
    every run healthy.
    """

    actions: list[RunAction] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


def supervise_once(
    run_store: FilesystemRunStore,
    run_ids: Iterable[str],
    notifier: Notifier | None = None,
) -> SupervisionReport:
    """One supervision pass over the given run ids.

    Kept short and free of agent calls on purpose: `iax loop` runs this on
    every iteration, and an overnight loop that pauses for a slow agent is a
    loop that is not watching anything.

    This is the body `MonitorDaemon._check_runs` used to run inline, per run
    id, extracted so both the daemon's tick and `run_loop`'s iteration share
    it. A run whose status cannot be read, or whose check itself raises, is
    skipped rather than aborting the whole pass -- one bad run must not end
    supervision of the rest -- but the failure is recorded in `errors`
    rather than swallowed; a caller who never sees "3 runs, 3 errors" cannot
    tell a quiet tick from a broken one.

    `MonitorDaemon._check_runs` already filters to run ids it has confirmed
    are active before calling this, so the status read below will never hit
    the "not in ACTIVE_RUN_STATES" branch for the daemon's own call. It is
    still done here, unconditionally, because this function also has to be
    correct for `run_loop`, which does not pre-filter against a live status
    read -- the double read is deliberate, not duplicated by accident.
    """
    ladder = EscalationLadder(run_store)
    report = SupervisionReport()
    for run_id in run_ids:
        try:
            status = run_store.read_status(run_id)
            if status.details.get(SYNTHETIC_STATUS_KEY):
                raise RuntimeError(status.error or "status unreadable")
        except Exception as exc:
            report.errors.append(f"{run_id}: {exc}")
            continue

        if status.status not in ACTIVE_RUN_STATES:
            continue

        try:
            action = _check_run(run_store, run_id, ladder)
        except Exception as exc:
            report.errors.append(f"{run_id}: {exc}")
            continue
        if action is None:
            continue

        report.actions.append(action)
        if notifier is not None and action.action in NOTIFY_ACTIONS:
            notifier.send(
                f"run {action.action}",
                f"{action.run_id}: {', '.join(action.reasons)}",
                run_id=action.run_id,
                action=action.action,
                reasons=action.reasons,
            )
    return report


def _check_run(
    run_store: FilesystemRunStore, run_id: str, ladder: EscalationLadder
) -> RunAction | None:
    backend = backend_for_run(run_store, run_id)
    diagnosis = backend.diagnose(run_id)
    decision = diagnosis.decision
    manifest = run_store.read_manifest(run_id)
    policy = manifest.monitoring if manifest else MonitorPolicy()

    if decision.decision == "kill":
        return _handle_fatal(run_store, run_id, backend, decision, policy)

    if decision.decision == "delegate_diagnosis":
        ladder_action = ladder.observe(run_id, decision, policy.escalation)
        if ladder_action == "invoke_agent":
            verdict = escalate(run_store, decision, policy.escalation)
            if verdict is not None and verdict.verdict == "kill":
                backend.cancel(run_id)
                run_store.append_event(
                    run_id,
                    RunEvent(
                        level="error",
                        message="run killed on agent verdict",
                        details={"reason": verdict.reason},
                    ),
                )
                return RunAction(
                    run_id=run_id,
                    decision=decision.decision,
                    action="killed_by_agent_verdict",
                    reasons=decision.reasons,
                )
            return RunAction(
                run_id=run_id,
                decision=decision.decision,
                action="escalated",
                reasons=decision.reasons,
            )
        if ladder_action in {"budget_exhausted", "cooling_down"}:
            return RunAction(
                run_id=run_id,
                decision=decision.decision,
                action=ladder_action,
                reasons=decision.reasons,
            )
        return RunAction(
            run_id=run_id,
            decision=decision.decision,
            action="suspicion_recorded",
            reasons=decision.reasons,
        )

    # Healthy or terminal: clear any stale escalation file.
    clear_escalation(run_store, run_id)
    return None


def _handle_fatal(
    run_store: FilesystemRunStore,
    run_id: str,
    backend: Any,
    decision: Any,
    policy: MonitorPolicy,
) -> RunAction:
    if "process_dead" in decision.reasons:
        # The worker is already gone; reap instead of cancelling. The
        # workload it was supervising can easily have outlived it -- it is
        # a separate process -- so reaping the *run* without also dealing
        # with the workload reports a clean death over a live GPU job.
        try:
            reaped = backend.reap(run_id)
        except Exception as exc:
            reaped = {"outcome": "reap_failed", "error": str(exc)}
        error = "worker process died without reporting a final status"
        summary = _ORPHAN_SUMMARY.get(str(reaped.get("outcome")))
        if summary:
            error = f"{error}; orphaned workload {summary}"
        run_store.update_status(
            run_id,
            status="failed",
            completed_at=utc_now(),
            error=error,
            details={"workload_reap": reaped},
        )
        run_store.append_event(
            run_id,
            RunEvent(
                level="error",
                message="run reaped: worker process dead",
                details={"workload_reap": reaped},
            ),
        )
        return RunAction(
            run_id=run_id,
            decision="kill",
            action="reaped_dead_process",
            reasons=decision.reasons,
            details={"workload_reap": reaped},
        )
    if policy.auto_kill:
        backend.cancel(run_id)
        run_store.update_status(run_id, error=f"auto-killed: {', '.join(decision.reasons)}")
        run_store.append_event(
            run_id,
            RunEvent(
                level="error",
                message="run auto-killed by monitor daemon",
                details={"reasons": decision.reasons},
            ),
        )
        return RunAction(
            run_id=run_id,
            decision="kill",
            action="auto_killed",
            reasons=decision.reasons,
        )
    escalate(run_store, decision, policy.escalation)
    return RunAction(
        run_id=run_id,
        decision="kill",
        action="escalated_fatal",
        reasons=decision.reasons,
    )
