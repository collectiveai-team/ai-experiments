"""The monitor daemon: programmatic checks first, agent escalation second.

Every tick the daemon:

1. Diagnoses each active run with the free programmatic rules
   (`monitoring.rules.diagnose_run`). Fatal conditions (NaN objective, hard
   timeout, dead worker) produce a ``kill`` decision — handled inline:
   auto-kill when the run's policy allows it, escalation otherwise.
2. Feeds suspicious-but-not-fatal decisions through the escalation ladder, so
   an agent is only consulted after repeated suspicion, with cooldown and a
   per-run call budget. Without a configured agent command the escalation is
   a file under ``<runs>/_escalations/`` — zero tokens spent.
3. Advances every active campaign one orchestrator step (collect results,
   analyze, plan, submit) — the auto-experiment loop.

Run it with ``iax daemon`` (foreground; supervise with systemd/launchd/tmux
in production).
"""

from __future__ import annotations

import json
import signal
import time
from types import FrameType
from typing import Any

from pydantic import BaseModel, Field

from ai_experiments.backends.factory import backend_for_run
from ai_experiments.monitoring.escalation import (
    EscalationLadder,
    clear_escalation,
    escalate,
)
from ai_experiments.notify import Notifier
from ai_experiments.orchestrator import CampaignOrchestrator
from ai_experiments.schemas import MonitorPolicy, RunEvent, utc_now
from ai_experiments.store import FilesystemRunStore
from ai_experiments.store.campaign import CampaignStore

NOTIFY_ACTIONS = {
    "auto_killed",
    "escalated",
    "escalated_fatal",
    "killed_by_agent_verdict",
    "reaped_dead_process",
}

ACTIVE_RUN_STATES = {"submitted", "running"}


class RunAction(BaseModel):
    run_id: str
    decision: str
    action: str
    reasons: list[str] = Field(default_factory=list)


class TickReport(BaseModel):
    timestamp: str
    runs_checked: int = 0
    campaigns_advanced: int = 0
    actions: list[RunAction] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class MonitorDaemon:
    def __init__(
        self,
        run_store: FilesystemRunStore,
        campaign_store: CampaignStore | None = None,
        orchestrator: CampaignOrchestrator | None = None,
        notifier: Notifier | None = None,
    ) -> None:
        self.run_store = run_store
        self.campaign_store = campaign_store or CampaignStore(run_store.root)
        self.orchestrator = orchestrator or CampaignOrchestrator(
            run_store, self.campaign_store
        )
        self.ladder = EscalationLadder(run_store)
        self.notifier = notifier or Notifier(run_store.root)
        self._stop = False

    # -- one tick --------------------------------------------------------------

    def tick(self) -> TickReport:
        report = TickReport(timestamp=utc_now().isoformat())
        self._check_runs(report)
        self._advance_campaigns(report)
        return report

    def _check_runs(self, report: TickReport) -> None:
        for run_id in sorted(self.run_store.list_runs()):
            status = self.run_store.read_status(run_id)
            if status.status not in ACTIVE_RUN_STATES:
                continue
            report.runs_checked += 1
            try:
                action = self._check_run(run_id)
            except Exception as exc:
                report.errors.append(f"{run_id}: {exc}")
                continue
            if action is not None:
                report.actions.append(action)
                if action.action in NOTIFY_ACTIONS:
                    self.notifier.send(
                        f"run {action.action}",
                        f"{action.run_id}: {', '.join(action.reasons)}",
                        run_id=action.run_id,
                        action=action.action,
                        reasons=action.reasons,
                    )

    def _check_run(self, run_id: str) -> RunAction | None:
        backend = backend_for_run(self.run_store, run_id)
        diagnosis = backend.diagnose(run_id)
        decision = diagnosis.decision
        manifest = self.run_store.read_manifest(run_id)
        policy = manifest.monitoring if manifest else MonitorPolicy()

        if decision.decision == "kill":
            return self._handle_fatal(run_id, backend, decision, policy)

        if decision.decision == "delegate_diagnosis":
            ladder_action = self.ladder.observe(run_id, decision, policy.escalation)
            if ladder_action == "invoke_agent":
                verdict = escalate(self.run_store, decision, policy.escalation)
                if verdict is not None and verdict.verdict == "kill":
                    backend.cancel(run_id)
                    self.run_store.append_event(
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
        clear_escalation(self.run_store, run_id)
        return None

    def _handle_fatal(
        self,
        run_id: str,
        backend: Any,
        decision: Any,
        policy: MonitorPolicy,
    ) -> RunAction:
        if "process_dead" in decision.reasons:
            # The worker is already gone; reap instead of cancelling.
            self.run_store.update_status(
                run_id,
                status="failed",
                completed_at=utc_now(),
                error="worker process died without reporting a final status",
            )
            self.run_store.append_event(
                run_id,
                RunEvent(level="error", message="run reaped: worker process dead"),
            )
            return RunAction(
                run_id=run_id,
                decision="kill",
                action="reaped_dead_process",
                reasons=decision.reasons,
            )
        if policy.auto_kill:
            backend.cancel(run_id)
            self.run_store.update_status(
                run_id, error=f"auto-killed: {', '.join(decision.reasons)}"
            )
            self.run_store.append_event(
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
        escalate(self.run_store, decision, policy.escalation)
        return RunAction(
            run_id=run_id,
            decision="kill",
            action="escalated_fatal",
            reasons=decision.reasons,
        )

    def _advance_campaigns(self, report: TickReport) -> None:
        for campaign_id in self.campaign_store.list_campaigns():
            try:
                state = self.campaign_store.read_state(campaign_id)
                if state.status in {"completed", "stopped", "failed", "paused"}:
                    continue
                after = self.orchestrator.advance(campaign_id)
                report.campaigns_advanced += 1
                if after.status in {"completed", "stopped", "failed"}:
                    best = next(
                        (t for t in after.trials if t.trial_id == after.best_trial_id),
                        None,
                    )
                    self.notifier.send(
                        f"campaign {after.status}",
                        f"{after.name} ({campaign_id}): {after.stop_reason}",
                        campaign_id=campaign_id,
                        stop_reason=after.stop_reason,
                        best_trial=(
                            {
                                "trial_id": best.trial_id,
                                "objective_value": best.objective_value,
                                "params": best.params,
                            }
                            if best
                            else None
                        ),
                    )
            except Exception as exc:
                report.errors.append(f"{campaign_id}: {exc}")

    # -- the loop ----------------------------------------------------------------

    def _handle_signal(self, signum: int, frame: FrameType | None) -> None:
        self._stop = True

    def run_forever(
        self, interval_seconds: int = 30, max_ticks: int | None = None
    ) -> None:
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)
        ticks = 0
        while not self._stop:
            report = self.tick()
            if report.actions or report.errors:
                print(json.dumps(report.model_dump(mode="json")), flush=True)
            ticks += 1
            if max_ticks is not None and ticks >= max_ticks:
                break
            deadline = time.monotonic() + interval_seconds
            while not self._stop and time.monotonic() < deadline:
                time.sleep(0.2)
