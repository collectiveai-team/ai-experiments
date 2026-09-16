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

from pydantic import BaseModel, Field

from ai_experiments.heartbeat import Heartbeat, write_heartbeat
from ai_experiments.monitoring.supervision import RunAction, supervise_once
from ai_experiments.notify import Notifier
from ai_experiments.orchestrator import CampaignOrchestrator
from ai_experiments.schemas import ACTIVE_RUN_STATES, utc_now
from ai_experiments.store import FilesystemRunStore
from ai_experiments.store.campaign import CampaignStore
from ai_experiments.store.filesystem import SYNTHETIC_STATUS_KEY


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
        self.notifier = notifier or Notifier(run_store.root)
        self._stop = False
        self.ticks = 0

    # -- one tick --------------------------------------------------------------

    def tick(self, interval_seconds: int | None = None) -> TickReport:
        report = TickReport(timestamp=utc_now().isoformat())
        self._check_runs(report)
        self._advance_campaigns(report)
        self.ticks += 1
        # Stamped even when the tick found nothing to do: the whole point is
        # that "nothing happened" and "nobody is watching" stop looking alike.
        try:
            write_heartbeat(
                self.run_store.root,
                Heartbeat(
                    interval_seconds=interval_seconds,
                    ticks=self.ticks,
                    runs_checked=report.runs_checked,
                    campaigns_advanced=report.campaigns_advanced,
                ),
            )
        except OSError as exc:
            report.errors.append(f"heartbeat: {exc}")
        return report

    def _check_runs(self, report: TickReport) -> None:
        from ai_experiments.tracking import TrackingSyncError, finalize_tracking

        active_run_ids: list[str] = []
        for run_id in sorted(self.run_store.list_runs()):
            # Reading the status is itself fallible (a torn or truncated
            # status.json), so it belongs inside the guard: one unreadable run
            # must not end the tick for every other run being supervised.
            try:
                status = self.run_store.read_status(run_id)
                if status.details.get(SYNTHETIC_STATUS_KEY):
                    raise RuntimeError(status.error or "status unreadable")
            except Exception as exc:
                report.errors.append(f"{run_id}: {exc}")
                continue

            if status.status not in ACTIVE_RUN_STATES:
                try:
                    if finalize_tracking(self.run_store, status):
                        report.actions.append(
                            RunAction(
                                run_id=run_id,
                                decision=status.status,
                                action="mlflow_synced",
                            )
                        )
                except TrackingSyncError as exc:
                    report.errors.append(f"{run_id}: {exc}")
                except Exception as exc:
                    report.errors.append(f"{run_id}: mlflow finalize: {exc}")
                continue
            report.runs_checked += 1
            active_run_ids.append(run_id)

        pass_report = supervise_once(self.run_store, active_run_ids, self.notifier)
        # Grouped by category, not sorted by run_id: mlflow-finalize actions/
        # errors above were appended in this loop's own run_id order, and
        # everything supervise_once found is appended after as one block --
        # report.actions/errors are no longer strictly run_id-ordered across
        # the two categories.
        report.actions.extend(pass_report.actions)
        report.errors.extend(pass_report.errors)

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
        self,
        interval_seconds: int = 30,
        max_ticks: int | None = None,
        heartbeat_seconds: int = 300,
    ) -> None:
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)
        ticks = 0
        next_heartbeat = 0.0
        while not self._stop:
            report = self.tick(interval_seconds=interval_seconds)
            if report.actions or report.errors:
                print(json.dumps(report.model_dump(mode="json")), flush=True)
            elif time.monotonic() >= next_heartbeat:
                # A quiet daemon that never speaks cannot be told from a dead
                # one in a log someone reads tomorrow morning.
                print(
                    json.dumps(
                        {
                            "timestamp": report.timestamp,
                            "heartbeat": True,
                            "runs_checked": report.runs_checked,
                            "campaigns_advanced": report.campaigns_advanced,
                            "next_tick_seconds": interval_seconds,
                        }
                    ),
                    flush=True,
                )
                next_heartbeat = time.monotonic() + heartbeat_seconds
            ticks += 1
            if max_ticks is not None and ticks >= max_ticks:
                break
            deadline = time.monotonic() + interval_seconds
            while not self._stop and time.monotonic() < deadline:
                time.sleep(0.2)
