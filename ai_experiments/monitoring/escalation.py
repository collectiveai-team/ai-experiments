"""Escalation ladder between programmatic checks and agent diagnosis.

Programmatic monitor ticks are free; agent diagnosis costs tokens. The ladder
only requests an agent after a run has looked suspicious for N consecutive
ticks, then enforces a cooldown and a hard per-run budget of agent calls.

When ``EscalationPolicy.agent_command`` is configured the daemon shells out to
it (e.g. ``claude -p "Diagnose iax run {run_id}" --output-format json``) and
interprets a JSON verdict. Without a command, the escalation is recorded as a
file under ``<runs>/_escalations/`` plus a run event, so an external agent
session can pick it up — zero tokens spent by the daemon itself.
"""

from __future__ import annotations

import json
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from ai_experiments.schemas import EscalationPolicy, MonitorDecision, RunEvent, utc_now
from ai_experiments.store import FilesystemRunStore

EscalationAction = Literal["none", "invoke_agent", "budget_exhausted", "cooling_down"]


class EscalationState(BaseModel):
    suspicious_ticks: int = 0
    agent_calls: int = 0
    last_agent_call_at: datetime | None = None


class AgentVerdict(BaseModel):
    verdict: Literal["kill", "continue", "inconclusive"] = "inconclusive"
    reason: str = ""
    raw_output: str = ""


class EscalationLadder:
    """Persists per-run escalation state in the run directory."""

    def __init__(self, store: FilesystemRunStore) -> None:
        self.store = store

    def _state_path(self, run_id: str) -> Path:
        return self.store.run_dir(run_id) / "escalation.json"

    def read_state(self, run_id: str) -> EscalationState:
        path = self._state_path(run_id)
        if not path.exists():
            return EscalationState()
        return EscalationState(**json.loads(path.read_text()))

    def _write_state(self, run_id: str, state: EscalationState) -> None:
        self._state_path(run_id).write_text(
            json.dumps(state.model_dump(mode="json"), indent=2)
        )

    def observe(
        self,
        run_id: str,
        decision: MonitorDecision,
        policy: EscalationPolicy,
        now: datetime | None = None,
    ) -> EscalationAction:
        """Record one monitor tick and decide whether to call the agent."""
        now = now or utc_now()
        state = self.read_state(run_id)

        if decision.decision != "delegate_diagnosis":
            if state.suspicious_ticks:
                state.suspicious_ticks = 0
                self._write_state(run_id, state)
            return "none"

        state.suspicious_ticks += 1
        action: EscalationAction = "none"
        if state.suspicious_ticks >= policy.after_suspicious_ticks:
            if state.agent_calls >= policy.max_agent_calls:
                action = "budget_exhausted"
            elif _in_cooldown(state.last_agent_call_at, policy, now):
                action = "cooling_down"
            else:
                action = "invoke_agent"
                state.agent_calls += 1
                state.last_agent_call_at = now
                state.suspicious_ticks = 0
        self._write_state(run_id, state)
        return action


class EscalationRequest(BaseModel):
    run_id: str
    created_at: datetime = Field(default_factory=utc_now)
    decision: MonitorDecision
    agent_calls_used: int = 0
    note: str = (
        "Programmatic checks flagged this run repeatedly. "
        "Diagnose with the diagnosing-experiments skill; cancel via "
        "`iax cancel <run_id>` if it cannot recover."
    )


def escalate(
    store: FilesystemRunStore,
    decision: MonitorDecision,
    policy: EscalationPolicy,
) -> AgentVerdict | None:
    """Hand a suspicious run to an agent; returns a verdict when one ran inline."""
    run_id = decision.run_id
    if policy.agent_command:
        verdict = _run_agent_command(store, run_id, policy)
        store.append_event(
            run_id,
            RunEvent(
                level="warning",
                message="agent diagnosis invoked",
                details={"verdict": verdict.verdict, "reason": verdict.reason},
            ),
        )
        return verdict

    request = EscalationRequest(run_id=run_id, decision=decision)
    escalations_dir = store.root / "_escalations"
    escalations_dir.mkdir(parents=True, exist_ok=True)
    (escalations_dir / f"{run_id}.json").write_text(
        json.dumps(request.model_dump(mode="json"), indent=2)
    )
    store.append_event(
        run_id,
        RunEvent(
            level="warning",
            message="escalation recorded for agent pickup",
            details={"reasons": decision.reasons},
        ),
    )
    return None


def clear_escalation(store: FilesystemRunStore, run_id: str) -> None:
    path = store.root / "_escalations" / f"{run_id}.json"
    path.unlink(missing_ok=True)


def list_escalations(store: FilesystemRunStore) -> list[EscalationRequest]:
    escalations_dir = store.root / "_escalations"
    if not escalations_dir.exists():
        return []
    requests = []
    for path in sorted(escalations_dir.glob("*.json")):
        requests.append(EscalationRequest(**json.loads(path.read_text())))
    return requests


def _in_cooldown(
    last_call: datetime | None, policy: EscalationPolicy, now: datetime
) -> bool:
    if last_call is None:
        return False
    if last_call.tzinfo is None:
        last_call = last_call.replace(tzinfo=timezone.utc)
    return (now - last_call).total_seconds() < policy.cooldown_minutes * 60


def _run_agent_command(
    store: FilesystemRunStore, run_id: str, policy: EscalationPolicy
) -> AgentVerdict:
    assert policy.agent_command is not None
    # Plain token replacement, not str.format: agent prompts legitimately
    # contain literal braces (JSON examples) that format() would reject.
    rendered = policy.agent_command.replace("{run_id}", run_id).replace(
        "{run_dir}", str(store.run_dir(run_id))
    )
    command = shlex.split(rendered)
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=policy.agent_timeout_seconds,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        return AgentVerdict(
            verdict="inconclusive", reason=f"agent command failed: {exc}"
        )

    output = result.stdout.strip()
    try:
        payload = json.loads(output)
        if isinstance(payload, dict) and payload.get("verdict") in {
            "kill",
            "continue",
            "inconclusive",
        }:
            return AgentVerdict(
                verdict=payload["verdict"],
                reason=str(payload.get("reason", "")),
                raw_output=output,
            )
    except json.JSONDecodeError:
        pass
    return AgentVerdict(
        verdict="inconclusive",
        reason="agent output was not a JSON verdict",
        raw_output=output[-2000:],
    )
