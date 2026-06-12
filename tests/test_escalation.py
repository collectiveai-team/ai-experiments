from __future__ import annotations

import sys
from datetime import timedelta

from ai_experiments.monitoring.escalation import (
    EscalationLadder,
    escalate,
    list_escalations,
)
from ai_experiments.schemas import (
    EscalationPolicy,
    ExperimentManifest,
    MonitorDecision,
    WorkloadSpec,
    utc_now,
)
from ai_experiments.store import FilesystemRunStore


def _setup(tmp_path) -> tuple[FilesystemRunStore, str]:
    store = FilesystemRunStore(tmp_path / "runs")
    manifest = ExperimentManifest(
        experiment="esc",
        workload=WorkloadSpec(entrypoint="python train.py"),
    )
    run_id, _ = store.create_run(manifest)
    return store, run_id


def _suspicious(run_id: str) -> MonitorDecision:
    return MonitorDecision(
        run_id=run_id, decision="delegate_diagnosis", reasons=["no_metric_progress"]
    )


def _healthy(run_id: str) -> MonitorDecision:
    return MonitorDecision(run_id=run_id, decision="continue_waiting")


def test_ladder_waits_for_consecutive_suspicion(tmp_path):
    store, run_id = _setup(tmp_path)
    ladder = EscalationLadder(store)
    policy = EscalationPolicy(after_suspicious_ticks=3)

    assert ladder.observe(run_id, _suspicious(run_id), policy) == "none"
    assert ladder.observe(run_id, _suspicious(run_id), policy) == "none"
    assert ladder.observe(run_id, _suspicious(run_id), policy) == "invoke_agent"


def test_healthy_tick_resets_the_ladder(tmp_path):
    store, run_id = _setup(tmp_path)
    ladder = EscalationLadder(store)
    policy = EscalationPolicy(after_suspicious_ticks=2)

    ladder.observe(run_id, _suspicious(run_id), policy)
    ladder.observe(run_id, _healthy(run_id), policy)
    assert ladder.observe(run_id, _suspicious(run_id), policy) == "none"


def test_cooldown_blocks_back_to_back_agent_calls(tmp_path):
    store, run_id = _setup(tmp_path)
    ladder = EscalationLadder(store)
    policy = EscalationPolicy(after_suspicious_ticks=1, cooldown_minutes=30)

    assert ladder.observe(run_id, _suspicious(run_id), policy) == "invoke_agent"
    assert ladder.observe(run_id, _suspicious(run_id), policy) == "cooling_down"

    later = utc_now() + timedelta(minutes=31)
    assert (
        ladder.observe(run_id, _suspicious(run_id), policy, now=later) == "invoke_agent"
    )


def test_budget_caps_agent_calls(tmp_path):
    store, run_id = _setup(tmp_path)
    ladder = EscalationLadder(store)
    policy = EscalationPolicy(
        after_suspicious_ticks=1, cooldown_minutes=0, max_agent_calls=2
    )

    assert ladder.observe(run_id, _suspicious(run_id), policy) == "invoke_agent"
    assert ladder.observe(run_id, _suspicious(run_id), policy) == "invoke_agent"
    assert ladder.observe(run_id, _suspicious(run_id), policy) == "budget_exhausted"


def test_escalate_without_agent_command_writes_request_file(tmp_path):
    store, run_id = _setup(tmp_path)

    verdict = escalate(store, _suspicious(run_id), EscalationPolicy())

    assert verdict is None
    requests = list_escalations(store)
    assert [r.run_id for r in requests] == [run_id]
    assert requests[0].decision.reasons == ["no_metric_progress"]


def test_escalate_runs_agent_command_and_parses_verdict(tmp_path):
    store, run_id = _setup(tmp_path)
    script = 'import json; print(json.dumps({"verdict": "kill", "reason": "stuck at step 0"}))'
    policy = EscalationPolicy(agent_command=f"{sys.executable} -c '{script}'")

    verdict = escalate(store, _suspicious(run_id), policy)

    assert verdict is not None
    assert verdict.verdict == "kill"
    assert verdict.reason == "stuck at step 0"


def test_escalate_handles_non_json_agent_output(tmp_path):
    store, run_id = _setup(tmp_path)
    policy = EscalationPolicy(
        agent_command=f"{sys.executable} -c 'print(\"thinking...\")'"
    )

    verdict = escalate(store, _suspicious(run_id), policy)

    assert verdict is not None
    assert verdict.verdict == "inconclusive"
