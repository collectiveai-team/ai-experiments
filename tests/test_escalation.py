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


def test_campaign_review_does_not_break_the_escalation_inbox(tmp_path):
    """A campaign review file has no run_id and no decision. Parsing every
    *.json as an EscalationRequest crashed `iax escalations` permanently from
    the first agent-review round on (#4)."""
    import json

    store, run_id = _setup(tmp_path)
    escalate(store, _suspicious(run_id), EscalationPolicy())

    reviews = store.root / "_escalations"
    (reviews / "campaign_cmp_abc123.json").write_text(
        json.dumps(
            {
                "type": "campaign_review",
                "created_at": utc_now().isoformat(),
                "campaign_id": "cmp_abc123",
                "summary": {"best": None, "history": []},
                "note": "review the history",
            }
        )
    )

    items = list_escalations(store)

    kinds = sorted(item.kind for item in items)
    assert kinds == ["campaign", "run"]
    campaign = next(i for i in items if i.kind == "campaign")
    assert campaign.campaign_id == "cmp_abc123"
    run = next(i for i in items if i.kind == "run")
    assert run.run_id == run_id


def test_unreadable_escalation_file_is_skipped_not_fatal(tmp_path):
    store, run_id = _setup(tmp_path)
    escalate(store, _suspicious(run_id), EscalationPolicy())
    (store.root / "_escalations" / "garbage.json").write_text("{not json")

    items = list_escalations(store)

    assert [i.kind for i in items] == ["run"]


def test_clear_campaign_review_removes_the_file(tmp_path):
    import json

    from ai_experiments.monitoring.escalation import clear_campaign_review

    store, _ = _setup(tmp_path)
    reviews = store.root / "_escalations"
    reviews.mkdir(parents=True, exist_ok=True)
    (reviews / "campaign_cmp_x.json").write_text(
        json.dumps(
            {
                "type": "campaign_review",
                "created_at": utc_now().isoformat(),
                "campaign_id": "cmp_x",
                "summary": {},
                "note": "",
            }
        )
    )

    clear_campaign_review(store, "cmp_x")

    assert list_escalations(store) == []
