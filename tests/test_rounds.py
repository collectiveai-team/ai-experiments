"""Round records answer "why did it try that, and did it help?" (#23).

state.json says what the campaign is now. Anyone reading an overnight run
needs the history instead: what each round proposed, on what hypothesis, and
what the trials it submitted actually measured.
"""

from __future__ import annotations

import json

from typer.testing import CliRunner

from ai_experiments.agents import StubAgentRunner
from ai_experiments.cli import app
from ai_experiments.improve.rounds import RoundLog, RoundRecord
from ai_experiments.orchestrator import CampaignOrchestrator
from ai_experiments.schemas import GoalSpec
from ai_experiments.store import FilesystemRunStore
from ai_experiments.store.campaign import CampaignStore
from tests.test_orchestrator import FakeBackend

runner = CliRunner()


def _goal(**overrides) -> GoalSpec:
    data: dict = {
        "goal": "minimize (x-2)^2",
        "name": "rounds",
        "objective": {"metric": "loss", "mode": "min", "target": 0.01},
        "search_space": {"x": {"type": "uniform", "low": -5.0, "high": 5.0}},
        "workload": {"entrypoint": "python toy.py"},
        "budget": {"max_trials": 4, "max_parallel": 1},
        "strategy": {"name": "adaptive", "seed": 3},
    }
    data.update(overrides)
    return GoalSpec(**data)


def _harness(tmp_path, agent_runner=None):
    store = FilesystemRunStore(tmp_path / "runs")
    backend = FakeBackend(store)
    orchestrator = CampaignOrchestrator(
        store,
        CampaignStore(store.root),
        backend_factory=lambda goal: backend,
        agent_runner_factory=(lambda goal, cid: agent_runner) if agent_runner else None,
    )
    return orchestrator, store


def _drive(orchestrator, state, limit=20):
    for _ in range(limit):
        state = orchestrator.advance(state.campaign_id)
        if state.status != "running":
            break
    return state


def test_a_campaign_writes_one_record_per_stage_in_order(tmp_path):
    orchestrator, store = _harness(tmp_path)

    state = _drive(orchestrator, orchestrator.start(_goal()))

    log = RoundLog(CampaignStore(store.root).campaign_dir(state.campaign_id))
    records = log.read()
    assert [r.stage for r in records][0] == "propose"
    assert "evaluate" in {r.stage for r in records}
    assert [r.round for r in records] == sorted(r.round for r in records)
    proposed = {tid for r in records if r.stage == "propose" for tid in r.trial_ids}
    assert proposed == {t.trial_id for t in state.trials if t.run_id}


def test_the_evaluate_record_carries_the_measured_values(tmp_path):
    orchestrator, store = _harness(tmp_path)

    state = _drive(orchestrator, orchestrator.start(_goal()))

    records = RoundLog(CampaignStore(store.root).campaign_dir(state.campaign_id)).read()
    evaluations = [r for r in records if r.stage == "evaluate"]
    assert evaluations
    values = evaluations[0].outcome["values"]
    assert evaluations[0].outcome["metric"] == "loss"
    assert all("objective_value" in v for v in values.values())


def test_an_agent_round_records_its_hypothesis_and_its_rejections(tmp_path):
    agent = StubAgentRunner(
        [
            {
                "hypothesis": "x is far from the optimum",
                "rationale": "bracket the minimum",
                "trials": [
                    {"params": {"x": 99.0}},
                    {"params": {"x": 1.5}},
                ],
            }
        ]
    )
    orchestrator, store = _harness(tmp_path, agent_runner=agent)

    state = orchestrator.start(
        _goal(strategy={"name": "agent", "seed": 1, "fallback": "random"})
    )

    proposal = RoundLog(
        CampaignStore(store.root).campaign_dir(state.campaign_id)
    ).read()[0]
    assert proposal.hypothesis == "x is far from the optimum"
    assert proposal.rationale == "bracket the minimum"
    assert proposal.agent_calls == 1
    assert any("x" in r["reason"] for r in proposal.rejected)


def test_a_corrupt_line_hides_only_itself(tmp_path):
    log = RoundLog(tmp_path)
    log.append(RoundRecord(campaign_id="c", round=1, stage="propose"))
    with log.path.open("a") as fh:
        fh.write("{not json\n")
    log.append(RoundRecord(campaign_id="c", round=2, stage="evaluate"))

    records = log.read()

    assert [r.round for r in records] == [1, 2]


def test_rounds_command_reads_the_history_back(tmp_path):
    orchestrator, store = _harness(tmp_path)
    state = _drive(orchestrator, orchestrator.start(_goal()))

    result = runner.invoke(
        app,
        [
            "campaign",
            "rounds",
            state.campaign_id,
            "--runs-dir",
            str(store.root),
            "--json",
        ],
    )

    assert result.exit_code == 0
    records = json.loads(result.stdout)
    assert records and records[0]["stage"] == "propose"


def test_trials_command_lists_every_trial_with_its_run_id(tmp_path):
    orchestrator, store = _harness(tmp_path)
    state = _drive(orchestrator, orchestrator.start(_goal()))

    result = runner.invoke(
        app,
        ["campaign", "trials", state.campaign_id, "--runs-dir", str(store.root)],
    )

    assert result.exit_code == 0
    for trial in state.trials:
        assert trial.trial_id in result.stdout
        if trial.run_id:
            assert trial.run_id in result.stdout


def test_campaign_status_reports_when_the_loop_last_advanced(tmp_path):
    orchestrator, store = _harness(tmp_path)
    state = _drive(orchestrator, orchestrator.start(_goal()))

    result = runner.invoke(
        app,
        [
            "campaign",
            "status",
            state.campaign_id,
            "--runs-dir",
            str(store.root),
            "--json",
        ],
    )

    summary = json.loads(result.stdout)
    assert summary["last_advanced_at"]
    assert summary["rounds"] >= 1


def test_rounds_and_trials_reject_an_unknown_campaign(tmp_path):
    for command in (["campaign", "rounds"], ["campaign", "trials"]):
        result = runner.invoke(
            app, [*command, "cmp_nope", "--runs-dir", str(tmp_path), "--json"]
        )
        assert result.exit_code == 1
        assert json.loads(result.stdout)["code"] == "not_found"
