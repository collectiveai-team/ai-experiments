"""`iax handoff`: the door out of the search and into development.

When a campaign stops because the code is wrong, the loop files a ticket. This
is what carries that ticket to a development flow. Three things have to hold:
the fix lands on its own experimentation branch, the same defect never buys
two flows, and workload output never becomes part of a command.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from typer.testing import CliRunner

from ai_experiments.cli import app
from ai_experiments.handoff import hand_off, plan_handoff, render_issue
from ai_experiments.monitoring.escalation import ChangeRequest, record_change_request
from ai_experiments.store import FilesystemRunStore

runner = CliRunner()


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "test")
    (repo / "toy.py").write_text("print('hi')\n")
    _git(repo, "add", "toy.py")
    _git(repo, "commit", "-qm", "first")
    return repo


def _flow(tmp_path: Path) -> tuple[list[str], Path]:
    """A stand-in for orq-lite: records every call, never spends a token."""
    log = tmp_path / "flow-calls.jsonl"
    script = tmp_path / "flow.py"
    script.write_text(
        "import json, os, sys\n"
        f"log = {str(log)!r}\n"
        "with open(log, 'a') as fh:\n"
        "    fh.write(json.dumps({'argv': sys.argv[1:], 'cwd': os.getcwd()}) + '\\n')\n"
        "sys.exit(int(os.environ.get('FLOW_EXIT', '0')))\n"
    )
    return [sys.executable, str(script), "--issue", "{issue}"], log


def _ticket(**overrides) -> ChangeRequest:
    data: dict = {
        "campaign_id": "camp_abc123",
        "title": "the loader crashes on an empty window",
        "rationale": "every trial dies in the same place",
        "files": ["toy.py"],
        "trial_ids": ["t001", "t002"],
        "run_ids": ["run_1", "run_2"],
        "error_tail": "t001: IndexError: list index out of range",
        "acceptance": "an empty window returns an empty batch",
        "source_key": "iax:camp_abc123:deadbeef1234",
    }
    data.update(overrides)
    return ChangeRequest(**data)


def _calls(log: Path) -> list[dict]:
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text().splitlines() if line]


def test_the_fix_gets_its_own_experimentation_branch(tmp_path):
    """Trials before and after a code change are not comparable."""
    repo = _repo(tmp_path)
    store = FilesystemRunStore(tmp_path / "runs")
    command, log = _flow(tmp_path)

    result = hand_off(store, _ticket(), repo=repo, command=command)

    assert result.status == "launched"
    assert result.plan.branch == "exp/camp_abc123-deadbeef1234"
    branches = subprocess.run(
        ["git", "-C", str(repo), "branch", "--list", result.plan.branch],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert result.plan.branch in branches
    # The original checkout is untouched: the campaign's branch stays measurable.
    head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert head == "main"
    assert (result.plan.worktree / "toy.py").exists()
    assert _calls(log)[0]["cwd"] == str(result.plan.worktree.resolve())


def test_the_flow_receives_a_path_never_the_workload_output(tmp_path):
    """`error_tail` is untrusted text. It must not reach a command line."""
    repo = _repo(tmp_path)
    store = FilesystemRunStore(tmp_path / "runs")
    command, log = _flow(tmp_path)
    nasty = "t001: boom `$(touch /tmp/pwned)` ``` still output"

    result = hand_off(store, _ticket(error_tail=nasty), repo=repo, command=command)

    argv = _calls(log)[0]["argv"]
    assert argv == ["--issue", str(result.plan.issue_path)]
    issue = result.plan.issue_path.read_text()
    assert nasty in issue
    # The fence outgrows the backticks inside the output, so the block holds.
    assert "````text" in issue


def test_the_same_defect_does_not_buy_two_development_flows(tmp_path):
    repo = _repo(tmp_path)
    store = FilesystemRunStore(tmp_path / "runs")
    command, log = _flow(tmp_path)

    first = hand_off(store, _ticket(), repo=repo, command=command)
    second = hand_off(store, _ticket(), repo=repo, command=command)

    assert first.status == "launched"
    assert second.status == "already_handled"
    assert len(_calls(log)) == 1

    again = hand_off(store, _ticket(), repo=repo, command=command, force=True)
    assert again.status == "launched"
    assert len(_calls(log)) == 2


def test_a_dry_run_names_the_branch_and_creates_nothing(tmp_path):
    repo = _repo(tmp_path)
    store = FilesystemRunStore(tmp_path / "runs")
    command, log = _flow(tmp_path)

    result = hand_off(store, _ticket(), repo=repo, command=command, dry_run=True)

    assert result.status == "planned"
    assert result.plan.branch == "exp/camp_abc123-deadbeef1234"
    assert not result.plan.issue_path.exists()
    assert not result.plan.worktree.exists()
    assert _calls(log) == []


def test_a_failing_flow_is_reported_not_swallowed(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    store = FilesystemRunStore(tmp_path / "runs")
    command, _ = _flow(tmp_path)
    monkeypatch.setenv("FLOW_EXIT", "1")

    result = hand_off(store, _ticket(), repo=repo, command=command)

    assert result.status == "failed"
    assert result.exit_code == 1
    assert result.error
    assert json.loads(result.plan.ledger_path.read_text())["status"] == "failed"


def test_a_ticket_without_a_source_key_still_gets_a_stable_branch(tmp_path):
    store = FilesystemRunStore(tmp_path / "runs")
    ticket = _ticket(source_key="")

    first = plan_handoff(store, ticket)
    second = plan_handoff(store, ticket)

    assert first.branch == second.branch
    assert first.branch.startswith("exp/camp_abc123-")


def test_the_issue_tells_the_reader_to_check_the_evidence(tmp_path):
    store = FilesystemRunStore(tmp_path / "runs")
    ticket = _ticket()

    issue = render_issue(ticket, plan_handoff(store, ticket))

    assert "`t001`" in issue and "`run_1`" in issue
    assert "an empty window returns an empty batch" in issue
    assert "Read the runs before you trust the diagnosis" in issue
    assert "exp/camp_abc123-deadbeef1234" in issue


def test_the_cli_lists_a_blocked_campaign_without_touching_the_repo(tmp_path):
    store = FilesystemRunStore(tmp_path / "runs")
    record_change_request(store, _ticket())

    result = runner.invoke(
        app,
        ["handoff", "--runs-dir", str(store.root), "--dry-run", "--json"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload[0]["status"] == "planned"
    assert payload[0]["plan"]["branch"] == "exp/camp_abc123-deadbeef1234"


def test_the_cli_says_so_when_nothing_is_blocked(tmp_path):
    store = FilesystemRunStore(tmp_path / "runs")
    store.root.mkdir(parents=True, exist_ok=True)

    result = runner.invoke(app, ["handoff", "--runs-dir", str(store.root)])

    assert result.exit_code == 0
    assert "No campaign is blocked" in result.stdout


def test_the_cli_refuses_an_unknown_campaign(tmp_path):
    store = FilesystemRunStore(tmp_path / "runs")
    record_change_request(store, _ticket())

    result = runner.invoke(
        app, ["handoff", "camp_nope", "--runs-dir", str(store.root), "--dry-run"]
    )

    assert result.exit_code == 1
    assert "camp_nope" in result.output
