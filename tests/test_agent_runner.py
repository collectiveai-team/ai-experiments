"""The agent boundary: text in, one validated JSON payload out.

An agent reply is untrusted text. Every failure mode here — a crash, a
timeout, a chatty reply with no JSON — must become an `AgentResult(ok=False)`
that the caller can fall back from, never an exception that kills the loop.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from ai_experiments.agents import (
    AgentResult,
    CliAgentRunner,
    StubAgentRunner,
    extract_json,
    unwrap_envelope,
)


def test_extract_json_finds_the_object_after_prose():
    text = 'Let me think about this.\n\n{"hypothesis": "lr too high", "trials": []}'

    assert extract_json(text) == {"hypothesis": "lr too high", "trials": []}


def test_extract_json_prefers_a_fenced_block():
    text = 'Consider {"draft": 1} first.\n\n```json\n{"final": 2}\n```\n'

    assert extract_json(text) == {"final": 2}


def test_extract_json_takes_the_last_object_when_the_agent_corrects_itself():
    text = '{"trials": [1]}\n\nActually, better:\n\n{"trials": [1, 2]}'

    assert extract_json(text) == {"trials": [1, 2]}


def test_extract_json_handles_nested_braces():
    text = 'ok: {"trials": [{"params": {"lr": 0.1}}]}'

    assert extract_json(text) == {"trials": [{"params": {"lr": 0.1}}]}


def test_extract_json_ignores_braces_inside_strings():
    text = '{"note": "use {curly} braces", "n": 1}'

    assert extract_json(text) == {"note": "use {curly} braces", "n": 1}


@pytest.mark.parametrize("text", ["", "no json here", "[1, 2, 3]", "{not json}"])
def test_extract_json_returns_none_when_there_is_nothing_to_parse(text):
    assert extract_json(text) is None


def test_unwrap_envelope_looks_through_the_claude_cli_wrapper():
    envelope = {
        "type": "result",
        "subtype": "success",
        "result": 'Here you go:\n```json\n{"verdict": "continue"}\n```',
    }

    assert unwrap_envelope(envelope) == {"verdict": "continue"}


def test_unwrap_envelope_leaves_a_plain_payload_alone():
    payload = {"verdict": "stop", "reason": "budget spent"}

    assert unwrap_envelope(payload) == payload


def _fake_agent(tmp_path: Path, body: str) -> str:
    script = tmp_path / "fake_agent.py"
    script.write_text(body)
    return f"{sys.executable} {script}"


def test_cli_runner_returns_the_payload_and_writes_a_transcript(tmp_path):
    command = _fake_agent(
        tmp_path,
        "import sys\n"
        "prompt = sys.stdin.read()\n"
        "print('thinking about', len(prompt), 'chars')\n"
        'print(\'{"hypothesis": "ok", "trials": []}\')\n',
    )
    runner = CliAgentRunner(command, transcript_dir=tmp_path / "agents")

    result = runner.run("plan the next round", role="planner")

    assert result.ok
    assert result.payload["hypothesis"] == "ok"
    transcript = tmp_path / "agents" / "planner" / "1"
    assert (transcript / "prompt.txt").read_text() == "plan the next round"
    assert "thinking about" in (transcript / "stdout.txt").read_text()
    assert json.loads((transcript / "result.json").read_text())["ok"] is True


def test_cli_runner_reports_a_nonzero_exit_instead_of_raising(tmp_path):
    command = _fake_agent(
        tmp_path, "import sys\nsys.stderr.write('rate limited\\n')\nsys.exit(7)\n"
    )
    runner = CliAgentRunner(command)

    result = runner.run("plan")

    assert not result.ok
    assert result.exit_code == 7
    assert "rate limited" in result.error


def test_cli_runner_reports_a_reply_without_json(tmp_path):
    command = _fake_agent(tmp_path, "print('I would try a smaller learning rate.')\n")
    runner = CliAgentRunner(command)

    result = runner.run("plan")

    assert not result.ok
    assert "no JSON" in result.error
    assert "smaller learning rate" in result.raw


def test_cli_runner_times_out_instead_of_hanging_the_loop(tmp_path):
    command = _fake_agent(tmp_path, "import time\ntime.sleep(30)\n")
    runner = CliAgentRunner(command, timeout_seconds=1)

    result = runner.run("plan")

    assert not result.ok
    assert "timed out" in result.error


def test_cli_runner_reports_a_missing_binary(tmp_path):
    runner = CliAgentRunner("iax-no-such-agent-binary")

    result = runner.run("plan")

    assert not result.ok
    assert "could not run" in result.error


def test_cli_runner_passes_the_prompt_on_stdin_not_as_an_argument(tmp_path):
    """Prompts carry untrusted campaign output; they must never be shell text."""
    command = _fake_agent(
        tmp_path,
        "import json, sys\n"
        "prompt = sys.stdin.read()\n"
        "print(json.dumps({'argv_len': len(sys.argv), 'prompt': prompt}))\n",
    )
    runner = CliAgentRunner(command)

    result = runner.run("$(rm -rf /) `whoami` ; echo pwned")

    assert result.ok
    assert result.payload["argv_len"] == 1
    assert result.payload["prompt"].strip() == "$(rm -rf /) `whoami` ; echo pwned"


def test_cli_runner_resolves_a_preset_name():
    runner = CliAgentRunner("claude")

    assert runner.argv[0] == "claude"
    assert "--output-format" in runner.argv


def test_cli_runner_rejects_an_empty_command():
    with pytest.raises(ValueError):
        CliAgentRunner("   ")


def test_stub_runner_replays_replies_then_repeats_the_last_one():
    runner = StubAgentRunner([{"round": 1}, {"round": 2}])

    assert runner.run("a").payload == {"round": 1}
    assert runner.run("b").payload == {"round": 2}
    assert runner.run("c").payload == {"round": 2}
    assert runner.prompts == ["a", "b", "c"]


def test_stub_runner_accepts_raw_text_and_prebuilt_results():
    runner = StubAgentRunner(
        ['prose then {"trials": []}', AgentResult(error="simulated outage")]
    )

    assert runner.run("a").payload == {"trials": []}
    assert not runner.run("b").ok
