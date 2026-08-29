"""Invoke an agent CLI under a strict contract, and keep the transcript.

The harness shells out; it imports no agent SDK, so `claude`, `codex`, or any
command that reads a prompt on stdin and writes a reply on stdout all work the
same way. The prompt goes in on **stdin**, never interpolated into a shell
string: prompts carry campaign output, and campaign output is untrusted
(CONVENTIONS.md §9).

Every call writes ``prompt.txt``, ``stdout.txt``, ``stderr.txt`` and
``result.json`` into its own transcript directory, so a loop that ran overnight
can be read back in the morning.
"""

from __future__ import annotations

import shlex
import subprocess
import time
from pathlib import Path
from typing import Protocol

from ai_experiments.agents.contracts import AgentResult, extract_json, unwrap_envelope

#: Preset argv for the agent CLIs the harness is tested against. Each one reads
#: the prompt from stdin and prints the reply on stdout.
PRESETS: dict[str, list[str]] = {
    "claude": ["claude", "-p", "--output-format", "json"],
    "codex": ["codex", "exec", "-"],
}

DEFAULT_TIMEOUT_SECONDS = 600


class AgentRunner(Protocol):
    """Anything that turns a prompt into a JSON payload."""

    def run(self, prompt: str, *, role: str = "planner") -> AgentResult: ...


class CliAgentRunner:
    """Run a local agent CLI as a subprocess.

    ``command`` is a preset name (``claude``, ``codex``) or a full command line
    (``"my-agent --json"``), split with :func:`shlex.split` once, at
    construction. It is operator-supplied configuration, like
    ``monitoring.escalation.agent_command``; the *prompt* never reaches it as
    an argument.
    """

    def __init__(
        self,
        command: str = "claude",
        *,
        transcript_dir: Path | None = None,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
        cwd: Path | None = None,
    ) -> None:
        self.argv = list(PRESETS.get(command, [])) or shlex.split(command)
        if not self.argv:
            raise ValueError("agent command must not be empty")
        self.transcript_dir = Path(transcript_dir) if transcript_dir else None
        self.timeout_seconds = timeout_seconds
        self.cwd = Path(cwd) if cwd else None
        self._calls = 0

    def run(self, prompt: str, *, role: str = "planner") -> AgentResult:
        self._calls += 1
        started = time.monotonic()
        try:
            completed = subprocess.run(
                self.argv,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                cwd=str(self.cwd) if self.cwd else None,
            )
        except subprocess.TimeoutExpired:
            result = AgentResult(
                error=f"agent timed out after {self.timeout_seconds}s",
                duration_seconds=time.monotonic() - started,
            )
            self._write_transcript(role, prompt, "", "", result)
            return result
        except OSError as exc:
            result = AgentResult(
                error=f"agent command could not run: {exc}",
                duration_seconds=time.monotonic() - started,
            )
            self._write_transcript(role, prompt, "", "", result)
            return result

        result = _interpret(completed, time.monotonic() - started)
        self._write_transcript(role, prompt, completed.stdout, completed.stderr, result)
        return result

    def _write_transcript(
        self, role: str, prompt: str, stdout: str, stderr: str, result: AgentResult
    ) -> None:
        if self.transcript_dir is None:
            return
        directory = self.transcript_dir / role / str(self._calls)
        try:
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "prompt.txt").write_text(prompt)
            (directory / "stdout.txt").write_text(stdout)
            (directory / "stderr.txt").write_text(stderr)
            (directory / "result.json").write_text(result.model_dump_json(indent=2))
        except OSError:
            # A transcript is evidence, not state. Losing it must not fail the
            # round that produced it.
            return


class StubAgentRunner:
    """A scripted agent, for tests and for dry runs of a loop.

    Takes a list of replies (payload dicts, raw strings, or AgentResults) and
    returns them in order; once they run out it repeats the last one, so a loop
    of unknown length still terminates.
    """

    def __init__(self, replies: list[dict | str | AgentResult]) -> None:
        if not replies:
            raise ValueError("StubAgentRunner needs at least one reply")
        self.replies = replies
        self.prompts: list[str] = []

    def run(self, prompt: str, *, role: str = "planner") -> AgentResult:
        self.prompts.append(prompt)
        reply = self.replies[min(len(self.prompts) - 1, len(self.replies) - 1)]
        if isinstance(reply, AgentResult):
            return reply
        if isinstance(reply, str):
            payload = extract_json(reply)
            if payload is None:
                return AgentResult(raw=reply, error="agent reply contained no JSON")
            return AgentResult(ok=True, payload=payload, raw=reply)
        return AgentResult(ok=True, payload=reply, raw="")


def _interpret(
    completed: subprocess.CompletedProcess[str], duration: float
) -> AgentResult:
    output = completed.stdout or ""
    tail = output[-4000:]
    if completed.returncode != 0:
        return AgentResult(
            raw=tail,
            error=(
                f"agent exited {completed.returncode}: "
                f"{(completed.stderr or '').strip()[-500:]}"
            ),
            exit_code=completed.returncode,
            duration_seconds=duration,
        )
    payload = extract_json(output)
    if payload is None:
        return AgentResult(
            raw=tail,
            error="agent reply contained no JSON object",
            exit_code=0,
            duration_seconds=duration,
        )
    return AgentResult(
        ok=True,
        payload=unwrap_envelope(payload),
        raw=tail,
        exit_code=0,
        duration_seconds=duration,
    )
