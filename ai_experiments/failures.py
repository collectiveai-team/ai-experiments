"""How a dead workload explains itself.

A trial's ``error`` is read by three audiences: a person scanning
``iax campaign trials``, the planner's evidence block, and an agent deciding
what to try next. "workload exited with code 1" serves none of them, and a
whole job log serves them worse — it drowns the one line that mattered and
costs tokens on every later prompt.

So every backend trims failure output the same way: the last few lines, to a
bounded length. The output is untrusted. It is quoted, never executed.
"""

from __future__ import annotations

#: How many of a failed workload's final output lines survive into its error.
ERROR_TAIL_LINES = 3
#: The hard ceiling on an error message, in characters.
ERROR_TAIL_CHARS = 400


def error_tail(lines: list[str]) -> str:
    """Join the last thing a workload said, trimmed to fit an error field."""
    kept = [line.strip() for line in lines if line.strip()][-ERROR_TAIL_LINES:]
    tail = " | ".join(kept)
    if len(tail) > ERROR_TAIL_CHARS:
        tail = tail[: ERROR_TAIL_CHARS - 1].rstrip() + "…"
    return tail


def failure_message(prefix: str, output: str | list[str]) -> str:
    """`prefix`, plus whatever the workload last said, when it said anything."""
    lines = output.splitlines() if isinstance(output, str) else list(output)
    tail = error_tail(lines)
    return f"{prefix}: {tail}" if tail else prefix
