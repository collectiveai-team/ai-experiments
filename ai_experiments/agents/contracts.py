"""What an agent must return, and how the harness digs it out of the reply.

An agent is a text process. It answers with prose, then a JSON object, and
sometimes wraps the whole thing in its own envelope. The harness never trusts
that text: it extracts one JSON object, validates it against a pydantic model,
and treats anything else as a failed call it can fall back from.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from collections.abc import Iterator

#: ```json ... ``` fences, the most common way an agent returns structured data.
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


class AgentResult(BaseModel):
    """One agent invocation, successful or not.

    ``ok`` is false for every failure mode — non-zero exit, timeout, no JSON in
    the reply — so a caller branches once instead of catching four exceptions.
    ``raw`` keeps the tail of the output for the transcript and the event log.
    """

    ok: bool = False
    payload: dict[str, Any] = Field(default_factory=dict)
    raw: str = ""
    error: str = ""
    exit_code: int | None = None
    duration_seconds: float = 0.0


def extract_json(text: str) -> dict[str, Any] | None:
    """Return the last JSON object in ``text``, or None.

    Agents put the answer at the end: they think out loud, then commit. Fenced
    blocks win over bare braces because a fence is an explicit "this is the
    answer" marker, and the last one wins because an agent that corrects itself
    corrects itself downward.
    """
    for candidate in reversed(_FENCE.findall(text)):
        parsed = _loads_object(candidate)
        if parsed is not None:
            return parsed
    for candidate in reversed(_brace_spans(text)):
        parsed = _loads_object(candidate)
        if parsed is not None:
            return parsed
    return None


def unwrap_envelope(  # ast-grep-ignore: no-dict-return-annotation
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Look through an agent CLI's own JSON wrapper.

    ``claude -p --output-format json`` returns metadata with the model's answer
    as a *string* under ``result``. The answer we want is inside that string.

    Both sides of this function are an agent's arbitrary JSON, not a shape the
    harness defines: the caller stores it unchanged in ``AgentResult.payload``
    and every reader of that field validates the keys it needs for itself.
    """
    inner = payload.get("result")
    if isinstance(inner, str):
        nested = extract_json(inner)
        if nested is not None:
            return nested
    return payload


def _loads_object(text: str) -> dict[str, Any] | None:
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _brace_spans(text: str) -> list[str]:
    """Return every balanced ``{...}`` span, outermost first, in source order.

    A regex cannot balance braces, and agent replies nest them (a params object
    inside a proposal). Scanning is short and exact.
    """
    spans: list[str] = []
    depth = 0
    start = -1
    for index, char in _outside_strings(text):
        if char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth:
            depth -= 1
            if depth == 0 and start >= 0:
                spans.append(text[start : index + 1])
    return spans


def _outside_strings(text: str) -> Iterator[tuple[int, str]]:
    """Yield ``(index, char)`` for every character outside a JSON string literal.

    A brace inside a string is data, not structure: in ``{"note": "}"}`` the
    middle ``}`` closes nothing, and a scanner that counted it would cut the
    object short. Tracking the escape state here keeps :func:`_brace_spans`
    about nesting alone.
    """
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if escaped:
            escaped = False
        elif in_string and char == "\\":
            escaped = True
        elif char == '"':
            in_string = not in_string
        elif not in_string:
            yield index, char
