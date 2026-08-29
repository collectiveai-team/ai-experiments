"""Agent invocation: one prompt in, one validated JSON payload out."""

from __future__ import annotations

from ai_experiments.agents.contracts import AgentResult, extract_json, unwrap_envelope
from ai_experiments.agents.runner import (
    PRESETS,
    AgentRunner,
    CliAgentRunner,
    StubAgentRunner,
)

__all__ = [
    "PRESETS",
    "AgentResult",
    "AgentRunner",
    "CliAgentRunner",
    "StubAgentRunner",
    "extract_json",
    "unwrap_envelope",
]
