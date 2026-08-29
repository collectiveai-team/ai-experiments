"""Round records: the loop's memory of what it tried and why.

`state.json` holds *what the campaign is now*. It cannot answer "why did round
3 try a smaller learning rate, and did it help?" — the question anyone asks
first when they read an overnight run. Round records answer it: one
append-only JSONL file, one record per stage.

The stages mirror the governed development pack — propose, apply, validate,
evaluate, review — because an experiment round is the same shape as a code
change: someone proposes a change, it is applied, it is checked, the result is
measured, and a reviewer decides what happens next.

Layout: ``<campaign_dir>/rounds.jsonl``. Append-only, like every event log
here: a round that went wrong is corrected by a later record, never by
rewriting an earlier one (CONVENTIONS.md §4).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from ai_experiments.schemas import utc_now

RoundStage = Literal["propose", "apply", "validate", "evaluate", "review"]

ROUNDS_FILE = "rounds.jsonl"


class RoundRecord(BaseModel):
    """One stage of one improvement round."""

    campaign_id: str
    round: int
    stage: RoundStage
    created_at: datetime = Field(default_factory=utc_now)
    strategy: str = ""
    hypothesis: str = ""
    rationale: str = ""
    trial_ids: list[str] = Field(default_factory=list)
    #: What the stage produced: submitted params, measured values, a verdict.
    outcome: dict[str, Any] = Field(default_factory=dict)
    agent_calls: int = 0
    used_fallback: bool = False
    #: Proposals the harness refused, with the reason. Kept because a pattern
    #: of rejections is itself a finding about the agent or the search space.
    rejected: list[dict[str, Any]] = Field(default_factory=list)


class RoundLog:
    """Append-only reader/writer for one campaign's round records."""

    def __init__(self, campaign_dir: str | Path) -> None:
        self.path = Path(campaign_dir) / ROUNDS_FILE

    def append(self, record: RoundRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as fh:
            fh.write(json.dumps(record.model_dump(mode="json")) + "\n")

    def read(self, limit: int | None = None) -> list[RoundRecord]:
        """Every readable record, oldest first.

        This is a history, not state: one corrupt line must not hide the rest,
        so unparseable lines are skipped.
        """
        if not self.path.exists():
            return []
        records: list[RoundRecord] = []
        for line in self.path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                records.append(RoundRecord(**json.loads(line)))
            except (json.JSONDecodeError, ValidationError):
                continue
        return records[-limit:] if limit is not None else records
