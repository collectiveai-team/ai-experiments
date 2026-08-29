from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Iterable

from ai_experiments.schemas import (
    CampaignState,
    GoalSpec,
    RunEvent,
    load_stored,
    utc_now,
)
from ai_experiments.store.filesystem import atomic_write_text


class CampaignStore:
    """Filesystem store for campaigns, living beside the run store.

    Layout: ``<root>/<campaign_id>/{goal.yaml, state.json, events.jsonl}``.
    """

    def __init__(self, runs_root: str | Path) -> None:
        self.root = Path(runs_root) / "_campaigns"

    def create_campaign(self, goal: GoalSpec) -> CampaignState:
        campaign_id = f"cmp_{uuid.uuid4().hex[:12]}"
        campaign_dir = self.root / campaign_id
        campaign_dir.mkdir(parents=True, exist_ok=False)
        (campaign_dir / "goal.yaml").write_text(goal.to_yaml())
        state = CampaignState(campaign_id=campaign_id, name=goal.name, goal=goal.goal)
        self.write_state(state)
        return state

    def campaign_dir(self, campaign_id: str) -> Path:
        return self.root / campaign_id

    def read_goal(self, campaign_id: str) -> GoalSpec:
        return load_stored(GoalSpec, self.campaign_dir(campaign_id) / "goal.yaml")

    def write_state(self, state: CampaignState) -> None:
        state.updated_at = utc_now()
        atomic_write_text(
            self.campaign_dir(state.campaign_id) / "state.json",
            json.dumps(state.model_dump(mode="json"), indent=2),
        )

    def read_state(self, campaign_id: str) -> CampaignState:
        path = self.campaign_dir(campaign_id) / "state.json"
        return CampaignState(**json.loads(path.read_text()))

    def append_event(self, campaign_id: str, event: RunEvent) -> None:
        path = self.campaign_dir(campaign_id) / "events.jsonl"
        with path.open("a") as fh:
            fh.write(json.dumps(event.model_dump(mode="json")) + "\n")

    def read_events(self, campaign_id: str, tail: int | None = None) -> list[RunEvent]:
        path = self.campaign_dir(campaign_id) / "events.jsonl"
        if not path.exists():
            return []
        lines = path.read_text().splitlines()
        if tail is not None:
            lines = lines[-tail:]
        return [RunEvent(**json.loads(line)) for line in lines if line.strip()]

    def list_campaigns(self) -> Iterable[str]:
        if not self.root.exists():
            return []
        return sorted(
            path.name
            for path in self.root.iterdir()
            if path.is_dir() and (path / "state.json").exists()
        )
