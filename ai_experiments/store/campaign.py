from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from ai_experiments.config_loading import load_stored
from ai_experiments.schemas import (
    CampaignState,
    GoalSpec,
    RunEvent,
    utc_now,
)
from ai_experiments.store.filesystem import atomic_write_text

if TYPE_CHECKING:
    from collections.abc import Iterable

    # import cycle: variants imports schemas
    from ai_experiments.improve.variants import VariantRecord


class CampaignStore:
    """Filesystem store for campaigns, living beside the run store.

    Layout: ``<root>/<campaign_id>/{goal.yaml, state.json, events.jsonl}``,
    plus ``variants.json`` once the campaign has been offered code variants.
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

    # -- variants --------------------------------------------------------------

    def _variants_path(self, campaign_id: str) -> Path:
        return self.campaign_dir(campaign_id) / "variants.json"

    def read_variants(self, campaign_id: str) -> list[VariantRecord]:
        """Every variant proposed to this campaign, accepted or not.

        A rejected variant stays on the list: the next proposal has to be able
        to read what already failed, or an unattended loop will retry it.
        """
        from ai_experiments.improve.variants import VariantRecord

        path = self._variants_path(campaign_id)
        if not path.exists():
            return []
        return [VariantRecord(**item) for item in json.loads(path.read_text())]

    def read_variant(self, campaign_id: str, variant_id: str) -> VariantRecord | None:
        for record in self.read_variants(campaign_id):
            if record.variant_id == variant_id:
                return record
        return None

    def write_variant(self, campaign_id: str, record: VariantRecord) -> None:
        """Append the record, or replace the one with the same id."""
        records = [
            item for item in self.read_variants(campaign_id) if item.variant_id != record.variant_id
        ]
        records.append(record)
        atomic_write_text(
            self._variants_path(campaign_id),
            json.dumps([item.model_dump(mode="json") for item in records], indent=2),
        )
