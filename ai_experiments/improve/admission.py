"""Admitting a code variant to a campaign: materialize, smoke-check, record.

This is the half of an improvement round the harness owns. Whoever proposes the
edits -- an agent, a person -- does not get to say whether they run: the
configured smoke command does, by its exit code, before a single trial is spent
on them. A variant that fails is deleted from disk and kept on the record, so
the next proposal can read why.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ai_experiments.improve.variants import (
    VariantError,
    discard_variant,
    materialize_variant,
    smoke_check,
    variants_root,
)
from ai_experiments.schemas import RunEvent

if TYPE_CHECKING:
    from ai_experiments.improve.variants import VariantEdit, VariantRecord
    from ai_experiments.store.campaign import CampaignStore


def admit_variant(
    campaign_store: CampaignStore,
    campaign_id: str,
    edits: list[VariantEdit],
    *,
    hypothesis: str = "",
    rationale: str = "",
    parent: str | None = None,
) -> VariantRecord:
    """Materialize a code variant of the workload, smoke-check it, and record the verdict.

    Raises `ValueError` when the goal did not enable variants or the edits
    are refused (a path outside `editable_paths`, an unknown parent).
    """
    goal = campaign_store.read_goal(campaign_id)
    spec = goal.variants
    if not spec.enabled:
        raise ValueError(
            f"campaign {campaign_id} did not enable code variants; set "
            "variants.enabled in the goal before proposing edits"
        )
    source = spec.source_dir or goal.workload.working_dir or "."
    try:
        record = materialize_variant(
            campaign_store.campaign_dir(campaign_id),
            source,
            edits,
            spec,
            parent=parent,
            hypothesis=hypothesis,
            rationale=rationale,
        )
    except VariantError as exc:
        raise ValueError(str(exc)) from exc

    record = smoke_check(record, spec)
    if record.smoke_ok is False:
        record = discard_variant(record)
    campaign_store.write_variant(campaign_id, record)
    campaign_store.append_event(
        campaign_id,
        RunEvent(
            message=(
                "variant rejected by its smoke check"
                if record.smoke_ok is False
                else "variant accepted"
            ),
            details={
                "variant_id": record.variant_id,
                "edited_paths": record.edited_paths,
                "hypothesis": record.hypothesis,
                "smoke_ok": record.smoke_ok,
                "discarded": record.discarded,
                "discard_error": record.discard_error,
            },
        ),
    )
    return record


def variant_dir(
    campaign_store: CampaignStore, campaign_id: str, variant_id: str | None
) -> str | None:
    """Where a trial's workload variant lives, if it has one."""
    if not variant_id:
        return None
    root = variants_root(campaign_store.campaign_dir(campaign_id)) / variant_id
    if not root.is_dir():
        raise ValueError(f"variant {variant_id} is missing from {root}")
    return str(root)
