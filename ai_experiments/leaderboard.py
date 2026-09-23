"""Ranking every campaign in a store by its best objective value.

The CLI's `iax leaderboard` and the server's `/api/leaderboard` answer the same
question and must answer it the same way, so the ranking lives here once and both
surfaces render what it returns.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from ai_experiments.planner.analysis import summarize_campaign
from ai_experiments.responses import LeaderboardRow

if TYPE_CHECKING:
    from ai_experiments.store.campaign import CampaignStore


def leaderboard_rows(campaign_store: CampaignStore) -> list[LeaderboardRow]:
    """Rank the campaigns that have a best trial, best first within each metric.

    Campaigns without one are left out: a row exists to compare a result, and a
    campaign that has not produced one has nothing to compare.
    """
    rows: list[LeaderboardRow] = []
    for campaign_id in campaign_store.list_campaigns():
        state = campaign_store.read_state(campaign_id)
        goal = campaign_store.read_goal(campaign_id)
        summary = summarize_campaign(state, goal)
        if summary.best is None:
            continue
        rows.append(
            LeaderboardRow(
                campaign_id=campaign_id,
                name=state.name,
                status=state.status,
                metric=goal.objective.metric,
                mode=goal.objective.mode,
                # best_trial() only sets `best` from trials with a non-None,
                # finite objective_value; cast makes that invariant visible here.
                best_value=cast("float", summary.best.objective_value),
                best_params=summary.best.params,
                best_run_id=summary.best.run_id,
                trials=len(state.trials),
                gpu_hours=summary.gpu_hours,
                estimated_cost=summary.estimated_cost,
                updated_at=state.updated_at.isoformat(),
            )
        )
    rows.sort(
        key=lambda r: (
            r.metric,
            -r.best_value if r.mode == "max" else r.best_value,
        )
    )
    return rows
