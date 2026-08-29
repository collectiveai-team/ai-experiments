"""Turns a goal + trial history into concrete experiment manifests."""

from __future__ import annotations

import json
from typing import Any

from ai_experiments.planner.strategies import get_strategy
from ai_experiments.schemas import ExperimentManifest, GoalSpec, TrialRecord


def plan_next_params(
    goal: GoalSpec, trials: list[TrialRecord], count: int
) -> list[dict[str, Any]]:
    if count <= 0:
        return []
    return get_strategy(goal.strategy.name).plan(goal, trials, count)


def build_trial_manifest(
    goal: GoalSpec,
    trial_id: str,
    params: dict[str, Any],
    backend_address: str | None = None,
    working_dir: str | None = None,
    variant_id: str | None = None,
) -> ExperimentManifest:
    """Instantiate the goal's workload template with one parameter assignment.

    Params are injected three ways so any workload style works:
    - ``{name}`` placeholders in ``workload.args`` are substituted;
    - params without a placeholder are appended as ``--name value`` args;
    - the full assignment is exported as ``IAX_PARAMS`` (JSON) in the env.

    ``working_dir`` redirects the trial at a materialized workload variant, so
    a round that changed code runs the changed code without touching the
    user's tree.
    """
    args: list[str] = []
    substituted: set[str] = set()
    for arg in goal.workload.args:
        rendered = arg
        for name, value in params.items():
            placeholder = "{" + name + "}"
            if placeholder in rendered:
                rendered = rendered.replace(placeholder, _format_value(value))
                substituted.add(name)
        args.append(rendered)
    for name in sorted(params):
        if name not in substituted:
            args.extend([f"--{name}", _format_value(params[name])])

    env = dict(goal.workload.env)
    env["IAX_PARAMS"] = json.dumps(params)
    env["IAX_TRIAL_ID"] = trial_id

    update: dict[str, Any] = {"args": args, "env": env}
    if working_dir is not None:
        update["working_dir"] = working_dir
    workload = goal.workload.model_copy(update=update)
    return ExperimentManifest(
        experiment=f"{goal.name}/{trial_id}",
        backend=goal.backend,
        backend_address=backend_address or goal.backend_address,
        workload=workload,
        resources=goal.resources,
        monitoring=goal.monitoring,
        tracking=goal.tracking,
        metadata={
            **goal.metadata,
            "campaign": goal.name,
            "trial_id": trial_id,
            "params": params,
            **({"variant_id": variant_id} if variant_id else {}),
        },
    )


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return format(value, ".6g")
    return str(value)
