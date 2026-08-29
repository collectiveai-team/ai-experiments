---
name: running-campaigns
description: Use when asked to pursue a training goal autonomously — "find the best lr", "tune these hyperparams", "keep experimenting until val_loss < X". Authors a goal manifest, starts an iax campaign, keeps the daemon running, and reviews campaign progress.
---

# Running Campaigns

A campaign is a goal pursued autonomously: the planner generates trials over a
search space, the orchestrator submits/collects them, and the monitor daemon
drives the loop until the target is reached or the budget is spent.

Use **autonomous-experimentation** instead when the user states an outcome and
wants the harness to reach it — `iax loop` runs the whole loop as one command
and reports whether the target was met. This skill is for driving the rounds
yourself.

## Author the goal manifest

Translate the user's ask into a `GoalSpec` YAML (full reference:
`examples/goal_toy.yaml`; schema: `ai_experiments/schemas.py::GoalSpec`):

```yaml
goal: "<the user's objective, in one sentence>"
name: short-slug
objective: { metric: val_loss, mode: min, target: 0.05 }   # target optional
search_space:
  lr: { type: loguniform, low: 1e-5, high: 1e-2 }
  batch_size: { type: choice, values: [16, 32, 64] }
  layers: { type: int, low: 1, high: 4 }
  dropout: { type: uniform, low: 0.0, high: 0.5 }
workload:
  entrypoint: "python train.py"
  args: ["--lr", "{lr}"]        # {param} placeholders substituted;
  working_dir: .                # params without placeholders appended as --name value
budget: { max_trials: 12, max_parallel: 2, max_hours: 8.0,
          max_gpu_hours: 100, gpu_hour_rate: 2.5 }   # gpu budget + $/gpu-h optional
strategy: { name: adaptive, seed: 7 }     # grid | random | adaptive
backend: local                            # or ray + cluster/backend_address
monitoring:
  timeout_seconds: 14400
  auto_kill: true
```

Constraints that matter:
- The workload **must** print `IAX_METRIC {"step": N, "<metric>": value}`
  lines (or call `ai_experiments.report.report_metric`). The objective metric
  name must match `objective.metric` — without it trials score `null`.
- `strategy: adaptive` needs ≥3 completed trials before it exploits; for tiny
  budgets (<4 trials) prefer `random` or `grid`.
- The full params dict also reaches the workload as `IAX_PARAMS` (JSON env var).

## Launch and drive

```bash
iax campaign validate goal.yaml
iax run goal.yaml                     # all-in-one: campaign + dashboard + loop;
                                      # blocks until done; Ctrl+C detaches safely
```

For long-lived or multi-campaign setups, split the pieces:

```bash
iax campaign start goal.yaml          # submits the first batch, prints campaign_id
iax daemon --interval 30              # REQUIRED: drives the loop; keep it running
                                      # (tmux/systemd/launchd; or `--once` per tick)
```

Without `iax run` or a running daemon the campaign does not advance.
`iax campaign advance <campaign_id>` performs a single step manually.

## Observe

```bash
iax campaign list
iax campaign status <campaign_id> --json   # best trial, history, gpu-hours, cost
iax leaderboard                             # rank all campaigns by best objective
iax artifacts <run_id>                      # checkpoints the workload saved
iax serve                                   # dashboard at http://127.0.0.1:8585
```

Workloads should save checkpoints/plots into `$IAX_ARTIFACTS_DIR`. Every run
also gets a repro bundle (git SHA, dirty diff, environment) — `iax repro
<run_id>` shows it, `iax rerun <run_id>` repeats the run exactly.

For MLflow mirroring add `tracking: {mlflow: true}` to the goal (needs the
`[mlflow]` extra). The harness creates the MLflow run, logs params/metrics/
artifacts/git-sha, and injects `MLFLOW_RUN_ID` into the workload env — on Ray
clusters the workload should `mlflow.log_artifact(...)` so checkpoints reach
the central artifact store (the local `artifacts/` dir is not visible across
machines).

## Refine mid-flight

When results cluster in a region, narrow the space without losing history:

```bash
iax campaign pause <campaign_id>             # active trials finish; no new ones
iax campaign edit <campaign_id> goal2.yaml   # tighter search_space / new budget;
                                             # the objective metric must not change
iax campaign resume <campaign_id>            # replans from the full trial history
```

A finished campaign writes `summary.json` in
`<runs>/_campaigns/<campaign_id>/`. `stop_reason` is one of:

| `stop_reason` | what it means |
|---|---|
| `target_reached` | the objective target was met; the best trial is the answer |
| `budget_exhausted` | `max_trials` were run without reaching the target |
| `max_hours_exceeded` | `budget.max_hours` elapsed |
| `gpu_hours_exhausted` | `budget.max_gpu_hours` were spent |
| `search_space_exhausted` | the planner ran out of points; widen the goal |
| `backend_unavailable` | no trial could be submitted; start the cluster |
| `objective_not_reported` | trials ran but never reported the objective metric |
| `agent_requested_stop` | the reviewing agent judged the campaign hopeless |
| `user_requested` | `iax campaign stop` |

Only `target_reached` answers the question. Every other reason means the
campaign stopped for a reason of its own, and the best trial so far is a
partial result — say which one it was when you report.

## Inject your own analysis (opt-in tokens)

With `analysis.agent_review: true`, each round drops
`<runs>/_escalations/campaign_<id>.json` containing the trial history. Review
it and act:

```bash
iax campaign suggest <campaign_id> --params '{"lr": 3e-4, "layers": 2}' \
  --note "best region per trial history; halve lr from t004"
iax campaign stop <campaign_id>       # when continuing is pointless
```

Suggested trials are submitted before strategy-planned ones on the next
advance and count toward `max_trials`.

## Stuck runs inside a campaign

The daemon already kills fatal runs (`auto_kill: true`) and escalates
suspicious ones — see `iax escalations` and the **monitoring-experiments** /
**diagnosing-experiments** skills. A killed/failed trial records its error in
the campaign state and the loop keeps going.
