# The goal file, field by field

The schema is `ai_experiments/schemas.py::GoalSpec`. `iax new goal goal.yaml`
writes a commented, valid one; this file explains the choices that file leaves
to you. `iax campaign validate goal.yaml` rejects anything wrong before a
single trial runs — always run it.

## objective

```yaml
objective: { metric: val_loss, mode: min, target: 0.05 }
```

- `metric` must be **exactly** a key in the `IAX_RESULT` line the workload's
  `evaluate` phase declares (`ai_experiments.report.report_result`, or print
  the line directly) — never a key from an `IAX_METRIC` progress line, which
  never scores. A workload that never prints `IAX_RESULT` scores `null`
  tagged `no_result`; a name mismatch scores `null` tagged `metric_absent`.
- `mode` is `min` or `max`. It decides what "best" means everywhere: the
  planner, the leaderboard, and the target check.
- `target` is optional. With it, the campaign stops the moment a trial reaches
  it (`stop_reason: target_reached`, `iax loop` exits 0). Without it, the
  campaign always spends its budget and `iax loop` always exits 4.

## search_space

Four parameter types, and nothing else:

```yaml
search_space:
  lr:         { type: loguniform, low: 1.0e-5, high: 1.0e-2 }   # low > 0
  dropout:    { type: uniform,    low: 0.0,    high: 0.5 }
  layers:     { type: int,        low: 1,      high: 4 }
  batch_size: { type: choice,     values: [16, 32, 64] }
```

- Use `loguniform` for anything that spans orders of magnitude — learning
  rates, weight decay, regularisation strengths. `uniform` over `1e-5..1e-2`
  puts almost every sample above `1e-3`.
- Use `choice` for anything not numeric, and for values a workload only
  accepts from a fixed set.
- Every key must reach the workload. Params appear as `IAX_PARAMS` (a JSON
  env var) and, when named in `workload.args`, as `{placeholder}`
  substitutions. A key the workload ignores wastes the whole search.
- Keep the space small. Each added parameter multiplies what the budget has
  to cover; three well-chosen knobs beat eight guessed ones.

## workload

```yaml
workload:
  # `entrypoint` is the single-phase fallback: with both `train` and
  # `evaluate` set, those two run instead and the entrypoint is unused.
  entrypoint: python
  # The trainer's flags live here, not in a top-level `args`: `args` is
  # appended to every phase, so it would hand the trainer's flags to the
  # evaluator too.
  train: "python train.py --epochs 20"
  evaluate: "python evaluate.py"
  working_dir: "."                     # relative paths resolve from here
  env: { CUDA_VISIBLE_DEVICES: "0" }
```

Params without a `{placeholder}` in `args` are appended as `--name value` —
and `args` itself is appended to **every** declared phase, `train` and
`evaluate` alike. Only `evaluate` may declare the result that scores the
trial; `train` may only report progress (`IAX_METRIC`), and a result printed
from `train` is discarded with a warning. Give the workload only the
environment it needs: it runs untrusted code paths and its stdout is
untrusted input.

## budget

```yaml
budget:
  max_trials: 24        # the hard ceiling on trials
  max_parallel: 4       # how many run at once
  max_hours: 8.0        # wall clock for the whole campaign
  max_gpu_hours: 100    # optional
  gpu_hour_rate: 2.50   # optional, display only
```

`max_parallel` should match what the backend can actually run at once. Above
that, trials queue and `max_hours` expires with the budget unspent.

For `strategy: adaptive`, keep `max_trials` at 8 or more: it needs three
completed trials before it exploits anything. Below that, use `random`.

## strategy

| name | picks params by | use when |
|---|---|---|
| `grid` | an even sweep of every axis | few params, and you want coverage |
| `random` | independent samples | a first look, or a tiny budget |
| `adaptive` | perturbing the best trials so far | the default; most campaigns |
| `agent` | asking an agent, with the full history | params interact, or failures are informative |

```yaml
strategy:
  name: adaptive
  seed: 7             # same seed, same trials — set it, so a run is repeatable
  exploration: 0.3    # adaptive: share of fresh random samples per round
  top_k: 2            # adaptive: perturb around one of the best k trials
  fallback: adaptive  # agent: plans the round when the agent cannot
```

## agent, analysis

Read only when `strategy.name: agent` or `analysis.review_between_rounds`.

```yaml
agent:
  command: claude          # claude | codex | any command reading stdin
  timeout_seconds: 600
  max_calls: 20            # cost ceiling for the campaign
analysis:
  review_between_rounds: true    # a verdict after each round
  apply_agent_changes: false     # true lets a verdict widen space or budget
```

Every agent reply is validated against the search space before use. Out-of-
range and repeated params are dropped, and a crash, a timeout, an exhausted
`max_calls`, or a reply without JSON costs the round to `strategy.fallback` —
never the campaign. Transcripts land under `<campaign_dir>/agents/`.

`apply_agent_changes` may widen `search_space` and `budget`. It can never
change `objective.metric`: past values were measured against it.

## backend

```yaml
backend: local                       # subprocesses on this machine
```

```yaml
backend: ray
backend_address: ray://head:10001    # an explicit cluster
# cluster: prod-gpu                  # or a named one from the cluster config
resources: { cpus: 8, gpus: 1 }      # per trial
```

Use `local` to prove the workload declares its result. Move to `ray` for
anything with real parallelism. `resources` is per trial, and `max_parallel`
times `resources` must fit the cluster.

## monitoring

```yaml
monitoring:
  interval_seconds: 30
  stuck_after_minutes: 10
  auto_kill: true          # kill a trial the harness judges fatal
  fatal_on_nan: true       # a NaN loss is a dead trial, not a slow one
```

Leave `auto_kill` on for unattended loops. A trial that has already diverged
only spends budget.

## tracking

```yaml
tracking: { mlflow: true, experiment: my-project }   # needs the [mlflow] extra
```

The harness mirrors params, metrics, artifacts and the git SHA, and injects
`MLFLOW_RUN_ID` into the workload — so remote Ray trials can log their own
artifacts to the tracking server, which a local `artifacts/` directory cannot
reach.
