# Industrial AI Experiments

Autonomous experiment harness for industrial AI training workloads.

`ai-experiments` turns a research **goal** into a self-driving campaign: it
plans experiments over a search space, submits them to a local or Ray backend
(any cloud Ray runs on), watches them with cheap programmatic checks — killing
runs that diverge, freeze, or time out — escalates to an agent only when the
free checks can't decide, and when trials finish it analyzes results and
schedules better ones until the target is reached or the budget is spent.

```
goal.yaml ──> planner ──> trials ──> backend (local | ray @ aws/gcp/azure)
                ▲                        │
                │   analysis             ▼
            orchestrator <── metrics ── runs   <── monitor daemon
                                                    (kill / escalate)
```

## Responsibilities

- Understand a goal manifest: objective metric, search space, budget, strategy.
- Plan trials (grid / random / adaptive) and orchestrate them on `local` or `ray`.
- Persist portable run + campaign state on the filesystem (greppable, agent-friendly).
- Monitor every run: heartbeats, metric staleness, NaN/divergence, plateaus,
  hard timeouts, dead workers — programmatic first, agent escalation second,
  with cooldowns and per-run agent-call budgets so tokens are spent only when needed.
- Auto-experiment loop: analyze finished trials, plan improvements, submit more.
- Serve a web dashboard (`iax serve`) and a JSON REST API over all of it.

## Non-responsibilities

- Model-specific training logic (workloads are arbitrary commands).
- Reimplementing cloud APIs — cluster provisioning delegates to Ray's own
  cluster launcher (`ray up` via `iax cluster up`).
- Workbench project/session persistence and UI rendering.

## Quick start: an autonomous campaign

One command — starts the campaign, serves the dashboard, and drives the
monitor/experiment loop until the goal is reached or the budget is spent:

```bash
iax run examples/goal_toy.yaml --open     # dashboard at http://127.0.0.1:8585
```

Ctrl+C detaches without killing the trials; resume the loop any time with
`iax daemon`. The same pieces are also available separately for long-lived /
multi-campaign setups:

```bash
iax campaign validate examples/goal_toy.yaml   # check the goal
iax campaign start examples/goal_toy.yaml      # submit the first batch
iax daemon --interval 30                       # the loop driver (supervised service)
iax serve                                      # dashboard for everything in the store
iax campaign status <campaign_id>
```

The workload contract is one line of stdout per observation:

```python
print('IAX_METRIC {"step": 12, "loss": 0.0734}')   # or:
from ai_experiments.report import report_metric
report_metric(step=12, loss=0.0734)
```

This works identically on the local backend (the worker tails stdout) and on
remote Ray clusters with no shared filesystem (metrics are extracted from job
logs).

## Artifacts and reproducibility

Workloads write checkpoints/plots/models to `$IAX_ARTIFACTS_DIR` (or use
`ai_experiments.report.artifacts_dir()`); the harness lists them with
`iax artifacts <run_id>` and serves them as download links in the run's
dashboard view. (Local backend; on remote Ray clusters artifacts stay on the
cluster's storage.)

Every submit also captures a repro bundle under `<run_dir>/repro/` — git SHA,
branch, dirty flag + `diff.patch` of uncommitted changes, python/platform,
and the installed package list. Then:

```bash
iax repro <run_id>     # show the captured context
iax rerun <run_id>     # resubmit the exact persisted manifest;
                       # warns when your git state differs from the recorded one
```

## MLflow integration

Enable per manifest/goal (needs `pip install 'ai-experiments[mlflow]'` — or
full `mlflow` if you also run the tracking server):

```yaml
tracking:
  mlflow: true
  tracking_uri: http://mlflow.internal:5000   # default: MLFLOW_TRACKING_URI env
  experiment: my-project                      # default: the campaign name
```

Two halves:

- **Harness mirroring (zero workload changes)**: iax creates the MLflow run
  at submit (tagged `iax.run_id`, campaign, trial, git commit from the repro
  bundle, with trial params logged), and the daemon finalizes it at terminal
  state — all `IAX_METRIC` points with steps, the local `artifacts/` dir,
  and the matching terminal status (`FINISHED`/`FAILED`/`KILLED`).
- **Workload handoff (solves remote artifacts)**: the workload env gets
  `MLFLOW_RUN_ID` + `MLFLOW_TRACKING_URI` — on the local backend and inside
  Ray `runtime_env`. A workload on a remote cluster node that calls
  `mlflow.log_artifact("ckpt.pt")` ships checkpoints straight to the central
  MLflow artifact store; both halves write into the *same* MLflow run.

Tracking is best-effort: missing mlflow or an unreachable server records a
warning event and never blocks a submit or a daemon tick.

For teams, point `tracking_uri` at an MLflow tracking server (`http://...`) —
that is also what lets remote Ray workloads deliver artifacts. For solo/local
use a `file://` store works: MLflow 3.x deprecates it behind
`MLFLOW_ALLOW_FILE_STORE`, but iax sets the opt-out automatically when a file
store is the configured choice (an explicit `MLFLOW_ALLOW_FILE_STORE=false`
is respected). Note `mlflow-skinny` has no SQL store — `sqlite:///` URIs need
the full `mlflow` package.

## Comparison and cost

```bash
iax leaderboard        # campaigns ranked by best objective (+ gpu-hours, cost)
```

The dashboard adds a leaderboard panel and **run comparison**: select runs
with checkboxes and overlay their metric curves on one chart.

GPU spend is first-class budget: trials record `gpu_hours`
(`resources.gpus` × runtime), and a campaign stops with
`gpu_hours_exhausted` when `budget.max_gpu_hours` is spent. Set
`budget.gpu_hour_rate` to see estimated cost in `campaign status`, the
leaderboard, and the dashboard.

## Notifications

The daemon pushes alerts when a campaign finishes or a run is killed/
escalated. Sinks (all best-effort, always also logged to
`<runs>/_notifications.jsonl`):

```bash
iax daemon --notify-webhook https://hooks.slack.com/services/...   # Slack-ready
iax daemon --notify-command "./send_email.sh"                      # JSON on stdin
# or env: IAX_NOTIFY_WEBHOOK / IAX_NOTIFY_COMMAND
```

## Mid-flight goal editing

Narrow the search space around the best region without losing trial history:

```bash
iax campaign pause <campaign_id>            # stop scheduling (active runs finish)
iax campaign edit <campaign_id> goal2.yaml  # new space/budget/strategy; same metric
iax campaign resume <campaign_id>           # strategy replans from full history
```

Pause/resume are also one click in the dashboard.

## Monitoring: programmatic first, agent second

Every daemon tick runs free checks per active run:

| Check | Verdict |
|---|---|
| NaN/inf in reported metrics (`fatal_on_nan`) | `kill` |
| `timeout_seconds` exceeded | `kill` |
| Local worker process dead | `kill` (reaped as failed) |
| Heartbeat stale (worker alive-ness, 15s cadence) | suspicious |
| No metric progress for `stuck_after_minutes` | suspicious |
| Objective plateau over `plateau_patience_points` | suspicious |

`kill` verdicts are handled inline: cancelled when the run's policy sets
`auto_kill: true`, escalated otherwise. Suspicious runs climb an escalation
ladder — only after `after_suspicious_ticks` consecutive suspicious ticks does
the daemon involve an agent, subject to `cooldown_minutes` and
`max_agent_calls`. With no `agent_command` configured the escalation is just a
file under `<runs>/_escalations/` plus `iax escalations` — an external agent
session picks it up and zero tokens are spent by the daemon.

```yaml
monitoring:
  auto_kill: true
  timeout_seconds: 14400
  escalation:
    after_suspicious_ticks: 3
    cooldown_minutes: 30
    max_agent_calls: 3
    agent_command: >-
      claude -p 'Diagnose iax run {run_id} in {run_dir}; reply JSON
      {"verdict": "kill"|"continue", "reason": "..."}' --output-format json
```

## Clusters (local, AWS, GCP, Azure)

Ray is the execution substrate everywhere; clouds differ only in how the
cluster comes up. Name your clusters once (see `examples/clusters.yaml`):

```bash
iax cluster list
iax cluster status vader        # pings the Ray dashboard
iax cluster up aws-gpu          # ray up <cluster_config> -y
iax cluster down aws-gpu
```

Goal manifests select a cluster by name (`cluster: aws-gpu`) or address
(`backend_address: http://head:8265`).

## Agent integration

The daemon and the planner are fully programmatic. Agents plug in at three
opt-in points:

- **Escalations** — `iax escalations` lists runs the free checks flagged;
  the `diagnosing-experiments` skill drives the triage.
- **Agent verdicts** — set `monitoring.escalation.agent_command` and the
  daemon itself invokes an agent and acts on a JSON `kill`/`continue` verdict.
- **Campaign review** — set `analysis.agent_review: true` and each round drops
  a review request with the trial history; the agent queues better trials with
  `iax campaign suggest <campaign_id> --params '{"lr": 3e-4}'`.

## Install

### Runtime (`iax` CLI)

Install the experiment runtime from git, pinned to a release tag so the CLI and
manifest contract stay reproducible:

```bash
# Public HTTPS
uv pip install "ai-experiments @ git+https://github.com/collectiveai-team/ai-experiments.git@v0.1.0"

# Private repo over SSH (uses your git credentials)
uv pip install "ai-experiments @ git+ssh://git@github.com/collectiveai-team/ai-experiments.git@v0.1.0"
```

For Ray Jobs API support, add the `ray` extra:

```bash
uv pip install "ai-experiments[ray] @ git+https://github.com/collectiveai-team/ai-experiments.git@v0.1.0"
```

Pin `@v0.1.0` to the tag matching the `version` in `pyproject.toml`; use `@main` or a
commit SHA to track unreleased work.

### Agent skills

The repo also ships agent skills (under `.claude/skills/`) that teach a Claude Code
agent to drive the harness. Install them with the open [`skills`](https://github.com/vercel-labs/skills) CLI:

```bash
# Public HTTPS
npx skills@latest add https://github.com/collectiveai-team/ai-experiments

# Private repo over SSH
npx skills@latest add git@github.com:collectiveai-team/ai-experiments.git
```

This installs the `submitting-experiments`, `monitoring-experiments`,
`diagnosing-experiments`, and `cancelling-experiments` skills into the project's
`.claude/skills/` (add `-g` for `~/.claude/skills/`).

During in-repo development, Workbench uses the uv workspace member at `packages/ai_experiments`.

## Manifest

```yaml
experiment: demand_forecast_baseline
backend: ray
workload:
  entrypoint: python
  args:
    - -m
    - ts_agents_lab.cli
    - train
    - configs/training.yaml
  working_dir: .
monitor:
  interval_seconds: 300
  stuck_after_seconds: 1800
metadata:
  project_id: example
```

## CLI

```bash
# single runs
iax validate experiment.yaml           # add --strict to fail on warnings
iax submit experiment.yaml --json      # warns on the same checks; --strict refuses
iax runs
iax status <run_id> --json
iax logs <run_id> --tail 200
iax metrics <run_id> --tail 50
iax artifacts <run_id>
iax repro <run_id>
iax rerun <run_id>
iax diagnose <run_id> --json
iax monitor <run_id> --json --quiet-when-waiting
iax cancel <run_id>
iax escalations
iax leaderboard

# campaigns (goal-driven auto-experiment loop)
iax campaign validate goal.yaml
iax campaign start goal.yaml           # --strict refuses a workload that cannot start
iax campaign list / status / advance / suggest / pause / edit / resume / stop

# infrastructure
iax daemon --interval 30        # monitor + loop driver (foreground)
iax serve --port 8585           # dashboard + REST API  (needs [server] extra)
iax cluster list / status / up / down
```

`iax monitor --quiet-when-waiting` is the Workbench scheduler integration point. It prints nothing while the run should keep waiting. It emits JSON only when the run has completed, failed, looks stuck, or needs delegated review.

## Example agent prompts

With the agent skills installed (see [Agent skills](#agent-skills)), you can drive the
harness in natural language — the agent authors the manifest, validates it, submits the
run, and watches it for you.

**Simple local run:**

> Run a quick training experiment locally: the command is `python train.py --epochs 5`
> in `./workloads/forecast`. Check on it every minute and tell me if it finishes or
> stalls for more than 10 minutes.

**On the local Ray cluster:**

> Launch the demand-forecast training on the Ray backend with 4 CPUs and 1 GPU
> (entrypoint `python -m forecast.train configs/base.yaml`). Keep monitoring it and only
> ping me if it fails or looks stuck — otherwise summarize the results when it completes.

## Local Ray cluster

The `ray` backend submits through the Ray Jobs API, which is served by the cluster
dashboard (default `http://127.0.0.1:8265`). To run a small single-node cluster on your
machine:

```bash
uv pip install "ai-experiments[ray] @ git+https://github.com/collectiveai-team/ai-experiments.git@v0.1.0"

# Start a head node — dashboard + Jobs API listen on :8265
ray start --head --num-cpus=4 --dashboard-host=127.0.0.1

# Point the backend at it (this is also the default)
export RAY_ADDRESS=http://127.0.0.1:8265
```

Any manifest with `backend: ray` now submits jobs to that cluster:

```bash
iax submit experiment.yaml --json
iax status <run_id> --json   # mirrors the Ray job state
```

Add worker nodes from other machines with `ray start --address=<head-host>:6379`. Tear
the cluster down with `ray stop`.

## Composition With Workbench

Workbench composes this runtime instead of owning it:

1. A training agent exports a generic experiment manifest.
2. Workbench persists the returned run handle in `experiment_runs`.
3. The scheduler runs `iax monitor <run_id> --json --quiet-when-waiting`.
4. Empty monitor output means no agent wake-up.
5. JSON monitor output updates the run record and can trigger the training-monitor agent.

When this package is extracted to its own repository, Workbench should depend on a released package version instead of a workspace source:

```toml
dependencies = [
  "ai-experiments>=0.1.0",
]
```

and remove the local uv source override:

```toml
[tool.uv.sources]
ai-experiments = { workspace = true }
```

## Release Checklist

1. Run package tests.
2. Build wheel and sdist.
3. Publish to the chosen package registry or GitHub Release assets.
4. Update Workbench to consume the released version.
5. Run Workbench backend, package, frontend, and scaffold verification.
