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

Installed from PyPI, with no repo checkout, scaffold the same two files first:

```bash
iax new workload train.py     # a workload that already reports IAX_METRIC
iax new goal goal.yaml        # a commented goal; point its workload at train.py
iax run goal.yaml --open
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

## One goal, one answer (`iax loop`)

`iax run` is for a human watching a dashboard. `iax loop` is for an agent: it
starts the campaign, drives it to a conclusion, prints one report, and exits.

```bash
iax loop goal.yaml --json            # blocks until the loop ends
echo $?                              # 0 = target reached, 4 = it was not
```

```json
{
  "campaign_id": "cmp_20260829_1",
  "status": "completed",
  "stop_reason": "target_reached",
  "target_reached": true,
  "rounds": 4,
  "trials": 12,
  "best": {"trial_id": "t_009", "objective_value": 0.0041, "params": {"x": 1.94}}
}
```

Exit code 4 is not an error: the work ran, the objective was not met. A caller
must be able to tell that apart from a bad goal file (2) or an unreachable
cluster (3), which is why it has a code of its own.

The loop stops by itself. `--max-rounds` and `--max-seconds` bound it further;
either one leaves the campaign `running`, so a later `iax loop --resume
<campaign_id>` continues the same campaign instead of starting a new one.

### Review between rounds

With `strategy: agent`, the agent can also be asked to judge the campaign
after each round, not just to propose the next one:

```yaml
analysis:
  review_between_rounds: true    # ask the agent for a verdict between rounds
  apply_agent_changes: true      # let an accepted verdict edit the goal
```

The verdict is `continue`, `stop`, or `change_goal`. A `stop` ends the campaign
with the agent's reason instead of burning the rest of the budget on a goal
that cannot be reached. A `change_goal` may widen the search space or the
budget — and nothing else: the objective metric stays fixed, because every
value already recorded was measured against it. Each review is appended to
`rounds.jsonl` as a `review` stage, so the decision is auditable afterwards.

## The same loop from python

`iax` is one caller of the library, not the library. Everything the CLI does
is a function in `ai_experiments.api`, taking plain data and returning plain
data, so an agent in a chat session can drive a campaign without a shell:

```python
from ai_experiments import api

goal = api.goal_from_dict({...})          # or api.goal_from_yaml("goal.yaml")
report = api.run_loop(goal, max_rounds=20)

if not report.target_reached:             # think, then continue the same one
    api.suggest_trial(report.campaign_id, {"lr": 3e-4}, note="the flat region")
    api.run_loop(goal, campaign_id=report.campaign_id)
```

| function | answers |
|---|---|
| `start_campaign(goal)` | begin, and submit the first round |
| `advance_campaign(id)` | one loop step, when the agent wants to think between rounds |
| `campaign_report(id)` | where it stands, without advancing it |
| `campaign_rounds(id)` | what the loop believed at each round, oldest first |
| `suggest_trial(id, params)` | queue one trial the agent chose itself |
| `run_loop(goal)` | all of the above, until it ends |

Failures raise `IaxError`, carrying the same `code` and `exit_code` the CLI
reports, so an agent handles a bad goal the same way whichever surface it uses.

## Agent-planned rounds (`strategy: agent`)

The built-in strategies search a fixed space by fixed rules. `strategy: agent`
asks an agent instead: it reads the goal, the search space, and every trial so
far — failures and their errors included — and answers with the next batch.

```yaml
strategy:
  name: agent
  fallback: adaptive      # used whenever the agent cannot deliver a round
agent:
  command: claude         # claude | codex | any command reading a prompt on stdin
  timeout_seconds: 600
  max_calls: 20           # hard ceiling on agent calls for this campaign
```

The harness never trusts the reply. Every proposal is validated against the
search space; out-of-range and already-tried params are dropped and recorded.
If the agent crashes, times out, exceeds `max_calls`, or answers without JSON,
the round is planned by `strategy.fallback` and the campaign keeps going — an
agent outage slows a campaign down, it does not stop it. The agent may end the
campaign deliberately by replying `{"stop": true, "rationale": "..."}`, which
stops it with `agent_requested_stop`.

The prompt goes to the agent on **stdin**, never as a command-line argument:
it quotes campaign output, and campaign output is untrusted. Every call leaves
a transcript under `<campaign_dir>/agents/<role>/<n>/`.

## Reading the loop back

Every round leaves a record in `<campaign_dir>/rounds.jsonl`: what it proposed,
on what hypothesis, which trials it submitted, and what those trials measured.
The stages mirror a code review — propose, apply, validate, evaluate, review —
because an experiment round has the same shape as a change.

```bash
iax campaign rounds <campaign_id>          # the loop's reasoning, oldest first
iax campaign trials <campaign_id>          # per-trial status, value, run id, error
```

The file is append-only. A round that went wrong is corrected by a later
record, never by rewriting an earlier one.

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

The check list is fixed; only the thresholds above are configurable.

`kill` verdicts are handled inline: cancelled when the run's policy sets
`auto_kill: true`, escalated otherwise. Suspicious runs climb an escalation
ladder — only after `after_suspicious_ticks` consecutive suspicious ticks does
the daemon involve an agent, subject to `cooldown_minutes` and
`max_agent_calls`. With no `agent_command` configured the escalation is just a
file under `<runs>/_escalations/` plus `iax escalations` — an external agent
session picks it up and zero tokens are spent by the daemon.

### Is the daemon alive?

Every tick stamps `<runs>/_daemon/heartbeat.json`, so a working daemon and a
dead one are distinguishable after the fact. A quiet daemon also prints one
line every `--heartbeat` seconds (default 300):

```json
{"timestamp": "...", "heartbeat": true, "runs_checked": 3, "campaigns_advanced": 1, "next_tick_seconds": 30}
```

When something is waiting on a daemon that is not ticking — an active run, a
running campaign, a campaign you just started — `iax runs`, `iax campaign list`
and `iax campaign status` say so on **stderr**, so `--json` stdout stays
parseable:

```console
$ iax campaign status camp_9f21 --json
No daemon tick for 45m (last: 2026-08-29T01:12:04+00:00, pid 4242). Start one with `iax daemon`.
{ ... }
```

A failed MLflow sync is a tick error, not a clean tick: the daemon reports it,
`status.details.mlflow_sync_error` keeps it, and it is retried three times
before iax gives up on that run and sets `mlflow_sync_failed`.

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

### From a chat

The intended path: the user states an outcome, the agent states a goal, the
harness runs the experiments.

```
user   > get val_loss under 0.05 on this model, you have 20 runs
agent  > iax new goal goal.yaml        # then fills in metric, space, budget
         iax campaign validate goal.yaml
         iax loop goal.yaml --json --max-rounds 20
         → exit 0, best val_loss 0.041 at lr 3.1e-4, depth 6 (14 trials)
```

The agent writes the goal and reads the report; it does not drive the rounds.
The procedure it follows — which four things to ask for, how to diagnose a
campaign that missed its target, what may never change mid-campaign — is the
`autonomous-experimentation` skill under `.claude/skills/`, with `AGENTS.md`
carrying the same entry point for agents that do not load skills.
`examples/goal_agentic.yaml` is a worked example, with a workload whose
failures the planner is meant to learn from.

### Programmatic hooks

The daemon and the planner are fully programmatic. Agents plug in at four
opt-in points:

- **Round planning** — `strategy: agent` hands the next batch to an agent,
  with `strategy.fallback` covering every way that can fail.

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

This installs the `autonomous-experimentation`, `running-campaigns`,
`submitting-experiments`, `monitoring-experiments`, `diagnosing-experiments`,
and `cancelling-experiments` skills into the project's `.claude/skills/` (add
`-g` for `~/.claude/skills/`). `autonomous-experimentation` is the one that
turns "get val_loss under 0.05" into a goal file and an `iax loop` run.

Agents that do not load Claude Code skills read `AGENTS.md` at the repo root:
the same entry point, the exit-code contract, and where the procedure lives.

During in-repo development, Workbench uses the uv workspace member at `packages/ai_experiments`.

## Environment variables

The harness **reads** these from its own environment:

| variable | default | what it does |
|---|---|---|
| `IAX_RUNS_DIR` | `<project>/outputs/experiments/runs` | where runs, campaigns and events are stored; `--runs-dir` overrides it |
| `IAX_CLUSTERS` | `./clusters.yaml`, then `~/.config/iax/clusters.yaml` | the cluster profile file `iax cluster` and `cluster:` resolve against |
| `RAY_ADDRESS` | `http://127.0.0.1:8265` | the Ray Jobs address, used when the manifest sets no `backend_address` |
| `IAX_NOTIFY_WEBHOOK` | unset | daemon notifications POST here as JSON; `--notify-webhook` overrides it |
| `IAX_NOTIFY_COMMAND` | unset | daemon notifications run this command with the JSON payload on **stdin**; `--notify-command` overrides it |
| `MLFLOW_TRACKING_URI` | unset (mlflow uses `./mlruns`) | used when `tracking.tracking_uri` is not set |

Without `IAX_RUNS_DIR` or `--runs-dir`, the store is found by walking up from
the current directory: the first ancestor that already holds
`outputs/experiments/runs`, else the first that holds a `.git` or
`pyproject.toml`. So `iax submit` at the repo root and `iax status` three
directories down read the same store. Every not-found error prints the store
path it read.

The harness **injects** these into every workload it starts:

| variable | value | for |
|---|---|---|
| `IAX_RUN_ID` | the run id | naming logs and checkpoints |
| `IAX_RUN_DIR` | the run's directory | anything the workload wants beside its run |
| `IAX_ARTIFACTS_DIR` | `<run_dir>/artifacts` | checkpoints and plots; `iax artifacts <run_id>` lists what lands here |
| `IAX_PARAMS` | the trial's params, as JSON | campaign trials — the whole params dict, whether or not it appears in `args` |
| `IAX_TRIAL_ID` | the trial id | campaign trials |
| `MLFLOW_RUN_ID`, `MLFLOW_TRACKING_URI` | the run the harness created | `tracking.mlflow: true`, so a workload on a remote Ray node logs to the same run |
| `MLFLOW_ALLOW_FILE_STORE` | `true` | set only for a file-backed tracking URI, which MLflow 3 gates; an explicit `false` is respected |

`IAX_METRIC` is not a variable. It is the stdout prefix a workload prints its
observations with:

```python
print('IAX_METRIC {"step": 12, "loss": 0.0734}')
```

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
monitoring:
  interval_seconds: 300
  stuck_after_minutes: 30
metadata:
  project_id: example
```

Manifests and goals are validated strictly: a key iax does not know fails
`iax validate` by name, with the field it most likely meant.

```console
$ iax validate experiment.yaml
Error: invalid manifest experiment.yaml: monitor: unknown field; did you mean 'monitoring'?
```

## CLI

```bash
# start from a template (works from a pip/uv install, no repo checkout)
iax new goal goal.yaml
iax new workload train.py
iax new manifest experiment.yaml
iax new manifest next.yaml --from-run <run_id>   # reuse a run that worked

# single runs
iax validate experiment.yaml
iax submit experiment.yaml --json
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

# one goal in, one answer out (the agent entry point)
iax loop goal.yaml --json --max-rounds 20    # exit 0 = target reached, 4 = not
iax loop goal.yaml --resume <campaign_id>    # continue a bounded loop

# campaigns (goal-driven auto-experiment loop)
iax campaign validate goal.yaml
iax campaign start goal.yaml
iax campaign list / status / advance / suggest / pause / edit / resume / stop
iax campaign trials <campaign_id>          # every trial: status, value, run, error
iax campaign rounds <campaign_id> --json   # why each round tried what it tried

# infrastructure
iax daemon --interval 30        # monitor + loop driver (foreground)
iax serve --port 8585           # dashboard + REST API  (needs [server] extra)
iax cluster list / status / up / down
```

### Exit codes and errors

Every command takes `--json` and every failure follows one contract, so an
agent can branch on the result instead of parsing prose:

| exit | code | meaning |
|---|---|---|
| 0 | — | success |
| 1 | `not_found` | the run, campaign, or bundle does not exist |
| 2 | `invalid_input` | bad manifest, bad goal, params outside the search space |
| 3 | `backend_unavailable` | the execution backend could not be reached — for `iax loop`, no trial could start |
| 4 | — | `iax loop` only: the loop ran, the objective was not reached |

With `--json` the error is one object on **stdout**; without it, one line on
**stderr**. Successful output always goes to stdout.

```console
$ iax status run_nope --json; echo "exit=$?"
{
  "error": "unknown run 'run_nope'; list them with `iax runs`",
  "code": "not_found",
  "details": {"run": "run_nope"}
}
exit=1
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
