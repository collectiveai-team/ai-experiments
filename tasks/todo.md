# Production experiment loop — implementation plan

Branch: `feat/production-experiment-loop`

Goal: turn `iax` into a production-grade autonomous experimentation system:
goal understanding → planning/orchestration on Ray (local/cloud) → monitoring
daemon (programmatic checks first, agent escalation second) → post-run analysis
→ auto-scheduling of improved experiments → web dashboard.

## Stages

- [x] 1. Metrics layer: `MetricPoint`, `ai_experiments.report` helper for
      workloads, worker parses `IAX_METRIC` lines + heartbeat thread, store
      read/write `metrics.jsonl`, Ray backend parses metrics from job logs.
- [x] 2. Monitoring v2: extended programmatic checks (NaN/inf objective,
      metric staleness, heartbeat loss, dead PID reaper, hard timeout,
      plateau), `kill` decision, escalation ladder (consecutive suspicious
      ticks, cooldown, max agent calls, optional agent command hook).
- [x] 3. Goal layer + planner: `GoalSpec` (objective, budget, search space,
      workload template), search-space sampling (choice/int/uniform/
      loguniform), strategies: grid, random, adaptive (explore→exploit around
      best), param substitution into trial manifests.
- [x] 4. Campaign store + orchestrator: `_campaigns/<id>/` state, trial
      records, `CampaignOrchestrator.advance()` — collect results, analyze,
      check stopping (target/budget), plan + submit next batch.
- [x] 5. Daemon + CLI: `iax daemon` (tick loop: check runs → act/escalate/
      kill; advance campaigns), `iax campaign start|status|list|advance|
      suggest|stop`, `iax runs|metrics|escalations`.
- [x] 6. Cluster profiles: `clusters.yaml` (local/aws/gcp/azure via Ray
      cluster launcher), `iax cluster list|status|up|down`, goal manifests
      reference clusters by name (`cluster:`).
- [x] 7. Web dashboard: `iax serve` — FastAPI REST API + self-contained
      HTML/JS dashboard (campaigns, runs, live metrics charts, escalations,
      cancel/stop actions).
- [x] 8. Docs + examples + skills: `examples/{goal_toy.yaml,toy_train.py,
      clusters.yaml}`, README rewrite, new `running-campaigns` skill,
      `monitoring-experiments` updated for `kill` + metrics + escalations.
- [x] 9. Verification: 70 tests + ruff clean. Live smoke: example campaign
      reached its target (loss 1.6e-06 < 0.01) after 2 trials driven by the
      real daemon; dashboard API served it; a frozen workload with
      `timeout_seconds: 5, auto_kill: true` was auto-killed by one daemon
      tick (process verified gone). Found+fixed: `list_runs` leaked
      `_campaigns`/`_escalations` as phantom runs (regression test added).

## Round 2 (user feedback)

- [x] `iax run goal.yaml` — single command: campaign + dashboard + loop.
- [x] Ray in the web UI: clusters panel with reachability, deep links to the
      Ray dashboard job page, ray condition + log tail in run details.

## Round 3 (user request: production gaps)

- [x] Artifacts: `$IAX_ARTIFACTS_DIR` convention, `iax artifacts`, API list +
      download with traversal protection, links in the run modal.
- [x] Reproducibility: `repro/` bundle (git SHA/branch/dirty + diff.patch,
      python/platform, environment.txt) captured on every submit;
      `iax repro`, `iax rerun` (exact resubmit with git-drift warnings).
- [x] Cross-campaign comparison: `/api/leaderboard` + `iax leaderboard`,
      dashboard leaderboard panel, multi-run metric overlay (checkbox
      select → compare) via a multi-series canvas chart.
- [x] Notifications: `Notifier` (webhook Slack-compatible, command sink,
      always-on `_notifications.jsonl`), daemon alerts on kills/escalations/
      campaign finish, `--notify-webhook/--notify-command` + env vars.
- [x] Mid-flight editing: campaign `pause`/`resume`/`edit` (CLI + API + UI
      buttons); objective metric locked, history preserved for replanning.
- [x] Cost accounting: `gpu_hours` per trial, `budget.max_gpu_hours` stop
      condition (`gpu_hours_exhausted`), `gpu_hour_rate` → estimated cost in
      status/leaderboard/dashboard.

## Round 4 (MLflow integration)

- [x] `TrackingSpec` on manifests + goals (`tracking: {mlflow, tracking_uri,
      experiment}`), passed through to trials; `[mlflow]` extra
      (mlflow-skinny).
- [x] `tracking.py`: harness creates the MLflow run at submit (params, iax/
      git tags), backends inject MLFLOW_RUN_ID + MLFLOW_TRACKING_URI into
      the workload env — incl. Ray runtime_env, which routes remote-cluster
      artifacts to the central MLflow store.
- [x] Daemon finalizes terminal tracked runs: metric history with steps,
      local artifacts upload, FINISHED/FAILED/KILLED status; idempotent via
      `mlflow_synced` detail. All best-effort (missing mlflow / dead server
      → warning event, never blocks).
- [x] Unit tests against a fake mlflow module + live smoke against real
      mlflow-skinny on a file store.

## Review

**What was built** — the harness now covers the four production requirements:

1. *Goal understanding*: `GoalSpec` YAML (objective metric/mode/target,
   typed search space, budget, strategy, backend, monitoring policy).
2. *Plan + orchestrate on Ray*: planner strategies (grid/random/adaptive)
   generate trial manifests; orchestrator submits to local or any Ray
   cluster; named cluster profiles for aws/gcp/azure delegate provisioning
   to Ray's own launcher (`iax cluster up/down`).
3. *Monitor daemon*: `iax daemon` runs free programmatic checks every tick
   (NaN/inf, timeout, dead worker → kill; heartbeat/metric staleness,
   plateau → suspicious) and only involves an agent after the escalation
   ladder (consecutive ticks, cooldown, per-run call budget). Zero-token
   default: escalation files + `iax escalations`.
4. *Auto-experiment loop*: on trial completion the orchestrator extracts the
   objective, updates the best, checks stopping conditions, and plans +
   submits the next batch; opt-in agent review can inject trials via
   `iax campaign suggest`.

Interface: web dashboard (`iax serve`, FastAPI + dependency-free dark UI
with canvas metric charts) — chosen over TUI/VS Code extension because it
works from any machine that sees the run store and is the natural base for
a future VS Code webview.

**Key design decisions**
- Metrics travel as `IAX_METRIC` stdout lines: one contract that works on
  local subprocesses and remote Ray clusters without a shared filesystem.
- Everything stays on the filesystem (runs + `_campaigns` + `_escalations`
  under one root): greppable, agent-friendly, no DB dependency.
- Strategies are deterministic given (seed, trial history) so a crashed
  loop replans identically.
- `fastapi`/`uvicorn` are a `[server]` extra; the core CLI stays light.
