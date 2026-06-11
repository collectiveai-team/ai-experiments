# Production experiment loop — implementation plan

Branch: `feat/production-experiment-loop`

Goal: turn `iax` into a production-grade autonomous experimentation system:
goal understanding → planning/orchestration on Ray (local/cloud) → monitoring
daemon (programmatic checks first, agent escalation second) → post-run analysis
→ auto-scheduling of improved experiments → web dashboard.

## Stages

- [ ] 1. Metrics layer: `MetricPoint`, `ai_experiments.report` helper for
      workloads, worker parses `IAX_METRIC` lines + heartbeat thread, store
      read/write `metrics.jsonl`, Ray backend parses metrics from job logs.
- [ ] 2. Monitoring v2: extended programmatic checks (NaN/inf objective,
      metric staleness, heartbeat loss, dead PID reaper, hard timeout,
      plateau), `kill` decision, escalation ladder (consecutive suspicious
      ticks, cooldown, max agent calls, optional agent command hook).
- [ ] 3. Goal layer + planner: `GoalSpec` (objective, budget, search space,
      workload template), search-space sampling (choice/int/uniform/
      loguniform), strategies: grid, random, adaptive (explore→exploit around
      best), param substitution into trial manifests.
- [ ] 4. Campaign store + orchestrator: `campaigns/<id>/` state, trial
      records, `CampaignOrchestrator.advance()` — collect results, analyze,
      check stopping (target/budget), plan + submit next batch.
- [ ] 5. Daemon + CLI: `iax daemon` (tick loop: check runs → act/escalate/
      kill; advance campaigns), `iax campaign start|status|list|stop`,
      `iax runs` listing.
- [ ] 6. Cluster profiles: `clusters.yaml` (local/aws/gcp/azure via Ray
      cluster launcher), `iax cluster list|status|up|down`, goal manifests
      reference clusters by name.
- [ ] 7. Web dashboard: `iax serve` — FastAPI REST API + self-contained
      HTML/JS dashboard (campaigns, runs, live metrics charts, decisions).
- [ ] 8. Docs + examples + skills: example goal/toy workload, README,
      update `.claude/skills/*` for the new flow.
- [ ] 9. Verification: full pytest, ruff, end-to-end campaign smoke test on
      local backend (toy objective converges, daemon kills a stuck run).

## Review

(filled in at the end)
