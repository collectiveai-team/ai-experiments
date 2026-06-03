# Design: Agent Skills for the `iax` Experiment Harness

**Date:** 2026-06-03
**Status:** Approved (pending spec review)

## Problem

`ai-experiments` ships the *code* harness for running detached training experiments:
the `iax` CLI (`validate`, `submit`, `status`, `logs`, `diagnose`, `monitor`,
`cancel`), a manifest contract, `local`/`ray` backends, a filesystem run store, and
monitor decisions that tell a scheduler when to wake an agent.

What is missing is the *agent-facing* layer: skills that teach a Claude Code agent how
to **drive** that harness through its lifecycle. Today an agent must rediscover the
command surface, manifest fields, run-store layout, and the meaning of each
`MonitorDecision` every time. This repo should carry that knowledge as version-controlled
skills that travel with the harness.

## Decisions

| Question | Decision |
|---|---|
| Audience / packaging | Repo-local `.claude/skills/` (version-controlled with the harness; no plugin, no package bundling). |
| Scope | Full lifecycle: author+submit, monitor+react, diagnose, cancel+cleanup. |
| Structure | Four focused skills, one per lifecycle stage. |
| Diagnose skill | Harness-specific; delegates the root-cause loop to `superpowers:systematic-debugging` rather than duplicating it. |

## Skills

All four live under `.claude/skills/<name>/SKILL.md`, gerund-named, each with frontmatter
(`name`, `description` containing explicit trigger phrases). Skills link to the
authoritative in-repo source (`ai_experiments/schemas.py`, `ai_experiments/monitoring/rules.py`)
instead of duplicating field lists, so they cannot drift from the code.

| Skill | Stage | Trigger phrases |
|---|---|---|
| `submitting-experiments` | author manifest → validate → submit | "run/launch an experiment", "set up a training run", needs a manifest |
| `monitoring-experiments` | interpret `iax monitor`/`status` + `MonitorDecision` → next action | scheduler wake-up fires, "check on a run", JSON appears from monitor |
| `diagnosing-experiments` | stuck/failed run → harness triage → hand off | decision is `delegate_diagnosis`/`training_failed`, "why did the run fail" |
| `cancelling-experiments` | cancel a run + run-store housekeeping | "stop/cancel the run", "clean up old runs" |

### `submitting-experiments`

The only skill needing the full manifest contract. Walks the agent through:

1. Gather workload: `entrypoint`, `args`, `working_dir`, `env`.
2. Pick backend: `local` (default) vs `ray`.
3. Set `resources` (`cpus`/`gpus`/`memory_gb`) and `monitoring` policy
   (`interval_seconds`, `stuck_after_minutes`, `no_event_after_minutes`,
   `timeout_seconds`, `checks`).
4. `iax validate <manifest>` → fix errors → `iax submit <manifest> --json`.
5. Capture the returned `run_id`, `run_dir`, `status_uri` from the `RunHandle`.

Contents:
- A worked YAML example (mirrors `README.md` / `manual_test/*.yaml`).
- A "common validation errors" table (e.g. empty `experiment`, missing `workload`).
- `reference/manifest.md` — field-by-field detail (progressive disclosure; keeps
  `SKILL.md` short).

### `monitoring-experiments`

Decision table mapping each `MonitorDecision.decision` to the agent's next move:

| Decision | Agent action |
|---|---|
| `continue_waiting` | Do nothing / stay quiet; keep the scheduler monitor active. |
| `training_complete` | Summarize results; tear down the monitor job. |
| `training_failed` | Invoke `diagnosing-experiments`. |
| `delegate_diagnosis` | Invoke `diagnosing-experiments`. |
| `unknown` | Inspect `status.json`; status file likely missing. |

Also documents:
- The `iax monitor --quiet-when-waiting` contract: **empty output = do nothing**;
  JSON output = act.
- Run-store layout: `run_dir/{manifest.yaml, status.json, events.jsonl, worker.log}`
  and `IAX_RUNS_DIR` / `--runs-dir` resolution.

### `diagnosing-experiments`

Harness-specific triage, then hand off:

1. Read `iax diagnose <run_id> --json` → inspect `reasons` and `recommendations`.
2. Pull `iax logs <run_id> --tail N`; inspect `status.json` and `worker.log`.
3. Recognize encoded reasons from `monitoring/rules.py`: `ray_resource_starved`,
   `ray_stuck_suspected`, `no_status_update_for_Nm`, `no_event_progress_for_Nm`,
   `no_run_events`, `status_error_present`.
4. **Hand off to `superpowers:systematic-debugging`** for the actual root-cause loop.
5. Conclude: recommend a fix, then re-submit via `submitting-experiments` or
   `cancelling-experiments` as appropriate.

### `cancelling-experiments`

- `iax cancel <run_id>`; verify the resulting `RunState`.
- Run-store housekeeping: list runs, identify terminal-state `run_dir`s
  (`completed`/`failed`/`cancelled`) that are safe to delete.
- **Guard:** never delete `running`/`submitted` runs.
- Notes `IAX_RUNS_DIR` / `--runs-dir` resolution.

## Verification

After writing, do a manual end-to-end dry run using the existing fixtures in
`manual_test/` (`experiment_ok_local.yaml`, `experiment_stuck.yaml`) against a temp
`--runs-dir`: `iax validate` / `submit` / `monitor` / `diagnose` — to confirm every
command and JSON shape each skill cites is real and current. Skills are documentation;
no new entries are added to the pytest suite.

## Out of Scope (YAGNI)

- Plugin packaging or bundling skills inside the Python package.
- Any change to the Python harness, schemas, or CLI surface.
- New `iax` subcommands.
- MLflow / W&B / experiment-tracking integration.
