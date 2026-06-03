# iax Agent Skills Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add four repo-local Claude Code skills that teach an agent to drive the `iax` experiment harness through its full lifecycle (submit, monitor, diagnose, cancel).

**Architecture:** Each skill is a self-contained `.claude/skills/<name>/SKILL.md` with `name`/`description` frontmatter. Skills cite exact `iax` commands and link to authoritative in-repo source (`ai_experiments/schemas.py`, `ai_experiments/monitoring/rules.py`) instead of duplicating field lists, so they can't drift. These are documentation artifacts — "tests" means running the real `iax` commands each skill cites against a temp `--runs-dir` and confirming the cited output/JSON shape is accurate.

**Tech Stack:** Markdown skill files; verification via `uv run iax ...` against fixtures in `manual_test/`.

**Conventions (verified against the live harness on 2026-06-03):**
- `iax submit <manifest> --json` → `RunHandle` JSON: `run_id`, `backend`, `status`, `status_uri`, `run_dir`, `submitted_at`.
- `iax monitor <run_id> --json --quiet-when-waiting` → prints **nothing** when decision is `continue_waiting`; otherwise prints a `DiagnosisReport`.
- Run dir layout: `<run_dir>/{manifest.yaml, status.json, events.jsonl, worker.log}`.
- `iax logs` line format: `[<ISO-8601 timestamp>] <level>: <message>`.
- `diagnose_run` emits these decisions: `training_complete`, `training_failed`, `delegate_diagnosis`, `continue_waiting`. It never emits `unknown` (present in the enum but unused; a missing status file becomes `delegate_diagnosis` / `status_error_present`).
- Encoded diagnosis reasons (from `monitoring/rules.py`): `status_error_present`, `ray_resource_starved`, `ray_stuck_suspected`, `no_status_update_for_<N>m`, `no_event_progress_for_<N>m`, `no_run_events`, `run_completed`, `run_active`.
- Run store root resolves from `--runs-dir`, else `$IAX_RUNS_DIR`, else `outputs/experiments/runs`.
- This shell aliases `ls` to `colorls`; verification commands must parse JSON (not `ls`) to extract `run_id`.

---

## File Structure

- Create: `.claude/skills/submitting-experiments/SKILL.md`
- Create: `.claude/skills/submitting-experiments/reference/manifest.md`
- Create: `.claude/skills/monitoring-experiments/SKILL.md`
- Create: `.claude/skills/diagnosing-experiments/SKILL.md`
- Create: `.claude/skills/cancelling-experiments/SKILL.md`

Each skill owns one lifecycle stage. Only `submitting-experiments` needs the full manifest contract, so it gets the `reference/manifest.md` progressive-disclosure file; the other three keep everything inline.

---

## Task 1: `submitting-experiments` skill

**Files:**
- Create: `.claude/skills/submitting-experiments/SKILL.md`
- Create: `.claude/skills/submitting-experiments/reference/manifest.md`

- [ ] **Step 1: Create `reference/manifest.md`**

Write `.claude/skills/submitting-experiments/reference/manifest.md` with this exact content:

````markdown
# Experiment Manifest Reference

Authoritative source: `ai_experiments/schemas.py` (`ExperimentManifest`). This file
summarizes the fields; if it disagrees with the code, the code wins.

## Top-level fields

| Field | Type | Default | Notes |
|---|---|---|---|
| `experiment` | string | — (required) | Must be non-empty. |
| `backend` | `local` \| `ray` | `local` | `ray` needs the `ai-experiments[ray]` extra. |
| `workload` | object | — (required) | See below. |
| `resources` | object | all defaults | See below. |
| `artifacts` | object | all defaults | `output_dir` (default `outputs/training`), `status_path`. |
| `monitoring` | object | all defaults | See below. |
| `metadata` | mapping | `{}` | Free-form (e.g. `project_id`). |

## `workload`

| Field | Type | Default | Notes |
|---|---|---|---|
| `entrypoint` | string | — (required) | e.g. `python`, `python3`. |
| `args` | list[string] | `[]` | e.g. `["-m", "pkg.cli", "train", "cfg.yaml"]`. |
| `working_dir` | string | `.` | Resolved by the backend. |
| `env` | mapping | `{}` | Extra environment variables. |

## `resources`

| Field | Type | Default |
|---|---|---|
| `cpus` | float | `1` |
| `gpus` | float | `0` |
| `memory_gb` | float \| null | `null` |

## `monitoring` (`MonitorPolicy`)

| Field | Type | Default |
|---|---|---|
| `interval_seconds` | int | `300` |
| `stuck_after_minutes` | int | `30` |
| `no_event_after_minutes` | int \| null | `null` |
| `timeout_seconds` | int \| null | `null` |
| `checks` | list[string] | `["no_status_update", "no_log_progress", "process_exit"]` |
````

- [ ] **Step 2: Create `SKILL.md`**

Write `.claude/skills/submitting-experiments/SKILL.md` with this exact content:

````markdown
---
name: submitting-experiments
description: Use when launching or setting up a detached training experiment with the iax harness — authoring an experiment manifest, validating it, and submitting a run. Triggers on "run an experiment", "launch a training run", "submit a job to iax", or when you need to produce a run_id for a training workload.
---

# Submitting Experiments

Drive the `iax` harness to launch a detached training run. Authoritative manifest
schema: `ai_experiments/schemas.py`. Full field reference: `reference/manifest.md`.

## Workflow

1. **Gather the workload.** What command runs the training? Map it to `workload`:
   `entrypoint` + `args`, the `working_dir` it must run from, and any `env`.
2. **Pick the backend.** `local` (default; runs as a detached subprocess) or `ray`
   (needs the `ai-experiments[ray]` extra and a reachable cluster).
3. **Set resources and monitoring.** `resources.cpus`/`gpus`/`memory_gb`; a
   `monitoring` policy with `interval_seconds` (how often the scheduler checks) and
   `stuck_after_minutes` (how long without progress before flagging). See
   `reference/manifest.md` for every field and its default.
4. **Validate, then submit:**

   ```bash
   iax validate experiment.yaml
   iax submit experiment.yaml --json
   ```

5. **Capture the run handle.** `submit --json` prints a `RunHandle`. Record
   `run_id` (everything downstream keys off it), `run_dir`, and `status_uri`. If you
   pass `--runs-dir <dir>`, pass the same `--runs-dir` to every later command, or set
   `IAX_RUNS_DIR` so the scheduler and agent agree on the store root.

## Worked example

```yaml
experiment: demand_forecast_baseline
backend: local
workload:
  entrypoint: python3
  args:
    - -m
    - ts_agents_lab.cli
    - train
    - configs/training.yaml
  working_dir: .
resources:
  cpus: 4
  gpus: 1
monitoring:
  interval_seconds: 300
  stuck_after_minutes: 30
metadata:
  project_id: example
```

## Common validation errors

| Symptom | Cause | Fix |
|---|---|---|
| `experiment must not be empty` | `experiment` blank/missing | Set a non-empty name. |
| `field required` on `workload` | `workload` block absent | Add `workload.entrypoint`. |
| `Input should be 'local' or 'ray'` | bad `backend` value | Use `local` or `ray`. |
| YAML parse error | indentation / tabs | Re-indent with spaces. |

## After submitting

Hand off the `run_id` to a scheduler running
`iax monitor <run_id> --json --quiet-when-waiting`, then use the
**monitoring-experiments** skill to interpret its output.
````

- [ ] **Step 3: Verify cited commands are accurate**

Run:
```bash
cd /Users/lionelchamorro/Projects/collectiveai/workbench/ai-experiments
uv run iax validate manual_test/experiment_ok_local.yaml
TMP=$(mktemp -d)
uv run iax submit manual_test/experiment_ok_local.yaml --runs-dir "$TMP" --json
rm -rf "$TMP"
```
Expected: `validate` prints `Manifest valid: ...`; `submit --json` prints a JSON object containing `run_id`, `run_dir`, and `status_uri`. Confirm the SKILL.md's described fields match.

- [ ] **Step 4: Commit**

```bash
git add .claude/skills/submitting-experiments/
git commit -m "Add submitting-experiments skill"
```

---

## Task 2: `monitoring-experiments` skill

**Files:**
- Create: `.claude/skills/monitoring-experiments/SKILL.md`

- [ ] **Step 1: Create `SKILL.md`**

Write `.claude/skills/monitoring-experiments/SKILL.md` with this exact content:

````markdown
---
name: monitoring-experiments
description: Use when a scheduler wake-up fires for an experiment, when JSON appears from `iax monitor`, or when asked to check on a running training run. Interprets `iax monitor`/`status` output and the MonitorDecision, then chooses the next action.
---

# Monitoring Experiments

A scheduler runs `iax monitor <run_id> --json --quiet-when-waiting` on the
`monitoring.interval_seconds` cadence. Your job: read its output and act.

## The quiet contract

`--quiet-when-waiting` prints **nothing** while the run is healthy (decision
`continue_waiting`). **Empty output → do nothing; let the scheduler keep waiting.**
Non-empty output is a `DiagnosisReport` JSON — act on its `decision`.

To inspect a run yourself at any time:

```bash
iax status <run_id> --json        # current RunStatus
iax monitor <run_id> --json       # DiagnosisReport (no quiet suppression)
```

Pass the same `--runs-dir` (or `IAX_RUNS_DIR`) the run was submitted with.

## Decision table

`decision` comes from `MonitorDecision` (`ai_experiments/schemas.py`); the logic lives
in `ai_experiments/monitoring/rules.py`.

| `decision` | Meaning | Your next action |
|---|---|---|
| `continue_waiting` | Run active, no concern | Nothing. Stay quiet; keep the monitor job alive. |
| `training_complete` | Status `completed` | Summarize results from `<run_dir>`, then tear down the monitor job. |
| `training_failed` | Status `failed`/`cancelled` | Invoke **diagnosing-experiments**. |
| `delegate_diagnosis` | Stuck / no progress / status error | Invoke **diagnosing-experiments**. |

> `MonitorDecision` also defines `unknown`, but the current `diagnose_run` never emits
> it — a missing status file surfaces as `delegate_diagnosis` with reason
> `status_error_present`. Treat any unexpected decision as `delegate_diagnosis`.

## Run-store layout

Each run lives under `<runs_dir>/<run_id>/`:

| File | Contents |
|---|---|
| `manifest.yaml` | The submitted manifest. |
| `status.json` | Current `RunStatus` (status, exit_code, error, details, timestamps). |
| `events.jsonl` | One JSON `RunEvent` per line; what `iax logs` reads. |
| `worker.log` | Raw stdout/stderr from the workload. |

`runs_dir` resolves from `--runs-dir`, else `$IAX_RUNS_DIR`, else
`outputs/experiments/runs`.
````

- [ ] **Step 2: Verify the quiet contract and a non-quiet decision**

Run:
```bash
cd /Users/lionelchamorro/Projects/collectiveai/workbench/ai-experiments
TMP=$(mktemp -d)
RID=$(uv run iax submit manual_test/experiment_ok_local.yaml --runs-dir "$TMP" --json | python3 -c "import sys,json;print(json.load(sys.stdin)['run_id'])")
echo "--- quiet on healthy run (expect EMPTY) ---"
uv run iax monitor "$RID" --runs-dir "$TMP" --json --quiet-when-waiting; echo "[exit=$?]"
echo "--- non-quiet diagnose (expect a decision) ---"
uv run iax monitor "$RID" --runs-dir "$TMP" --json | python3 -c "import sys,json;print('decision=',json.load(sys.stdin)['decision']['decision'])"
rm -rf "$TMP"
```
Expected: the quiet command prints nothing (just `[exit=0]`); the non-quiet command prints a `decision=...` value. Confirms the quiet contract and the report shape the SKILL.md describes.

- [ ] **Step 3: Commit**

```bash
git add .claude/skills/monitoring-experiments/
git commit -m "Add monitoring-experiments skill"
```

---

## Task 3: `diagnosing-experiments` skill

**Files:**
- Create: `.claude/skills/diagnosing-experiments/SKILL.md`

- [ ] **Step 1: Create `SKILL.md`**

Write `.claude/skills/diagnosing-experiments/SKILL.md` with this exact content:

````markdown
---
name: diagnosing-experiments
description: Use when an iax run is stuck or failed — when a monitor decision is `delegate_diagnosis` or `training_failed`, or when asked why a training run failed or stalled. Does harness-specific triage, then hands off the root-cause loop to systematic-debugging.
---

# Diagnosing Experiments

You were delegated a stuck or failed run. Gather the harness evidence, then run a
disciplined root-cause loop.

## 1. Read the diagnosis report

```bash
iax diagnose <run_id> --json
```

This `DiagnosisReport` gives you `decision`, `decision.reasons`, the embedded
`status`, recent `events`, and `recommendations`. The `reasons` are the fastest
pointer to what tripped. Recognized reasons (`ai_experiments/monitoring/rules.py`):

| Reason | Meaning |
|---|---|
| `status_error_present` | `status.json` carries an `error` (or is missing). |
| `ray_resource_starved` | Ray reports the job can't get resources. |
| `ray_stuck_suspected` | Ray heuristics suspect a hang. |
| `no_status_update_for_<N>m` | `status.updated_at` older than `stuck_after_minutes`. |
| `no_event_progress_for_<N>m` | Last event older than `stuck_after_minutes`. |
| `no_run_events` | Submitted/running but `events.jsonl` is empty. |

## 2. Pull the raw evidence

```bash
iax logs <run_id> --tail 200          # parsed events: "[<ts>] <level>: <msg>"
iax status <run_id> --json            # exit_code, error, pid, details
```

Then read the files directly under `<run_dir>/`:
`worker.log` (raw stdout/stderr — the real stack traces live here),
`status.json`, `events.jsonl`. Use the same `--runs-dir`/`IAX_RUNS_DIR` as the run.

## 3. Hand off to the root-cause loop

You now have symptoms (reasons), logs, and exit state. **Do NOT guess a fix from
here.** Invoke the **superpowers:systematic-debugging** skill and feed it this
evidence to reproduce, isolate, and confirm the root cause.

## 4. Conclude

Once the root cause is confirmed:
- **Config/manifest fix** → re-author via the **submitting-experiments** skill and
  submit a fresh run.
- **Run should be stopped** → use the **cancelling-experiments** skill.
- Record the cause and the fix in your summary so the next wake-up has context.
````

- [ ] **Step 2: Verify the diagnose surface and reasons**

Run:
```bash
cd /Users/lionelchamorro/Projects/collectiveai/workbench/ai-experiments
TMP=$(mktemp -d)
uv run iax diagnose missing_run_xyz --runs-dir "$TMP" --json | python3 -c "import sys,json;d=json.load(sys.stdin);print('decision=',d['decision']['decision'],'reasons=',d['decision']['reasons'])"
rm -rf "$TMP"
grep -n 'no_status_update_for\|ray_resource_starved\|no_run_events\|status_error_present' ai_experiments/monitoring/rules.py
```
Expected: the diagnose of a non-existent run prints `decision= delegate_diagnosis reasons= ['status_error_present']`; the grep confirms every reason string the SKILL.md cites exists verbatim in `rules.py`.

- [ ] **Step 3: Commit**

```bash
git add .claude/skills/diagnosing-experiments/
git commit -m "Add diagnosing-experiments skill"
```

---

## Task 4: `cancelling-experiments` skill

**Files:**
- Create: `.claude/skills/cancelling-experiments/SKILL.md`

- [ ] **Step 1: Create `SKILL.md`**

Write `.claude/skills/cancelling-experiments/SKILL.md` with this exact content:

````markdown
---
name: cancelling-experiments
description: Use when asked to stop or cancel a running iax experiment, or to clean up old experiment runs from the run store. Covers `iax cancel` and safe run-store housekeeping.
---

# Cancelling Experiments

## Cancel a run

```bash
iax cancel <run_id>
iax status <run_id> --json     # verify the result
```

Pass the same `--runs-dir` (or `IAX_RUNS_DIR`) the run was submitted with, or cancel
hits the wrong store. After cancelling, confirm `status` is `cancelled` (or already
terminal — `completed`/`failed`). If a scheduler is monitoring this run, stop that
monitor job too.

## Run-store housekeeping

Runs live as directories under the store root (`--runs-dir`, else `$IAX_RUNS_DIR`,
else `outputs/experiments/runs`). To clean up, inspect each run's status first:

```bash
for d in <runs_dir>/run_*; do
  rid=$(basename "$d")
  st=$(iax status "$rid" --runs-dir <runs_dir> --json | python3 -c "import sys,json;print(json.load(sys.stdin)['status'])")
  echo "$rid -> $st"
done
```

**Safe to delete:** only `run_dir`s whose status is terminal — `completed`,
`failed`, or `cancelled`.

**Never delete:** runs that are `running` or `submitted` (you'd orphan a live
workload and lose its state). Cancel them first, confirm `cancelled`, then delete.

Deleting a `run_dir` removes its `manifest.yaml`, `status.json`, `events.jsonl`, and
`worker.log` — i.e. all record of the run. Archive anything you need first.
````

- [ ] **Step 2: Verify cancel + status round-trip**

Run:
```bash
cd /Users/lionelchamorro/Projects/collectiveai/workbench/ai-experiments
TMP=$(mktemp -d)
RID=$(uv run iax submit manual_test/experiment_ok_local.yaml --runs-dir "$TMP" --json | python3 -c "import sys,json;print(json.load(sys.stdin)['run_id'])")
uv run iax cancel "$RID" --runs-dir "$TMP"
uv run iax status "$RID" --runs-dir "$TMP" --json | python3 -c "import sys,json;print('status=',json.load(sys.stdin)['status'])"
rm -rf "$TMP"
```
Expected: `cancel` prints `Cancelled <run_id>`; `status` then prints a terminal state. Confirms the commands the SKILL.md cites work.

- [ ] **Step 3: Commit**

```bash
git add .claude/skills/cancelling-experiments/
git commit -m "Add cancelling-experiments skill"
```

---

## Task 5: Final cross-skill verification

**Files:** none (verification only).

- [ ] **Step 1: Confirm all four skills are discoverable and well-formed**

Run:
```bash
cd /Users/lionelchamorro/Projects/collectiveai/workbench/ai-experiments
for f in .claude/skills/*/SKILL.md; do
  echo "=== $f ==="
  head -4 "$f"
done
```
Expected: four skills listed, each with a frontmatter block containing `name:` and `description:` (description starting with "Use when").

- [ ] **Step 2: Confirm no skill cites a command or path that doesn't exist**

Run:
```bash
cd /Users/lionelchamorro/Projects/collectiveai/workbench/ai-experiments
grep -rhoE 'iax [a-z]+' .claude/skills | sort -u
uv run iax --help | grep -E '^\s+(validate|submit|status|logs|diagnose|monitor|cancel)'
```
Expected: every `iax <subcommand>` referenced in the skills appears in `iax --help`. No skill references a non-existent subcommand.

- [ ] **Step 3: Final commit (if any fixes were needed)**

```bash
git add -A
git commit -m "Verify iax agent skills end-to-end" || echo "nothing to commit"
```
