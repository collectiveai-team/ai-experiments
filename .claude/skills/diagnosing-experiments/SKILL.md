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
