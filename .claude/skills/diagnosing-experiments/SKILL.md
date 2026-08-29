---
name: diagnosing-experiments
description: Use when an iax run is stuck or failed — when a monitor decision is `delegate_diagnosis` or `training_failed`, or when asked why a training run failed or stalled. Does harness-specific triage, then runs a disciplined root-cause loop.
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
| `run_completed` | Run reached `completed`. |
| `failed` | Run reached `failed`. |
| `cancelled` | Run reached `cancelled`. |
| `run_active` | Run is active and no concern is currently detected. |
| `status_error_present` | `status.json` carries an `error`. A missing status file sets this same `error`, so both cases surface here. |
| `ray_resource_starved` | Ray reports the job can't get resources. |
| `ray_stuck_suspected` | Ray heuristics suspect a hang. |
| `no_status_update_for_<N>m` | `status.updated_at` older than `stuck_after_minutes`. |
| `no_event_progress_for_<N>m` | Last event older than `stuck_after_minutes`. |
| `no_run_events` | Submitted/running but `events.jsonl` is empty. |

## 2. Pull the raw evidence

```bash
iax logs <run_id> --tail 200          # parsed events: "[<ts>] <level>: <msg>"
iax logs <run_id> --tail 200 --json   # same events as JSON, for programmatic scanning
iax status <run_id> --json            # exit_code, error, pid, details
```

Then read the files directly under `<run_dir>/`:
`status.json`, `events.jsonl`, and for local runs `worker.log` (raw stdout/stderr,
where local stack traces live). For Ray runs, do not expect `worker.log`; read
`status.error`, `status.details.ray_log_tail`, and `status.details.ray_condition`.
`exit_code` is local-backend-only. On Ray, use `status`, `error`, and `details.ray_*`
fields instead. Use the same `--runs-dir`/`IAX_RUNS_DIR` as the run.

## 3. Hand off to the root-cause loop

You now have symptoms (reasons), logs, and exit state. **Do NOT guess a fix from
here.** If your harness ships a systematic-debugging skill, invoke it and feed it
this evidence. Otherwise run the same loop yourself:

1. **Reproduce.** `iax repro <run_id>` writes the exact command, environment and
   working tree the run used. Reproduce the failure outside the harness first —
   a bug you cannot reproduce is a bug you cannot confirm you fixed.
2. **Isolate.** Shrink the failing case: fewer steps, smaller batch, one GPU,
   the single param the last passing run did not have. `iax runs` and
   `iax compare` give you the last passing neighbour to diff against.
3. **Explain.** State the mechanism, not the correlation: which line, given
   which value, produced which observed symptom.
4. **Confirm.** Change one thing, rerun the reproduction, and check the symptom
   is gone. If it is not, your explanation was wrong — go back to step 2.

## 4. Conclude

Once the root cause is confirmed:
- **Config/manifest fix** → re-author via the **submitting-experiments** skill and
  submit a fresh run.
- **Run should be stopped** → use the **cancelling-experiments** skill.
- Record the cause and the fix in your summary so the next wake-up has context.
