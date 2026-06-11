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
| `training_failed` | Status `failed`/`cancelled` | Invoke **diagnosing-experiments** — unless you just cancelled this run on purpose (an intentional `cancel` also reports `training_failed`), in which case no diagnosis is needed. |
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
| `worker.log` | Local backend only: raw stdout/stderr from the workload. Ray run dirs have only `manifest.yaml`, `status.json`, and `events.jsonl`. |

`runs_dir` resolves from `--runs-dir`, else `$IAX_RUNS_DIR`, else
`outputs/experiments/runs`.

On Ray, `iax logs` returns harness events only. Worker stdout and tracebacks surface
through `status.error` and `status.details.ray_log_tail`; Ray condition hints live in
`status.details.ray_condition`. `exit_code` is local-backend-only and remains `None`
for Ray runs.
