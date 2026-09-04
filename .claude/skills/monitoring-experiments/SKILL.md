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
| `kill` | Fatal & unrecoverable: NaN/inf metric, `timeout_exceeded`, or `process_dead` | If `iax daemon` is running it already acts (auto-kill or escalate). Otherwise `iax cancel <run_id>` — except `process_dead`, where the worker is already gone and the daemon reaps it as `failed`. |

> `MonitorDecision` also defines `unknown`, but the current `diagnose_run` never emits
> it — a missing status file surfaces as `delegate_diagnosis` with reason
> `status_error_present`. Treat any unexpected decision as `delegate_diagnosis`.

## Run-store layout

Each run lives under `<runs_dir>/<run_id>/`:

| File | Contents |
|---|---|
| `manifest.yaml` | What was executed: the submitted manifest with `workload.working_dir` resolved to an absolute path, so it means the same thing from any directory. |
| `manifest.source.yaml` | What was submitted, verbatim — the portable form, present only when the author wrote a relative `working_dir`. `iax rerun --portable` resubmits this one. |
| `status.json` | Current `RunStatus` (status, exit_code, error, details, timestamps). |
| `status.lock` | Harness bookkeeping for serialized status updates; ignore it when inspecting or summarizing a run. |
| `events.jsonl` | One JSON `RunEvent` per line; what `iax logs` reads. |
| `metrics.jsonl` | `MetricPoint`s parsed from the workload's `IAX_METRIC` stdout lines; what `iax metrics` reads. Present on both backends once the workload reports. |
| `escalation.json` | Escalation-ladder state (suspicious tick count, agent-call budget) written by the daemon. |
| `cancel.requested` | Local backend only: created by `iax cancel` before it signals. Its presence is what lets the supervisor report a stop it was asked for as `cancelled`, and one it was not (a kernel OOM kill) as `failed`. |
| `worker.log` | Local backend only: the *supervisor's* own stdout/stderr, including any traceback that killed it — `iax logs <run> --worker` prints it. The workload's output goes to `events.jsonl`. Ray run dirs lack it. |

`runs_dir` resolves from `--runs-dir`, else `$IAX_RUNS_DIR`, else
`outputs/experiments/runs` under the nearest project root above the current
directory. A not-found error prints the store it read; check that path before
concluding a run is gone.

**Orphan reaping needs Linux or the `psutil` extra.** The daemon kills a
workload whose supervisor died only when it can prove the recorded pid still
names the same process. It reads that proof from `/proc` on Linux, and from
`psutil` when there is no `/proc` (macOS, Windows). With neither, the run
records `orphan reaping is disabled on this machine` at start, the reap report
comes back `identity_unverifiable` with a `hint`, and the workload keeps
running — read `status.details.workload_pid` and kill it by hand. Install
`ai-experiments[psutil]` to make the reaper work everywhere.

On Ray, `iax logs` returns harness events only. Worker stdout and tracebacks surface
through `status.error` and `status.details.ray_log_tail`; Ray condition hints live in
`status.details.ray_condition`. `exit_code` is local-backend-only and remains `None`
for Ray runs.
