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
