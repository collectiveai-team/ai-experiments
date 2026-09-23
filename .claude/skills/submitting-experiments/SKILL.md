---
name: submitting-experiments
description: Use when launching or setting up a detached training experiment with the iax harness — authoring an experiment manifest, validating it, and submitting a run. Triggers on "run an experiment", "launch a training run", "submit a job to iax", or when you need to produce a run_id for a training workload.
---

# Submitting Experiments

Drive the `iax` harness to launch a detached training run. Authoritative manifest
schema: `ai_experiments/schemas.py`. Full field reference: `reference/manifest.md`.

## Workflow

1. **Gather the workload.** What command runs the training? Map it to
   `workload`: `entrypoint` + `args` for one command, or `train:` + `evaluate:`
   when the trainer and its scorer must stay separate — declaring both makes
   `entrypoint` unused, and only `evaluate`'s declared result (`IAX_RESULT`)
   scores. Either way, add the `working_dir` it must run from and any `env`.
   `args` is appended to **every** phase you declare, not only `entrypoint` —
   see `reference/manifest.md`.
2. **Pick the backend.** `local` (default; runs as a detached subprocess) or `ray`
   (needs the `ai-experiments[ray]` extra and a reachable Ray dashboard / Jobs API).
   Ray address resolution is `backend_address` in the manifest, then `RAY_ADDRESS`,
   then `http://127.0.0.1:8265`. For remote Ray clusters, start the dashboard with
   `--dashboard-host 0.0.0.0` so the submitting machine can reach it.

   The two backends do not enforce the phase split equally. On `local` it is
   structural: train and evaluate are two separately supervised processes, and
   the supervisor reading a train phase's stdout is the one that discards its
   result. On `ray` one job runs both commands into one log stream, so the
   harness reconstructs the phase from a per-run random token echoed by the
   entrypoint — careful (an unmarked, mis-marked or wrongly-tokened result is
   discarded) but defensive, not structural: the token lives in the job's
   entrypoint string and a workload that reads its parent's command line can
   recover it. When the workload is code an agent wrote rather than code a
   person read, `local` is the backend whose guarantee does not rest on the
   workload behaving.
3. **Set resources and monitoring.** `resources.cpus`/`gpus`/`memory_gb`; a
   `monitoring` policy with `interval_seconds` (how often the scheduler checks) and
   `stuck_after_minutes` (how long without progress before flagging). See
   `reference/manifest.md` for every field and its default.
4. **Validate, then submit:**

   ```bash
   iax validate experiment.yaml
   iax submit experiment.yaml --json
   ```

   Both commands check that the `working_dir` exists and that every command
   `phases()` would actually run resolves — the single `entrypoint`, or both
   `train` and `evaluate` when declared — and print `Warning:` lines on
   **stderr**; stdout stays parseable. They are warnings because a Ray
   workload resolves its commands on the cluster, not here. Pass `--strict`
   to refuse instead of warn — use it when the backend is `local`, where a
   warning is always a failure a second later.

5. **Capture the run handle.** `submit --json` prints a `RunHandle`. Record
   `run_id` (everything downstream keys off it), `run_dir`, and `status_uri`. If you
   pass `--runs-dir <dir>`, pass the same `--runs-dir` to every later command, or set
   `IAX_RUNS_DIR` so the scheduler and agent agree on the store root.

Ray `iax submit` uploads `workload.working_dir` for each run through the Ray SDK.
Those upload progress logs go to stderr; JSON output remains clean on stdout.

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

Two-phase form — the trainer's own flags go in `train:`, not in a top-level
`args`, because `args` is appended to every declared phase:

```yaml
experiment: demand_forecast_two_phase
backend: local
workload:
  entrypoint: python3        # unused: train/evaluate run instead, but still required
  train: "python3 -m ts_agents_lab.cli train configs/training.yaml"
  evaluate: "python3 -m ts_agents_lab.cli evaluate configs/training.yaml"
  working_dir: .
resources:
  cpus: 4
  gpus: 1
```

Only `evaluate`'s declared result (`IAX_RESULT`) scores; a result printed
from `train` is discarded with a warning.

## Reading exit codes

Every command accepts `--json` and reports failures the same way, so branch on
the exit code instead of on the message text:

| exit | `code` | what to do |
|---|---|---|
| 0 | — | success; parse stdout as JSON |
| 1 | `not_found` | the id is wrong — list with `iax runs` / `iax campaign list` |
| 2 | `invalid_input` | fix the manifest, the goal, or the params and retry |
| 3 | `backend_unavailable` | the backend is unreachable; check Ray, then retry |

With `--json`, the failure is one object on stdout: `{"error": ..., "code":
..., "details": {...}}`. Without it, one line on stderr. Never treat a non-zero
exit as an empty result.

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
