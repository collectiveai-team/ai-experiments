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
else `outputs/experiments/runs` under the nearest project root above the current
directory). To clean up, inspect each run's status first:

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
for local runs `worker.log` — i.e. all record of the run. Ray run stores do not have
`worker.log`. Archive anything you need first.
