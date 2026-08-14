# Experiment Manifest Reference

Authoritative source: `ai_experiments/schemas.py` (`ExperimentManifest`). This file
summarizes the fields; if it disagrees with the code, the code wins.

## Top-level fields

| Field | Type | Default | Notes |
|---|---|---|---|
| `experiment` | string | — (required) | Must be non-empty. |
| `backend` | `local` \| `ray` | `local` | `ray` needs the `ai-experiments[ray]` extra. |
| `backend_address` | string \| null | `null` | Ray-only Jobs API/dashboard URL. Resolution order: manifest `backend_address`, then `RAY_ADDRESS`, then `http://127.0.0.1:8265`. Must start with `http://` or `https://`. |
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
| `working_dir` | string | `.` | Relative paths are resolved against the directory you submit from, once, at submit time; the run stores both the resolved manifest and the original. Keep it relative to stay portable. |
| `env` | mapping | `{}` | Extra environment variables. |

## `resources`

| Field | Type | Default |
|---|---|---|
| `cpus` | float | `1` |
| `gpus` | float | `0` |
| `memory_gb` | float \| null | `null` |

## `artifacts` (`ArtifactSpec`)

| Field | Type | Default |
|---|---|---|
| `output_dir` | string | `outputs/training` |
| `status_path` | string \| null | `null` |

## `monitoring` (`MonitorPolicy`)

| Field | Type | Default |
|---|---|---|
| `interval_seconds` | int | `300` |
| `stuck_after_minutes` | int | `30` |
| `no_event_after_minutes` | int \| null | `null` |
| `timeout_seconds` | int \| null | `null` |
| `checks` | list[string] | `["no_status_update", "no_log_progress", "process_exit"]` |

## Ray backend address

For `backend: ray`, the dashboard / Jobs API must be reachable from the machine
running `iax`. Remote clusters should start Ray with `--dashboard-host 0.0.0.0`.
`iax submit` uploads `workload.working_dir` through the Ray SDK for every run; SDK
upload progress logs go to stderr so `--json` output stays parseable on stdout.
