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
| `entrypoint` | string | — (required) | e.g. `python`, `python3`. Always required by the schema, even when `train`/`evaluate` are set — but then it is never run. |
| `args` | list[string] | `[]` | e.g. `["-m", "pkg.cli", "train", "cfg.yaml"]`. Appended to **every** phase that runs — `entrypoint` alone, or both `train` and `evaluate` when declared. `{name}` placeholders a campaign substitutes per trial live here, not in `train`/`evaluate` (those two are static strings, never templated). |
| `train` | string \| null | `null` | Optional first phase's command. Requires `evaluate` too — declaring only one is rejected. When both are set, `entrypoint` is unused and `train`/`evaluate` run instead, in order. May report progress (`IAX_METRIC`); a result (`IAX_RESULT`) it prints is discarded with a warning. |
| `evaluate` | string \| null | `null` | Optional second phase's command. Requires `train` too. The only phase whose declared result (`IAX_RESULT`) scores. |
| `working_dir` | string | `.` | Relative paths are resolved against the directory you submit from, once, at submit time; the run stores both the resolved manifest and the original. Keep it relative to stay portable. |
| `env` | mapping | `{}` | Extra environment variables, merged into every phase. |
| `data` | object | all defaults | `train`/`val`/`test` string references (`DataSpec`). Exported as `IAX_DATA_TRAIN`/`IAX_DATA_VAL` to every phase; `IAX_DATA_TEST` only to `evaluate` — a `train` phase never receives it **as an environment variable**, and that is the whole of the guarantee: the run's `manifest.yaml` records `data.test` verbatim and `IAX_RUN_DIR` points every phase at that file, so the held-out reference is a boundary train code is declared not to cross, not one the harness enforces. `data.test` requires an `evaluate` phase. |

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
| `timeout_seconds` | int \| null | `null` |
| `auto_kill` | bool | `false` |
| `fatal_on_nan` | bool | `true` |

Every field above is the whole list. An unknown key is rejected by name, so a
manifest never runs under configuration it did not get.

## Ray backend address

For `backend: ray`, the dashboard / Jobs API must be reachable from the machine
running `iax`. Remote clusters should start Ray with `--dashboard-host 0.0.0.0`.
`iax submit` uploads `workload.working_dir` through the Ray SDK for every run; SDK
upload progress logs go to stderr so `--json` output stays parseable on stdout.
