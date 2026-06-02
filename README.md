# Industrial AI Experiments

Detached experiment runtime for industrial AI training workloads.

`industrial-ai-experiments` is intentionally separate from agent prompts and Workbench UI code. It provides a small manifest contract, pluggable execution backends, filesystem run state, and scheduler-friendly monitoring commands for arbitrary model-training experiments.

## Responsibilities

- Validate an experiment manifest before a run starts.
- Submit a detached workload to a backend (`local` or `ray`).
- Persist a portable run handle, status, logs, and diagnosis state.
- Emit quiet monitor results so an external scheduler can wake an agent only when intervention is useful.
- Keep Ray-specific behavior behind the Ray backend and Ray monitoring rules.

## Non-responsibilities

- Agent orchestration and prompt design.
- Workbench project/session persistence.
- UI rendering.
- Model-specific training logic.
- Cloud provisioning for Ray clusters.

## Install

```bash
uv pip install industrial-ai-experiments
```

For Ray Jobs API support:

```bash
uv pip install "industrial-ai-experiments[ray]"
```

During in-repo development, Workbench uses the uv workspace member at `packages/industrial_ai_experiments`.

## Manifest

```yaml
experiment: demand_forecast_baseline
backend: ray
workload:
  entrypoint: python
  args:
    - -m
    - ts_agents_lab.cli
    - train
    - configs/training.yaml
  working_dir: .
monitor:
  interval_seconds: 300
  stuck_after_seconds: 1800
metadata:
  project_id: example
```

## CLI

```bash
iax validate experiment.yaml
iax submit experiment.yaml --json
iax status <run_id> --json
iax logs <run_id> --tail 200
iax diagnose <run_id> --json
iax monitor <run_id> --json --quiet-when-waiting
iax cancel <run_id>
```

`iax monitor --quiet-when-waiting` is the Workbench scheduler integration point. It prints nothing while the run should keep waiting. It emits JSON only when the run has completed, failed, looks stuck, or needs delegated review.

## Composition With Workbench

Workbench composes this runtime instead of owning it:

1. A training agent exports a generic experiment manifest.
2. Workbench persists the returned run handle in `experiment_runs`.
3. The scheduler runs `iax monitor <run_id> --json --quiet-when-waiting`.
4. Empty monitor output means no agent wake-up.
5. JSON monitor output updates the run record and can trigger the training-monitor agent.

When this package is extracted to its own repository, Workbench should depend on a released package version instead of a workspace source:

```toml
dependencies = [
  "industrial-ai-experiments>=0.1.0",
]
```

and remove the local uv source override:

```toml
[tool.uv.sources]
industrial-ai-experiments = { workspace = true }
```

## Release Checklist

1. Run package tests.
2. Build wheel and sdist.
3. Publish to the chosen package registry or GitHub Release assets.
4. Update Workbench to consume the released version.
5. Run Workbench backend, package, frontend, and scaffold verification.
