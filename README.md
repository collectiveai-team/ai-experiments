# Industrial AI Experiments

Detached experiment runtime for industrial AI training workloads.

`ai-experiments` is intentionally separate from agent prompts and Workbench UI code. It provides a small manifest contract, pluggable execution backends, filesystem run state, and scheduler-friendly monitoring commands for arbitrary model-training experiments.

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

### Runtime (`iax` CLI)

Install the experiment runtime from git, pinned to a release tag so the CLI and
manifest contract stay reproducible:

```bash
# Public HTTPS
uv pip install "ai-experiments @ git+https://github.com/collectiveai-team/ai-experiments.git@v0.1.0"

# Private repo over SSH (uses your git credentials)
uv pip install "ai-experiments @ git+ssh://git@github.com/collectiveai-team/ai-experiments.git@v0.1.0"
```

For Ray Jobs API support, add the `ray` extra:

```bash
uv pip install "ai-experiments[ray] @ git+https://github.com/collectiveai-team/ai-experiments.git@v0.1.0"
```

Pin `@v0.1.0` to the tag matching the `version` in `pyproject.toml`; use `@main` or a
commit SHA to track unreleased work.

### Agent skills

The repo also ships agent skills (under `.claude/skills/`) that teach a Claude Code
agent to drive the harness. Install them with the open [`skills`](https://github.com/vercel-labs/skills) CLI:

```bash
# Public HTTPS
npx skills@latest add https://github.com/collectiveai-team/ai-experiments

# Private repo over SSH
npx skills@latest add git@github.com:collectiveai-team/ai-experiments.git
```

This installs the `submitting-experiments`, `monitoring-experiments`,
`diagnosing-experiments`, and `cancelling-experiments` skills into the project's
`.claude/skills/` (add `-g` for `~/.claude/skills/`).

During in-repo development, Workbench uses the uv workspace member at `packages/ai_experiments`.

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

## Example agent prompts

With the agent skills installed (see [Agent skills](#agent-skills)), you can drive the
harness in natural language — the agent authors the manifest, validates it, submits the
run, and watches it for you.

**Simple local run:**

> Run a quick training experiment locally: the command is `python train.py --epochs 5`
> in `./workloads/forecast`. Check on it every minute and tell me if it finishes or
> stalls for more than 10 minutes.

**On the local Ray cluster:**

> Launch the demand-forecast training on the Ray backend with 4 CPUs and 1 GPU
> (entrypoint `python -m forecast.train configs/base.yaml`). Keep monitoring it and only
> ping me if it fails or looks stuck — otherwise summarize the results when it completes.

## Local Ray cluster

The `ray` backend submits through the Ray Jobs API, which is served by the cluster
dashboard (default `http://127.0.0.1:8265`). To run a small single-node cluster on your
machine:

```bash
uv pip install "ai-experiments[ray] @ git+https://github.com/collectiveai-team/ai-experiments.git@v0.1.0"

# Start a head node — dashboard + Jobs API listen on :8265
ray start --head --num-cpus=4 --dashboard-host=127.0.0.1

# Point the backend at it (this is also the default)
export RAY_ADDRESS=http://127.0.0.1:8265
```

Any manifest with `backend: ray` now submits jobs to that cluster:

```bash
iax submit experiment.yaml --json
iax status <run_id> --json   # mirrors the Ray job state
```

Add worker nodes from other machines with `ray start --address=<head-host>:6379`. Tear
the cluster down with `ray stop`.

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
  "ai-experiments>=0.1.0",
]
```

and remove the local uv source override:

```toml
[tool.uv.sources]
ai-experiments = { workspace = true }
```

## Release Checklist

1. Run package tests.
2. Build wheel and sdist.
3. Publish to the chosen package registry or GitHub Release assets.
4. Update Workbench to consume the released version.
5. Run Workbench backend, package, frontend, and scaffold verification.
