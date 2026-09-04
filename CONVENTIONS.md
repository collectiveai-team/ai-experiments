# Conventions

Repository rules for `ai-experiments` (the `iax` harness). These bind every
agent and human change. Where this file and a nearby file disagree, this file
wins; where this file is silent, copy the surrounding module.

## 1. Toolchain and commands

`uv` is the single authoritative package manager. There is no pip, poetry, or
conda surface. `requires-python = ">=3.10"`; the default dev environment
resolves to CPython 3.13.

Dev tools (`pytest`, `ruff`, `httpx`, `fastapi`, `uvicorn`) live in the `dev`
extra, not in the default dependency set. **Always pass `--extra dev`.** A bare
`uv run pytest` silently falls through to a globally installed `pytest` that
cannot import `ai_experiments`, and the whole suite errors on collection.

```bash
uv run --extra dev ruff check .     # lint gate
uv run --extra dev pytest -q        # test gate
uv build                            # packaging check (CI does this)
```

Both gates must exit zero from a fresh clone before any change is proposed.
Never weaken a gate to make it pass. Never add a blanket `# noqa`, a
`--exit-zero`, or a bare `pytest.mark.skip` to silence a real failure.

## 2. Code style

- Start every module with `from __future__ import annotations`.
- Use built-in generics and PEP 604 unions: `list[RunEvent]`, `dict[str, Any]`,
  `str | Path`, `int | None`. Do not import `List`, `Dict`, or `Optional`.
- Type every public function signature, including the return type.
- Import absolutely from the package root: `from ai_experiments.schemas import
  RunStatus`. No relative imports across subpackages.
- `ruff` runs with its default rule set and no per-project configuration. Keep
  it that way: fix the finding rather than adding a config exception. If an
  exception is genuinely legitimate, configure it centrally in
  `pyproject.toml`, never as a scattered inline suppression.
- Docstrings explain layout and intent, not the signature. Module and class
  docstrings that record an on-disk layout or a non-obvious constraint are the
  house style — keep them accurate when the layout changes.
- Comments explain *why*. A comment restating the code is noise.

## 3. Architecture boundaries

```
cli.py / server/app.py   entry points  (typer CLI `iax`, FastAPI + dashboard)
orchestrator.py          campaign loop: plan -> submit -> analyze -> replan
planner/                 search space, strategies, analysis   (pure, no I/O)
backends/                execution substrate: local | ray, behind ExperimentBackend
monitoring/              programmatic rules, then agent escalation
store/                   filesystem persistence for runs and campaigns
schemas.py               pydantic models shared by every layer
daemon.py                the monitor tick loop
```

Rules:

- `schemas.py` is the shared contract. Every layer depends on it; it depends on
  nothing in the package. Never import a backend, store, or server module there.
- **Every backend implements `ExperimentBackend` (`backends/base.py`)** —
  `submit`, `inspect`, `logs`, `cancel`, `diagnose`. Callers program against
  that ABC and reach a concrete backend only through `backends/factory.py`.
  Never branch on `isinstance(backend, RayBackend)` in orchestrator, daemon, or
  server code.
- `planner/` stays pure: it takes state and returns trials. No filesystem, no
  network, no clock reads that are not injected. This is what keeps planning
  testable without a backend.
- The server is a read/command layer over the store and the orchestrator. Put
  business logic in the orchestrator or the planner, not in a route handler.
- Do not reimplement cloud APIs. Cluster provisioning delegates to Ray's own
  cluster launcher (`ray up` via `iax cluster up`).
- Model-specific training logic is out of scope. Workloads are arbitrary
  commands that report metrics; the harness never assumes a framework.

## 4. State and durability

- The filesystem store is the source of truth, and it is deliberately
  greppable and agent-friendly: JSON and JSONL under a run/campaign directory.
- **All state writes go through `atomic_write_text`** (`store/filesystem.py`).
  A partially written `state.json` is a corrupted campaign. Never open a state
  file with a plain `open(..., "w")`.
- `atomic_write_text` makes a write *indivisible*, not a read-modify-write
  *serializable*. `FilesystemRunStore.update_status` is the only mutation path
  for an existing status, and it takes `fcntl.flock` on the per-run
  `status.lock` sidecar. Do not add a mutation path that bypasses it, and
  prove a cross-process claim with a test that forks processes.
- Events are append-only JSONL. Never rewrite or truncate an event log to
  correct history; append a corrective event.
- Run and campaign directory layouts are a public contract — external agents
  read them. Changing a layout is a breaking change (see §6).

## 5. Tracking, notifications, and other best-effort edges

MLflow mirroring, webhooks, and notify commands are **best-effort by
definition**. A missing dependency, an unreachable server, or a non-zero notify
command records a warning event and continues. It must never block a submit,
fail a daemon tick, or change a run's terminal status. Preserve that property
in any change to `tracking.py`, `notify.py`, or the backends.

MLflow specifics worth keeping intact: the harness creates the run at submit
and finalizes it at the terminal state, and it hands `MLFLOW_RUN_ID` plus
`MLFLOW_TRACKING_URI` to the workload (locally and in the Ray `runtime_env`) so
both halves write into the *same* run. A change that breaks that linkage is the
class of bug issue #6 was — invisible to the stubbed unit suite.

## 6. Compatibility promises

Treat these as public API. Breaking one needs an explicit note in the PR
description and, where users have persisted data, a migration or a documented
upgrade step:

- the `iax` CLI command names, arguments, and exit codes;
- the REST routes served by `server/app.py`;
- the on-disk run and campaign layout, and the `state.json` / `events.jsonl`
  schemas;
- the goal manifest (`goal.yaml`) keys and the `IAX_METRIC` report format;
- the `ExperimentBackend` ABC.

Add fields; do not repurpose or silently drop them. Keep a removed key accepted
and ignored for at least one release rather than erroring on an old manifest.

## 7. Test strategy

- `tests/` is the deterministic suite. It must pass with no Docker, no network,
  no credentials, and no `.env`. It must not depend on test ordering or on wall
  clock timing beyond injected values.
- The unit suite deliberately stubs `JobSubmissionClient` and the `mlflow`
  module. That is intentional: it covers states a live cluster will not produce
  on demand (resource starvation, a job stuck `PENDING`). Keep the stubs.
- `tests/integration/` exercises **real** Ray and **real** MLflow, marked
  `integration`. It self-skips when the services are unreachable, so the
  deterministic gate stays green without Docker. Keep that skip behavior — do
  not let an integration test fail the default gate.
- Version pinning is load-bearing for integration: the Ray Jobs API is
  versioned, so the client and cluster must match (`rayproject/ray:2.37.0-py312`,
  client `ray[default]==2.37.0`). `ray` publishes no wheels for 3.13+, so that
  venv pins Python 3.12.

  ```bash
  docker compose -f tests/integration/compose.yaml up -d
  uv run --python 3.12 --extra ray --extra mlflow --extra dev \
      python -m pytest tests/integration -v
  docker compose -f tests/integration/compose.yaml down -v
  ```

- **A regression test must be proven to catch its bug.** Check out the broken
  code into the package, watch the new test fail, then restore:

  ```bash
  git checkout <bad-commit> -- ai_experiments/
  uv run --extra dev pytest -q          # expect the new test to fail
  git checkout HEAD -- ai_experiments/
  ```

  `git stash` does not work for this once the fix is committed — it stashes only
  uncommitted changes, so the test passes against the fix while appearing to
  test the old code. Stating "the test fails without the fix" without having run
  it is not acceptable evidence.

## 8. Verifying the dashboard and the API

The dashboard is a single hand-written `ai_experiments/server/static/index.html`
served by FastAPI. There is no bundler, no npm, and no build step — edit that
file directly and do not introduce a frontend toolchain to change it.

For an API change, prefer a FastAPI `TestClient` test in `tests/test_server.py`
over manual checking; that is the durable evidence.

For a visual or interactive change, start the real app and exercise it:

```bash
uv run --extra dev iax serve --port 8000      # then open http://127.0.0.1:8000
curl -s http://127.0.0.1:8000/api/campaigns | head
```

Check the browser console and the network tab for errors, and attach what you
observed. A screenshot or a console transcript is evidence; "it should work" is
not. Encode any check that should stay true as a `TestClient` test so the test
gate protects it.

## 9. Security invariants

- **Never commit secrets.** No tokens, webhook URLs, cloud keys, or tracking
  credentials in source, tests, fixtures, goal manifests, or committed run
  directories. They come from the environment.
- `repro.py` captures a `diff.patch` of the uncommitted worktree into the run
  bundle. Assume a run directory may contain whatever was in the tree — never
  publish one, attach one to an issue, or copy it into a PR without reading it
  first.
- Redact credentials in logs and events. A tracking URI or webhook URL may
  embed a token; do not echo one into an event, a notification payload, or an
  error message.
- `monitoring.escalation.agent_command` and `--notify-command` execute a
  configured command. They are operator-supplied by design. Never build one by
  interpolating untrusted run output into a shell string; pass data as JSON on
  stdin, which is the existing contract.
- Do not widen a workload's environment beyond what it needs. The `runtime_env`
  handoff carries the MLflow run linkage, not the operator's whole environment.
- Treat metrics and logs reported by a workload as untrusted input: validate
  before acting, and never `eval` them.

## 10. Generated and vendored files

- `uv.lock` is generated and committed. Change it only through `uv` — never by
  hand — and commit it in the same change as the `pyproject.toml` edit.
- Do not edit anything under `.orquestalite/`, `.venv/`, `.agents/`, or
  `manual_test/`. They are local runtime state and are gitignored.
- `team.json` is local orq-lite configuration and is gitignored. Do not commit
  it and do not add project source rules to it; repository rules belong here.
- Keep `README.md` true. A change to the CLI surface, the goal manifest, the
  monitoring rules, or the dashboard updates the README in the same PR.

## 11. Change hygiene

- One objective per change. Do not fold an unrelated refactor into a fix.
- Do not leave dead code, commented-out blocks, or a debug `print`. The harness
  emits events; use those.
- Delete code you replace. A superseded code path kept "just in case" is a
  maintenance cost and a second thing to keep correct.
- Report outcomes honestly. If a gate fails, say so and show the output. If a
  step was skipped, say which and why.
