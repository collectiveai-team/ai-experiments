# Conventions

Rules for changing `ai_experiments` (the `iax` harness). These describe what the
codebase already does; follow them rather than importing habits from elsewhere.

## Commands

```bash
uv run --extra dev ruff check .        # lint gate
uv run --extra dev pytest -q          # test gate (CI: uv run --extra dev python -m pytest tests)
```

`--extra dev` is required. A bare `uv run pytest` resolves a pytest outside the
project venv and every test fails to import `ai_experiments`.

Integration tests (`tests/integration/`, marker `integration`) need real Ray and
MLflow via `docker compose -f tests/integration/compose.yaml up -d`; they skip
when the services are down, so `pytest tests` stays green without Docker. Never
make the default suite depend on Docker, the network, or `.env`.

## Style

- `from __future__ import annotations` at the top of every module with code.
  Re-export-only `__init__.py` files are the only exception.
- Fully annotate every function and method, including `-> None`. Use builtin
  generics (`list[str]`, `dict[str, object]`, `str | Path | None`). `**kwargs` is
  typed `**updates: object`, not `Any`.
- Data types are **pydantic v2 `BaseModel`**, not dataclasses, and live in
  `schemas.py`. `Field(default_factory=...)` for mutable defaults, `Literal`
  aliases instead of `enum.Enum`, `@field_validator` + `@classmethod`,
  `@model_validator(mode="after")` returning `self`, discriminated unions via
  `Annotated[Union[...], Field(discriminator="type")]`. Serialize with
  `model_dump(mode="json")` and `yaml.safe_dump(..., sort_keys=False)`.
- Absolute first-party imports (`from ai_experiments.store import ...`). No
  relative imports. Order: stdlib, third party, first party.
- 88 columns, ruff/black defaults. There is no ruff config file; do not add
  per-file `noqa` to dodge the gate.
- `# type: ignore[code]` only with an explicit error code, sparingly.
- **There is no `logging`.** Machine-readable state goes to
  `store.append_event(run_id, RunEvent(...))`; user-facing output goes through
  `typer.echo(...)`, errors via `typer.echo(..., err=True)` then
  `raise typer.Exit(1)`; daemon machine output is `print(json.dumps(...), flush=True)`.
- Docstrings are prose that explains *why*, on modules, classes, and non-obvious
  functions only. No `Args:`/`:param:` blocks. Obvious methods carry none.

## Error handling

- Catch the specific exception when the failure mode is known
  (`OSError`, `json.JSONDecodeError`, `ProcessLookupError`, ...).
- A bare `except Exception` is only acceptable on a genuinely best-effort side
  path, and must carry a comment saying why the failure is swallowed. Repro
  capture, MLflow tracking and notifications are best-effort **by contract**:
  they must never block a submit or a daemon tick.
- Optional dependencies (`ray`, `fastapi`, `mlflow`) are imported lazily inside
  the function that needs them and converted to a clear error at the import site.

## Run-store invariants

The run store is a published interface: agents, skills, and every issue repro
read these files directly with `cat`/`jq`/`json.load`. Changing the layout breaks
consumers outside this repo.

- `<runs>/<run_id>/{manifest.yaml, manifest.source.yaml, status.json, status.lock, events.jsonl, metrics.jsonl, artifacts/, repro/}`.
- Underscore-prefixed entries under the run root (`_campaigns/`, `_escalations/`,
  `_notifications.jsonl`) are **not** runs; `list_runs()` filters them.
- `status.json` is written through `atomic_write_text` (temp sibling +
  `os.replace`) so no reader ever sees a partial document. Do not reintroduce a
  plain `write_text` on it.
- `write_handle` **establishes** a status and refuses to overwrite an existing
  one; `update_status` is the only way to modify one. It serializes
  cross-process mutations with `fcntl.flock` on the per-run `status.lock`
  sidecar. Do not add a new mutation path that bypasses it.
- A synthesized status (missing/corrupt file, `SYNTHETIC_STATUS_KEY`) describes
  the store's inability to answer, not the run. Never persist one.
- `events.jsonl` / `metrics.jsonl` are append-only, one JSON object per line,
  written with a single small `O_APPEND` write.
- The workload contract is one stdout line, `IAX_METRIC {json}`. A workload must
  never be required to import the harness. Do not break it.
- Planner strategies are deterministic given `(seed, prior trials)` so replanning
  after a crash reproduces the same decisions.

## Concurrency

`status.json` has concurrent writers in **different processes**: the worker main
thread and its 15s heartbeat thread, the daemon reaper, MLflow finalize, CLI
cancel, and `RayBackend.inspect` from whichever of CLI/daemon/server polls. A
`threading.Lock` protects none of that. `FilesystemRunStore.update_status` is
the only mutation path for an existing status, and it serializes cross-process
read-modify-write sequences with `fcntl.flock` on the per-run `status.lock`
sidecar. Assume any new read-modify-write can interleave with another process,
and prove cross-process claims with a test that actually forks processes — not
threads, not a mocked clock.

## Tests

- Flat `tests/*.py`, named after the module under test; test functions are full
  sentences (`test_daemon_auto_kills_timed_out_run`).
- No root `conftest.py`. The fixture convention is a module-level `_`-prefixed
  factory taking `tmp_path` (`_store(tmp_path)`, `_running_run(store, ...)`).
  Use a real `@pytest.fixture()` only when composition demands it.
- `tmp_path` is the isolation mechanism; pass `capture_repro=False` when the test
  does not exercise repro capture.
- Assert through the store's API (`store.read_status(run_id).status`), not by
  reading raw files — except where the file layout itself is the contract.
- Hand-written fake backends beat `Mock`. Use `monkeypatch.setenv` /
  `monkeypatch.chdir` to isolate config lookups.
- A test for a race must be able to fail on the unfixed code. Bound it with a
  timeout and keep it deterministic enough to run in CI.

## Changes

- Fix the cause, not the symptom, and do not weaken a gate to make a run green.
- Keep the diff scoped to the ticket; no drive-by reformatting.
- Any change to a run-store file layout, the `IAX_METRIC` line, or a manifest
  field is a compatibility change: say so explicitly and update
  `.claude/skills/*/SKILL.md` and `README.md` in the same change.
