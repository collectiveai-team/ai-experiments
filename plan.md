# Implementation Plan — Make iax skills truthful for Ray + `backend_address` field

Tracks [PRD #1](https://github.com/collectiveai-team/ai-experiments/issues/1). Tasks are
ordered so each is independently grabbable; the code tasks (1–4) form one vertical slice,
the doc tasks (5–8) are parallelizable, and the test tasks (9–10) gate the code.

Resolution order being implemented everywhere: **manifest `backend_address` → `RAY_ADDRESS` env → `http://127.0.0.1:8265`**.

---

## Task 1 — Add `backend_address` to the manifest schema

**Files:** `ai_experiments/schemas.py`

- [ ] Add optional top-level field `backend_address: str | None = None` to `ExperimentManifest`.
- [ ] Add a light `field_validator` for `backend_address`: when non-null, reject empty/whitespace and require a URL-ish shape (e.g. starts with `http://`/`https://`). Keep it shape-only — cluster reachability stays a runtime concern.
- [ ] Confirm it round-trips through `to_yaml`/`from_yaml` and is persisted by the run store's `create_run` (writes `manifest.yaml`) — no extra persistence wiring expected.

**Done when:** `iax validate` on a manifest with a malformed `backend_address` fails; a valid one passes; `backend: local` manifests are unaffected.

---

## Task 2 — Resolve the address inside `RayBackend`

**Files:** `ai_experiments/backends/ray.py`

- [ ] Generalize the constructor so the resolved address is computed in one place with precedence: explicit `address` arg → `RAY_ADDRESS` env → `http://127.0.0.1:8265`.
- [ ] Ensure `dashboard_url` on the `RunHandle` and `details.ray_address` reflect the resolved address.

**Done when:** constructing `RayBackend(address=...)` overrides env; with no arg, env wins; with neither, the default is used.

---

## Task 3 — Thread the address through the factory + submit path

**Files:** `ai_experiments/backends/factory.py`, `ai_experiments/cli.py`

- [ ] Add an optional `address: str | None = None` parameter to `get_backend` and pass it into `RayBackend` (keep the factory surface narrow — pass an address, not the whole manifest).
- [ ] In `cli.py submit`, resolve `manifest.backend_address` and pass it to `get_backend`.
- [ ] `backend: local` must ignore the address cleanly.

**Done when:** `iax submit` against a manifest with `backend_address` reaches that cluster (verified via the stub client_factory in tests, not a live cluster).

---

## Task 4 — Reconstruct the address on inspect-side commands

**Files:** `ai_experiments/cli.py` (+ small helper, e.g. in `store` or `cli`)

- [ ] `status`, `logs`, `diagnose`, `monitor`, `cancel` only know the run's backend *name* from `status.json`. For `ray`, read the persisted `manifest.yaml` from the run store to recover `backend_address`, then pass it to `get_backend`.
- [ ] Fall back to `RAY_ADDRESS` → default when the persisted manifest has no field (older runs).
- [ ] Extract the "resolve address for a run_id" logic into a single small, unit-testable helper so it isn't duplicated across five commands.

**Done when:** a `diagnose`/`cancel` on a run submitted to a remote cluster talks to that cluster, not `127.0.0.1:8265`.

---

## Task 5 — Doc: `submitting-experiments` SKILL.md + `reference/manifest.md`

**Files:** `.claude/skills/submitting-experiments/SKILL.md`, `.../reference/manifest.md`

- [ ] Document the address resolution order and the `http://127.0.0.1:8265` default.
- [ ] State the Ray dashboard must be reachable — cluster started with `--dashboard-host 0.0.0.0`.
- [ ] Add `backend_address` to the manifest field table: type `string | null`, default `null`, Ray-only, note precedence.
- [ ] Note `iax submit` re-uploads `working_dir` per run, and Ray SDK upload logs go to **stderr** (JSON stays clean on stdout).

---

## Task 6 — Doc: `monitoring-experiments` SKILL.md

**Files:** `.claude/skills/monitoring-experiments/SKILL.md`

- [ ] Mark `worker.log` in the run-store layout table as **local-backend-only**; state a Ray `run_dir` has only `manifest.yaml`, `status.json`, `events.jsonl`.
- [ ] Note `iax logs` on Ray returns only harness events; worker stdout/tracebacks live in `status.error` / `details.ray_log_tail`.
- [ ] Note `exit_code` is local-backend-only and stays `None` on Ray.

---

## Task 7 — Doc: `diagnosing-experiments` SKILL.md

**Files:** `.claude/skills/diagnosing-experiments/SKILL.md`

- [ ] Step 2: make the "read the raw evidence" guidance backend-aware — local → `worker.log`; Ray → `status.error`, `details.ray_log_tail`, `details.ray_condition`.
- [ ] Add `run_completed`, `failed`, `cancelled`, `run_active` to the "Recognized reasons" table (prefer listing over a scoping caveat).
- [ ] Note `exit_code` is local-backend-only; on Ray use `status` + `error` + `details.ray_*`.

---

## Task 8 — Doc: `cancelling-experiments` SKILL.md

**Files:** `.claude/skills/cancelling-experiments/SKILL.md`

- [ ] In the "deleting a run_dir removes … `worker.log`" line, note `worker.log` is local-only so the housekeeping description holds for Ray run stores.

---

## Task 9 — Test: backend address resolution

**Files:** `tests/test_ray_backend.py`

- [ ] Table-driven unit test over precedence: (a) `backend_address` set wins even with `RAY_ADDRESS` also set; (b) no field, `RAY_ADDRESS` set → env wins; (c) neither → `http://127.0.0.1:8265`.
- [ ] Capture the resolved address via the injected `client_factory` (already receives the address) — no network.
- [ ] Assert external behavior (the address handed to the client), not private attributes.

---

## Task 10 — Test: `rules.py` reason coverage

**Files:** `tests/` (new or existing rules test)

- [ ] Assert `diagnose_run` emits `run_completed` for `completed`, `failed`/`cancelled` for those states, and `run_active` for a healthy active run.
- [ ] Build statuses via `FilesystemRunStore` in a tmp dir; assert on emitted `decision.reasons`.

**Done when:** the doc table (Task 7) and code can't silently drift apart.

---

## Verification (whole slice)

- [ ] `ruff check` + `ruff format` clean; `pyright` zero errors.
- [ ] `pytest` green, including Tasks 9–10.
- [ ] Re-read each edited SKILL.md against current source (`backends/*.py`, `monitoring/rules.py`, `schemas.py`) — docs match code.
- [ ] (Optional, manual) one `iax submit`/`diagnose` round-trip against the vader cluster with `backend_address` set, confirming no `127.0.0.1:8265` fallback.

## Out of scope (do not do here)

- `iax submit --address` CLI flag.
- Synthesizing `exit_code` for Ray; redesigning per-run `working_dir` upload.
- Any `MonitorDecision` / `DiagnosisReport` schema or decision-logic change.
- Cluster provisioning, auth, or TLS for the dashboard.
