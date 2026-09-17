# House Standards Adoption Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Get `uvx prek run --all-files` to exit 0 on this repo and apply the CES standards that the scaffolding install exposed as unapplied.

**Architecture:** One branch, `feat/house-standards`, landed as a single PR. Work moves from zero-risk config, through a pure-formatting commit, to mechanical lint fixes, and only then to the genuine refactors (CES-76 settings, CES-79 no-dict boundaries, CES-71 `cli.py` split, CES-110 complexity). Each task ends green on the checks it owns, so a bisect lands on a small blast radius even though the PR is one unit.

**Tech Stack:** Python 3.10+, uv, ruff 0.15.22, pyrefly, ast-grep, prek, pytest, Typer, FastAPI, Pydantic v2.

**Spec:** None — this plan is driven by measured tool output, inventoried on 2026-09-17 and reproduced in Task 1. The `.agents/rules/<slug>.md` catalog is authoritative for every CES cited (per `engineering-rules`: where both exist, `.agents/rules/` wins over the skill bundle).

## Global Constraints

- **Branch:** all work on `feat/house-standards`, cut from `main`. Never commit to `main`.
- **Commit messages:** Conventional Commits (`agents-conventional-commits`, CES-75) **and** no AI co-authorship or "Generated with" trailers (`no-ai-coauthorship`, CES-91, enforced by `commit-policy.yml` in CI).
- **Line length:** `line-length = 100`, already set in `pyproject.toml`. Decided 2026-09-17; the resulting reformat is isolated in Task 3.
- **Test gate:** `119 passed, 6 skipped` is the baseline. Two integration tests in `tests/integration/test_local_mlflow.py` (`test_daemon_mirrors_a_completed_run_as_finished`, `test_run_artifacts_are_uploaded_to_mlflow`) **fail before this plan starts** — they assert against a real MLflow on `:5000` that this host does not own. They are out of scope. Never "fix" them by weakening an assertion; if they start passing, note it, don't chase it.
- **Never run `uv sync`** in the primary working tree — four other worktrees share it and a sync prunes packages another branch needs. `uv lock` is safe (lockfile only). To add a dependency to the venv, use `.venv/bin/python -m pip install <pkg>`, which is additive. Run tests with `.venv/bin/python -m pytest` and linters with `uvx <tool>`.
- **Two classes of work, don't confuse them.** *Gates* are checks that fail the `prek` hook and therefore block every commit once hooks are installed: ruff-format, ruff-check, pyrefly, `file-size-guard`, `deptry`, `complexipy`, and the 41 `no-dict` ast-grep **errors**. *Adoptions* are the 17 ast-grep **warnings** (`settings-module` ×12, `log-no-print` ×3, `cli-typed-framework` ×2) — they never fail the hook, and they are in this plan because the user explicitly asked for the unapplied standards to be applied. A gate may not be skipped; an adoption may be suppressed with a visible, justified `ast-grep-ignore` where the rule's own "Suppressing" section allows it.
- **Behavior stays stable.** This is a refactor (`engineering-refactor` §3). No feature changes, no API changes, no changed CLI output, unless a CES demands it and the task says so.
- **Entrypoint contract:** `[project.scripts] iax = "ai_experiments.cli:app"` must keep resolving. Task 11 changes `cli.py` into a package; `ai_experiments.cli:app` must still be importable.

---

## File Structure

| File | Responsibility after this plan |
|---|---|
| `pyproject.toml` | ruff/pyrefly/deptry config; gains `[tool.deptry]` and bugbear immutable-calls |
| `ai_experiments/settings.py` | **new** — `BaseSettings` + cached `get_settings()`; the only env reader (CES-76) |
| `ai_experiments/core/logger.py` | **new** — the structlog `get_logger()` drop-in (CES-74, CES-45, CES-46) |
| `ai_experiments/cli/__init__.py` | **new** — assembles Typer apps, exports `app` |
| `ai_experiments/cli/runs.py` | **new** — run-level commands (validate…serve, run_goal) |
| `ai_experiments/cli/campaigns.py` | **new** — `campaign_*` commands |
| `ai_experiments/cli/clusters.py` | **new** — `cluster_*` commands |
| `ai_experiments/schemas.py` | gains the result models that replace returned raw dicts (CES-79) |
| `ai_experiments/server/app.py` | response models instead of dict returns; `create_app` decomposed |
| `ai_experiments/worker.py` | argparse → Typer; argv contract unchanged (CES-67) |
| 20 files listed in Task 10 | dict returns/annotations replaced |

---

### Task 1: Branch and reproduce the baseline

**Files:**
- Create: none
- Modify: none

**Interfaces:**
- Produces: the branch `feat/house-standards`; the file `/tmp/baseline-prek.txt` that later tasks diff against.

- [ ] **Step 1: Cut the branch from a clean main**

```bash
cd /home/lio/Projects/collectiveai/ai-experiments
git status --porcelain          # expect only: M .gitignore, M pyproject.toml, plus untracked scaffolding files
git checkout -b feat/house-standards
```

The scaffolding changes from 2026-09-16 are uncommitted on `main`. They come along onto the branch — that is intended; this PR carries both.

- [ ] **Step 2: Commit the scaffolding baseline first**

```bash
git add -A -- ':!docs/assessment-y-plan-2026-09-07.md'
git commit -m "chore: adopt house scaffolding (prek, ast-grep, AGENTS.md, CI, skills manifest)"
```

This separates "the config arrived" from "the code was fixed", so the rest of the PR reads as the fix.

- [ ] **Step 3: Record the test baseline**

Run: `.venv/bin/python -m pytest tests -q 2>&1 | tail -3`
Expected: `2 failed, 119 passed, 6 skipped` — the two known MLflow integration failures.

- [ ] **Step 4: Record the hook baseline**

```bash
uvx prek run --all-files > /tmp/baseline-prek.txt 2>&1; echo "exit=$?"
git checkout -- ai_experiments tests examples
```

Expected: `exit=1`. The `git checkout` is mandatory — `ruff format` and `ruff check --fix` rewrite files in place during the run, and this task must not leave code changes behind.

---

### Task 2: Config-only fixes (no code touched)

Three checks are failing on configuration, not on code. Fixing them first shrinks the real inventory before anyone edits a line of Python.

**Files:**
- Modify: `pyproject.toml`

**Interfaces:**
- Produces: `deptry` and the `.agents/` half of `pyrefly` passing; `B008` (35 hits) gone from the ruff inventory.

- [ ] **Step 1: Exclude scaffolding snippets from type checking**

`pyrefly` currently type-checks `.agents/snippets/`, which are template drop-ins, not repo code. They import `structlog` and `pydantic_settings`, which this repo does not depend on — 3 of the 21 pyrefly errors are this. In `pyproject.toml`, under `[tool.pyrefly]`, add `".agents/**"` to `project-excludes`:

```toml
[tool.pyrefly]
project-excludes = [
  "**/node_modules",
  "**/__pycache__",
  "**/venv/**/*",
  "**/.venv/**/*",
  "**/.uv_cache/**/*",
  ".agents/**",
]
```

- [ ] **Step 2: Verify those three errors are gone**

Run: `uvx pyrefly check --config pyproject.toml 2>&1 | grep -c 'missing-import'`
Expected: `3` (down from 6) — the remaining three are the `test_orchestrator` / `test_tracking` cross-test imports, fixed in Task 7.

- [ ] **Step 3: Teach deptry the two module→package mappings**

`yaml` ships in `pyyaml`, `mlflow` in `mlflow-skinny`; deptry cannot infer either, so it reports both a missing import and an unused dependency for each. The dev tools are declared in `[project.optional-dependencies].dev`, which deptry does not read as dev by default. Append to `pyproject.toml`:

```toml
# MARK: deptry
# CES-109: module→distribution mappings deptry cannot infer, plus the dev extra.
[tool.deptry]
known_first_party = ["ai_experiments"]
pep621_dev_dependency_groups = ["dev"]

[tool.deptry.package_module_name_map]
pyyaml = ["yaml"]
mlflow-skinny = ["mlflow"]
```

- [ ] **Step 4: Verify deptry is clean**

Run: `uvx deptry .`
Expected: `Success! No dependency issues found` — all 8 findings were config.

- [ ] **Step 5: Declare the Typer/FastAPI default-argument idiom immutable**

All 35 `B008` hits are `typer.Option(...)` / `typer.Argument(...)` in `cli.py` — the framework's documented calling convention, not a mutable-default bug. Silencing it per-file would also hide real B008s, so declare the calls immutable instead. Add to `pyproject.toml` under the ruff lint config:

```toml
[tool.ruff.lint.flake8-bugbear]
extend-immutable-calls = [
  "typer.Option",
  "typer.Argument",
  "fastapi.Depends",
  "fastapi.Query",
  "fastapi.Path",
  "fastapi.Body",
]
```

- [ ] **Step 6: Verify the inventory shrank**

Run: `uvx ruff@0.15.22 check --statistics 2>&1 | tail -1`
Expected: `Found 165 errors` — 200 minus the 35 B008.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml
git commit -m "build: configure deptry mappings, pyrefly excludes and bugbear immutable calls"
```

---

### Task 3: The formatting commit

Isolated on purpose: `line-length = 100` reflows 33 files. Mixed into a logic commit it would make review impossible, and it must be rebased onto the 13 open PRs as one identifiable commit.

**Files:**
- Modify: 33 files under `ai_experiments/`, `tests/`, `examples/` (formatting only)

**Interfaces:**
- Produces: a tree where `ruff format --check` passes, so every later diff is pure logic.

- [ ] **Step 1: Format**

Run: `uvx ruff@0.15.22 format .`
Expected: `33 files reformatted` (approximately; count is not a gate).

- [ ] **Step 2: Prove it is formatting-only**

Run: `git diff --stat | tail -1`
Expected: roughly `33 files changed, ~260 insertions(+), ~350 deletions(-)` — a net reduction, because reflowing to 100 columns collapses wrapped calls.

Run: `.venv/bin/python -m pytest tests -q 2>&1 | tail -1`
Expected: `2 failed, 119 passed, 6 skipped` — identical to baseline. Formatting cannot change behavior; if this number moves, stop and investigate.

- [ ] **Step 3: Commit**

```bash
git add -u
git commit -m "style: reformat to line-length 100 (ruff format)"
```

- [ ] **Step 4: Tell the humans**

This commit is the one that conflicts with the 13 open PRs. Note its SHA in the PR description so each PR author can rebase with `git rebase feat/house-standards` and resolve formatting conflicts with `git checkout --theirs` plus a re-run of `ruff format`.

---

### Task 4: Mechanical ruff fixes

**Files:**
- Modify: files across `ai_experiments/`, `tests/`, `examples/` as ruff selects them

**Interfaces:**
- Produces: ruff down from 165 to roughly 120, with zero judgment applied.

- [ ] **Step 1: Apply the safe fixes**

Run: `uvx ruff@0.15.22 check --fix`
Expected: about 40 fixed (`I001` import sorting, `TC001`/`TC003` typing-only imports moved into `TYPE_CHECKING` blocks, `RUF100`, `PT001`, `UP007`).

- [ ] **Step 2: Run the tests**

Run: `.venv/bin/python -m pytest tests -q 2>&1 | tail -1`
Expected: `2 failed, 119 passed, 6 skipped`.

`TC001`/`TC003` move imports into `if TYPE_CHECKING:` blocks. That is safe only when the annotation is not evaluated at runtime — Pydantic models **do** evaluate theirs. If any test fails with `NameError` or a Pydantic `class not fully defined` error, revert that specific move and add `from __future__ import annotations` to the file instead.

- [ ] **Step 3: Review the unsafe fixes before applying any**

Run: `uvx ruff@0.15.22 check --unsafe-fixes --diff | head -120`

Read the diff. `--unsafe-fixes` is allowed to change behavior. Apply only the categories you have read in full; do not run it blind across the repo.

- [ ] **Step 4: Commit**

```bash
git add -u
git commit -m "style: apply ruff safe autofixes"
```

---

### Task 5: Remaining ruff findings by category

Roughly 120 findings, each needing a decision. Work category by category, committing per category so a bad call is revertible on its own.

**Files:**
- Modify: `ai_experiments/**`, `tests/**`, `examples/toy_train.py`

**Interfaces:**
- Produces: `uvx ruff check` exiting 0.

- [ ] **Step 1: B904 — `raise ... from` inside except (13 hits)**

Real defect class: swallowing the cause makes tracebacks lie. Fix, never ignore:

```python
# before
except ValueError:
    raise typer.BadParameter("invalid goal file")

# after
except ValueError as exc:
    raise typer.BadParameter("invalid goal file") from exc
```

Use `from None` only where the cause is genuinely noise (e.g. re-raising a parse error as a user-facing message where the inner traceback is meaningless).

- [ ] **Step 2: D205 — blank line after docstring summary (20 hits)**

Purely mechanical:

```python
# before
"""Return the run store.
Uses the configured runs directory."""

# after
"""Return the run store.

Uses the configured runs directory.
"""
```

- [ ] **Step 3: S603/S607 — subprocess calls (16 hits)**

These are deliberate: the repo shells out to `ray`, `git`, and training commands. `S607` (partial path) is the one worth fixing where the binary is known — resolve via `shutil.which` so a hijacked `PATH` cannot substitute it. Where the command is user-supplied (the whole point of a training runner), suppress with a justification on the line:

```python
result = subprocess.run(cmd, check=False, capture_output=True)  # noqa: S603  # user-supplied training command, documented in CES-8 boundary
```

Every `noqa` in this step needs that trailing justification. A bare `noqa` is a review failure.

- [ ] **Step 4: PLW1510 — `subprocess.run` without `check` (3 hits)**

Pass `check=` explicitly at each site. Read the surrounding code first: if the caller already inspects `returncode`, pass `check=False`; if it ignores it, `check=True` is the fix and changes behavior deliberately — call that out in the commit body.

- [ ] **Step 5: S311 — non-cryptographic random (6 hits)**

These are the planner's search sampling — legitimate. Add a scoped ignore rather than six `noqa`s:

```toml
[tool.ruff.lint.per-file-ignores]
"tests/**" = ["S101", "INP001"]
"ai_experiments/planner/**" = ["S311"]   # search sampling, not security
```

- [ ] **Step 6: The tail (SIM105, SIM102, SIM117, RET504, PERF401, PERF203, C901, PT011, PT018, B905, S110, S112, S310, D401, PLW0108, RUF059, INP001)**

Roughly 30 findings, each a one-to-three-line local edit. Two carry judgment:
- `C901` (4 hits) overlaps Task 11 — leave them, Task 11 removes them.
- `S110`/`S112` (try-except-pass/continue, 4 hits) hide failures. Either log at debug via the Task 7 logger, or add a justification comment. Do not delete the handler.

- [ ] **Step 7: Verify and commit**

```bash
uvx ruff@0.15.22 check          # expect: All checks passed! (C901 may remain until Task 11)
.venv/bin/python -m pytest tests -q 2>&1 | tail -1
git add -u && git commit -m "fix: resolve remaining ruff findings"
```

---

### Task 6: pyrefly type errors (18 remaining)

Task 2 removed the 3 `.agents/snippets/` errors; Task 7 removes the 3 cross-test `missing-import`s. This task clears the other 15, which are real type defects. All but one are in `tests/` — the production code is nearly clean, and the tests are where the type contract is being quietly bypassed.

**Files:**
- Modify: `ai_experiments/planner/strategies.py:116`, `tests/test_daemon.py:43,44`, `tests/test_manifest.py:16,17`, `tests/test_monitoring_v2.py:38,39`, `tests/test_ray_backend.py:163`, `tests/test_tracking.py:93,119,123,227`, `tests/test_e2e_campaign.py:87`
- Possibly modify: `ai_experiments/schemas.py` (export the two `Literal` aliases), `ai_experiments/tracking.py` (narrow `last_client`)

**Interfaces:**
- Consumes: nothing.
- Produces: `ai_experiments.schemas.Backend` and `ai_experiments.schemas.RunState` — the two `Literal` aliases, named once instead of repeated inline. Later tasks' models reuse them.

- [ ] **Step 1: Name the two Literal types (7 `bad-argument-type` errors)**

Every one of these is a test passing a plain `str` where a `Literal` is declared:

```
Argument `str` is not assignable to parameter `backend` with type `Literal['local', 'ray']`
Argument `str` is not assignable to parameter `status` with type `Literal['cancelled', 'completed', ...]`
```

In `ai_experiments/schemas.py`, hoist the inline literals into named aliases and use them in the model fields:

```python
Backend = Literal["local", "ray"]
RunState = Literal["cancelled", "completed", "failed", "running", "submitted", "unknown"]
```

This is not the fix by itself — it makes the fix readable. Do not widen either field to `str`; the literal *is* the contract this repo is enforcing.

- [ ] **Step 2: Fix the test helpers that erase the type**

The cause is a dict-then-splat helper, e.g. `tests/test_daemon.py:38-44`:

```python
base = {..., "backend": "local", "details": {...}}
base.update(status_overrides)          # base is inferred dict[str, object]
store.write_status(RunStatus(**base))  # every value is now `object`
```

`base.update(status_overrides)` is also the `MutableMapping.update` overload error (3 hits) — same root cause. Build the model directly instead of splatting an untyped dict:

```python
def _seeded_run(tmp_path, **overrides) -> str:
    status = RunStatus(
        run_id=run_id,
        backend="local",
        status="running",
        status_uri=str(store.status_path(run_id)),
        run_dir=str(run_dir),
        started_at=utc_now() - timedelta(minutes=10),
        details={"heartbeat_at": utc_now().isoformat()},
    )
    store.write_status(status.model_copy(update=overrides))
    return run_id
```

`model_copy(update=...)` keeps the override ergonomics the helpers exist for, while the base object stays typed. Apply the same shape in `test_manifest.py`, `test_monitoring_v2.py`, `test_ray_backend.py` and `test_tracking.py`.

- [ ] **Step 3: Run the tests for those files**

Run: `.venv/bin/python -m pytest tests/test_daemon.py tests/test_manifest.py tests/test_monitoring_v2.py tests/test_ray_backend.py tests/test_tracking.py -q`
Expected: the same pass/fail counts as before the change. `model_copy(update=)` does not re-validate by default — if a test relied on an override being coerced, it will surface here; fix it by passing the value already typed, not by reverting to the splat.

- [ ] **Step 4: Fix the `NoneType` attribute errors (3 hits)**

`tests/test_tracking.py:119,123,227` read `fake.last_client.runs`, where `last_client` is declared `Client | None` and only set as a side effect of `start_run`. Pyrefly is right: nothing proves it is set. Assert it, which also documents the precondition:

```python
client = fake.last_client
assert client is not None, "start_run must have created a client"
assert client.runs[mlflow_run_id]["tags"]["iax.run_id"] == run_id
```

- [ ] **Step 5: Fix the two comparison errors**

`ai_experiments/planner/strategies.py:116` sorts `TrialRecord` by `objective_value`, which is `float | None` — unorderable, and it already carries a blanket `# type: ignore[arg-type, return-value]` that hides the real problem. Decide what a `None` objective means and encode it:

```python
scored = [t for t in trials if t.objective_value is not None]
ranked = sorted(scored, key=lambda t: t.objective_value or 0.0, reverse=reverse)
```

Read the surrounding code first — `scored` may already be filtered upstream, in which case a narrowing assert is the honest fix and the `or 0.0` is not. Delete the `type: ignore` either way; a suppression over a real defect is worse than the defect.

`tests/test_e2e_campaign.py:87` is the same `float | None` in a `max(...)` key. Fix it the same way.

- [ ] **Step 6: Verify and commit**

```bash
uvx pyrefly check --config pyproject.toml 2>&1 | tail -1     # expect: 3 errors (the cross-test imports, Task 7)
.venv/bin/python -m pytest tests -q 2>&1 | tail -1
git add -u && git commit -m "fix: resolve pyrefly type errors in the test helpers and planner ranking"
```

---

### Task 7: CES-74/45/46 the house logger, and the test-import errors

**Files:**
- Create: `ai_experiments/core/__init__.py`, `ai_experiments/core/logger.py`, `tests/test_logger.py`
- Modify: `pyproject.toml` (add `structlog`), `ai_experiments/daemon.py:290`, `examples/toy_train.py:32,43`, `tests/test_orchestrator.py`, `tests/test_tracking.py`, `tests/conftest.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `ai_experiments.core.logger.get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger`, used by Tasks 7 and 10.

Read `.agents/rules/core-logger.md`, `.agents/rules/log-get-logger.md`, `.agents/rules/log-no-print.md`, and the drop-in at `.agents/snippets/core/logger.py`.

The snippet is built on **structlog**, which this repo does not yet depend on. Add it. `general-respect-local-repo` protects an *existing* local contract; this repo has no logging module at all, so there is nothing to respect — CES-74 is the contract here, and a stdlib reimplementation would diverge from every other house repo. `ai_experiments/core/` does not exist yet; this task creates it.

- [ ] **Step 1: Add the dependency without disturbing the shared venv**

Add `"structlog>=25.1"` to `[project].dependencies` in `pyproject.toml`, then:

```bash
uv lock                          # updates uv.lock only — the uv-lock hook requires this
.venv/bin/python -m pip install structlog
```

`uv lock` does not touch the virtualenv, and `pip install` into the existing venv is additive. **Do not run `uv sync`** — four other worktrees share this tree and a sync prunes packages another branch needs.

- [ ] **Step 2: Write the failing test**

```python
# tests/test_logger.py
"""The house logger is the only logging entry point (CES-74, CES-45)."""

import structlog

from ai_experiments.core.logger import get_logger


def test_get_logger_returns_a_bound_structlog_logger():
    log = get_logger("ai_experiments.daemon")
    assert isinstance(log, structlog.stdlib.BoundLogger)


def test_events_are_emitted_as_key_value_pairs(capsys):
    get_logger("ai_experiments.daemon").info("run_finished", run_id="abc123")
    captured = capsys.readouterr()
    assert "run_finished" in captured.out
    assert "abc123" in captured.out


def test_bind_carries_context_into_the_event(capsys):
    get_logger("ai_experiments.daemon").bind(campaign_id="c1").info("tick")
    assert "c1" in capsys.readouterr().out
```

- [ ] **Step 3: Run it and watch it fail**

Run: `.venv/bin/python -m pytest tests/test_logger.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ai_experiments.core'`.

- [ ] **Step 4: Copy the snippet in verbatim**

```bash
mkdir -p ai_experiments/core
touch ai_experiments/core/__init__.py
cp .agents/snippets/core/logger.py ai_experiments/core/logger.py
```

Copy it unchanged. It already carries `# ast-grep-ignore: settings-module` on its two bootstrap `os.getenv` calls (logging configures before settings exists), so it does not fight Task 9. The only permitted edit is the package path in its docstring.

- [ ] **Step 5: Run the test**

Run: `.venv/bin/python -m pytest tests/test_logger.py -v`
Expected: PASS (3 tests).

- [ ] **Step 6: Replace the three prints**

`ai_experiments/daemon.py:290` is library code — it becomes a logger call:

```python
# top of module
from ai_experiments.core.logger import get_logger

log = get_logger(__name__)

# at the call site — key/value pairs, not an f-string
log.info("tick_complete", actions=len(report.actions))
```

`examples/toy_train.py:32,43` is a standalone example script whose stdout *is* its output. CES-46 governs library code; an example that logs instead of printing teaches the wrong thing. Keep the prints, suppress each visibly:

```python
print(f"epoch {epoch} loss {loss:.4f}")  # ast-grep-ignore: log-no-print  # example script, stdout is the artifact
```

- [ ] **Step 7: Fix the cross-test imports (3 pyrefly `missing-import` errors)**

`tests/test_orchestrator.py` and `tests/test_tracking.py` import helpers by bare module name, which resolves only because pytest injects rootdir into `sys.path` — pyrefly cannot see it. Move the shared helpers into `tests/conftest.py` as fixtures. Prefer fixtures over a `tests/helpers.py`: `test-in-memory-adapters` (CES-64) wants shared fakes reachable as fixtures, and `no-utils` (CES-63) would reject a grab-bag module name anyway.

- [ ] **Step 8: Verify and commit**

```bash
uvx --from ast-grep-cli ast-grep scan 2>&1 | grep -c log-no-print          # expect 0
uvx pyrefly check --config pyproject.toml 2>&1 | grep -c missing-import    # expect 0
.venv/bin/python -m pytest tests -q 2>&1 | tail -1                         # expect 2 failed, 122 passed
git add -A && git commit -m "refactor: adopt the house structlog logger and drop library prints (CES-74, CES-45, CES-46)"
```

---

### Task 8: CES-67 typed CLI for the worker (2 sites)

**Files:**
- Modify: `ai_experiments/worker.py:3` (and its `main()` / arg plumbing), `examples/toy_train.py:10`
- Test: `tests/test_run_command.py` (or a new `tests/test_worker_cli.py` if no worker-invocation test exists)

**Interfaces:**
- Consumes: nothing.
- Produces: no new public names. `python -m ai_experiments.worker <args>` must accept exactly the arguments it accepts today.

Read `.agents/rules/cli-typed-framework.md`. CES-67 is a **warning**, deliberately: "modern tooling is encouraged, not mandated… a nudge to adopt the house pattern on the next touch." The repo already uses Typer for its real CLI, so `worker.py`'s `argparse` is the odd one out, and the user asked for the unapplied standards to be applied.

- [ ] **Step 1: Pin the worker's argument contract**

`worker.py` is spawned as a subprocess by the backends — its argv is a wire protocol between processes, and breaking it breaks run submission silently. Find every spawn site before touching it:

```bash
grep -rn "ai_experiments.worker\|worker.py" ai_experiments/ tests/
```

Write a test that invokes it exactly the way the backend does, with the real flags, and assert the observable result (the run directory it writes, the exit code). Run it against the unchanged `argparse` implementation and confirm it passes — this is the characterization gate for the rewrite.

- [ ] **Step 2: Convert to Typer**

Typer is already a dependency, so there is no new dependency decision. Derive the parser from the signature:

```python
import typer

app = typer.Typer(add_completion=False)


@app.command()
def main(
    manifest: Path = typer.Option(..., "--manifest", help="Path to the run manifest."),
    run_dir: Path = typer.Option(..., "--run-dir", help="Directory to write run state into."),
) -> None:
    """Execute one run and stream its metrics into the store."""
    ...


if __name__ == "__main__":
    app()
```

Copy the flag **names and defaults verbatim** from the existing `add_argument` calls. A renamed flag is a broken spawn. `typer.Option` is already covered by the `extend-immutable-calls` config from Task 2, so this introduces no new B008.

- [ ] **Step 3: Run the characterization test**

Run: `.venv/bin/python -m pytest tests/test_run_command.py -v`
Expected: PASS, unchanged.

- [ ] **Step 4: Leave the example alone, visibly**

`examples/toy_train.py:10` is a standalone training script a user copies and adapts — its `argparse` is deliberately dependency-free, and the module docstring sells exactly that ("Works on the local backend and on any Ray cluster"). Making the example depend on Typer would make it a worse example. Suppress with the reason:

```python
import argparse  # ast-grep-ignore: cli-typed-framework  # standalone example: stays dependency-free
```

- [ ] **Step 5: Verify and commit**

```bash
uvx --from ast-grep-cli ast-grep scan 2>&1 | grep -c cli-typed-framework   # expect 0 beyond the justified ignore
.venv/bin/python -m pytest tests -q 2>&1 | tail -1
git add -u && git commit -m "refactor(worker): replace argparse with Typer (CES-67)"
```

---

### Task 9: CES-76 settings module (12 sites)

**Files:**
- Create: `ai_experiments/settings.py`, `tests/test_settings.py`
- Modify: `pyproject.toml` (add `pydantic-settings`), `ai_experiments/backends/local.py:64`, `ai_experiments/backends/ray.py:32`, `ai_experiments/clusters.py:53`, `ai_experiments/notify.py:40,41`, `ai_experiments/report.py:37`, `ai_experiments/store/filesystem.py:54`, `ai_experiments/tracking.py:55,69,70`, `ai_experiments/worker.py:47`, `examples/toy_train.py:38`

**Interfaces:**
- Consumes: nothing.
- Produces: `ai_experiments.settings.get_settings() -> Settings`, an `lru_cache`d accessor. **Call sites use `get_settings()`, never `Settings()`** — CES-76 names ad-hoc construction at call sites as the anti-pattern.

Read `.agents/rules/settings-module.md` and `.agents/snippets/settings.py`. The snippet uses `pydantic-settings`; `pydantic>=2.6` is already a dependency, so this is an in-family addition, not a new paradigm.

- [ ] **Step 1: Add the dependency**

Add `"pydantic-settings>=2.7"` to `[project].dependencies`, then `uv lock` and `.venv/bin/python -m pip install pydantic-settings`. Same constraint as Task 7 Step 1: never `uv sync` in this tree.

- [ ] **Step 2: Inventory the real variables before writing a single field**

```bash
uvx --from ast-grep-cli ast-grep scan --json=compact 2>/dev/null | python3 -c "
import json, sys
for x in json.load(sys.stdin):
    if x['ruleId'] == 'settings-module':
        print(x['file'], x['range']['start']['line'] + 1, x['lines'].strip())
"
```

Record each variable name, its inline default, and its type. Do not invent fields and do not rename variables — a rename is a deployment break, and this task is behavior-stable.

Cross-check against `.env.schema`, the committed contract, which must list every variable this task centralizes.

> **Blocked sub-step — raise it, do not route around it.** The agent cannot read `.env.schema`: `.claude/settings.json` denies `Read(./.env.*)`, and that deny overrides its own `Read(./.env.schema)` allow (an upstream template defect). Ask the user to paste the file or to narrow the deny pattern. If no answer arrives, finish the rest of the task and report the reconciliation as outstanding — never guess its contents.

- [ ] **Step 3: Write the failing test**

Tests are exempt from CES-76, so they may set env directly. `get_settings` is cached, so any test that changes env must clear it.

```python
# tests/test_settings.py
import pytest

from ai_experiments.settings import Settings, get_settings


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_defaults_do_not_require_any_environment(monkeypatch):
    monkeypatch.delenv("IAX_RUNS_DIR", raising=False)
    assert get_settings().runs_dir.name == "runs"


def test_environment_overrides_the_default(monkeypatch, tmp_path):
    monkeypatch.setenv("IAX_RUNS_DIR", str(tmp_path / "elsewhere"))
    assert get_settings().runs_dir == tmp_path / "elsewhere"


def test_binding_is_case_insensitive(monkeypatch, tmp_path):
    monkeypatch.setenv("iax_runs_dir", str(tmp_path / "lower"))
    assert get_settings().runs_dir == tmp_path / "lower"


def test_get_settings_is_cached():
    assert get_settings() is get_settings()


def test_unknown_keys_are_ignored_not_fatal(monkeypatch):
    monkeypatch.setenv("IAX_NOT_A_SETTING", "1")
    Settings()  # asserts no raise: model_config sets extra="ignore"
```

Replace `IAX_RUNS_DIR` / `runs_dir` with the real names from Step 2, and set `env_prefix` to the prefix those variables actually use. If they share no prefix, use `env_prefix=""` and name the fields after the full variable names — do not rename the variables to fit a prefix.

- [ ] **Step 4: Run it and watch it fail**

Run: `.venv/bin/python -m pytest tests/test_settings.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ai_experiments.settings'`.

- [ ] **Step 5: Implement from the snippet**

Base `ai_experiments/settings.py` on `.agents/snippets/settings.py`: one typed field per variable from Step 2, each current inline default preserved exactly, `case_sensitive=False`, `env_file=".env"`, and the `@lru_cache def get_settings()` accessor. Export `Settings` too, so tests can construct it directly.

- [ ] **Step 6: Run the test**

Run: `.venv/bin/python -m pytest tests/test_settings.py -v`
Expected: PASS (5 tests).

- [ ] **Step 7: Convert the 11 in-package call sites**

```python
# before
runs_dir = Path(os.environ.get("IAX_RUNS_DIR", "runs"))

# after
from ai_experiments.settings import get_settings

runs_dir = get_settings().runs_dir
```

Read each site before converting: several read env *inside* a function precisely so a test can monkeypatch it. Those tests now need `get_settings.cache_clear()`. Find them first:

```bash
grep -rn "monkeypatch.setenv\|os.environ\[" tests/ | head -30
```

Fix each affected test in the same commit as the call site it covers — a settings refactor whose tests pass only by accident is worse than no refactor.

- [ ] **Step 8: Handle the example**

`examples/toy_train.py:38` sits outside the package. Suppress rather than import package settings into a standalone script:

```python
epochs = int(os.environ.get("TOY_EPOCHS", "3"))  # ast-grep-ignore: settings-module  # standalone example
```

- [ ] **Step 9: Verify and commit**

```bash
uvx --from ast-grep-cli ast-grep scan 2>&1 | grep -c settings-module   # expect 0 beyond the justified ignores
.venv/bin/python -m pytest tests -q 2>&1 | tail -1
git add -A && git commit -m "refactor: centralize environment reads behind get_settings (CES-76)"
```

---

### Task 10: CES-79 no raw dicts at boundaries (41 sites, 20 files)

The real refactor. Read `.agents/rules/no-dict.md` and `.agents/snippets/no-dict-boundary.py` before starting. Two sub-rules, different work:

- `no-dict-return-annotation` (24) — the signature says `dict`. Mostly mechanical once the model exists.
- `no-dict-literal-return` (17) — a dict literal is constructed and returned. Needs the model *designed*.

Work **file by file**, not rule by rule: a file's annotation and its literal are the same boundary, and splitting them leaves the file half-migrated.

**Files:**
- Modify: `ai_experiments/schemas.py` (gains the models), then in this order:
  1. `ai_experiments/server/app.py` — 12 sites (9 annotation, 3 literal). Biggest win: these are HTTP responses, so the models double as the API contract.
  2. `ai_experiments/tracking.py` — 6 sites
  3. `ai_experiments/clusters.py` — 6 sites
  4. `ai_experiments/backends/ray.py` — 3 sites
  5. `ai_experiments/planner/` (`analysis.py` ×2, `search_space.py` ×2, `strategies.py` ×1)
  6. `ai_experiments/notify.py`, `repro.py`, `report.py`, `schemas.py:330` — 1 each
  7. `tests/` — 5 sites (`conftest.py`, `test_ray_backend.py` ×2, the two integration tests)

**Interfaces:**
- Consumes: `ai_experiments.settings` (Task 9) where a boundary reads config.
- Produces: named models in `ai_experiments/schemas.py`. Later tasks import them by name; keep names domain-shaped (`RunSummary`, `ClusterStatus`, `TrackingHandle`), never `XxxDict` or `XxxResult`.

- [ ] **Step 1: Pin the HTTP contract with tests before touching the server**

`server/app.py` is the riskiest file: its dict returns are the JSON API. Before changing it, assert the current response shape so the refactor cannot silently change it.

```python
# tests/test_server.py — add
def test_run_status_response_shape_is_stable(client, seeded_run):
    body = client.get(f"/runs/{seeded_run}").json()
    assert set(body) == {"run_id", "status", "metrics", "started_at", "finished_at"}
```

Replace the key set with the keys the endpoint actually returns today — read them off a live response, do not guess:

Run: `.venv/bin/python -m pytest tests/test_server.py -q` and inspect, or add a temporary `print(body)`.

- [ ] **Step 2: Run the new test against unchanged code**

Run: `.venv/bin/python -m pytest tests/test_server.py -v`
Expected: PASS. This is a characterization test — it must pass *before* the refactor, then keep passing after.

- [ ] **Step 3: Model one boundary**

```python
# ai_experiments/schemas.py
class RunSummary(BaseModel):
    """One run as returned by the runs API and the CLI's `status` command."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    status: RunStatus
    metrics: dict[str, float] = Field(default_factory=dict)
    started_at: datetime | None = None
    finished_at: datetime | None = None
```

`metrics: dict[str, float]` stays a dict — CES-79 governs *boundaries*, not every mapping. A homogeneous key→value map with no fixed schema is data, not a boundary. Read `.agents/rules/no-dict.md` for where that line sits, and if a site is genuinely a map, suppress it with `# ast-grep-ignore: no-dict-return-annotation  # homogeneous metric map, not a boundary` rather than inventing a model with dynamic fields.

- [ ] **Step 4: Convert the call site**

```python
# before
@app.get("/runs/{run_id}")
def get_run(run_id: str) -> dict:
    return {"run_id": run_id, "status": run.status, ...}

# after
@app.get("/runs/{run_id}")
def get_run(run_id: str) -> RunSummary:
    return RunSummary(run_id=run_id, status=run.status, ...)
```

FastAPI serializes the model to the same JSON, so the characterization test from Step 1 keeps passing unchanged. If it fails, the shape *did* change — fix the model, not the test.

- [ ] **Step 5: Run the tests for that file**

Run: `.venv/bin/python -m pytest tests/test_server.py -v`
Expected: PASS, including the Step 1 characterization test.

- [ ] **Step 6: Commit that file, then repeat Steps 3–5 for each remaining file**

```bash
git add -A && git commit -m "refactor(server): return typed models instead of raw dicts (CES-79)"
```

One commit per file from the ordered list. Each commit is independently revertible — that is the whole reason for the ordering.

- [ ] **Step 7: Verify the rule is clean**

```bash
uvx --from ast-grep-cli ast-grep scan 2>&1 | grep -cE 'no-dict'   # expect 0, or only justified suppressions
.venv/bin/python -m pytest tests -q 2>&1 | tail -1                # expect 2 failed, 119 passed
```

---

### Task 11: CES-71 split `cli.py` (801 lines, limit 700)

**Files:**
- Delete: `ai_experiments/cli.py`
- Create: `ai_experiments/cli/__init__.py`, `ai_experiments/cli/runs.py`, `ai_experiments/cli/campaigns.py`, `ai_experiments/cli/clusters.py`
- Modify: `pyproject.toml` if the wheel's package list needs it

**Interfaces:**
- Consumes: `ai_experiments.settings` (Task 9), `ai_experiments.core.logger` (Task 7), the models from Task 10.
- Produces: `ai_experiments.cli:app` — unchanged import path, unchanged CLI surface.

The file already has three Typer apps (`app`, `campaign_app`, `cluster_app`), so the seam is drawn for you (`arch-vocabulary`: this is the existing seam, not a new abstraction).

- [ ] **Step 1: Pin the CLI surface first**

```python
# tests/test_cli.py — add
def test_cli_command_surface_is_stable():
    from typer.main import get_command

    from ai_experiments.cli import app

    names = sorted(get_command(app).commands)
    assert names == [
        "artifacts", "campaign", "cancel", "cluster", "daemon", "diagnose",
        "escalations", "leaderboard", "logs", "metrics", "monitor", "repro",
        "rerun", "run", "runs", "serve", "status", "submit", "validate",
    ]
```

Correct the list against the real output before committing — run `.venv/bin/python -c "from typer.main import get_command; from ai_experiments.cli import app; print(sorted(get_command(app).commands))"` and paste what it prints.

- [ ] **Step 2: Run it against the unsplit file**

Run: `.venv/bin/python -m pytest tests/test_cli.py -v`
Expected: PASS. Characterization test, same contract as Task 10 Step 1.

- [ ] **Step 3: Move, do not rewrite**

```bash
mkdir ai_experiments/cli
git mv ai_experiments/cli.py ai_experiments/cli/runs.py
```

Then cut the `campaign_*` commands into `campaigns.py` and the `cluster_*` commands into `clusters.py`, moving the lines verbatim. No renaming, no logic edits, no signature changes in this task — `py-no-formatter-churn` and reviewability both depend on this being a pure move.

- [ ] **Step 4: Assemble in `__init__.py`**

```python
"""The `iax` command-line interface.

Entry point for `[project.scripts] iax = "ai_experiments.cli:app"`.
"""

import typer

from ai_experiments.cli.campaigns import campaign_app
from ai_experiments.cli.clusters import cluster_app
from ai_experiments.cli.runs import app

app.add_typer(campaign_app, name="campaign")
app.add_typer(cluster_app, name="cluster")

__all__ = ["app"]
```

Check how `add_typer` is called in the current `cli.py` and preserve the exact names and help strings.

- [ ] **Step 5: Verify the surface and the entrypoint**

```bash
.venv/bin/python -m pytest tests/test_cli.py -v          # the Step 1 test must still pass
.venv/bin/python -c "from ai_experiments.cli import app; print(app)"
.venv/bin/iax --help                                      # if installed; else: .venv/bin/python -m ai_experiments.cli --help
uvx prek run --all-files file-size-guard                  # expect: Passed
```

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "refactor(cli): split the command module into a package (CES-71)"
```

- [ ] **Step 7: Decide on `orchestrator.py` (412 lines)**

This is a *warning*, not a failure — the guard errors at 700 and warns at 400. Do not split it reflexively. Apply the deletion test from `arch-deep-modules`: if `CampaignOrchestrator` is one coherent module with a narrow interface, 412 lines is fine and the warning is noise. Record the decision in the PR description either way.

---

### Task 12: CES-110 cognitive complexity (6 functions)

**Files:**
- Modify: `ai_experiments/server/app.py` (`create_app`, 29), `ai_experiments/monitoring/rules.py` (`_suspicious_reasons`, 26), `ai_experiments/daemon.py` (`MonitorDaemon._check_runs`, 21), `ai_experiments/cli/runs.py` (`run_goal`, 20), `ai_experiments/orchestrator.py` (`CampaignOrchestrator._stop_reason`, 20), `ai_experiments/report.py:54` (`parse_metric_line`, 16)

**Interfaces:**
- Consumes: the models from Task 10, the package layout from Task 11.
- Produces: every function at or under complexity 15.

Ceiling is 15. These also clear the 4 remaining ruff `C901` findings.

- [ ] **Step 1: Take `_suspicious_reasons` (26) first**

It is the clearest case of `spaghetti-mixed-orchestration` (CES-8): a chain of independent heuristics in one function. Extract one predicate per reason, then collect:

```python
def _suspicious_reasons(run: Run, policy: MonitorPolicy) -> list[str]:
    """Return every reason `run` looks suspicious under `policy`."""
    checks = (
        _stale_heartbeat_reason,
        _no_metric_progress_reason,
        _excessive_restart_reason,
    )
    return [reason for check in checks if (reason := check(run, policy)) is not None]
```

Each `_*_reason` returns `str | None` and is independently testable. Write a test per predicate before extracting it — that is the point of the split.

- [ ] **Step 2: Run the monitoring tests**

Run: `.venv/bin/python -m pytest tests/test_rules.py tests/test_daemon.py -v`
Expected: PASS, unchanged count.

- [ ] **Step 3: Commit, then repeat for the other five**

`create_app` (29) is usually route registration plus wiring — extract per-concern registration helpers (`_register_run_routes(app, store)`), not a generic loop. `run_goal` (20) mixes CLI parsing, orchestration and output: push orchestration into `ai_experiments/orchestrator.py` and leave the command thin (CES-8).

- [ ] **Step 4: Verify the ceiling**

```bash
uvx complexipy --max-complexity-allowed 15 ai_experiments tests >/dev/null 2>&1; echo "exit=$?"   # expect exit=0
uvx ruff@0.15.22 check 2>&1 | grep -c C901                                                        # expect 0
```

---

### Task 13: Full green and the PR

**Files:**
- Modify: none (verification only)

**Interfaces:**
- Produces: a green `prek run --all-files` and an open PR.

- [ ] **Step 1: The whole suite**

```bash
uvx prek run --all-files; echo "exit=$?"
```
Expected: `exit=0`. If `ruff format` or `ruff check --fix` modify anything at this point, commit that diff — it means an earlier task left unformatted code.

- [ ] **Step 2: The tests**

Run: `.venv/bin/python -m pytest tests -q 2>&1 | tail -1`
Expected: `2 failed, 119 passed, 6 skipped` — the same two pre-existing MLflow integration failures, no new ones, and no lost passes.

- [ ] **Step 3: Install the hooks now that they are green**

```bash
prek install -t pre-commit -t commit-msg
```

Deliberately last: installing earlier would have blocked every commit in this plan.

- [ ] **Step 4: Open the PR**

```bash
git push -u origin feat/house-standards
gh pr create --title "refactor: adopt house engineering standards" --body-file -
```

The body must carry: the SHA of the Task 3 formatting commit and rebase instructions for the 13 open PRs; the `orchestrator.py` decision from Task 11 Step 7; every `ast-grep-ignore` and `noqa` added, with its justification; and the note that the two MLflow integration failures predate this branch.

- [ ] **Step 5: Confirm CI**

Run: `gh pr checks --watch`
Expected: `tests`, `zizmor`, `osv-scanner`, `dependency-review`, `commit-policy`, `conventional-commits` all pass. `tests.yml` runs `uvx prek run --all-files`, so Step 1 already predicted this result.

---

## Risks

- **The formatting commit (Task 3) conflicts with all 13 open PRs.** Accepted deliberately on 2026-09-17. Mitigation is the rebase note in the PR body; there is no way to make a line-length change conflict-free.
- **Task 10 changes the HTTP response contract if a model is wrong.** The characterization tests in Step 1 are the only thing standing between this refactor and a silent API break. Do not skip them, and do not edit them to match new output.
- **The plan runs in the primary working tree**, shared with four other worktrees and other sessions. Every task commits before the next begins, so no task leaves a dirty tree for someone else to trip over.
- **`.env.schema` is unreadable to the agent** (Task 9 Step 2). That blocks only the reconciliation sub-step, not the task.
