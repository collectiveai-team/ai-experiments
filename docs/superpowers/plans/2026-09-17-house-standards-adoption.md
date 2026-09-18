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
| 15 files, 42 findings (Tasks 10 + 10b) | dict returns/annotations replaced or justified |

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
git commit -m "refactor: apply ruff fixes (safe autofixes plus reviewed unsafe fixes)"
```

The type is `refactor:`, not `style:`. Step 3 invites unsafe fixes, and two of the categories
ruff offers here are not stylistic: `TC001` changes when a first-party import is evaluated, and
`SIM105` changes the construct used to suppress an exception. `style:` is for changes with no
semantic effect; claiming it for this commit would understate what the commit contains.


---

### Task 5: Remaining ruff findings by category

95 findings remain after Task 4 (it over-delivered by applying reviewed unsafe fixes), each needing a decision. Work category by category, committing per category so a bad call is revertible on its own.

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
- `C901` (4 hits) overlaps **Task 12** — leave them. All four (`cli.py:391 run_goal`, `monitoring/rules.py:92 _suspicious_reasons`, `report.py:54 parse_metric_line`, `server/app.py:27 create_app`) are on Task 12's complexipy decomposition list, and Task 12 is where `ruff-check` stops being skipped.
- `TC003` (10 hits) was deferred here from Task 4, which skipped the rule whole rather than cherry-pick: 9 of the 10 moves are safe, but `ai_experiments/cli.py`'s would move `Path` into a `TYPE_CHECKING` block, and Typer resolves parameter annotations at runtime to build the CLI. Apply the 9; on the `cli.py` one use `# noqa: TC003  # Typer resolves this annotation at runtime`.
- `S110`/`S112` (try-except-pass/continue, 4 hits) hide failures. Either log at debug via the Task 7 logger, or add a justification comment. Do not delete the handler.

- [ ] **Step 7: Verify and commit**

```bash
uvx ruff@0.15.22 check          # expect: exactly the 4 C901 findings, nothing else (Task 12 removes those)
.venv/bin/python -m pytest tests -q 2>&1 | tail -1
git add -u && git commit -m "fix: resolve remaining ruff findings"
```

---

### Task 6: pyrefly type errors (18 remaining)

Task 2 removed the 3 `.agents/snippets/` errors; Task 7 removes the 3 cross-test `missing-import`s. This task clears the other 15, which are real type defects. All but one are in `tests/` — the production code is nearly clean, and the tests are where the type contract is being quietly bypassed.

**Files:**
- Modify: `ai_experiments/planner/strategies.py:116`, `ai_experiments/planner/analysis.py:43-44`, `tests/test_daemon.py:43,44`, `tests/test_manifest.py:16,17`, `tests/test_monitoring_v2.py:38,39`, `tests/test_ray_backend.py:163`, `tests/test_tracking.py:93,119,123,227`, `tests/test_e2e_campaign.py:87`
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

`ai_experiments/planner/analysis.py:43-44` is the third instance, and the clearest: `best_trial`
already filters `scored` to `t.objective_value is not None and math.isfinite(...)`, then carries
**two** blanket suppressions — `# type: ignore[operator]` on the lambda and `# type: ignore[arg-type]`
on the `min(...)` — because the narrowing cannot survive into a lambda. The filter is the proof;
the suppressions only hide it. Make the narrowing something the checker can see, delete both
`type: ignore` comments, and keep `best_trial`'s behaviour identical (`min` over a negated key
for `mode == "max"`).

Task 3's reviewer flagged this same line for a second reason: at 118 characters it is the only
line in the repo over the 100-column limit, escaping `ruff check` solely because E501 exempts
lines ending in `# type: ignore[...]`. Removing the suppression removes the exemption, so the
rewrite must land under 100 columns or `ruff check` will start failing on it.

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

Copy it, then make exactly three edits — no more. The snippet already carries
`# ast-grep-ignore: settings-module` on its two bootstrap `os.getenv` calls (logging configures
before settings exists), so it does not fight Task 9.

1. The package path in its docstring.
2. `_is_prod`'s single line is **104 columns** and would fail `ruff check` under this repo's
   `line-length = 100`. The suppression must stay trailing on the `os.getenv` line (CES-46's
   "Suppressing" section shows the same-line form), so split the statement instead:

   ```python
   def _is_prod() -> bool:
       env = os.getenv("ENV", "dev")  # ast-grep-ignore: settings-module
       return env.lower() in {"prod", "production"}
   ```

3. `logging.getLevelNamesMapping()` is Python 3.11+, and this repo declares
   `requires-python = ">=3.10"`. CES-30 (`general-respect-local-repo`) makes the repo's pin the
   binding contract, so use a 3.10-compatible lookup:

   ```python
   def _level() -> int:
       name = os.getenv("LOG_LEVEL", "INFO").upper()  # ast-grep-ignore: settings-module
       level = logging.getLevelName(name)  # 3.10-compatible; getLevelNamesMapping is 3.11+
       return level if isinstance(level, int) else logging.INFO
   ```

Both snippet defects are upstream bugs in `collectiveai-team/scaffolding` and should be reported
there.

- [ ] **Step 5: Run the test**

Run: `.venv/bin/python -m pytest tests/test_logger.py -v`
Expected: PASS (3 tests).

- [ ] **Step 6: Replace the three prints**

`ai_experiments/daemon.py:289` is **not** a stray diagnostic — it is `run_forever`'s output
contract. It emits the whole tick report as one JSON line with `flush=True`, and `iax daemon`
deliberately sends its human banner to stderr (`cli.py`, `typer.echo(..., err=True)`) to keep
stdout machine-parseable. Replacing it with `log.info("tick_complete", ...)` would drop the
report body (including `report.errors`) and, in dev mode, render colored console text over the
JSON stream. CES-46 exempts output where "stdout *is* the product"; this is that case. Keep the
print and suppress it visibly, the same way this task treats the example script:

```python
print(  # ast-grep-ignore: log-no-print  # run_forever's stdout is the daemon's JSON stream
    json.dumps(report.model_dump(mode="json")), flush=True
)
```

The logger module still lands, still ships with tests, and Task 10 consumes it.

`examples/toy_train.py:32,43` is a standalone example script whose stdout *is* its output. CES-46 governs library code; an example that logs instead of printing teaches the wrong thing. Keep the prints, suppress each visibly:

```python
print(f"epoch {epoch} loss {loss:.4f}")  # ast-grep-ignore: log-no-print  # example script, stdout is the artifact
```

- [ ] **Step 7: Fix the cross-test imports (3 pyrefly `missing-import` errors)**

The importers are `tests/test_daemon.py:103` (`from test_orchestrator import FakeBackend, _goal`)
and `tests/test_ray_backend.py:170,190` (`from test_tracking import FakeMlflowModule`) — the bare
module names resolve only because pytest injects rootdir into `sys.path`, which pyrefly cannot
see. The three shared helpers are therefore `FakeBackend` and `_goal` (defined in
`test_orchestrator.py`) and `FakeMlflowModule` (defined in `test_tracking.py`). Move the shared helpers into `tests/conftest.py` as fixtures. Prefer fixtures over a `tests/helpers.py`: `test-in-memory-adapters` (CES-64) wants shared fakes reachable as fixtures, and `no-utils` (CES-63) would reject a grab-bag module name anyway.

- [ ] **Step 8: Verify and commit**

```bash
uvx --from ast-grep-cli ast-grep scan 2>&1 | grep -c log-no-print          # expect 0
uvx pyrefly@latest check --config pyproject.toml 2>&1 | tail -1           # expect 0 errors
.venv/bin/python -m pytest tests -q 2>&1 | tail -1              # expect 2 failed, 123 passed, 6 skipped
git add pyproject.toml uv.lock ai_experiments tests examples
git commit -m "refactor: adopt the house structlog logger and drop library prints (CES-74, CES-45, CES-46)"
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

The worker takes exactly two flags today (`worker.py:142-143`), both required, no defaults:
`--run-id` and `--runs-dir`. Those two names are the wire protocol — `backends/local.py:53-62`
builds `[sys.executable, "-m", "ai_experiments.worker", "--run-id", run_id, "--runs-dir",
str(self.store.root)]`. A renamed flag is a silently broken spawn.

```python
import typer

app = typer.Typer(add_completion=False)


@app.command()
def main(
    run_id: str = typer.Option(..., "--run-id", help="Run id to supervise."),
    runs_dir: Path = typer.Option(..., "--runs-dir", help="Run store root."),
) -> None:
    """Supervise one run: spawn its workload and stream metrics into the store."""
    FilesystemRunStore(runs_dir).…


if __name__ == "__main__":
    app()
```

`runs_dir` may be typed `Path`: `FilesystemRunStore.__init__` accepts `str | Path | None`
(`store/filesystem.py:54`), so `Path` is a tightening, not a change. `run_id` stays `str`.
`typer.Option` is already covered by the `extend-immutable-calls` config from Task 2, so this
introduces no new B008.

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
uvx --from ast-grep-cli ast-grep scan 2>&1 | grep -c cli-typed-framework   # expect 0
.venv/bin/python -m pytest tests -q 2>&1 | tail -1     # expect 2 failed, 123+ passed, 6 skipped
git add ai_experiments examples tests
git commit -m "refactor(worker): replace argparse with Typer (CES-67)"
```

A visible `ast-grep-ignore` suppresses the finding, so the expected count is 0, not "0 beyond the
ignore". Use explicit paths, not `git add -u` — Step 1 may add a new test file, which `-u` skips.

---

### Task 9: CES-76 settings module (12 findings = 7 conversions + 5 suppressions)

**Files:**
- Create: `ai_experiments/settings.py`, `tests/test_settings.py`
- Modify (convert — 7 genuine config reads): `ai_experiments/notify.py:42,43`, `ai_experiments/clusters.py:55`, `ai_experiments/store/filesystem.py:55`, `ai_experiments/report.py:37`, `ai_experiments/tracking.py:59`, `ai_experiments/backends/ray.py:34`
- Modify (suppress — 5 sites that are not config reads): `ai_experiments/worker.py:53`, `ai_experiments/backends/local.py:64`, `ai_experiments/tracking.py:73,74`, `examples/toy_train.py:41`
- Modify: `pyproject.toml` (add `pydantic-settings`), `tests/conftest.py` (cache-clearing fixture), `tests/test_server.py`, `tests/test_ray_backend.py`, `tests/test_tracking.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `ai_experiments.settings.get_settings() -> Settings`, an `lru_cache`d accessor, plus the `Settings` class. **Call sites use `get_settings()`, never `Settings()`** — CES-76 names ad-hoc construction at call sites as the anti-pattern.

Read `.agents/rules/settings-module.md` and `.agents/snippets/settings.py`. The snippet uses `pydantic-settings`; `pydantic>=2.6` is already a dependency, so this is an in-family addition, not a new paradigm.

**This task is behavior-stable.** Nothing it touches may change what any variable resolves to.

- [ ] **Step 1: Add the dependency**

Add `"pydantic-settings>=2.7"` to `[project].dependencies`, then `uv lock` and `.venv/bin/python -m pip install pydantic-settings`. Same constraint as Task 7 Step 1: never `uv sync` in this tree.

- [ ] **Step 2: Know which findings are config, and which are not**

The inventory is already done — these are the twelve `settings-module` findings, verified against the live tree. Do not re-derive them, and do not invent fields.

**Seven genuine config reads → convert:**

| Site | Variable | Today's expression |
|---|---|---|
| `notify.py:42` | `IAX_NOTIFY_WEBHOOK` | `webhook_url or os.environ.get("IAX_NOTIFY_WEBHOOK")` |
| `notify.py:43` | `IAX_NOTIFY_COMMAND` | `command or os.environ.get("IAX_NOTIFY_COMMAND")` |
| `clusters.py:55` | `IAX_CLUSTERS` | `os.environ.get("IAX_CLUSTERS")`, then `if env: return Path(env)` |
| `store/filesystem.py:55` | `IAX_RUNS_DIR` | `Path(root or os.environ.get("IAX_RUNS_DIR", "outputs/experiments/runs"))` |
| `report.py:37` | `IAX_ARTIFACTS_DIR` | `os.environ.get("IAX_ARTIFACTS_DIR")`, then `if not value: return None` |
| `tracking.py:59` | `MLFLOW_TRACKING_URI` | `tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", "")` |
| `backends/ray.py:34` | `RAY_ADDRESS` | `os.environ.get("RAY_ADDRESS")`, then `if env_address and env_address.strip()` |

**Five findings that are not config reads → keep the call, add a visible suppression:**

- `worker.py:53` and `backends/local.py:64` are `env = os.environ.copy()`. They propagate the *whole parent environment* into a child process. There is no variable and no default here — nothing a `Settings` field could hold. Suppress:
  ```python
  env = os.environ.copy()  # ast-grep-ignore: settings-module  # child-process env propagation, not config
  ```
- `tracking.py:73,74` live inside `_file_store_optout`, which *writes* `MLFLOW_ALLOW_FILE_STORE` with `os.environ.setdefault(...)` because mlflow reads that variable out of `os.environ` inside its own library code, then reads it back to propagate the identical value into the workload's env. CES-76 governs reading config; a `setdefault` write is not a read a settings module can own, and routing line 74 through a cached `get_settings()` would break the docstring's stated contract ("An explicit MLFLOW_ALLOW_FILE_STORE=false set by the user is respected") — a value `setdefault` installs at call time is invisible to a settings object constructed earlier. Suppress both, citing that:
  ```python
  # mlflow reads this out of os.environ itself; we set it for mlflow and mirror it into the
  # workload env. Not app config, and a cached settings read would not see the setdefault.
  os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")  # ast-grep-ignore: settings-module
  return {"MLFLOW_ALLOW_FILE_STORE": os.environ["MLFLOW_ALLOW_FILE_STORE"]}  # ast-grep-ignore: settings-module
  ```
- `examples/toy_train.py:41` is a standalone example kept deliberately dependency-free — the same justification Task 8 already recorded on its `argparse` import. Suppress with the matching wording (Step 8).

Cross-check against `.env.schema`, the committed contract, which must list every variable this task centralizes.

> **Ruling (already made — do not stall on it).** `.env.schema` exists but is unreadable to agents: `.claude/settings.json` denies `Read(./.env.*)` and that deny shadows its own `Read(./.env.schema)` allow (an upstream template defect). This task centralizes **no new variable** — all seven already exist in the tree today — so `.env.schema` needs no edit and the reconciliation is a verification, not a change. Do not guess its contents, do not edit it, and do not ask. Note the unverified reconciliation in your report and move on.

- [ ] **Step 3: Write the failing test**

Tests are exempt from CES-76, so they may set env directly. `get_settings` is cached, so any test that changes env must clear it.

```python
# tests/test_settings.py
"""CES-76: every environment read in the package resolves through this one module."""

from __future__ import annotations

from ai_experiments.settings import Settings, get_settings


def test_defaults_do_not_require_any_environment(monkeypatch):
    monkeypatch.delenv("IAX_RUNS_DIR", raising=False)
    assert get_settings().runs_dir == "outputs/experiments/runs"


def test_environment_overrides_the_default(monkeypatch):
    monkeypatch.setenv("IAX_RUNS_DIR", "/elsewhere/runs")
    assert get_settings().runs_dir == "/elsewhere/runs"


def test_binding_is_case_insensitive(monkeypatch):
    monkeypatch.setenv("iax_runs_dir", "/lower/runs")
    assert get_settings().runs_dir == "/lower/runs"


def test_unprefixed_third_party_variables_bind_too(monkeypatch):
    monkeypatch.setenv("RAY_ADDRESS", "http://ray:8265")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "file:///mlruns")
    settings = get_settings()
    assert settings.ray_address == "http://ray:8265"
    assert settings.mlflow_tracking_uri == "file:///mlruns"


def test_get_settings_is_cached():
    assert get_settings() is get_settings()


def test_unknown_keys_are_ignored_not_fatal(monkeypatch):
    monkeypatch.setenv("IAX_NOT_A_SETTING", "1")
    Settings()  # asserts no raise: model_config sets extra="ignore"
```

`test_binding_is_case_insensitive` and `test_unprefixed_third_party_variables_bind_too` are the two that actually pin the config: they fail loudly if the alias wiring in Step 5 is wrong. If either fails, fix `settings.py` — never the test.

There is no `_clear_settings_cache` fixture in this file. Step 3b puts one in `tests/conftest.py` instead, where it protects the whole suite.

- [ ] **Step 3b: Clear the cache for every test, suite-wide**

`@lru_cache` makes `get_settings()` read the environment once per *process*. Every one of the seven sites reads env at *call* time today, and three test files rely on that by monkeypatching env mid-run: `tests/test_server.py:220,232` (`IAX_CLUSTERS`), `tests/test_ray_backend.py:127,129` (`RAY_ADDRESS`), `tests/test_tracking.py:73,74,89,102` (`MLFLOW_*`). Without a cache reset those tests would pass or fail depending on which test ran first — exactly the order-dependence CES-111 is about to start randomizing.

Add to `tests/conftest.py` (created in Task 7):

```python
@pytest.fixture(autouse=True)
def _clear_settings_cache():
    """CES-76: get_settings() is process-cached, so env-mutating tests must start clean."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
```

One autouse fixture covers every current and future test and needs no per-test edits. Prefer it over touching the eight monkeypatch sites individually.

**The accepted behavior change:** in production, env is now read once per process instead of on every call. That is the point of CES-76 and is deliberate. Say so in your report; do not smuggle it in.

- [ ] **Step 4: Run it and watch it fail**

Run: `.venv/bin/python -m pytest tests/test_settings.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ai_experiments.settings'`.

- [ ] **Step 5: Implement**

Two decisions the snippet cannot make for you, both load-bearing:

1. **`env_prefix` must be `""`, with an explicit `validation_alias` on every field.** The seven variables span three families: five `IAX_*`, plus `MLFLOW_TRACKING_URI` and `RAY_ADDRESS` — and those last two are third-party contracts that mlflow and Ray read from env themselves, so they cannot be renamed to fit a prefix. With `env_prefix=""` a field named `runs_dir` binds bare `RUNS_DIR`, *not* `IAX_RUNS_DIR`, silently. The alias is what makes it bind correctly.
2. **Every field is typed `str` or `str | None`** — exactly what `os.environ.get` returns today — and the conversion logic (`Path(...)`, `.strip()`, `if not value`) stays at the call site where it already is. Typing `runs_dir: Path` would change behavior: `IAX_ARTIFACTS_DIR=""` returns `None` today but would become `Path(".")`, and `IAX_CLUSTERS=""` is ignored today but would become `Path(".")` too. This task centralizes reads; it does not retype them.

```python
# ai_experiments/settings.py
"""CES-76 · the one module that reads the environment.

Every `os.getenv` / `os.environ` read in the package resolves here, through the cached
`get_settings()`. Fields are typed as the environment delivers them (`str`), and callers keep
whatever coercion they already did -- this module centralizes the reads without changing what
any of them mean.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,  # IAX_RUNS_DIR and iax_runs_dir both bind `runs_dir`
        extra="ignore",
    )

    # No env_prefix: MLFLOW_TRACKING_URI and RAY_ADDRESS are third-party contracts that cannot
    # be renamed, so every field names its variable outright.
    runs_dir: str = Field(default="outputs/experiments/runs", validation_alias="IAX_RUNS_DIR")
    artifacts_dir: str | None = Field(default=None, validation_alias="IAX_ARTIFACTS_DIR")
    clusters_config: str | None = Field(default=None, validation_alias="IAX_CLUSTERS")
    notify_webhook: str | None = Field(default=None, validation_alias="IAX_NOTIFY_WEBHOOK")
    notify_command: str | None = Field(default=None, validation_alias="IAX_NOTIFY_COMMAND")
    mlflow_tracking_uri: str = Field(default="", validation_alias="MLFLOW_TRACKING_URI")
    ray_address: str | None = Field(default=None, validation_alias="RAY_ADDRESS")


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings, constructed (and validated) once."""
    return Settings()
```

If `case_sensitive=False` turns out not to apply to `validation_alias` in the installed pydantic-settings version, `test_binding_is_case_insensitive` will say so. Fix it with `AliasChoices` covering both cases rather than dropping the test.

- [ ] **Step 6: Run the test**

Run: `.venv/bin/python -m pytest tests/test_settings.py -v`
Expected: PASS (6 tests).

- [ ] **Step 7: Convert the seven call sites**

Each conversion replaces only the `os.environ.get(...)` expression. The surrounding logic is the behavior contract — leave it byte-for-byte:

```python
# ai_experiments/store/filesystem.py:55
self.root = Path(root or get_settings().runs_dir)

# ai_experiments/report.py:37
value = get_settings().artifacts_dir

# ai_experiments/clusters.py:55
env = get_settings().clusters_config

# ai_experiments/notify.py:42-43
self.webhook_url = webhook_url or get_settings().notify_webhook
self.command = command or get_settings().notify_command

# ai_experiments/tracking.py:59
resolved = tracking_uri or get_settings().mlflow_tracking_uri

# ai_experiments/backends/ray.py:34
env_address = get_settings().ray_address
```

Import as `from ai_experiments.settings import get_settings` (CES-5: absolute imports, `settings` is a leaf module nothing else imports). Drop the now-unused `import os` only where nothing else in the file uses it — `worker.py`, `backends/local.py` and `tracking.py` still need it.

- [ ] **Step 8: Handle the example**

`examples/toy_train.py:41` sits outside the package. Suppress rather than import package settings into a standalone script — matching the justification Task 8 recorded on the same file's `argparse` import:

```python
    artifacts = os.environ.get("IAX_ARTIFACTS_DIR")  # ast-grep-ignore: settings-module  # standalone example: stays dependency-free
```

That line exceeds 100 columns with the comment attached; put the suppression comment on its own line above if `ruff check` complains, keeping `# ast-grep-ignore: settings-module` on the flagged line itself (ast-grep matches the comment on or immediately above the node — verify with a scan, do not assume).

- [ ] **Step 9: Verify and commit**

```bash
uvx --from ast-grep-cli ast-grep scan 2>&1 | grep -c settings-module            # expect 0
uvx ruff@0.15.22 check . 2>&1 | tail -1                                         # expect: Found 4 errors.
uvx ruff@0.15.22 format --check . 2>&1 | tail -1                                # expect: all formatted
uvx pyrefly@latest check --config pyproject.toml 2>&1 | tail -1                 # expect: 0 errors
uvx deptry@latest . 2>&1 | tail -1                                              # pydantic-settings must be declared
.venv/bin/python -m pytest tests -q 2>&1 | tail -1                              # expect: 2 failed, 131 passed, 6 skipped
git ls-files -m -o --exclude-standard | xargs awk 'length>100 {print FILENAME":"FNR}'   # expect: nothing
git add pyproject.toml uv.lock ai_experiments tests examples
git commit -m "refactor: centralize environment reads behind get_settings (CES-76)"
```

The 2 failures are the pre-existing `tests/integration/test_local_mlflow.py` ones; 125 + 6 new settings tests = 131. If any *other* test moves, a conversion changed behavior — find it, do not adjust the expectation.

---

### Task 10: CES-79 the HTTP contract and the two helpers it embeds (15 sites)

`no-dict` is enforced as an **error**, not a warning: with `C901` it is the last red gate. The 42
findings split across two tasks at the line where the risk changes. This one owns the twelve in
`ai_experiments/server/app.py` — those dict returns *are* the JSON API the dashboard consumes, the
only place in the 42 whose consumer is outside this repo — **plus the three sites in the two
helpers those handlers splice into their responses**, because a handler cannot be modeled while the
helper it embeds still returns a raw dict.

Read `.agents/rules/no-dict.md` first.

**Files:**
- Modify: `ai_experiments/server/app.py` (12 sites), `ai_experiments/planner/analysis.py`
  (`summarize_campaign`, line 54), `ai_experiments/repro.py` (`capture_repro` line 50,
  `read_repro` line 92)
- Modify (call sites only, no dict returns of their own): `ai_experiments/cli.py:242,268,310,575`,
  `ai_experiments/orchestrator.py:372,388`, `ai_experiments/tracking.py:115`
- Modify: `ai_experiments/schemas.py` (gains the models)
- Modify: `tests/test_server.py` (characterization tests first), `tests/test_artifacts_repro.py`
  (its `assert read_repro(run_dir) == context` at line 76 compares two of these values)

**Interfaces:**
- Consumes: `ai_experiments.settings.get_settings()` (Task 9) — do not reintroduce env reads.
- Produces: models in `ai_experiments/schemas.py`, named for the domain — never `XxxDict`,
  `XxxResult` or `XxxResponse`. Task 10b reuses them rather than defining near-duplicates.

**Most of the twelve are already models wearing a dict costume.** Five handlers build a real
pydantic model and then immediately `.model_dump(mode="json")` it:

| Line | Handler | What it actually returns |
|---|---|---|
| 49 | `run_detail` | `run_store.read_status(run_id)` — a `RunStatus` |
| 66 | `run_diagnosis` | `diagnose_run(...)` (`monitoring/rules.py:159`) |
| 156 | `campaign_stop` | `orchestrator.stop(...)` — a `CampaignState` |
| 163 | `campaign_pause` | `orchestrator.pause(...)` |
| 172 | `campaign_resume` | `orchestrator.resume(...)` |

For those five the change is: **delete the `.model_dump(mode="json")` and put the real model in the
return annotation.** No new model, no new field. FastAPI serializes the returned model through
`response_model`; the characterization tests in Step 1 are what prove that serialization matches
the hand-dumped JSON byte for byte.

The other four need a model written:

| Line | Handler | Shape |
|---|---|---|
| 38, 39 | `health` | `{"status", "runs_root"}` — trivial |
| 71, 74 | `run_cancel` | `{"run_id", "cancelled"}` — trivial |
| 91 | `run_repro` | `read_repro(...)`'s context, plus a `has_diff` key the handler sets at line 97 |
| 146, 150 | `campaign_detail` | `{"state": CampaignState, "summary": summarize_campaign(...)}` |

- [ ] **Step 1: Characterize every one of the nine endpoints before touching any of them**

This is the whole safety net; do not shortcut it. For each handler in the two tables above, add a
test to `tests/test_server.py` asserting the exact top-level key set of a real response:

```python
def test_health_response_shape_is_stable(client):
    body = client.get("/api/health").json()
    assert set(body) == {"status", "runs_root"}
```

Read the real key sets off live responses — print them, do not guess, and do not copy the example's
keys. Cover both branches wherever a handler answers differently on different paths (a found run vs.
a 404, `campaign_pause`'s success vs. its 409). For the five `.model_dump()` handlers assert on
**values too, not only keys**: the whole risk there is that FastAPI's serializer and
`model_dump(mode="json")` disagree on some field (a `datetime`, an enum, a `None`), which a key-set
assertion cannot see. Compare the full body against the dumped model:

```python
def test_run_detail_body_matches_the_stored_status(client, store, run_id):
    assert client.get(f"/api/runs/{run_id}").json() == store.read_status(run_id).model_dump(mode="json")
```

**Watch for the `None` trap:** FastAPI serializes an unset optional field to `null` rather than
omitting the key. `read_repro`'s context has conditionally-absent keys (`git_dirty` is set only when
a sha was found, `repro.py:57-70`), so a model with optional fields produces `{"git_dirty": null}`
where the dict produced nothing. That is a shape change the dashboard can see. Where you find one,
either keep the key absent (`response_model_exclude_none=True` on the route) or deliberately accept
the change and say so in your report — never let it pass unnoticed.

- [ ] **Step 2: Run the characterization tests against unchanged code**

Run: `.venv/bin/python -m pytest tests/test_server.py -v`
Expected: all PASS. They must pass *before* the refactor. Any that fail describe the code wrong —
fix the test, re-run, and only then change a handler.

```bash
git add tests/test_server.py
git commit -m "test(server): characterize the JSON response shapes (CES-79)"
```

- [ ] **Step 3: Drop `.model_dump()` from the five handlers that already return a model**

```python
# before
@app.get("/api/runs/{run_id}")
def run_detail(run_id: str) -> dict[str, Any]:
    _ensure_run(run_store, run_id)
    return run_store.read_status(run_id).model_dump(mode="json")

# after
@app.get("/api/runs/{run_id}")
def run_detail(run_id: str) -> RunStatus:
    _ensure_run(run_store, run_id)
    return run_store.read_status(run_id)
```

`RunStatus` already exists at `ai_experiments/schemas.py:149`. Do not invent a new model for any of
these five; import the one the callee already returns.

Run `.venv/bin/python -m pytest tests/test_server.py -q` after each. If a body changed, the
serializers disagree — fix the route (`response_model_exclude_none`, a field serializer), never the
test.

- [ ] **Step 4: Write models for `health` and `run_cancel`**

```python
# ai_experiments/schemas.py
class HealthStatus(BaseModel):
    """The /api/health body."""

    model_config = ConfigDict(extra="forbid")

    status: str
    runs_root: str


class CancelAck(BaseModel):
    """The /api/runs/{run_id}/cancel body."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    cancelled: bool
```

```python
@app.get("/api/health")
def health() -> HealthStatus:
    return HealthStatus(status="ok", runs_root=str(run_store.root))
```

Run the server tests. Commit Steps 3-4 together:
```bash
git add ai_experiments/schemas.py ai_experiments/server/app.py
git commit -m "refactor(server): return the models the handlers already build (CES-79)"
```

- [ ] **Step 5: Model `summarize_campaign` (`planner/analysis.py:54`) and update its six callers**

Its return is a fixed schema with two nested objects and a list — read the whole function
(`planner/analysis.py:54-100`) and model every key, including `budget`, `objective`, `history` and
the optional `best`. `trials_by_status` is a `status -> count` map with no fixed key set: that field
stays a raw `dict[str, int]` (only *returns* are flagged, not fields).

The six callers are `cli.py:310`, `cli.py:575`, `orchestrator.py:372`, `orchestrator.py:388`,
`server/app.py:111`, `server/app.py:152`. Read each one: those that subscript the result
(`summary["gpu_hours"]`) become attribute access; those that serialize it to JSON for the CLI
(`cli.py`) need an explicit `.model_dump(mode="json")` at the *print* site so the CLI's JSON output
does not change. `tests/test_planner.py` and the CLI tests that parse stdout as JSON are your check
that it did not.

- [ ] **Step 6: Model the repro context (`repro.py:50` and `:92`) and update its callers**

`capture_repro` builds the context and writes it to `repro/context.json`; `read_repro` parses that
same file back. One model serves both: `capture_repro` returns it, `read_repro` validates the parsed
JSON into it. Include `has_diff` as an optional field — `server/app.py:97` sets it on the way out,
and a model cannot take an undeclared key.

**`read_repro` returns `| None` and three of its callers rely on that**: `cli.py:268` and
`tracking.py:115` both write `read_repro(...) or {}` and then subscript. Convert those to an
explicit `is None` check with attribute access; a `or {}` against a model is a silent type error
that pyrefly will catch but the tests will not.

`tests/test_artifacts_repro.py:76` asserts `read_repro(run_dir) == context`. Pydantic models compare
by field value, so this keeps working once both sides are the same model — but run it and confirm
rather than assuming.

- [ ] **Step 7: Model `campaign_detail` and `run_repro`, the two composites**

Now that both helpers return models, these two are straightforward: a `CampaignDetail` with a
`state: CampaignState` and a `summary: CampaignSummary`, and `run_repro` returning the repro model
directly.

- [ ] **Step 8: Verify and commit**

```bash
uvx --from ast-grep-cli ast-grep scan 2>&1 | grep -E 'server/app.py|planner/analysis.py|repro.py'   # expect nothing
uvx pyrefly@latest check --config pyproject.toml 2>&1 | tail -1
.venv/bin/python -m pytest tests -q 2>&1 | tail -1
git add ai_experiments/schemas.py ai_experiments/server/app.py ai_experiments/planner/analysis.py \
        ai_experiments/repro.py ai_experiments/cli.py ai_experiments/orchestrator.py \
        ai_experiments/tracking.py tests/test_server.py tests/test_artifacts_repro.py
git commit -m "refactor: model the repro context and campaign summary (CES-79)"
```

Expected suite: `2 failed, <133 + your new tests> passed, 6 skipped`. The 2 failures are the
pre-existing `tests/integration/test_local_mlflow.py` ones. Any other movement is a regression.
pyrefly must stay at `0 errors` — it is not skipped for this task.

---

### Task 10b: CES-79 the remaining 27 sites

The rest, none of which crosses an HTTP boundary or a helper shared between modules. **Most of these
are suppressions, not conversions** — the rule's own "Suppressing (rare, must be visible)" section is
the operative part of this task, and getting the convert/suppress call right matters more than the
volume.

The line, from `.agents/rules/no-dict.md`: a raw dict is correct when the keys are **not a fixed
schema** — a `name -> object` registry, a sampled hyperparameter assignment, a third-party payload
passed through, an env-var mapping spliced into `os.environ`. It is wrong when the keys *are* a
fixed schema that a caller has to guess at.

**Files:** `ai_experiments/schemas.py`, `clusters.py`, `planner/search_space.py`,
`planner/strategies.py`, `notify.py`, `report.py`, `backends/ray.py`, `tracking.py`,
`tests/test_ray_backend.py`, `tests/integration/{conftest.py,test_local_mlflow.py,test_ray_mlflow.py}`

**Interfaces:** consumes the models Task 10 added to `schemas.py`; reuse them rather than defining
near-duplicates.

I have already ruled each site. Implement the ruling; if you believe one is wrong, say so in your
report rather than quietly doing the other thing.

**Convert — a fixed schema a caller has to guess at (7 sites):**

| Site | Why |
|---|---|
| `clusters.py:105` + its three return literals (`:108,:119,:126`) | `cluster_status` returns `name`/`reachable`/`address`/`ray_version`/`error` across three branches — one fixed schema with optional fields. One model, three constructions; make sure all three construct the *same* model rather than three near-identical ones. |
| `notify.py:45` | a fixed *core* (`timestamp`/`title`/`message`/`text`) that `**details` then splices arbitrary caller keys into, flattened at the top level. Model the core with `model_config = ConfigDict(extra="allow")` so the extras still serialize flat. **The flattening is a wire contract** -- `text` is the Slack-compatible field and the webhook POSTs this object verbatim -- so pin it with a test asserting that a `send(..., run_id="r1")` payload has `run_id` at the top level, not nested. If `extra="allow"` will not round-trip the extras into the POST body, suppress instead and say so in your report; do not nest them. |
| `report.py:91` | `{"step": ..., "values": ...}` — a fixed metric line. `values` stays a raw map *inside* the model. |

**Suppress — genuinely not a fixed schema (20 sites).** Each keeps its `dict` and gains a visible
`# ast-grep-ignore: <slug>` carrying a reason:

| Site | Reason the comment must give |
|---|---|
| `clusters.py:68,71` | `load_clusters` is a `profile name -> ClusterProfile` registry; the keys are the operator's profile names. |
| `planner/search_space.py:21,63`, `planner/strategies.py:113` | a sampled hyperparameter assignment; the keys are the user's own search-space parameter names. |
| `schemas.py:330` | `search_space_not_empty` is a pydantic field validator — its signature must return exactly the type it validates. |
| `backends/ray.py:143,231,233` | `_ray_details`/`_job_info_dict` pass Ray's own job-info keys through into the free-form `RunStatus.details` blob. |
| `tracking.py:64,73,78,168,178,192,200` | env-var mappings spliced into a subprocess environment; the consumer is `env.update(...)`, not a typed caller. |
| `tests/test_ray_backend.py:29,31`, `tests/integration/conftest.py:68`, `tests/integration/test_local_mlflow.py:93`, `tests/integration/test_ray_mlflow.py:70` | test fakes imitating a third-party API's dict shape; modeling them would make the fake diverge from the thing it fakes. |

A suppression whose comment just repeats the slug is not a justification. Say what the keys are and
where they come from. Re-measure the line numbers before you start — Task 10 edits `tracking.py`, so
its seven will have moved.

- [ ] **Step 1: Convert the seven, one file per commit**

For each: write the model in `schemas.py`, convert every return in that file, run that file's tests,
commit. `clusters.py` first (three branches that must all construct the same model), then
`notify.py`, then `report.py`.

Run after each: `.venv/bin/python -m pytest tests -q 2>&1 | tail -1` — the count must not move
except for tests you added.

- [ ] **Step 2: Suppress the twenty, one commit**

Verify placement as you go: ast-grep matches the flagged **token**, not the enclosing statement, so a
comment above a multi-line `return {` may not attach to it. This bit Task 9 in `tracking.py` — check
with a scan rather than assuming, and hoist the expression to its own line if that is what it takes.

```bash
git add -u && git commit -m "refactor: justify the raw-dict boundaries CES-79 does not govern"
```

- [ ] **Step 2b: Type `list_artifacts` at the store layer (carried over from Task 10)**

Task 10's review left `ArtifactEntry` as an `extra="forbid"` hand-mirror of an untyped producer:
`FilesystemRunStore.list_artifacts` (`store/filesystem.py:198-217`) returns raw dicts that the
server's `/api/runs/{run_id}/artifacts` handler feeds straight into the model. A fourth key added at
the store becomes a 500 at the boundary. ast-grep does not flag it (the annotation is
`list[dict[...]]`, a shape the rule does not match), so this is judgment, not a finding.

Return `list[ArtifactEntry]` from `list_artifacts` itself and let the handler pass it through. The
shape is already fixed and confirmed -- `modified_at` is an ISO string at the source, so nothing
leans on a datetime coercion. `tests/test_server.py::test_run_artifacts_body_shape_is_stable` writes
a real file through the real store, so it covers this hop; run it.

- [ ] **Step 3: Verify the gate is clean**

```bash
uvx --from ast-grep-cli ast-grep scan 2>&1 | grep -c 'no-dict'      # expect 0
uvx ruff@0.15.22 check . 2>&1 | tail -1                             # expect: Found 4 errors.
uvx pyrefly@latest check --config pyproject.toml 2>&1 | tail -1     # expect: 0 errors
.venv/bin/python -m pytest tests -q 2>&1 | tail -1
git ls-files '*.py' | xargs awk 'length>100 {print FILENAME":"FNR}' # expect: nothing
```

After this task ast-grep reports zero findings repo-wide, and the only `ruff check` errors left are
the four `C901` reserved for Task 12.

---

### Task 11: CES-71 split `cli.py` (755 lines, limit 700)

`file-size-guard` errors at 700 lines and warns at 400. `cli.py` is 755, so this is a hard gate.

**Files:**
- Delete: `ai_experiments/cli.py`
- Create: `ai_experiments/cli/__init__.py`, `cli/runs.py`, `cli/goals.py`, `cli/serving.py`,
  `cli/campaigns.py`, `cli/clusters.py`, `cli/__main__.py`
- Modify: `pyproject.toml` only if the wheel's package list enumerates modules (check; it may not)

**Interfaces:**
- Consumes: `ai_experiments.settings` (Task 9), `ai_experiments.core.logger` (Task 7), the models
  from Tasks 10 and 10b.
- Produces: `ai_experiments.cli:app` — **unchanged import path, unchanged CLI surface.**
  `[project.scripts] iax = "ai_experiments.cli:app"` (pyproject.toml:36) must keep resolving.

The file already builds three Typer apps (`app` at line 13, `campaign_app` at 19, `cluster_app` at
24) and wires them with bare `app.add_typer(campaign_app)` / `add_typer(cluster_app)` at lines 29-30
— **no `name=` argument; the names come from each `typer.Typer(name=...)` constructor.** Preserve
that exactly. Passing `name=` at the `add_typer` call instead would work but is a gratuitous change
to a line you are only moving.

Measured spans in the current file, so you can cut rather than hunt:

| Target file | What moves | Current lines | Size |
|---|---|---|---|
| `cli/runs.py` | `validate` … `leaderboard` (14 commands) | 44-343 | ~300 |
| `cli/serving.py` | `daemon`, `serve` | 346-391 | ~46 |
| `cli/goals.py` | `run_goal` (the `run` command) + `_start_dashboard_thread` | 394-488 | ~95 |
| `cli/campaigns.py` | `_orchestrator` + the 10 `campaign_*` commands | 491-698 | ~208 |
| `cli/clusters.py` | the 4 `cluster_*` commands | 701-755 | ~55 |
| `cli/__init__.py` | imports, the three `Typer(...)` constructors, `_echo_json`, `_backend_for_run`, the `add_typer` wiring | 1-41 | ~60 |

Cutting only campaigns and clusters would leave `runs.py` at 499 lines — under the 700 error but over
the 400 warning, i.e. the same design smell one file to the left. Do the full split.

**Naming note:** `ai_experiments/cli/clusters.py` sits alongside the existing
`ai_experiments/clusters.py`. That is legal and the import paths are unambiguous, but be careful
which one you are editing, and use absolute imports (`from ai_experiments.clusters import ...`) in
the new module so the intent is visible at the import line.

- [ ] **Step 1: Pin the CLI surface first**

```python
# tests/test_cli.py — add
def test_cli_command_surface_is_stable():
    """The split must not add, drop, or rename a single command."""
    from typer.main import get_command

    from ai_experiments.cli import app

    root = get_command(app)
    assert sorted(root.commands) == [
        "artifacts", "campaign", "cancel", "cluster", "daemon", "diagnose",
        "escalations", "leaderboard", "logs", "metrics", "monitor", "repro",
        "rerun", "run", "runs", "serve", "status", "submit", "validate",
    ]
    assert sorted(root.commands["campaign"].commands) == [
        "advance", "edit", "list", "pause", "resume", "start", "status",
        "stop", "suggest", "validate",
    ]
    assert sorted(root.commands["cluster"].commands) == ["down", "list", "status", "up"]
```

Those three lists are **verified against the live tree** — I ran `get_command` and captured them, so
use them as given. A sub-app is a `click.Group`, which is why `.commands` works on it.

- [ ] **Step 2: Run it against the unsplit file**

Run: `.venv/bin/python -m pytest tests/test_cli.py -v`
Expected: PASS. This is a characterization test — it must pass *before* the split.

```bash
git add tests/test_cli.py
git commit -m "test(cli): pin the command surface before the split (CES-71)"
```

- [ ] **Step 3: Move, do not rewrite**

```bash
mkdir ai_experiments/cli
git mv ai_experiments/cli.py ai_experiments/cli/runs.py
```

Then cut each block out of `runs.py` into its target file per the table, **moving the lines
verbatim**. No renaming, no logic edits, no signature changes, no reordering in this task — both
`py-no-formatter-churn` and this diff's reviewability depend on it being a pure move. The only new
code is the import headers each new module needs and the `__init__.py` below.

- [ ] **Step 4: Assemble in `__init__.py`**

The three `typer.Typer(...)` constructors move here, and each command module imports the app object
it decorates. That makes the modules import-for-side-effect, so `__init__.py` must import all five
*after* constructing the apps, and the imports need a `# noqa: E402` if they end up below other
statements. Keep the bare `add_typer` calls:

```python
app.add_typer(campaign_app)
app.add_typer(cluster_app)
```

Watch for a circular import: `cli/campaigns.py` importing `campaign_app` from `cli/__init__.py`
while `__init__.py` imports `cli/campaigns.py`. If it bites, put the three constructors and the two
shared helpers in a small `cli/app.py` that every command module imports, and let `__init__.py`
import only for registration. **Do not name that module `utils.py`, `helpers.py` or `common.py`** —
CES-63 forbids exactly those names and the `no-utils` prek hook will reject the commit.

- [ ] **Step 5: Add the missing `__main__` guard**

`python -m ai_experiments.cli <anything>` currently exits 0 and prints nothing — the module has no
`if __name__ == "__main__"` block, so only the `iax` console script works. Task 8 gave `worker.py`
such a guard, so the package's two entry points disagree today. Close it:

```python
# ai_experiments/cli/__main__.py
"""Allow `python -m ai_experiments.cli`, matching how `ai_experiments.worker` is launched."""

from ai_experiments.cli import app

if __name__ == "__main__":
    app()
```

Pin it, because nothing else would notice it regressing:

```python
def test_module_entry_point_runs():
    """`python -m ai_experiments.cli` must work, not silently exit 0 doing nothing."""
    result = subprocess.run(  # noqa: S603  # fixed argv, our own module
        [sys.executable, "-m", "ai_experiments.cli", "--help"],
        capture_output=True, text=True, timeout=30, check=False,
        env={**os.environ, "COLUMNS": "200"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Detached experiment runtime" in result.stdout
```

- [ ] **Step 6: Verify the surface, the entrypoint, and the gate**

```bash
.venv/bin/python -m pytest tests/test_cli.py -v
.venv/bin/python -c "from ai_experiments.cli import app; print(app)"
.venv/bin/iax --help
.venv/bin/python -m ai_experiments.cli --help
uvx prek run --all-files file-size-guard
wc -l ai_experiments/cli/*.py
```

Every new file must be under 400 lines, not merely under 700. `iax --help` and
`python -m ai_experiments.cli --help` must print the same command list.

- [ ] **Step 7: Commit**

```bash
git add ai_experiments/cli tests/test_cli.py && git rm --cached ai_experiments/cli.py 2>/dev/null
git commit -m "refactor(cli): split the command module into a package (CES-71)"
```

- [ ] **Step 8: Decide on `orchestrator.py` (412 lines) — a decision, not a split**

A *warning*, not a failure. Do not split it reflexively. Apply the deletion test from
`arch-deep-modules`: if `CampaignOrchestrator` is one coherent module behind a narrow interface, 412
lines is fine and the warning is noise. Record the decision and its reasoning in your report either
way; the PR description will carry it.

---

### Task 12: CES-110 cognitive complexity (6 functions)

**Files:**
- Modify: `ai_experiments/server/app.py` (`create_app`), `ai_experiments/monitoring/rules.py`
  (`_suspicious_reasons`), `ai_experiments/daemon.py` (`MonitorDaemon._check_runs`),
  `ai_experiments/cli/goals.py` (`run_goal`, after Task 11 moves it),
  `ai_experiments/orchestrator.py` (`CampaignOrchestrator._stop_reason`),
  `ai_experiments/report.py` (`parse_metric_line`)
- Test: `tests/test_rules.py`, `tests/test_daemon.py`, `tests/test_report.py`,
  `tests/test_orchestrator.py`, `tests/test_server.py`

**Interfaces:**
- Consumes: the models from Tasks 10 and 10b, the `cli/` package layout from Task 11.
- Produces: every function at or under cognitive complexity 15 and cyclomatic complexity 10.

**Two gates, not one.** `complexipy` measures *cognitive* complexity (ceiling 15); ruff `C901`
measures *cyclomatic* complexity (ceiling 10). They disagree about which functions fail — nesting
depth drives one, branch count the other. The union must clear. Scores measured at `cd997fe`:

| Function | File:line | ruff C901 (>10) | complexipy (>15) |
|---|---|---|---|
| `create_app` | `server/app.py:39` | 27 | 27 |
| `_suspicious_reasons` | `monitoring/rules.py:93` | 13 | 26 |
| `MonitorDaemon._check_runs` | `daemon.py:95` | — | 21 |
| `run_goal` | `cli.py:395` → `cli/goals.py` | 11 | 20 |
| `CampaignOrchestrator._stop_reason` | `orchestrator.py:258` | — | 19 |
| `parse_metric_line` | `report.py:56` | 11 | 16 |

**Re-measure before you start.** Task 11 moves `run_goal` into `cli/goals.py`, so its line number
is stale by construction, and every other row shifts if Task 11's split touched the file. Run both
gates first and work from your own reading:

```bash
uvx complexipy ai_experiments 2>&1 | sed 's/\x1b\[[0-9;]*m//g' | grep FAILED
.venv/bin/ruff check --select C901 --output-format=concise
```

**No behaviour changes.** Every one of these six is a pure restructuring. The existing tests are
your regression net and their counts must not fall: `test_rules.py` 2, `test_daemon.py` 7,
`test_monitoring_v2.py` 7, `test_server.py` 29, `test_report.py` 6, `test_orchestrator.py` 10.
None of them names a private function — they drive these through public seams (`diagnose_run`,
`create_app`, the CLI), which is CES-65 working as intended. Keep it that way: test the new
predicates through the same seams, or add direct tests only where a predicate has a branch the
seam cannot reach.

- [ ] **Step 1: `_suspicious_reasons` (26) — the clearest case**

The real signature, read from `ai_experiments/monitoring/rules.py:93`:

```python
def _suspicious_reasons(
    status: RunStatus,
    policy: MonitorPolicy,
    metrics: list[MetricPoint],
    events: list[RunEvent],
) -> list[str]:
```

It is `spaghetti-mixed-orchestration` (CES-8): independent heuristics inlined into one body, each
appending a reason string. The score is nesting depth, not branch count — which is why ruff only
scores it 13 while complexipy scores 26.

Two structural facts to preserve, both load-bearing:

1. `threshold` is computed once from `policy.stuck_after_minutes or status.details.get(
   "stuck_after_minutes", 30)` and used by three separate checks. Pass it in; do not recompute it
   per predicate, and do not let the `or` fallback drift.
2. The `if metrics: ... else: ...` split is **not** cosmetic. With metrics present it checks metric
   age and plateau; with none it falls back to status age, then event age, then the `no_run_events`
   case. Those are two different rule sets, not one set with a guard. Extracting
   `_reasons_with_metrics(...)` and `_reasons_without_metrics(...)` respects that; a flat list of
   predicates each re-testing `if metrics` does not, and will silently change behaviour.

The existing eleven reason strings (`status_error_present`, `ray_resource_starved`,
`ray_stuck_suspected`, `heartbeat_stale_for_{n}m`, `no_metric_progress_for_{n}m`,
`objective_plateau:{name}:{n}_points`, `no_status_update_for_{n}m`, `no_event_progress_for_{n}m`,
`no_run_events`) are an observable output — `diagnose_run` surfaces them and the CLI prints them.
Not one may change spelling or order.

- [ ] **Step 2: Run the monitoring tests**

Run: `.venv/bin/python -m pytest tests/test_rules.py tests/test_daemon.py tests/test_monitoring_v2.py -v`
Expected: PASS, 16 tests, unchanged.

- [ ] **Step 3: `_stop_reason` (19) — same shape, simpler**

`ai_experiments/orchestrator.py:258`. Four independent stop conditions, each returning a reason
string, evaluated in priority order: `target_reached`, `max_hours_exceeded`, `gpu_hours_exhausted`,
`budget_exhausted`. The nesting inside the first two (a three-way `and`, then a nested `if reached`;
a `tzinfo is None` normalisation, then a threshold compare) is what earns the 19.

Extract one `str | None` predicate per condition and compose them in the same order — first
non-`None` wins. **Priority order is observable**: a campaign that has both hit its target and
exhausted its budget reports `target_reached`, and `tests/test_orchestrator.py` will catch a
reordering.

Keep the naive-datetime guard (`created.replace(tzinfo=timezone.utc)`) with the `max_hours` check
that needs it. It exists because persisted `created_at` values may lack a timezone.

- [ ] **Step 4: `parse_metric_line` (16 / C901 11)**

`ai_experiments/report.py:56`. One parse followed by one coercion loop. The loop is the complexity:
for each key it distinguishes bool (skip), int/float (accept), and the string spellings of
non-finite values (`nan`, `inf`, `-inf`, `infinity`, `-infinity`). Extract it:

```python
def _coerce_metric_value(value: object) -> float | None:
    """One metric value as a float, or None when it is not a usable number.

    Bools are rejected before ints: `isinstance(True, int)` is True in Python, and a
    reported flag is not a measurement.
    """
```

That drops both scores at once and gives the non-finite string handling a test seam it does not
have today — `tests/test_report.py` has 6 tests and none reaches the `"infinity"` spellings.
Add one.

Two traps: the `isinstance(value, bool)` check **must** stay ahead of the int/float check, and
`lowered.replace("infinity", "inf")` is what makes `-infinity` parse — keep both.

- [ ] **Step 5: `_check_runs` (21)**

`ai_experiments/daemon.py:95`. One `for run_id in sorted(...)` loop whose body carries three
separate `try/except Exception` guards and two `continue`s. The score is loop-times-guard nesting,
not logic.

Split the body into `_check_one_run(self, run_id: str, report: TickReport) -> None` — the loop then
has no nesting at all — and lift the terminal-run branch into
`_sync_finished_run(self, run_id: str, status: RunStatus, report: TickReport) -> None`.

**The per-run isolation is the point of the function, not an accident.** The comment at
`daemon.py:99-101` says so: one unreadable `status.json` must not end the tick for every other run
being supervised. Every guard survives the split verbatim, and that comment moves with the code it
explains. A `continue` becomes an early `return` in the extracted method — check each one.

- [ ] **Step 6: `create_app` (27) — this is the APIRouter split, not a complexity refactor**

`ai_experiments/server/app.py:39`. It scores 27 because it nests **twenty route handlers inside one
function body**, closing over `store`. Its own control flow is unremarkable. So the fix is CES-17's
boundary layout — group the handlers into `APIRouter`s by resource and include them — and the score
falls out for free. Do not try to shave points any other way.

The handlers currently close over `store`; an `APIRouter` cannot, so each router module needs the
store injected. Use FastAPI's `Depends` with a module-level provider, or build the routers inside
factory functions (`def build_run_router(store) -> APIRouter`). The factory form is a smaller change
and keeps the wiring explicit — prefer it unless you find a reason not to, and say which you chose.

`tests/test_server.py` has 29 tests hitting these routes through `TestClient`. They must all pass
with no edit: if a route path, method, status code or response shape changes, the split is wrong.

- [ ] **Step 7: `run_goal` (20)**

In `ai_experiments/cli/goals.py` after Task 11. It mixes CLI argument handling, campaign
orchestration and output formatting. Push the orchestration into `ai_experiments/orchestrator.py`
and leave the command thin (CES-8). `tests/test_run_command.py` drives it end to end.

- [ ] **Step 8: Verify both ceilings**

```bash
uvx complexipy ai_experiments >/dev/null 2>&1; echo "complexipy exit=$?"   # expect 0
.venv/bin/ruff check --select C901 --output-format=concise                 # expect "All checks passed!"
.venv/bin/python -m pytest -q
uvx prek run --all-files --hook-stage pre-commit 2>&1 | tail -30
```

Commit each function separately, so a reviewer can reject one restructuring without rejecting all
six:

```bash
git commit -m "refactor(monitoring): extract one predicate per suspicion rule (CES-110)"
```

---

### Task 13: CES-111, full green, and the PR

**Files:**
- Modify: `pyproject.toml` (add `pytest-randomly` to the dev extra), `uv.lock`
- Modify: whatever the randomized test order exposes (see Step 2)

**Interfaces:**
- Produces: a green `prek run --all-files` and a PR *prepared* — see Step 6, which is a stop point.

- [ ] **Step 1: Adopt CES-111 (test order randomization)**

The last unapplied standard and the only `[dependency]` one. Verified on 2026-09-18:
`pytest-randomly` appears nowhere in `pyproject.toml` **and** `import pytest_randomly` raises
`ModuleNotFoundError`, so it is not merely unconfigured here — it is absent. Every "the suite is
green" result in this plan was therefore measured in pytest's *default* order, and none of them says
anything about order-independence. Read `.agents/rules/test-order-randomization-pytest-randomly.md`.

`[tool.pytest.ini_options]` currently sets `testpaths` and three markers and has no `addopts`, so
nothing is pinning a seed today. Add `"pytest-randomly>=3.15"` to
`[project.optional-dependencies].dev` (the group at `pyproject.toml:27-33`), then:

```bash
uv lock
.venv/bin/python -m pip install pytest-randomly
```

Do **not** run `uv sync` — five worktrees share this venv.

- [ ] **Step 2: Run the suite five times and fix what order exposes**

```bash
for i in 1 2 3 4 5; do .venv/bin/python -m pytest tests -q 2>&1 | tail -1; done
```

All five lines must be **identical**. That identity is the invariant — do not predict the number
from an earlier baseline, because every task in this plan added tests. As of Task 10b fix round 1
the suite reads `2 failed, 156 passed, 6 skipped`; Tasks 11 and 12 add more. The two failures are
`tests/integration/test_local_mlflow.py`, which predate this branch and need a live MLflow server
that does not exist in this environment.

A test that passes in one order and fails in another is a state leak, and CES-111 exists to surface
exactly that. Fix the leak — a missing fixture teardown, a module-level singleton, a shared tmp path
— **never by pinning the seed**. Known candidates, in order of likelihood:

- `ai_experiments/core/logger.py`'s `_configured` module global, against
  `tests/test_logger.py`'s reset fixture.
- The `Notifier` / `MonitorDaemon` construction in the CLI (`ai_experiments/cli/` after Task 11).
- Any module-level singleton added by Tasks 11 or 12 that did not exist when this list was written.

**Already ruled out, so do not chase it:** `get_settings()` is `@lru_cache`d, but
`tests/conftest.py:184` clears it in an `autouse` fixture on both sides of every test. That is the
shape of guard the other candidates need; it is the pattern to copy, not a bug to find.

```bash
git add pyproject.toml uv.lock tests
git commit -m "test: randomize test order (CES-111)"
```

- [ ] **Step 3: The whole hook suite**

```bash
uvx prek run --all-files; echo "exit=$?"
```
Expected: `exit=0`. This is the first point in the plan where nothing may be skipped. If
`ruff format` or `ruff check --fix` modify anything here, commit that diff — it means an earlier
task left unformatted code.

- [ ] **Step 4: Confirm the hooks are installed (they already are)**

Not an install step. `prek install` was run at the start of this session, before this plan existed:
`/home/lio/Projects/collectiveai/ai-experiments/.git/hooks/pre-commit` and `commit-msg` exist and
are headed `# File generated by prek`. That is *why* every task in this plan has carried a SKIP
list. Verify, do not reinstall:

```bash
head -2 "$(git rev-parse --git-common-dir)/hooks/pre-commit"   # expect the prek banner
```

**These hooks are shared.** `git rev-parse --git-common-dir` resolves to the primary repo's `.git`
for every worktree, so all five worktrees — and any other session committing in them — run the same
hooks. Do not install, uninstall, or reconfigure them.

- [ ] **Step 5: Assemble the PR body**

Write it to `.superpowers/sdd/2026-09-17-house-standards-adoption/pr-body.md`. It must carry:

- the SHA of the Task 3 formatting commit, with rebase instructions for the 13 open PRs;
- the `orchestrator.py` (412 lines) decision from Task 11 Step 8, with its reasoning;
- every `ast-grep-ignore` and `noqa` added by this branch, each with its justification;
- the note that the two MLflow integration failures predate this branch and need a live server;
- the deferred `NamedTuple` conversion for the two integration fixtures
  (`test_local_mlflow.py`, `test_ray_mlflow.py`), flagged as a follow-up for whoever next has a live
  MLflow/Ray environment — it was ruled a deliberate, revisitable exception, not a dismissal;
- the five defects found in upstream `collectiveai-team/scaffolding`, to be filed there:
  the `core/logger.py` snippet's 104-column `_is_prod` line; its use of 3.11-only
  `getLevelNamesMapping` against a `requires-python = ">=3.10"`; its `get_logger` annotated
  `-> structlog.stdlib.BoundLogger` when `make_filtering_bound_logger` guarantees it never returns
  one; the `prek.toml` `python` vs `python3` defect; and the `.env.schema` deny-rule shadowing.

- [ ] **Step 6: STOP — the push and the PR need the user's approval**

`git push -u origin feat/house-standards` and `gh pr create` are side effects outside this worktree,
against a shared repository. That is one of the four things that stops a running plan. Do **not**
run them autonomously.

Present to the user: the branch's commit count, the final gate results, the prepared PR body, and
the list of rulings made during execution. Then ask. On approval:

```bash
git push -u origin feat/house-standards
gh pr create --title "refactor: adopt house engineering standards" \
  --body-file .superpowers/sdd/2026-09-17-house-standards-adoption/pr-body.md
```

- [ ] **Step 7: Confirm CI**

Run: `gh pr checks --watch`
Expected: `tests`, `zizmor`, `osv-scanner`, `dependency-review`, `commit-policy`,
`conventional-commits` all pass. `tests.yml` runs `uvx prek run --all-files`, so Step 3 already
predicted that result.

---

## Risks

- **The formatting commit (Task 3) conflicts with all 13 open PRs.** Accepted deliberately on 2026-09-17. It is **`e97b0ff`** (`style: reformat to line-length 100 (ruff format)`); that SHA is what the rebase note in the PR body must name. There is no way to make a line-length change conflict-free.
- **Task 10 changes the HTTP response contract if a model is wrong.** The characterization tests in Step 1 are the only thing standing between this refactor and a silent API break. Do not skip them, and do not edit them to match new output.
- **The plan runs in the `house-standards` worktree, not the primary tree** -- but the two share one
  `.git`, so hooks, the stash stack and `refs/` are common to all five worktrees and to the other
  live sessions using them. Three consequences that bit during execution: prek's hooks were already
  installed before the plan began (which is why every task carries a SKIP list); never use bare
  `git stash`; and do not commit controller-side artifacts while an implementer has unstaged edits,
  because prek's patch save/restore will run across them.
- **`.env.schema` is unreadable to the agent** (Task 9 Step 2). That blocks only the reconciliation sub-step, not the task.
