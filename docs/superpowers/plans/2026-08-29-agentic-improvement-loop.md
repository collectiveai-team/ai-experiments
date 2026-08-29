# Agentic Improvement Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Turn `ai-experiments` into a library an agent drives from one chat message — state a goal, and the harness runs as many agent-guided improvement rounds on Ray as the budget allows until the objective is met.

**Architecture:** Three new packages (`agents/` invokes an agent CLI under a strict JSON contract, `improve/` records rounds and sandboxes workload-code variants, `api.py` is the public Python surface) plus one new strategy (`agent`) and one new command (`iax loop`). Round stages mirror the orq-lite governed pack: propose → apply → validate → evaluate → review. Everything the agent touches speaks JSON and exits with a documented code.

**Tech Stack:** Python 3.10+, pydantic v2, typer, Ray Jobs API, pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-08-29-agentic-improvement-loop.md`

## Global Constraints

- Python `>=3.10`; `from __future__ import annotations` in every module.
- No new required dependency. Agent invocation shells out; it does not import an SDK.
- Every new public model is a pydantic v2 `BaseModel` in `ai_experiments/schemas.py` or a package-local module, never a bare dict.
- Tests never call a real agent CLI, a real Ray cluster, or the network. `StubAgentRunner` and the existing local backend cover the loop.
- `uv run ruff check .` and `uv run pytest -q` stay green after every task.
- Commit after every task, message in imperative mood, no attribution footer.

---

## Phase 1 — Make the loop trustworthy (blocks everything else)

An unattended loop that lies is worse than no loop. These five defects each
break the closed loop specifically.

### Task 1: A failed trial can never win the campaign (#12)

**Files:** Modify `ai_experiments/planner/analysis.py`, Test `tests/test_orchestrator.py`

- [x] Step 1: Failing test — a campaign whose only scored trial has `status="failed"` must not stop with `target_reached`.
- [x] Step 2: `best_trial()` considers only `status == "completed"` trials.
- [x] Step 3: Regression test that a completed trial still wins, and that a failed trial's value still appears in `summarize_campaign()["history"]` (the agent must see failures).
- [x] Step 4: `uv run pytest -q`; commit.

### Task 2: A typo'd `objective.metric` fails loudly (#11)

**Files:** Modify `ai_experiments/planner/analysis.py`, `ai_experiments/orchestrator.py`, Test `tests/test_orchestrator.py`

- [x] Step 1: Failing test — a completed trial that reported metrics, none of them the objective metric, records a `trial.error` naming the metric and the keys that *were* reported, and emits a warning event.
- [x] Step 2: `extract_objective()` returns the observed metric names alongside the value; `_refresh_trials` turns "metrics present, objective absent" into an explicit error on the trial record.
- [x] Step 3: Distinguish it from "no metrics at all" (a separate reason: the workload never reported).
- [x] Step 4: `uv run pytest -q`; commit.

### Task 3: Grid exhaustion ends the campaign instead of wedging it (#15)

**Files:** Modify `ai_experiments/orchestrator.py`, Test `tests/test_orchestrator.py`

- [x] Step 1: Failing test — a `grid` campaign with `max_trials: 50` over a 4-point space reaches a terminal status with `stop_reason == "search_space_exhausted"`.
- [x] Step 2: In `_fill_capacity`, when the planner returns fewer params than requested and nothing is active or planned, set that stop reason.
- [x] Step 3: `uv run pytest -q`; commit.

### Task 4: Suggested trials respect the budget and the search space (#13)

**Files:** Create `ai_experiments/planner/validation.py`, Modify `ai_experiments/orchestrator.py`, `ai_experiments/cli.py`, Test `tests/test_planner.py`, `tests/test_orchestrator.py`

**Produces:** `validate_params(space: dict[str, ParamSpec], params: dict[str, Any]) -> dict[str, Any]` (raises `ParamValidationError` with every violation), used by `suggest()` and by `AgentStrategy` in Task 8.

- [x] Step 1: Failing tests — unknown key, out-of-range `uniform`, non-member `choice`, non-integer `int` each raise; a valid dict round-trips with ints coerced from floats.
- [x] Step 2: Implement `validate_params`.
- [x] Step 3: Failing test — `suggest()` past `max_trials` raises rather than queueing.
- [x] Step 4: Wire both into `suggest()`; map the error to exit code 2 in the CLI.
- [x] Step 5: `uv run pytest -q`; commit.

### Task 5: `iax escalations` survives campaign reviews (#4)

**Files:** Modify `ai_experiments/monitoring/escalation.py`, Test `tests/test_escalation.py`

- [x] Step 1: Failing test — a `campaign_<id>.json` review payload in `_escalations/` plus a run escalation: `list_escalations()` returns both, typed, and never raises.
- [x] Step 2: Split the directory into two typed readers (`list_run_escalations`, `list_campaign_reviews`) behind one `list_escalations()` that skips unparseable files with a warning instead of raising.
- [x] Step 3: `uv run pytest -q`; commit.

---

## Phase 2 — The agent-facing contract

### Task 6: One JSON and exit-code convention on every command (#18, #16)

**Files:** Create `ai_experiments/cli_support.py`, Modify `ai_experiments/cli.py`, Test `tests/test_cli.py`

**Produces:** `emit(obj, json_mode)`, `fail(message, code)`, `IaxError`; exit codes `0` ok, `1` not-found, `2` invalid-input, `3` backend-unavailable.

- [x] Step 1: Failing tests — `iax status nonexistent --json` exits 1 with `{"error": ..., "code": "not_found"}` on stdout and no traceback; `iax validate bad.yaml --json` exits 2.
- [x] Step 2: Implement the helper; convert every command that takes an id.
- [x] Step 3: Document the convention in the README and in `submitting-experiments`.
- [x] Step 4: `uv run pytest -q`; commit.

### Task 7: `iax new` scaffolds goals and instrumented workloads (#19)

**Files:** Create `ai_experiments/scaffold.py`, `ai_experiments/templates/`, Modify `ai_experiments/cli.py`, Test `tests/test_scaffold.py`

- [x] Step 1: Failing test — `iax new goal --name demo --metric val_loss` writes a `GoalSpec`-valid YAML; `iax new workload` writes a runnable script that prints `IAX_METRIC` lines and honours `IAX_PARAMS` and `IAX_SMOKE`.
- [x] Step 2: Implement; the generated workload must pass `iax campaign validate` and a smoke run.
- [x] Step 3: `uv run pytest -q`; commit.

---

## Phase 3 — The agent in the loop

### Task 8: `ai_experiments/agents/` — invoke an agent, parse a contract

**Files:** Create `ai_experiments/agents/{__init__,runner,contracts,prompts}.py`, Test `tests/test_agent_runner.py`

**Produces:** `AgentRunner` protocol, `CliAgentRunner`, `StubAgentRunner`, `AgentResult(ok, payload, raw, error)`, `extract_json(text)`.

- [x] Step 1: Failing tests for `extract_json` — JSON after prose, fenced JSON, two objects (last wins), no JSON (`None`).
- [x] Step 2: Implement `extract_json`.
- [x] Step 3: Failing tests for `CliAgentRunner` against a fake agent script — success, non-zero exit, timeout, unparseable output; transcript files written.
- [x] Step 4: Implement the runner and the `claude`/`codex`/`command` presets.
- [x] Step 5: `uv run pytest -q`; commit.

### Task 9: `strategy: agent` proposes the next round

**Files:** Create `ai_experiments/planner/agent_strategy.py`, Modify `ai_experiments/schemas.py`, `ai_experiments/planner/strategies.py`, Test `tests/test_agent_strategy.py`

- [x] Step 1: Add `AgentSpec` to the goal and `"agent"` to `StrategySpec.name` with a `fallback` field.
- [x] Step 2: Failing tests — a stub returning valid params yields them; out-of-range params are dropped; all-invalid or a runner failure falls back to `adaptive` and records the reason.
- [x] Step 3: Implement; the round brief includes failed trials and their errors.
- [x] Step 4: `uv run pytest -q`; commit.

### Task 10: Round records — the loop's memory

**Files:** Create `ai_experiments/improve/{__init__,rounds}.py`, Modify `ai_experiments/store/campaign.py`, `ai_experiments/orchestrator.py`, Test `tests/test_rounds.py`

**Produces:** `RoundRecord(round, stage, hypothesis, rationale, trial_ids, outcome, agent_calls)` persisted to `<campaign_dir>/rounds.jsonl`.

- [x] Step 1: Failing test — a two-round campaign writes two records, readable in order, each naming its trials and its rationale.
- [x] Step 2: Implement; `iax campaign rounds <id> --json` reads them back (also closes #23's trials listing).
- [x] Step 3: `uv run pytest -q`; commit.

### Task 11: Workload variants — improvement over code

**Files:** Create `ai_experiments/improve/variants.py`, Modify `ai_experiments/schemas.py`, `ai_experiments/planner/planner.py`, Test `tests/test_variants.py`

- [x] Step 1: Failing tests — materializing a variant copies `working_dir`; an edit path escaping the root raises; the trial manifest's `working_dir` points at the variant; the smoke check gates promotion.
- [x] Step 2: Implement `VariantSpec`, `VariantRecord`, `materialize_variant`, `smoke_check`.
- [x] Step 3: `uv run pytest -q`; commit.

### Task 12: `iax loop` — the closed loop, one command

**Files:** Create `ai_experiments/loop.py`, Modify `ai_experiments/cli.py`, Test `tests/test_loop.py`

- [x] Step 1: Failing test — `run_loop()` over the local backend with a stub agent reaches `target_reached` and returns a report naming rounds, trials and the best params.
- [x] Step 2: Implement: start → advance until terminal → agent stages between rounds → final report; `--max-rounds`, `--json`, non-zero exit when the target is missed.
- [x] Step 3: `uv run pytest -q`; commit.

### Task 13: `ai_experiments/api.py` — the public Python surface

**Files:** Create `ai_experiments/api.py`, Modify `ai_experiments/__init__.py`, Test `tests/test_api.py`

**Produces:** `goal_from_dict`, `start_campaign`, `advance_campaign`, `campaign_report`, `run_loop`, `suggest_trial`.

- [x] Step 1: Failing test — a goal dict drives a whole campaign to `target_reached` through the API alone, with no CLI and no YAML file.
- [x] Step 2: Implement as a thin façade; re-export from `ai_experiments/__init__.py`.
- [x] Step 3: `uv run pytest -q`; commit.

---

## Phase 4 — The chat experience

### Task 14: The `autonomous-experimentation` skill

**Files:** Create `.claude/skills/autonomous-experimentation/SKILL.md` and `reference/goal.md`, mirror into `.agents/skills/`, Modify `.claude/skills/running-campaigns/SKILL.md`

- [x] Step 1: Write the skill: objective → `GoalSpec` → scaffold → validate → `iax loop` → read rounds → report. Every command in it verified against the built CLI.
- [x] Step 2: Mirror to `.agents/skills/` for Codex; add `AGENTS.md` pointing at it.
- [x] Step 3: Cross-link from `running-campaigns`; commit.

### Task 15: End-to-end example and docs

**Files:** Create `examples/goal_agentic.yaml`, `examples/agentic_train.py`, Modify `README.md`, `docs/`

- [x] Step 1: An example that runs to target on the local backend in under a minute with a stub agent, and on Ray with a real one.
- [x] Step 2: README section: the chat-driven flow, the round stages, the env-var table (#20).
- [x] Step 3: Run it; commit.

---

## Verification (whole plan)

- [x] `uv run ruff check .` and `uv run ruff format --check .` clean.
- [x] `uv run pytest -q` green, including every new test — 288 passed, 14 skipped
  (the skips need the `mlflow` extra or a Ray cluster on the default port).
- [x] A real end-to-end: `iax loop examples/goal_agentic.yaml` reaches its target on the local backend, driven by a real `claude -p` agent, with the rounds file showing the agent's rationale.
  The agent reached `loss=0.331` (target 0.40) in 6 trials over 2 rounds, and its
  round-2 rationale cites the out-of-memory error trial t002 reported. The
  `adaptive` fallback spends all 30 trials on the same goal and misses.
- [x] The same goal with `backend: ray` against a local Ray cluster — 30 trials over
  15 rounds through the Ray Jobs API, metrics parsed from the job logs.
- [x] Every command quoted in the new skill executed verbatim and its output matched.
  This is what found the scaffold defect: `iax new goal` + `iax new workload` +
  `iax loop` exited 4 for a new user. Fixed, and pinned by
  `tests/test_scaffold.py::test_the_scaffolded_goal_and_workload_reach_their_target`.

### Defects this verification found, and fixed

1. A failed trial recorded only its exit code. The agent's whole advantage is
   reading failures, and it had nothing to read. Now the trial carries the tail
   of the workload's output (`ai_experiments/failures.py`).
2. A failed Ray trial recorded up to 20,000 characters of job log in that same
   field, which the planner puts in every later prompt. Same trimming rule.
3. The scaffolded goal and workload missed their target, so a new user's first
   `iax loop` exited 4.

### Known, not fixed

- An unreachable Ray cluster ends a campaign as `search_space_exhausted` instead
  of `backend_unavailable`. The submit error is recorded on the trial, so the
  cause is visible, but the campaign's stop reason lies about it.
