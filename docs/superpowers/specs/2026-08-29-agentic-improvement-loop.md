# Design: Agentic improvement loop (`iax loop`)

**Status:** accepted (2026-08-29)
**Supersedes nothing.** Extends the campaign layer built in `tasks/todo.md`.

## Problem

Today `ai-experiments` pursues a goal by *hyperparameter search*: `GoalSpec`
carries a fixed `search_space` and a fixed `workload`, and `grid`/`random`/
`adaptive` strategies pick the next points. Everything else — writing the
goal, instrumenting the workload, reading the history, deciding what to change
next, changing the code — is human work.

The target experience is different:

> A user states a goal in a Claude Code or Codex chat. The agent loads one
> skill, and from there `ai_experiments` runs however many improvement rounds
> are needed — on Ray — until the objective is met or the budget is spent.

Two gaps separate the two:

1. **No agent in the loop.** The agent can only drop a file
   (`analysis.agent_review`) and poll, or push single trials with
   `iax campaign suggest`. There is no strategy that *asks* an agent what to
   try next, and no command that closes the loop end to end.
2. **The loop can only change numbers, not code.** Real improvement cycles
   change the workload — a different loss, a schedule, an architecture — not
   just a point in a fixed box.

A third gap blocks both: **the loop is not yet trustworthy enough to leave
alone overnight.** A crashed trial can win a campaign (#12), a typo'd metric
name fails silently (#11), grid exhaustion wedges a campaign in `running`
forever (#15), agent-suggested trials bypass the budget and the search space
(#13), and the escalation file that carries the agent's work item crashes
every `iax escalations` call after the first campaign review (#4).

## Goals

- G1. An agent can drive a full campaign from one skill, with machine-readable
  output from every command it touches.
- G2. `strategy: agent` — the next round's parameters come from an agent that
  sees the trial history, with a programmatic strategy as the fallback.
- G3. Improvement rounds may change the **workload code**, not only params,
  inside a sandboxed variant directory that never mutates the user's tree.
- G4. `iax loop goal.yaml` runs the whole thing unattended to a terminal state.
- G5. The loop is honest: it never reports a target as reached from a failed
  trial, never wedges, never silently scores `null`.

## Non-goals

- Replacing Ray Tune / Optuna as a sampler. The programmatic strategies stay
  deliberately simple; the agent is the intelligence.
- Letting the agent write to the user's source tree. Variants are copies.
- Running the agent inside the monitor daemon's tick by default (tokens).

## Architecture

```
                  ┌──────────── one round ────────────┐
 goal.yaml ─┬────►│ propose ─► apply ─► validate ─►    │
            │     │ evaluate (trials on ray) ─► review │
            │     └──────────────┬────────────────────┘
            │                    │ keep / revert / stop
            └────────────────────┴──────► next round …
```

The stage names are lifted from the orq-lite `development` governed pack
(`ticket_planner → coder → qa → adversary → critic → integrator`) because the
same shape applies: a proposal is cheap, an evaluation is expensive, and a
separate reviewer decides whether the change survives.

### New packages

| Package | Responsibility |
|---|---|
| `ai_experiments/agents/` | Invoke an agent CLI, capture the transcript, parse a strict JSON contract. Knows nothing about campaigns. |
| `ai_experiments/improve/` | Round records, hypotheses, workload variants, the keep/revert decision. |
| `ai_experiments/api.py` | The public Python surface an agent (or a skill) imports. |

### `ai_experiments/agents/`

- `AgentRunner` protocol: `run(prompt: str, *, role: str) -> AgentResult`.
- `CliAgentRunner`: renders `command` with a `{prompt_file}` token, runs it
  with a timeout, writes `stdout`/`stderr`/`prompt.md` under
  `<campaign_dir>/agents/<role>/<n>/`, and extracts the **last** balanced JSON
  object from stdout (agent CLIs wrap JSON in prose).
- `StubAgentRunner`: returns scripted results; the whole loop is testable with
  zero tokens and zero network.
- `AgentSpec` in the goal selects and configures the runner.

Presets so a goal does not carry a command line:
`claude` → `claude -p --output-format json` (prompt on stdin),
`codex` → `codex exec --json`, `command` → the user's own argv.

### `strategy: agent`

`AgentStrategy.plan(goal, trials, count)`:
1. Build the round brief: goal text, objective, search space, full trial
   history with values and errors, budget remaining.
2. Ask the agent for `{"params": [...], "rationale": "..."}`.
3. **Validate every proposal against the search space** and drop duplicates —
   the same validator that fixes #13, shared with `campaign suggest`.
4. On failure (timeout, bad JSON, zero valid proposals) fall back to
   `strategy.fallback` (default `adaptive`) and record the reason as an event.

The agent never bypasses the budget: `_fill_capacity` already caps by
`max_trials`, and the validator caps the proposal count.

### Code variants (`ai_experiments/improve/`)

A `VariantRecord` is a materialized copy of `workload.working_dir` under
`<campaign_dir>/variants/<variant_id>/`, plus the diff that produced it and
the hypothesis that motivated it. Trials carry `variant_id`; the trial
manifest's `working_dir` points at the variant.

`variants.enabled: false` by default. When enabled, a round may return
`{"kind": "code", "edits": [{"path": ..., "content": ...}], "rationale": ...}`
instead of (or alongside) params. Guardrails:
- paths are resolved against the variant root and rejected if they escape it;
- a variant runs a **smoke check** (`variants.smoke_command`, default: the
  workload with `IAX_SMOKE=1`) before it costs a real trial — this is the
  `qa` stage, and it is what makes #9's class of failure cheap;
- the baseline advances only when the reviewer keeps the variant.

### `iax loop`

`iax loop goal.yaml` = `campaign start` + an in-process advance loop + the
agent round hooks + a terminal report. It is `iax run` with the agent stages
and without the dashboard requirement, and it exits non-zero when the campaign
ends without reaching the target.

### The skill

`.claude/skills/autonomous-experimentation/SKILL.md` (mirrored to
`.agents/skills/` for Codex) teaches the chat-side agent to: turn the user's
sentence into a `GoalSpec`, scaffold and instrument a workload when there is
none (`iax new`), validate, launch `iax loop`, and read the round records back.

## Risks

- **Token burn.** Every agent stage is opt-in and budgeted
  (`agent.max_calls`), and the fallback strategy keeps the campaign moving
  when the budget runs out.
- **A convincing but wrong agent.** The reviewer stage sees measured
  objective values, not the proposer's claims, and the search-space validator
  refuses out-of-range params regardless of the rationale.
- **Runaway code edits.** Variants are copies under the campaign dir, path
  traversal is rejected, and the smoke check gates the expensive trial.
