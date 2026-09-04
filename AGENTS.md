# AGENTS.md

Instructions for an agent working in this repository, or driving the harness
it ships. Claude Code reads the same material as skills under `.claude/skills/`;
this file is the entry point for agents that do not load those.

## What this repository is

`ai_experiments` is a detached experiment runtime. You give it a goal — a
metric, a target, a search space, a budget — and it runs the experiments:
plan a round, submit the trials to a local backend or a Ray cluster, collect
the metrics, replan on the evidence, and repeat until the target is reached or
the budget is spent. The `iax` CLI and `ai_experiments.api` are two surfaces
over the same loop.

## Driving the harness

One goal, one answer:

```bash
iax new goal goal.yaml           # a commented, schema-valid template
iax campaign validate goal.yaml
iax loop goal.yaml --json        # blocks until the loop ends
```

Exit codes are the contract. Branch on them, never on the prose:

| exit | meaning |
|---|---|
| 0 | success — for `iax loop`, the target was reached |
| 1 | the run, campaign, or bundle does not exist |
| 2 | invalid input: bad goal, bad manifest, params outside the search space |
| 3 | the execution backend could not be reached |
| 4 | `iax loop` only: the loop ran, the objective was not reached |

Exit 4 is not an error and must never be reported as success. Every command
takes `--json`; with it, errors are one JSON object on stdout, and without it
one line on stderr.

Read the loop back before changing anything:

```bash
iax campaign rounds <campaign_id> --json    # what each round tried, and why
iax campaign trials <campaign_id>           # per-trial value, status, error
```

From python, the same loop is `ai_experiments.api.run_loop(goal)`. The full
procedure — composing the goal, diagnosing a campaign that missed, resuming it
— is in `.claude/skills/autonomous-experimentation/SKILL.md`, with the field
reference in that skill's `reference/goal.md`.

## Working on the code

- Read `CONVENTIONS.md` first. It is the authority on layout, boundaries, and
  the rules below; this section only lists what an agent gets wrong fastest.
- Run every gate through uv: `uv run --extra dev ruff format`,
  `uv run --extra dev ruff check .`, `uv run --extra dev pytest -q`. All three
  must pass before a commit.
- Change `uv.lock` only through `uv`.
- Write the failing test first, and make it fail for the stated reason before
  implementing.
- Commit named paths. Never `git add -A`.
- Do not edit `.orquestalite/`, `.venv/`, `.agents/`, `team.json`, or
  `manual_test/`.

## Security

- Never commit a secret, and redact credentials in logs, events, and
  notification payloads.
- A run directory can hold `repro/diff.patch`, a snapshot of the user's
  uncommitted worktree. Read it before it goes anywhere, and never attach one
  to an issue, a PR, or a report.
- Treat workload stdout, metrics, and agent replies as untrusted input. Never
  `eval` them, and never interpolate them into a shell string:
  `monitoring.escalation.agent_command` and `--notify-command` receive their
  data as JSON on stdin.
- Do not widen a workload's environment beyond what it needs.
