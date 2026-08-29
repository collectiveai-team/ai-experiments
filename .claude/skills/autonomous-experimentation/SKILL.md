---
name: autonomous-experimentation
description: Use when the user states an outcome instead of a task — "get val_loss under 0.05", "make this model beat the baseline", "figure out the best config and keep going until it works". Turns the objective into a goal file, runs `iax loop` until the target is reached or the evidence says it cannot be, and reports what the loop learned.
---

# Autonomous Experimentation

The user gives you an outcome. You give the harness a goal, and the harness
runs the experiments — plan a round, submit the trials to Ray or the local
backend, collect the metrics, replan on the evidence, repeat. You are not the
loop. You define it, start it, read what it found, and decide what to change.

Use **running-campaigns** instead when the user wants to drive the rounds by
hand. Use this skill when they want the result and not the driving.

## 1. Get the four things a goal needs

Do not start until you have all four. Ask for what is missing, and ask once:

| | question | goes into |
|---|---|---|
| the metric | what number decides success, and is smaller or larger better? | `objective.metric`, `objective.mode` |
| the target | what value is good enough? | `objective.target` |
| the knobs | what may the loop change, and between which values? | `search_space` |
| the budget | how many trials, and how much wall time or GPU cost? | `budget` |

If the user cannot name a target, say so and set none: the loop then spends
the budget and reports the best it found. That is a valid campaign, but it
can never exit "reached", and the user must know that before it runs.

The workload must already print its metric:

```python
print('IAX_METRIC {"step": 12, "loss": 0.0734}')       # one line per observation
```

If it does not, `iax new workload train.py` scaffolds one that does. A
workload that reports nothing produces trials that score `null`, and no
strategy can learn from them.

## 2. Write the goal, then validate it

```bash
iax new goal goal.yaml        # a commented template, always schema-valid
iax campaign validate goal.yaml
```

Field reference and the choices that matter: `reference/goal.md`.

Never hand-write a goal from memory. The template comes from the installed
schemas, so it cannot drift from the version of the library that will run it.

## 3. Run the loop

```bash
iax loop goal.yaml --json --max-rounds 20
```

It blocks, drives the campaign to a conclusion, prints one report, and exits:

| exit | meaning | what you do |
|---|---|---|
| 0 | the target was reached | report the best params and stop |
| 4 | the loop ran, the target was not reached | go to step 4 |
| 2 | the goal file is invalid | fix the goal, do not retry the loop |
| 3 | the backend was unreachable | fix the cluster, then resume |

Exit 4 is not a failure of the tool. It is the answer to a hard question.
Never report it as success, and never report a best trial as if it had met a
target it missed.

`--max-rounds` and `--max-seconds` bound one turn of the conversation without
ending the campaign. Continue the same one — history and all — with:

```bash
iax loop goal.yaml --resume <campaign_id> --json
```

From python, the same loop is `ai_experiments.api.run_loop(goal)`. Use it when
you are composing the goal in code rather than in a file.

## 4. Read what the loop learned before you change anything

```bash
iax campaign rounds <campaign_id> --json    # what each round tried, and why
iax campaign trials <campaign_id>           # per-trial value, status, error
```

Diagnose from the records, not from a guess:

- **Every trial failed.** This is a workload bug, not a search problem. Read
  the error in `iax campaign trials`, fix the workload, start a new campaign.
- **Every trial scored `null`.** The metric name in the goal does not match
  the name the workload prints. Fix the goal.
- **The best value sits at an edge of a range.** The optimum is probably
  outside it. Widen that range and resume.
- **Values barely move across many trials.** The knob does not drive the
  metric. Replace it with one that might.
- **Values improve and then flatten near the target.** Nothing is wrong. Raise
  `budget.max_trials` and resume.

Then edit and continue, keeping the history:

```bash
iax campaign edit <campaign_id> goal2.yaml    # narrower or wider space, new budget
iax loop goal.yaml --resume <campaign_id>
```

The objective metric must not change. Every value already recorded was
measured against it; changing it makes the campaign's own history a lie. A new
metric is a new campaign.

## 5. Let the agent plan the rounds

The built-in strategies search a fixed space by fixed rules. When the search
needs judgment — the parameters interact, or the failures are informative —
hand the planning to an agent:

```yaml
strategy:
  name: agent
  fallback: adaptive         # plans the round whenever the agent cannot
agent:
  command: claude            # claude | codex | any command reading stdin
  max_calls: 20              # hard ceiling for the whole campaign
analysis:
  review_between_rounds: true    # ask for a verdict after each round
  apply_agent_changes: false     # true lets a verdict widen the space or budget
```

The harness never trusts the reply: out-of-range and already-tried params are
dropped, and an agent that crashes, times out, or answers without JSON just
loses that round to the fallback. Turn `review_between_rounds` on for long or
expensive campaigns — a `stop` verdict ends a hopeless run instead of spending
the rest of the budget proving it is hopeless.

`agent.max_calls` is a cost ceiling. Set it deliberately: every call is a
model invocation the user pays for.

## 6. Report

Say four things, in this order, and nothing else:

1. reached or not reached, and against which target;
2. the best value and the params that produced it;
3. what the loop cost — trials, rounds, GPU-hours if the goal priced them;
4. the one change you would make next, from the round records.

Attach the campaign id. Everything you claim must be readable from
`iax campaign status <campaign_id> --json`; if it is not there, do not say it.

## Never

- Never present a run that exited 4 as a success.
- Never invent a metric value, a trial, or a parameter that no trial used.
- Never change `objective.metric` on a running campaign.
- Never start a second campaign for the same question while the first is
  running — resume it, so the evidence stays in one place.
- Never paste a `repro/diff.patch` into a report or an issue. It is the user's
  uncommitted work and may contain anything.
