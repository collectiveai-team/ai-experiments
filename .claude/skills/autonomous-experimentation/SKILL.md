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

## 1. Get the five things a goal needs

Do not start until you have all five. Ask for what is missing, and ask once:

| | question | goes into |
|---|---|---|
| the metric | what number decides success, and is smaller or larger better? | `objective.metric`, `objective.mode` |
| the target | what value is good enough? | `objective.target` |
| the knobs | what may the loop change, and between which values? | `search_space` |
| the budget | how many trials, and how much wall time or GPU cost? | `budget` |
| the bar | what would this campaign have to show for its result to count? | `success_criteria` |

If the user cannot name a target, say so and set none: the loop then spends
the budget and reports the best it found. That is a valid campaign, but it
can never exit "reached", and the user must know that before it runs.

The bar is the one an agent is tempted to skip, and it is the one that decides
whether the run produced evidence or a number. Use **defining-goals** to set
it, along with `objective.aggregate`, `objective.baseline_metric` and the
observation count — those four decisions cannot be made honestly once the
results are in.

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
| 0 | the declared success criteria were met — or, with none declared, the target was reached | report the best params and stop |
| 4 | the loop ran and the bar was not cleared | go to step 4 |
| 2 | the goal file is invalid | fix the goal, do not retry the loop |
| 3 | the backend was unreachable | fix the cluster, then resume |

When the goal declares `success_criteria`, they decide the exit code and
`target_reached` does not: a target can be tripped by one lucky observation,
which is exactly what the criteria exist to catch.

Exit 4 is not a failure of the tool. It is the answer to a hard question.
Never report it as success, and never report a best trial as if it had met a
bar it missed.

`--max-rounds` and `--max-seconds` bound one turn of the conversation without
ending the campaign. Continue the same one — history and all — with:

```bash
iax loop goal.yaml --resume <campaign_id> --json
```

A report with a non-empty `pending_trials` stopped with work still running.
Those trials have no value recorded yet, so resume before you conclude
anything — the best trial may be one the loop never got to read.

From python, the same loop is `ai_experiments.api.run_loop(goal)`. Use it when
you are composing the goal in code rather than in a file.

## 4. Read what the loop learned before you change anything

```bash
iax campaign rounds <campaign_id> --json    # what each round tried, and why
iax campaign trials <campaign_id>           # per-trial source, value, status, error
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

## 6. When the code is the blocker, not the parameters

Some campaigns cannot be saved by any parameter: every trial dies on the same
error, the workload reports no metric at all, or the harness returns NaN at
the edge of the space. A review that sees this answers `needs_change` instead
of `stop`. The loop then stops the campaign with `blocked_on_change` and files
a ticket with the evidence — failed trials, run ids, the error tail.

```bash
iax escalations                       # the ticket, with kind "change"
iax handoff <campaign_id> --dry-run   # what branch and issue it would create
iax handoff <campaign_id> --repo /path/to/workload
```

`iax handoff` creates `exp/<campaign>-<digest>` in its own worktree and hands
the issue to a development flow. The default plans the work and stops; `--run`
lets the flow execute it, and it spends tokens, so ask the user first.

The fix goes on that branch and nowhere else. Trials measured before a code
change and trials measured after it are not comparable, so the campaign that
found the defect stays closed. When the fix lands, start a new campaign from
the same goal inside the worktree, and compare the two campaigns, never the
trials across them.

## 7. Report

Say four things, in this order, and nothing else:

1. met or not met, and which criterion failed — the `success` block says it;
2. the best value **with its interval**, and the params that produced it —
   the `verdict` block says whether that value is separated from the runner-up
   and whether it clears the baseline;
3. what the loop cost — trials, rounds, machine time, GPU-hours if the goal
   priced them;
4. the one change you would make next, from the round records.

Attach the campaign id. Everything you claim must be readable from
`iax campaign status <campaign_id> --json`; if it is not there, do not say it.
The human-readable form of points 1 and 2 is what the CLI already prints, from
`ai_experiments.planner.analysis.result_lines` — quote it rather than
paraphrasing, so the report and the harness cannot disagree.

`false` and `null` in those blocks are different answers. `separated: false`
means the campaign measured the lead and it sits inside the noise;
`separated: null` means nothing measured it. Say which.

## Never

- Never present a run that exited 4 as a success.
- Never invent a metric value, a trial, or a parameter that no trial used.
- Never announce a winner without saying whether it is distinguishable from
  the runner-up, and never turn an unmeasured comparison into a negative one.
- Never add or loosen `success_criteria` after seeing the results. A bar moved
  to fit the number is not a bar.
- Never change `objective.metric` on a running campaign.
- Never start a second campaign for the same question while the first is
  running — resume it, so the evidence stays in one place.
- Never paste a `repro/diff.patch` into a report or an issue. It is the user's
  uncommitted work and may contain anything.
- Never fix a defect on the branch the campaign ran on. Use the branch
  `iax handoff` created, so the before and after stay separable.
