---
name: defining-goals
description: Use before starting any iax campaign, and whenever a campaign reported a best trial that proves nothing — decides what one observation is, whether the objective needs a baseline, how many observations a trial needs, and writes down what would count as success, so the verdict at the end is arithmetic instead of an argument.
---

# Defining Goals

A campaign answers exactly the question its goal file asked. Everything that
decides whether the answer means anything — what an observation is, how many a
trial needs, what would count as success — has to be chosen before the first
trial runs. Chosen afterwards, it is chosen to fit the number that came out.

This is the half you do. **running-campaigns** and
**autonomous-experimentation** are the half the harness does: once the goal
file is written, `ai_experiments/planner/analysis.py` scores the trials,
aggregates the observations, compares the winner against the runner-up and
rules on the success criteria, and hands the same verdict to the CLI, the loop
and the dashboard. Your job is to give that code something to rule on.

## The rule

**Write the bar into the goal file. Let the code say whether the campaign
cleared it.** Anything you would claim in a report that the verdict does not
already contain is a claim the campaign did not measure.

## 1. Decide what one observation is

The workload prints one `IAX_METRIC` line per observation. What those lines
*are* decides `objective.aggregate`, and nothing downstream can recover from
getting it wrong:

| one line per… | the lines are | `aggregate` |
|---|---|---|
| epoch, step | successive states of one model — the last one is the model you keep | `best` |
| fold, seed, split, held-out window | independent evaluations of one configuration | `mean` |
| bootstrap replicate | resamples of **one** evaluation, not new evidence | `bootstrap` |

`bootstrap` averages like `mean` but reports the spread itself as the standard
error, without dividing by √k: the replicates *are* the sampling distribution,
so dividing again would shrink the interval by exactly the factor the
resampling exists to expose. It also means the replicate count is a
computational knob, not evidence — `min_observations` counts how long the
workload chose to resample, so gate thin evidence by having the workload report
no objective instead.

`best` over independent evaluations is max-of-k: biased upward by exactly the
noise the evaluations exist to measure, and it carries no standard error. With
no standard error the harness reports `separated: null` and
`beats_baseline: null` — *not measured*, which fails any criterion that asked
for a measurement. `mean` reports the average and the standard error over the
observations, which is what makes a lead checkable.

## 2. Decide whether the objective needs a baseline

If the search space can change the data or the labels a trial is scored on,
raw scores are not comparable between trials: the search will find the easiest
slice, not the best model. Set `baseline_metric` and have the workload report
both numbers **in the same line**, so the lift is paired within one
observation:

```python
from ai_experiments.report import report_metric
report_metric(step=fold, pr_auc=0.41, baseline_pr_auc=0.19)
```

```yaml
objective: { metric: pr_auc, baseline_metric: baseline_pr_auc, mode: max, aggregate: mean }
```

The harness then scores `pr_auc - baseline_pr_auc` per observation. Taking the
best metric and the best baseline separately would report a lift no single
observation achieved, so the pairing is not optional.

## 3. Mark the knobs that move the data, and hide the ones nobody reads

```yaml
search_space:
  window_days: { type: choice, values: [30, 90, 180], changes_data: true }
  model:       { type: choice, values: [hist_gb, logreg] }
  learning_rate:
    { type: loguniform, low: 0.01, high: 0.3, when: { model: [hist_gb] } }
```

- `changes_data` marks a dimension that changes what the trial is evaluated
  on rather than how it is fit. Preflight warns when one exists and
  the objective has no `baseline_metric`.
- `when` draws a dimension only for the trials that read it. Without it, a
  `logreg` trial still gets a `learning_rate` it ignores: the trials become
  duplicates the deduplicator cannot see, and their spread reads as evidence
  that the knob does nothing.
- `when` may only name unconditional keys, and the schema rejects one that
  names a key the space does not define.

## 4. Decide how many observations before deciding anything else

With `aggregate: mean`, the smallest difference the campaign can detect at
95% confidence and 80% power is

```
MDE ≈ 2.80 × sd / √k
```

where `sd` is the spread of the per-observation score within one trial and `k`
is the number of observations. Get `sd` from a previous campaign's per-fold
records if there is one, or from a single pilot trial; then pick `k` so the
MDE sits below the effect that would actually change a decision:

```bash
uv run python -c "print([(k, round(2.80*0.113/k**0.5, 3)) for k in (5,10,12,20,40)])"
```

Run this *before* the campaign, not after. The worked example — where the
previous design could not have detected its own best result — is
`examples/maintenance_events/README.md`, section "Cuántos folds".

**The formula assumes `sd` holds still when `k` changes. Check that it does.**
When the observations come from splitting one fixed dataset, more of them means
less data in each, `sd` grows with `k`, and the √k in the denominator buys
nothing. In that same worked example the measured MDE is flat at ~0.115 from 12
folds to 20 while the formula promised 0.091 → 0.071. Raising `k` only helps
when each new observation brings new data. When it does not, the honest moves
are a lower-variance metric, a bigger effect, or accepting that the campaign
answers a coarser question — and `k` is then set by whatever else depends on it,
like `min_observations`.

## 5. Write down what would count as success

```yaml
success_criteria:
  min_objective: 0.05          # the bar; `mode: min` reads it as "at most"
  min_observations: 10         # refuse a score resting on too few
  require_separation: true     # beat the runner-up by more than the noise
  require_beats_baseline: true # the interval must clear the baseline
```

`min_objective` is the only one that can be met by a single lucky trial;
`min_observations`, `require_separation` and `require_beats_baseline` are what
make it mean something. Declare them together.

A goal that declares none of these gets `met: null` at the end — not a pass,
and `iax loop` falls back to the bare target check. When criteria *are*
declared they are the authority: `iax loop` exits 4 whenever they are unmet,
even if `target_reached` fired, because a lucky single observation can trip a
target.

Every criterion has a prerequisite the preflight checks:
`require_separation` needs `aggregate: mean` or `bootstrap`,
`require_beats_baseline` needs
`baseline_metric`. A criterion whose evidence was never measured **fails** —
a campaign cannot satisfy "show me the lead is real" by not looking.

## 6. Validate before you run

```bash
iax new goal goal.yaml          # the template ships these fields; never hand-write from memory
iax campaign validate goal.yaml --strict
```

`--strict` exits non-zero on any warning. Get it to zero before starting: the
warnings are the ways a goal can be valid, run to completion and still settle
nothing — no criteria declared, a data-moving knob with no baseline, a
criterion whose evidence the objective cannot produce, a search-space key the
workload's `--help` does not accept.

Then hand off: **running-campaigns** to drive the rounds yourself,
**autonomous-experimentation** to let `iax loop` run it to a conclusion.

## 7. Report the verdict, not the number

`iax campaign status <id> --json` carries `verdict` and `success`; the
human-readable form comes from `result_lines` and says the same thing. Quote
it. In particular, keep `false` and `null` apart: `false` means the campaign
looked and the evidence was not there, `null` means it never looked.

## Never

- Never choose the bar after seeing the number.
- Never use `aggregate: best` for folds, seeds or splits.
- Never use `aggregate: bootstrap` for lines that are independent evaluations:
  its standard error assumes the lines resample one evaluation, and using it on
  folds inflates the interval by √k.
- Never compare trials whose data differs without a `baseline_metric`.
- Never quote a mean without its interval, or a winner without whether it is
  separated from the runner-up.
- Never report a campaign that declared no `success_criteria` as a success.
