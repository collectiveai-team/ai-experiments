---
name: proposing-variants
description: Use when a campaign has stalled and the next honest move is to change the workload's code, not its parameters — a new feature, a new model, a different loss. Writes the variant into a sandboxed copy, lets the harness smoke-check it, and spends trials on it only if it runs.
---

# Proposing Variants

A hyperparameter search cannot fix a wrong feature set. When the values stop
moving and the best trial is still short of the bar, the next move is to
change the code — and that is the one part of a campaign an agent actually
writes.

The split is the point. **You** write the file: the hypothesis, the feature,
the model. **The harness** decides whether it runs: it copies the workload,
applies whole files, runs the configured smoke command, and discards anything
that exits non-zero. You never get to say your own variant is good, and the
campaign's verdict machinery — see **defining-goals** — scores it against the
same criteria as every other trial.

## Before anything: is this a code problem?

Read the campaign back first, with **autonomous-experimentation** §4 or
`iax campaign rounds <id> --json`. A variant is the right move when the values
barely move across the space, or the best trial sits mid-range rather than at
an edge. It is the wrong move when every trial failed (a bug), when values
still improve at the budget ceiling (raise `max_trials`), or when the best sits
at an edge of a range (widen that range).

## 1. The goal has to allow it

```yaml
variants:
  enabled: true                       # off by default; a loop that edits code unasked is a surprise
  source_dir: examples/my_workload    # what gets copied; defaults to workload.working_dir
  editable_paths:                     # the files a variant may write
    - "my_workload/features.py"
    - "my_workload/models.py"
  smoke_command: ["uv", "run", "--project", ".", "python", "-m", "my_workload.train", "--self-test"]
  smoke_timeout_seconds: 300
```

Two rules the preflight enforces, and `iax campaign validate --strict` refuses
a goal that breaks either:

- **A smoke command is mandatory in practice.** Without one the harness records
  `smoke_ok: null` — unknown, not passed — and a variant that cannot import
  costs a whole round of trials that all fail the same way.
- **`editable_paths` must name the files.** Empty means a variant may rewrite
  anything in the copy, *including the evaluation it is scored by*. Leave the
  scoring code outside the sandbox: a variant that can edit its own thermometer
  makes every number after it its own.

`source_dir` must be the directory the entrypoint works from, because a trial
on a variant runs that same entrypoint with the copy as its working directory.
An entrypoint that names the source path from outside (`--project
examples/my_workload`) breaks inside the sandbox; make it relative (`--project
.`).

## 2. Write the file, then hand it over

Write the new version as a real file on disk, complete — whole files, not
patches, because a patch that applies with fuzz produces code nobody proposed
and there is no reviewer here to notice.

```bash
iax campaign variant <campaign_id> \
  --edit my_workload/features.py=/tmp/features_with_lag.py \
  --hypothesis "7-day lag of the vacuum ratio; fouling shows up as a trend, not a level" \
  --json
```

`--edit` is repeatable and takes `<path inside the workload>=<local file>`.
The exit code is the smoke check's verdict:

| exit | meaning | what you do |
|---|---|---|
| 0 | the variant was copied, edited and started | go to step 3 |
| 2 | it was rejected — a path outside `editable_paths`, or a failed smoke check | read the output, fix the file, propose again |

A rejected variant is deleted from disk but kept on the record, so
`iax campaign variants <campaign_id>` shows what has already been tried and
why it did not run. Read that before proposing: an unattended loop that does
not will re-propose the same broken idea.

## 3. Spend trials on it

```bash
iax campaign variants <campaign_id>       # variant_id, verdict, edited files
iax campaign suggest <campaign_id> --params '{"model": "hist_gb", "window_days": 90}' \
  --variant var_1a2b3c4d --note "baseline params, new features"
```

Suggested trials run before strategy-planned ones on the next advance and
count against `max_trials`. Run the incumbent's best params first: a variant
compared at different params is not a comparison.

If the variant adds a *value* the search space does not contain — a new model
name, a new label source — widen the space before suggesting it, or
`validate_params` will reject the suggestion:

```bash
iax campaign pause <campaign_id>
iax campaign edit <campaign_id> goal2.yaml     # the wider `choice`; the metric must not change
iax campaign resume <campaign_id>
```

## 4. Read the result as a result

A variant is not better because it was written to be. Its trials carry the
same `objective_value`, the same standard error, and the same place in
`verdict.separated` as every other trial. Quote those. A variant whose lead
over the incumbent sits inside the noise has not been shown to work, however
good the hypothesis was.

Raise `budget.max_trials` before a variant round if the budget is nearly
spent: a variant given one trial is a single noisy observation, not evidence.

## Never

- Never edit the user's working tree to test an idea. That is what the copy is
  for, and two variants editing one tree race.
- Never put the evaluation, the metric reporting or the data split inside
  `editable_paths`.
- Never run a variant whose smoke check failed, and never work around a failing
  smoke check by removing the smoke command.
- Never present a variant's trial as an improvement without saying whether it
  is distinguishable from the incumbent.
- Never propose a variant nobody asked for while the parameter search is still
  improving.
