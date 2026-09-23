# The CES convention

`AGENTS.md` (and `CLAUDE.md`, a symlink to it) carries the index of the house engineering
standards. This document is the convention that index is written in: what a rule is made of,
what its tags promise, where its parts live, and how to change one.

## A rule's three names

Every standard is a **CES — Collective Engineering Standard** and has three identifiers:

| Name | Example | Used by |
|---|---|---|
| **Code** | `CES-71` | Humans, commit messages, review comments. Cite this. |
| **Slug** | `file-size-guard` | Tooling: the rule's detail filename, its `ast-grep/rules/<slug>.yml`, and `# ast-grep-ignore: <slug>` suppressions. |
| **Title** | *keep files small* | Prose. Never load-bearing. |

The number in the code is the issue number in
[`collectiveai-team/scaffolding`](https://github.com/collectiveai-team/scaffolding/issues),
where the rule was proposed and argued. Each detail file links back to it under **Tracker**.
The numbers are therefore not contiguous and carry no ordering — `CES-4` is not older or more
important than `CES-119`.

## Enforcement tags

The tag after a rule's title says *what will catch a violation*, which is the difference
between a rule you can forget and a rule you must remember. Current inventory, all 31 entries:

| Tag | Meaning | Count |
|---|---|---|
| `[ast-grep]` | A structural pattern in `ast-grep/rules/<slug>.yml`, run by the `ast-grep` prek hook. | 8 |
| `[prek]` | A prek hook that is not ast-grep — a linter, a script, a commit-msg gate. | 10 |
| `[ci]` | A GitHub Actions workflow only. **No local hook**: it first fails on the PR. | 4 |
| `[dependency]` | A dev-dependency that activates on install; nothing separate to run. | 2 |
| `[script]` | A repo script maintains the artifact; the check is that you ran it. | 1 |
| `[snippet]` | Ships canonical drop-in code under `.agents/snippets/`. | 2 |
| `[judgment]` | No machine check. A reviewer or agent applies it. | 11 |

A rule may carry two tags — `CES-91` is `[prek]` `[ci]`, a local commit-msg hook for fast
feedback plus `commit-policy.yml` as the authority that cannot be bypassed with `--no-verify`.
Where a local hook and CI disagree, **CI is the source of truth.**

`[judgment]` is the largest group. That is deliberate: the rules worth having mostly cannot be
pattern-matched, and tagging one `[judgment]` is an honest statement that nothing will stop you.
Never tag a rule `[prek]` or `[ast-grep]` unless the hook exists — a false tag reads as "a
machine checks this" and buys silence instead of attention.

## Where the parts live

```
AGENTS.md                      the index: one entry per rule (CLAUDE.md -> AGENTS.md)
.agents/rules/<slug>.md        the detail: directive, rationale, examples, tracker link
.agents/snippets/**            canonical drop-in code for [snippet] rules
ast-grep/rules/<slug>.yml      the pattern for [ast-grep] rules
prek.toml                      the hooks for [prek] rules
.github/workflows/             the workflows for [ci] rules
.agents/skills/                installed agent skills -- derived, gitignored (CES-107)
```

`.agents/rules/` and `.agents/snippets/` are **tracked**. They were not always: a blanket
`.agents` entry in `.gitignore` once made the whole catalog unreachable from a clone while
`AGENTS.md` linked to all 30 files. Only `.agents/skills/` stays ignored, because
`skills-lock.json` is the tracked manifest it is restored from.

Every rule links to its detail file except `CES-77`, which points at a `pyproject.toml` comment
because the thing it governs *is* that comment.

## Authority

1. **`.agents/rules/<slug>.md`** — authoritative wherever it exists.
2. The `engineering-rules` skill bundle — the same standards carried into repos that have no
   `.agents/rules/`. Several ids are shared. Where both are present, `.agents/` wins.
3. **`CES-30` outranks both.** An existing deliberate local choice — a pin, a layout, a
   configured linter — beats a house default. Adopt a standard as an explicit migration, never
   as a silent overwrite.

## Suppressing a rule

`ast-grep` rules suppress with a trailing comment naming the slug; ruff uses `noqa`:

```python
env = os.getenv("ENV", "dev")  # ast-grep-ignore: settings-module
```

Two requirements, both load-bearing:

- **Name the slug.** A bare `# ast-grep-ignore` disables every rule on that line, including the
  ones you have not thought about.
- **Say why, truthfully.** A suppression whose stated reason is false is worse than no
  suppression: it replaces the next reader's judgment with a wrong answer instead of prompting
  them to form their own.

## The snippets are vendored, not ours

`.agents/snippets/` is upstream template code, kept **byte-identical** to the scaffolding repo
so `diff` still answers "has upstream fixed this yet?". It is therefore excluded from this
repo's linters — `[tool.ruff] extend-exclude`, `ast-grep --globs`, `[tool.pyrefly]
project-excludes` — because a defect in it is an upstream issue to file, not a gate this repo
can pass. Fix the *installed* copy (for `CES-74`, `ai_experiments/core/logger.py`), leave the
snippet alone, and file upstream.

## Changing a rule

A rule is a shared standard across repos, so it changes upstream first:

1. Open or find the issue in `collectiveai-team/scaffolding`; the issue number becomes the code.
2. Land the detail file, any hook/pattern, and the `AGENTS.md` entry together. A rule whose
   index entry claims an enforcement that does not exist is the defect this convention exists
   to prevent.
3. Pull it into this repo, then re-run `uvx prek run --all-files` — a new hook that fails on
   existing code needs either fixes or a documented, time-boxed exception in the same change.

The index and the catalog can be checked against each other mechanically: every `@.agents/rules/`
link in `AGENTS.md` should resolve to a file, and every file in `.agents/rules/` should be linked
from exactly one entry.
