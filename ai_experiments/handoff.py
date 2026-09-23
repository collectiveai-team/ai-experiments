"""Hand a blocked campaign to a development flow.

A campaign answers a question about parameters. When the answer is "the code
is wrong", the search has nothing left to try: `iax loop` stops the campaign
with `blocked_on_change` and writes a :class:`ChangeRequest` into the inbox.
This module is what picks that ticket up and turns it into development work.

Three decisions shape it:

**The change lands on an experimentation branch.** Trials measured before a
code change and trials measured after it are not comparable, so the fix must
not touch the branch the blocked campaign ran on. Every hand-off creates
`exp/<campaign>-<digest>` in its own git worktree; the successor campaign runs
there, and the two branches stay separately measurable.

**Nothing spends an agent by itself.** The loop only records the ticket. A
person or an agent runs `iax handoff`, and even then the default is
`--no-run`: orq-lite plans the work and stops. Passing `--run` is the explicit
"spend tokens on this" step.

**Workload output never reaches a shell.** `error_tail` is text a failing
workload printed. It travels into a markdown file, and the flow receives that
file *by path* in an argv list. No interpolation, no `shell=True`.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from ai_experiments.monitoring.escalation import ChangeRequest
from ai_experiments.schemas import utc_now
from ai_experiments.store import FilesystemRunStore

#: Where a hand-off keeps its issue file and its ledger entry.
HANDOFF_DIR = "_handoffs"
BRANCH_PREFIX = "exp/"
#: Plan the work and stop. Add `--run` to let the flow execute.
DEFAULT_COMMAND = ("orq-lite", "intake", "--issue", "{issue}", "--no-run")
OUTPUT_TAIL_CHARS = 4000

_UNSAFE_REF = re.compile(r"[^a-zA-Z0-9._-]+")

HandoffStatus = Literal["planned", "launched", "already_handled", "failed"]


class HandoffPlan(BaseModel):
    """What a hand-off will do, before it does any of it."""

    campaign_id: str
    source_key: str
    branch: str
    worktree: Path
    issue_path: Path
    ledger_path: Path
    command: list[str]


class HandoffResult(BaseModel):
    plan: HandoffPlan
    status: HandoffStatus
    created_at: str = Field(default_factory=lambda: utc_now().isoformat())
    exit_code: int | None = None
    output: str = ""
    error: str = ""
    #: The command a caller runs once the fix lands, to measure the branch.
    next_step: str = ""


def slug(text: str) -> str:
    """Make one path- and git-ref-safe token out of arbitrary text."""
    cleaned = _UNSAFE_REF.sub("-", text).strip("-.")
    return cleaned or "unnamed"


def digest_of(request: ChangeRequest) -> str:
    """A short, stable id for this defect, so a retry is recognisable."""
    if request.source_key:
        tail = request.source_key.rsplit(":", 1)[-1]
        if tail and tail != request.source_key:
            return slug(tail)[:12]
    material = "\x00".join([request.campaign_id, request.title, *sorted(request.files)])
    return hashlib.sha256(material.encode()).hexdigest()[:12]


def plan_handoff(
    store: FilesystemRunStore,
    request: ChangeRequest,
    *,
    worktree_root: Path | None = None,
    command: tuple[str, ...] | list[str] = DEFAULT_COMMAND,
) -> HandoffPlan:
    digest = digest_of(request)
    branch = f"{BRANCH_PREFIX}{slug(request.campaign_id)}-{digest}"
    key = request.source_key or f"iax:{request.campaign_id}:{digest}"
    stem = slug(key)
    directory = store.root / HANDOFF_DIR
    root = worktree_root or (store.root / HANDOFF_DIR / "worktrees")
    worktree = root / f"{slug(request.campaign_id)}-{digest}"
    issue_path = directory / f"{stem}.md"
    rendered = [
        part.replace("{issue}", str(issue_path)).replace("{branch}", branch)
        for part in command
    ]
    return HandoffPlan(
        campaign_id=request.campaign_id,
        source_key=key,
        branch=branch,
        worktree=worktree,
        issue_path=issue_path,
        ledger_path=directory / f"{stem}.json",
        command=rendered,
    )


def render_issue(request: ChangeRequest, plan: HandoffPlan) -> str:
    """Write the ticket as markdown a development flow can read.

    The workload output goes inside a fence long enough to survive whatever
    backticks the output itself contains.
    """
    lines = [
        f"# {request.title}",
        "",
        (
            f"`iax` stopped campaign `{request.campaign_id}` because no choice of "
            "parameters can fix what it found. The evidence below comes from the "
            "campaign's own records."
        ),
        "",
        "## Why the search stopped",
        "",
        request.rationale or "(the review gave no reason)",
        "",
        "## Evidence",
        "",
        f"- Campaign: `{request.campaign_id}`",
        f"- Failed trials: {_join(request.trial_ids)}",
        f"- Runs: {_join(request.run_ids)}",
        f"- Suspected files: {_join(request.files)}",
        f"- Reported at: {request.created_at.isoformat()}",
        "",
        "Read the runs before you trust the diagnosis: "
        f"`iax logs <run_id>` and `iax describe {request.campaign_id}`.",
        "",
    ]
    if request.error_tail:
        fence = "`" * max(3, _longest_backtick_run(request.error_tail) + 1)
        lines += [
            "## What the workload printed",
            "",
            fence + "text",
            request.error_tail,
            fence,
            "",
        ]
    lines += [
        "## Acceptance",
        "",
        request.acceptance or "(the review named no check; define one before you code)",
        "",
        "## Where the change belongs",
        "",
        (
            f"Work on branch `{plan.branch}`. It exists already, checked out at "
            f"`{plan.worktree}`. Do not put the fix on the branch the campaign "
            "ran on: trials measured before and after a code change are not "
            "comparable, and mixing them silently invalidates the comparison."
        ),
        "",
        "When the fix lands, the campaign is re-run on that branch:",
        "",
        "```bash",
        f"cd {plan.worktree}",
        f"iax loop <goal.yaml> --name {request.campaign_id}-retry",
        "```",
        "",
    ]
    return "\n".join(lines)


def hand_off(
    store: FilesystemRunStore,
    request: ChangeRequest,
    *,
    repo: Path,
    base: str = "HEAD",
    worktree_root: Path | None = None,
    command: tuple[str, ...] | list[str] = DEFAULT_COMMAND,
    dry_run: bool = False,
    force: bool = False,
) -> HandoffResult:
    """Create the experimentation branch and give the ticket to the flow."""
    plan = plan_handoff(store, request, worktree_root=worktree_root, command=command)
    next_step = f"cd {plan.worktree} && iax loop <goal.yaml>"

    if plan.ledger_path.exists() and not force:
        previous = _read_ledger(plan.ledger_path)
        return HandoffResult(
            plan=plan,
            status="already_handled",
            created_at=str(previous.get("created_at", "")),
            output=str(previous.get("output", "")),
            next_step=next_step,
        )

    if dry_run:
        return HandoffResult(plan=plan, status="planned", next_step=next_step)

    plan.issue_path.parent.mkdir(parents=True, exist_ok=True)
    plan.issue_path.write_text(render_issue(request, plan))

    branch_error = _ensure_worktree(repo, plan.branch, plan.worktree, base)
    if branch_error:
        result = HandoffResult(
            plan=plan, status="failed", error=branch_error, next_step=next_step
        )
        _write_ledger(plan, result)
        return result

    completed = subprocess.run(
        plan.command,
        cwd=plan.worktree,
        capture_output=True,
        text=True,
        check=False,
    )
    output = (completed.stdout + completed.stderr)[-OUTPUT_TAIL_CHARS:]
    result = HandoffResult(
        plan=plan,
        status="launched" if completed.returncode == 0 else "failed",
        exit_code=completed.returncode,
        output=output,
        error="" if completed.returncode == 0 else "the development flow failed",
        next_step=next_step,
    )
    _write_ledger(plan, result)
    return result


def _ensure_worktree(repo: Path, branch: str, worktree: Path, base: str) -> str:
    """Put `branch` in its own checkout. Return an error string, or ''."""
    if (worktree / ".git").exists():
        return ""
    worktree.parent.mkdir(parents=True, exist_ok=True)
    exists = _git(repo, "rev-parse", "--verify", "--quiet", f"refs/heads/{branch}")
    if exists.returncode == 0:
        added = _git(repo, "worktree", "add", str(worktree), branch)
    else:
        added = _git(repo, "worktree", "add", "-b", branch, str(worktree), base)
    if added.returncode != 0:
        return (added.stderr or added.stdout).strip()[-OUTPUT_TAIL_CHARS:]
    return ""


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def _write_ledger(plan: HandoffPlan, result: HandoffResult) -> None:
    plan.ledger_path.parent.mkdir(parents=True, exist_ok=True)
    plan.ledger_path.write_text(result.model_dump_json(indent=2))


def _read_ledger(path: Path) -> dict[str, object]:
    try:
        loaded = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _join(values: list[str]) -> str:
    return ", ".join(f"`{v}`" for v in values) if values else "(none reported)"


def _longest_backtick_run(text: str) -> int:
    return max((len(m) for m in re.findall(r"`+", text)), default=0)
