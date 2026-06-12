"""Reproducibility capture: stamp every run with enough context to re-run it.

On submit, each run dir gets a ``repro/`` bundle:

- ``context.json`` — git SHA, branch, dirty flag, python/platform, timestamp
- ``diff.patch``   — uncommitted changes in the workload's working dir (if any)
- ``environment.txt`` — installed distributions (name==version)

Re-running is then one command: ``iax rerun <run_id>`` resubmits the persisted
manifest (params are baked into args/env), warning when the current git SHA
differs from the recorded one. Capture is best-effort — a missing git binary
or non-repo working dir never blocks a submit.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

from ai_experiments.schemas import utc_now

GIT_TIMEOUT = 10
MAX_DIFF_BYTES = 512_000


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def capture_repro(run_dir: Path, working_dir: str | Path) -> dict[str, Any]:
    """Write the repro bundle into ``run_dir/repro/``; returns the context."""
    repro_dir = run_dir / "repro"
    repro_dir.mkdir(parents=True, exist_ok=True)
    cwd = Path(working_dir).resolve()

    sha = _git(["rev-parse", "HEAD"], cwd)
    context: dict[str, Any] = {
        "captured_at": utc_now().isoformat(),
        "git_sha": sha,
        "git_branch": _git(["rev-parse", "--abbrev-ref", "HEAD"], cwd) if sha else None,
        "git_dirty": None,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "working_dir": str(cwd),
    }

    if sha is not None:
        status = _git(["status", "--porcelain"], cwd)
        context["git_dirty"] = bool(status)
        if status:
            diff = _git(["diff", "HEAD"], cwd) or ""
            (repro_dir / "diff.patch").write_text(diff[:MAX_DIFF_BYTES])

    try:
        lines = sorted(
            f"{dist.metadata['Name']}=={dist.version}"
            for dist in metadata.distributions()
            if dist.metadata["Name"]
        )
        (repro_dir / "environment.txt").write_text("\n".join(lines) + "\n")
    except Exception:
        pass

    (repro_dir / "context.json").write_text(json.dumps(context, indent=2))
    return context


def read_repro(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "repro" / "context.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def current_git_sha(working_dir: str | Path) -> str | None:
    return _git(["rev-parse", "HEAD"], Path(working_dir).resolve())
