"""Checks that a workload could plausibly start, before anything is submitted.

``validate`` used to check the manifest's *shape* only, so a manifest naming an
entrypoint that does not exist passed, submitted, and only failed once a
detached supervisor tried to spawn it. These checks move the cheapest half of
that discovery back to the author's terminal.

They are warnings, never errors: a Ray workload resolves its entrypoint on the
cluster, not here, so a path that is missing locally can be perfectly valid.
``iax validate --strict`` is for callers (CI, an agent) that would rather stop.
"""

from __future__ import annotations

import os
import shlex
import shutil
from pathlib import Path

from ai_experiments.schemas import ExperimentManifest, GoalSpec, WorkloadSpec

#: What every caller prefixes a warning with, so one grep finds them all.
WARNING_PREFIX = "Warning: "


def workload_warnings(
    source: ExperimentManifest | GoalSpec | WorkloadSpec,
) -> list[str]:
    """Reasons this workload looks unable to start on this machine.

    Takes a manifest, a goal, or the workload itself: a campaign runs every
    trial from one `GoalSpec.workload`, so the same check answers for the
    whole campaign (#32).

    Checks every command `phases()` would actually run, not just
    `entrypoint`: for a two-phase workload the entrypoint is the one command
    that never executes, so a typo in `evaluate:` would otherwise pass here
    and surface only after a full training run.
    """
    workload = source if isinstance(source, WorkloadSpec) else source.workload
    warnings: list[str] = []
    working_dir = Path(workload.working_dir)
    if not working_dir.is_dir():
        warnings.append(f"working_dir does not exist: {working_dir}")

    try:
        phases = workload.phases()
    except ValueError as exc:
        # phases() raises on a half-declared workload; this function only
        # ever returns warnings, so that becomes one instead of a crash.
        warnings.append(str(exc))
        return warnings

    # A single-entrypoint workload is `[("evaluate", entrypoint)]`; its
    # messages must stay byte-identical to before this check existed, so the
    # phase name is only prefixed once there is more than one to tell apart.
    prefix = "{phase}: " if len(phases) > 1 else ""
    for phase, command in phases:
        for warning in _command_warnings(command, working_dir):
            warnings.append(prefix.format(phase=phase) + warning)
    return warnings


def _command_warnings(entrypoint: str, working_dir: Path) -> list[str]:
    """Reasons this one command looks unable to start."""
    try:
        argv = shlex.split(entrypoint)
    except ValueError as exc:
        return [f"entrypoint is not a valid command line: {exc}"]
    if not argv:
        return ["entrypoint is empty"]

    program = argv[0]
    if _resolves(program, working_dir):
        return []
    if os.sep in program or program.startswith("."):
        return [
            f"entrypoint {program!r} is not an executable file "
            f"(resolved against working_dir {working_dir})"
        ]
    return [f"entrypoint {program!r} is not on PATH"]


def _resolves(program: str, working_dir: Path) -> bool:
    """Whether the child process would find ``program``.

    Mirrors what ``execvp`` does: a name containing a separator is a path
    relative to the child's CWD (the working dir), and a bare name is looked
    up on PATH -- the CWD is *not* searched.
    """
    if os.sep in program or program.startswith("."):
        candidate = Path(program)
        if not candidate.is_absolute():
            candidate = working_dir / candidate
        return candidate.is_file() and os.access(candidate, os.X_OK)
    return shutil.which(program) is not None
