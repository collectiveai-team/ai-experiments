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
import re
import shlex
import shutil
import subprocess
from pathlib import Path

from ai_experiments.planner.planner import flag_for
from ai_experiments.schemas import ExperimentManifest, GoalSpec, WorkloadSpec

#: What every caller prefixes a warning with, so one grep finds them all.
WARNING_PREFIX = "Warning: "

#: How long the ``--help`` probe may take. A workload that imports torch is
#: slow to start, and a workload that ignores ``--help`` and starts training
#: must not hold up the campaign that was only asking a question.
HELP_TIMEOUT_SECONDS = 20

#: A long option as an argument parser prints it in its usage or options
#: block: ``--window-days``, ``--window-days N``, ``--window-days=N``.
_LONG_OPTION = re.compile(r"--[A-Za-z0-9][A-Za-z0-9_-]*")


def workload_warnings(
    source: ExperimentManifest | GoalSpec | WorkloadSpec,
) -> list[str]:
    """Reasons this workload looks unable to start on this machine.

    Takes a manifest, a goal, or the workload itself: a campaign runs every
    trial from one `GoalSpec.workload`, so the same check answers for the
    whole campaign (#32).
    """
    workload = source if isinstance(source, WorkloadSpec) else source.workload
    warnings: list[str] = []
    working_dir = Path(workload.working_dir)
    if not working_dir.is_dir():
        warnings.append(f"working_dir does not exist: {working_dir}")

    try:
        argv = shlex.split(workload.entrypoint)
    except ValueError as exc:
        return [*warnings, f"entrypoint is not a valid command line: {exc}"]
    if not argv:
        return [*warnings, "entrypoint is empty"]

    program = argv[0]
    if _resolves(program, working_dir):
        # The args go into the probe too: `uv run` and `python -m` are
        # launchers, and `uv run --help` answers for uv, not for the workload
        # the campaign will actually start.
        return warnings + _flag_warnings(source, [*argv, *workload.args], working_dir)
    if os.sep in program or program.startswith("."):
        warnings.append(
            f"entrypoint {program!r} is not an executable file "
            f"(resolved against working_dir {working_dir})"
        )
    else:
        warnings.append(f"entrypoint {program!r} is not on PATH")
    return warnings


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


def _flag_warnings(
    source: ExperimentManifest | GoalSpec | WorkloadSpec,
    argv: list[str],
    working_dir: Path,
) -> list[str]:
    """Search space keys this workload's parser does not appear to accept.

    Every trial is handed every key on its command line, so one key the
    parser never declared fails *all* of them, identically, with the same
    exit code -- the most expensive failure a campaign can have and the
    cheapest one to catch, since the answer is in ``--help``.

    Silent when the probe cannot answer: no help output, a parser that
    prints nothing recognizable, a workload that ignores ``--help``. Absence
    of evidence is not evidence the flags are wrong, and a false warning on
    every campaign start would teach people to ignore the real one.

    ``argv`` must be the whole command line, entrypoint *and* args. Probing
    the entrypoint alone reads the launcher's help -- `uv run --help` lists
    uv's options, none of which are the workload's, so every search space key
    looks undeclared.
    """
    if not isinstance(source, GoalSpec) or not source.search_space:
        return []
    declared = _declared_options(argv, working_dir)
    if not declared:
        return []
    style = source.workload.flag_style
    missing = [
        (name, flag_for(name, style))
        for name in sorted(source.search_space)
        if flag_for(name, style) not in declared
    ]
    return [
        f"the workload's --help does not declare {flag} "
        f"for search space key {name!r}; every trial would be sent it"
        for name, flag in missing
    ]


def _declared_options(argv: list[str], working_dir: Path) -> set[str]:
    """The long options this workload's ``--help`` mentions."""
    try:
        completed = subprocess.run(  # noqa: S603 - the user's own entrypoint
            [*argv, "--help"],
            cwd=working_dir,
            capture_output=True,
            text=True,
            timeout=HELP_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return set()
    if completed.returncode != 0:
        return set()
    return set(_LONG_OPTION.findall(completed.stdout + completed.stderr))


def goal_warnings(goal: GoalSpec) -> list[str]:
    """Ways a goal can be valid, run to completion, and still prove nothing.

    Separate from :func:`workload_warnings` because these are about the
    question, not the machinery: nothing here will make a trial crash. They
    are the failures that only show up at the end, when the campaign has a
    best trial and no rule that says whether that number was the point.
    """
    warnings: list[str] = []
    objective = goal.objective
    criteria = goal.success_criteria

    if not criteria.declared:
        warnings.append(
            "this goal declares no success_criteria, so the campaign will "
            "report a best trial without saying whether it is good enough; "
            "decide the bar now, not after seeing the number"
        )

    if objective.baseline_metric is None:
        moving = sorted(
            name for name, spec in goal.search_space.items() if spec.changes_data
        )
        if moving:
            warnings.append(
                f"{', '.join(moving)} change the data a trial is scored on, but "
                f"objective '{objective.metric}' has no baseline_metric; trials "
                "on different slices are not comparable and the search will "
                "reward the easiest slice"
            )

    if criteria.require_separation and objective.aggregate not in ("mean", "bootstrap"):
        warnings.append(
            "success_criteria.require_separation needs an interval, which "
            "only objective.aggregate='mean' or 'bootstrap' produces; as "
            "written the criterion can never be met"
        )

    if criteria.min_observations is not None and objective.aggregate == "bootstrap":
        warnings.append(
            "success_criteria.min_observations counts observations, and under "
            "objective.aggregate='bootstrap' those are resamples of one "
            "evaluation: their number is how long the workload chose to "
            "resample, not how much evidence it had. A criterion any workload "
            "can satisfy by looping more is not a criterion — let the workload "
            "report no objective when its evidence is too thin instead"
        )

    if criteria.require_beats_baseline and objective.baseline_metric is None:
        warnings.append(
            "success_criteria.require_beats_baseline has no baseline to beat: "
            "objective.baseline_metric is not set, so the criterion can never "
            "be met"
        )

    if goal.variants.enabled:
        if not goal.variants.smoke_command:
            warnings.append(
                "variants.enabled is on with no variants.smoke_command; a "
                "variant that cannot start would be handed a whole round of "
                "trials that all fail the same way"
            )
        if not goal.variants.editable_paths:
            warnings.append(
                "variants.enabled is on with no variants.editable_paths, so a "
                "variant may rewrite any file in the workload — including the "
                "evaluation it is scored by; name the files that may change"
            )

    return warnings
