"""The shipped skills are part of the product, so they are held to the code.

A skill that names a command the CLI does not have sends an agent into a
loop of exit-2 retries. These tests fail the build instead.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from ai_experiments.cli import app
from ai_experiments.cli_support import EXIT_GOAL_NOT_REACHED

SKILLS_DIR = Path(__file__).resolve().parent.parent / ".claude" / "skills"
SKILL_FILES = sorted(p for p in SKILLS_DIR.glob("*/SKILL.md") if p.is_file())

#: `iax <verb>` in a fenced block, with an optional sub-verb.
COMMAND = re.compile(r"^\s*iax\s+([a-z-]+)(?:\s+([a-z-]+))?", re.MULTILINE)


def _leaf(command) -> str:
    """Typer leaves `name` as a placeholder when the decorator did not set it."""
    if isinstance(command.name, str) and command.name:
        return command.name
    return command.callback.__name__.replace("_", "-")


def _command_names() -> set[str]:
    names = {_leaf(command) for command in app.registered_commands}
    for group in app.registered_groups:
        sub = group.typer_instance
        assert sub is not None, f"registered group {group.name} has no typer instance"
        group_name = group.name if isinstance(group.name, str) else (sub.info.name or "")
        names.add(group_name)
        names.update(f"{group_name} {_leaf(c)}" for c in sub.registered_commands)
    return names


def test_the_repo_ships_the_autonomous_experimentation_skill():
    """It is the entry point the README and AGENTS.md both point at."""
    assert (SKILLS_DIR / "autonomous-experimentation" / "SKILL.md").is_file()
    assert (SKILLS_DIR / "autonomous-experimentation" / "reference" / "goal.md").is_file()


@pytest.mark.parametrize("path", SKILL_FILES, ids=lambda p: p.parent.name)
def test_every_skill_declares_a_name_and_a_description(path):
    text = path.read_text()
    assert text.startswith("---\n")
    front = yaml.safe_load(text.split("---\n")[1])
    assert front["name"] == path.parent.name
    assert front["description"].strip()


@pytest.mark.parametrize("path", SKILL_FILES, ids=lambda p: p.parent.name)
def test_every_iax_command_a_skill_names_exists(path):
    known = _command_names()
    for verb, sub in COMMAND.findall(path.read_text()):
        if sub and f"{verb} {sub}" in known:
            continue
        assert verb in known, f"{path.parent.name} names unknown command `iax {verb}`"


def test_agents_md_documents_the_real_exit_codes():
    """An agent branches on these numbers; they cannot drift from the code."""
    text = (SKILLS_DIR.parent.parent / "AGENTS.md").read_text()
    for code in (
        EXIT_GOAL_NOT_REACHED,
        2,
        3,
    ):
        assert f"| {code} |" in text


def test_the_repo_ships_the_defining_goals_skill():
    """The pre-flight half of the contract.

    What an agent decides before a single trial runs, and cannot honestly decide afterwards.
    """
    assert (SKILLS_DIR / "defining-goals" / "SKILL.md").is_file()


def test_the_goal_template_would_settle_something():
    """`iax new goal` is the file agents copy, so it teaches whatever it contains.

    A template that draws a goal warning teaches the warning.
    """
    from ai_experiments.preflight import goal_warnings
    from ai_experiments.scaffold import render
    from ai_experiments.schemas import GoalSpec

    goal = GoalSpec(**yaml.safe_load(render("goal")))
    assert goal_warnings(goal) == []


#: The fields that decide whether a campaign's answer means anything. A skill
#: that tells an agent to run campaigns without naming these is teaching the
#: version of iax that reported a best trial and proved nothing.
VERDICT_FIELDS = (
    "aggregate",
    "baseline_metric",
    "changes_data",
    "when",
    "success_criteria",
    "min_objective",
    "min_observations",
    "require_separation",
    "require_beats_baseline",
)


def _schema_field_names() -> set[str]:
    from ai_experiments.schemas import (
        GoalSpec,
        ObjectiveSpec,
        ParamBase,
        SuccessCriteria,
    )

    names: set[str] = set()
    for model in (GoalSpec, ObjectiveSpec, SuccessCriteria, ParamBase):
        names.update(model.model_fields)
    return names


@pytest.mark.parametrize("field", VERDICT_FIELDS)
def test_defining_goals_documents_every_field_that_decides_a_verdict(field):
    """Both directions: the skill names the field, and the field is real."""
    assert field in _schema_field_names(), f"{field} is not a schema field any more"
    text = (SKILLS_DIR / "defining-goals" / "SKILL.md").read_text()
    assert f"`{field}`" in text, f"defining-goals never mentions {field}"
