"""The shipped skills are part of the product, so they are held to the code.

A skill that names a command the CLI does not have sends an agent into a
loop of exit-2 retries. These tests fail the build instead.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

import ai_experiments.cli as cli_module
from ai_experiments.cli import app

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
        group_name = group.name if isinstance(group.name, str) else sub.info.name
        names.add(group_name)
        names.update(f"{group_name} {_leaf(c)}" for c in sub.registered_commands)
    return names


def test_the_repo_ships_the_autonomous_experimentation_skill():
    """It is the entry point the README and AGENTS.md both point at."""
    assert (SKILLS_DIR / "autonomous-experimentation" / "SKILL.md").is_file()
    assert (
        SKILLS_DIR / "autonomous-experimentation" / "reference" / "goal.md"
    ).is_file()


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
        cli_module.EXIT_GOAL_NOT_REACHED,
        2,
        3,
    ):
        assert f"| {code} |" in text
