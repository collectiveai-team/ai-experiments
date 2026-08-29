"""Every stop reason the orchestrator can emit must be documented (#25).

A campaign ends with one word, and that word is the whole answer for the
agent reading `summary.json`. When someone adds a tenth reason and forgets
the docs, the agent meets a word nothing explains. These tests read the
source, not a hand-written list, so the drift is caught here.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from ai_experiments import orchestrator
from ai_experiments.orchestrator import STOP_REASONS

SOURCE = Path(orchestrator.__file__)
#: The functions that decide how a campaign ends.
DECIDERS = ("_stop_reason", "_exhausted_reason", "_objective_contract_broken")
SKILL = (
    Path(__file__).resolve().parents[1] / ".claude/skills/running-campaigns/SKILL.md"
)


def _module() -> ast.Module:
    return ast.parse(SOURCE.read_text())


def _literals_returned_by(name: str) -> set[str]:
    """Collect what a decider function can hand back, resolving constants."""
    module = _module()
    globals_ = {
        target.id: node.value.value
        for node in module.body
        if isinstance(node, ast.Assign | ast.AnnAssign)
        for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
        if isinstance(target, ast.Name)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    }
    found: set[str] = set()
    for node in ast.walk(module):
        if not isinstance(node, ast.FunctionDef) or node.name != name:
            continue
        for inner in ast.walk(node):
            if not isinstance(inner, ast.Return) or inner.value is None:
                continue
            value = inner.value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                found.add(value.value)
            elif isinstance(value, ast.Name) and value.id in globals_:
                found.add(globals_[value.id])
    return found


def test_the_deciders_are_still_named_what_this_test_thinks():
    """A renamed decider would make every assertion below vacuously pass."""
    defined = {
        node.name for node in ast.walk(_module()) if isinstance(node, ast.FunctionDef)
    }

    assert set(DECIDERS) <= defined


@pytest.mark.parametrize("name", DECIDERS)
def test_every_reason_a_decider_returns_is_documented(name):
    returned = _literals_returned_by(name)

    assert returned, f"{name} returned no string literal; did it get rewritten?"
    assert returned <= set(STOP_REASONS), (
        f"{name} can return {sorted(returned - set(STOP_REASONS))}, "
        "which STOP_REASONS does not document"
    )


def test_the_default_stop_reason_is_documented():
    """`iax campaign stop` takes the default of CampaignOrchestrator.stop."""
    defaults: set[str] = set()
    for node in ast.walk(_module()):
        if isinstance(node, ast.FunctionDef) and node.name == "stop":
            defaults = {
                default.value
                for default in node.args.defaults
                if isinstance(default, ast.Constant) and isinstance(default.value, str)
            }

    assert defaults
    assert defaults <= set(STOP_REASONS)


def test_no_reason_is_documented_that_nothing_can_emit():
    """A stale row teaches the agent to expect a word that never comes."""
    emitted = {"user_requested"}
    for name in DECIDERS:
        emitted |= _literals_returned_by(name)

    assert set(STOP_REASONS) == emitted


def test_the_campaign_skill_explains_every_reason():
    text = SKILL.read_text()

    missing = [reason for reason in STOP_REASONS if reason not in text]

    assert not missing, f"{SKILL.name} never mentions {missing}"


def test_the_failure_reasons_are_a_subset_of_the_documented_ones():
    assert orchestrator.FAILURE_STOP_REASONS <= set(STOP_REASONS)
