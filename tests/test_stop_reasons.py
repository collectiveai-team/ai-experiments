"""Every stop reason a campaign can end with must be documented (#25).

A campaign ends with one word, and that word is the whole answer for the
agent reading `summary.json`. When someone adds a tenth reason and forgets
the docs, the agent meets a word nothing explains. These tests read the
source, not a hand-written list, so the drift is caught here.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from ai_experiments import orchestrator, stopping
from ai_experiments.stopping import STOP_REASONS

#: `stopping` owns the vocabulary and the checks; the orchestrator only owns the
#: default `iax campaign stop` hands in, which is read from its own source below.
SOURCE = Path(stopping.__file__)
ORCHESTRATOR_SOURCE = Path(orchestrator.__file__)
#: The leaf functions that name how a campaign ends. Each returns its reason as
#: a literal, which is what the tests below read.
DECIDERS = (
    "exhausted_reason",
    "objective_contract_broken",
    "all_trials_failing_reason",
    "target_reached_reason",
    "max_hours_reason",
    "gpu_hours_reason",
    "budget_exhausted_reason",
)
#: The composer: a priority list of calls to the leaves, with no reason of its
#: own. It is checked separately because there is no literal in it to read.
COMPOSER = "stop_reason"
#: What `stop_reason` must keep delegating to. `exhausted_reason` is absent on
#: purpose: `advance` calls it directly, when a round submitted nothing.
COMPOSED = tuple(d for d in DECIDERS if d != "exhausted_reason")
SKILL = Path(__file__).resolve().parents[1] / ".claude/skills/running-campaigns/SKILL.md"


def _module() -> ast.Module:
    return ast.parse(SOURCE.read_text())


def _string_globals(  # ast-grep-ignore: no-dict-return-annotation
    module: ast.Module,
) -> dict[str, str]:
    """Module-level ``NAME = "literal"`` bindings, so a returned name resolves.

    A name-to-literal map is what this is; a model would only rename it.
    """
    bindings: dict[str, str] = {}
    for node in module.body:
        if not isinstance(node, ast.Assign | ast.AnnAssign):
            continue
        if not (isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if isinstance(target, ast.Name):
                bindings[target.id] = node.value.value
    return bindings


def _strings_in(expression: ast.expr, bindings: dict[str, str]) -> set[str]:
    """Every string a returned expression can evaluate to.

    Walked, not matched: a decider may return the reason outright or through a
    conditional (``"target_reached" if reached else None``).
    """
    found: set[str] = set()
    for node in ast.walk(expression):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            found.add(node.value)
        elif isinstance(node, ast.Name) and node.id in bindings:
            found.add(bindings[node.id])
    return found


def _literals_returned_by(name: str) -> set[str]:
    """Collect what a decider function can hand back, resolving constants."""
    module = _module()
    bindings = _string_globals(module)
    found: set[str] = set()
    for node in ast.walk(module):
        if not isinstance(node, ast.FunctionDef) or node.name != name:
            continue
        for inner in ast.walk(node):
            if isinstance(inner, ast.Return) and inner.value is not None:
                found |= _strings_in(inner.value, bindings)
    return found


def test_the_deciders_are_still_named_what_this_test_thinks():
    """A renamed decider would make every assertion below vacuously pass."""
    defined = {node.name for node in ast.walk(_module()) if isinstance(node, ast.FunctionDef)}

    assert {*DECIDERS, COMPOSER} <= defined


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
    orchestrator_module = ast.parse(ORCHESTRATOR_SOURCE.read_text())
    defaults: set[str] = set()
    for node in ast.walk(orchestrator_module):
        if isinstance(node, ast.FunctionDef) and node.name == "stop":
            defaults = {
                default.value
                for default in node.args.defaults
                if isinstance(default, ast.Constant) and isinstance(default.value, str)
            }

    assert defaults
    assert defaults <= set(STOP_REASONS)


def test_the_composer_delegates_rather_than_naming_a_reason_itself():
    """`stop_reason` is a priority list of calls to the leaves.

    A reason inlined there instead of delegated is one the per-decider tests
    above would never read, so the drift they exist to catch would slip past.
    """
    called: set[str] = set()
    for node in ast.walk(_module()):
        if not isinstance(node, ast.FunctionDef) or node.name != COMPOSER:
            continue
        for inner in ast.walk(node):
            if isinstance(inner, ast.Call):
                func = inner.func
                called.add(
                    func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
                )

    assert set(COMPOSED) <= called
    assert not _literals_returned_by(COMPOSER), (
        f"{COMPOSER} names a reason itself; give it a leaf decider instead"
    )


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
    assert set(STOP_REASONS) >= stopping.FAILURE_STOP_REASONS
