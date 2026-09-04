"""`iax new`: emit a runnable starting point instead of a blank file.

The wheel ships `ai_experiments` only, so a pip-installed user never sees
`examples/` and has nothing to copy (#19). These templates live inside the
package, and :func:`render` parses each one through the real pydantic model
before returning it — a template that drifts from the schema fails here rather
than at the user's first `iax validate`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml

from ai_experiments.schemas import ExperimentManifest, GoalSpec
from ai_experiments.store import FilesystemRunStore

TemplateKind = Literal["manifest", "goal", "workload"]

TEMPLATES: dict[str, str] = {
    "manifest": "manifest.yaml",
    "goal": "goal.yaml",
    "workload": "workload.py",
}

#: Default filename for each kind, used when the user names a directory.
DEFAULT_NAMES: dict[str, str] = {
    "manifest": "experiment.yaml",
    "goal": "goal.yaml",
    "workload": "train.py",
}

_TEMPLATE_DIR = Path(__file__).parent / "templates"


class ScaffoldError(Exception):
    """A template is unusable, or the target path is not writable."""


def render(kind: TemplateKind) -> str:
    """The template text for ``kind``, proven to parse against its schema."""
    try:
        text = (_TEMPLATE_DIR / TEMPLATES[kind]).read_text()
    except KeyError:
        raise ScaffoldError(
            f"unknown template '{kind}'; choose one of {', '.join(sorted(TEMPLATES))}"
        ) from None
    except OSError as exc:
        raise ScaffoldError(f"template '{kind}' is missing: {exc}") from exc
    _check(kind, text)
    return text


def _check(kind: str, text: str) -> None:
    model = {"manifest": ExperimentManifest, "goal": GoalSpec}.get(kind)
    if model is None:
        return
    try:
        model(**(yaml.safe_load(text) or {}))
    except Exception as exc:  # pragma: no cover - guards a packaging mistake
        raise ScaffoldError(
            f"the '{kind}' template no longer validates: {exc}"
        ) from exc


def resolve_target(kind: TemplateKind, path: Path) -> Path:
    """Where the template lands. A directory gets the conventional filename."""
    if path.is_dir():
        return path / DEFAULT_NAMES[kind]
    return path


def write(kind: TemplateKind, path: Path, force: bool = False) -> Path:
    """Write the template to ``path``; never clobber without ``force``."""
    target = resolve_target(kind, path)
    if target.exists() and not force:
        raise ScaffoldError(f"{target} already exists; pass --force to overwrite")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render(kind))
    return target


def manifest_from_run(store: FilesystemRunStore, run_id: str) -> str:
    """Rebuild a submittable manifest from a stored run.

    The run keeps the exact manifest it was submitted with, params baked in.
    Reusing it is how a user turns a run that worked into the template for the
    next one, without retyping the workload.
    """
    manifest = store.read_manifest(run_id)
    if manifest is None:
        raise ScaffoldError(f"run '{run_id}' has no stored manifest")
    return manifest.to_yaml()
