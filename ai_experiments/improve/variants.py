"""Workload variants: improvement cycles that change the code, not just the params.

A hyperparameter search cannot fix a wrong loss function. When the objective
stalls, the next honest move is to change the workload — and an unattended
loop must be able to do that without editing the user's working tree.

A variant is a copy of the workload's ``working_dir`` under
``<campaign_dir>/variants/<variant_id>/``, with a set of whole-file edits
applied. Trials run against the copy, so:

* the user's tree is never modified, and two variants never race;
* a bad variant is discarded by deleting a directory;
* the exact code of any trial stays readable after the fact.

Every edit path is resolved inside the variant root before it is written. A
path that escapes — ``../``, an absolute path, a symlink pointing out — is
rejected, not clamped: an agent that proposes one has misunderstood the task,
and silently rewriting its intent hides that.
"""

from __future__ import annotations

import shutil
import subprocess
import uuid
from datetime import datetime  # noqa: TC003  # pydantic resolves this field at runtime
from fnmatch import fnmatch
from pathlib import Path

from pydantic import BaseModel, Field

from ai_experiments.schemas import VariantSpec, utc_now

VARIANTS_DIR = "variants"

#: Directories never worth copying into a variant, and often enormous.
SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "node_modules",
    ".ipynb_checkpoints",
}


class VariantError(Exception):
    """A variant cannot be built: an unsafe path, or a source that is missing."""


class VariantEdit(BaseModel):
    """One whole file, written into the variant.

    Whole files, not patches: a patch that applies with fuzz produces code
    nobody proposed, and there is no reviewer here to notice.
    """

    path: str
    content: str
    rationale: str = ""


class VariantRecord(BaseModel):
    """A materialized variant, and what the smoke check made of it."""

    variant_id: str
    root: str
    source_dir: str
    created_at: datetime = Field(default_factory=utc_now)
    parent: str | None = None
    edited_paths: list[str] = Field(default_factory=list)
    hypothesis: str = ""
    rationale: str = ""
    smoke_ok: bool | None = None
    smoke_output: str = ""
    discarded: bool | None = None
    discard_error: str = ""


def variants_root(campaign_dir: str | Path) -> Path:
    return Path(campaign_dir) / VARIANTS_DIR


def resolve_edit_path(root: Path, relative: str, spec: VariantSpec) -> Path:
    """Return the absolute path an edit may write, or raise.

    ``root`` must already exist and be resolved, so that a symlink planted
    inside the copy cannot redirect a write outside it.
    """
    if not relative or relative.strip() != relative:
        raise VariantError(f"invalid edit path: {relative!r}")
    candidate = Path(relative)
    if candidate.is_absolute():
        raise VariantError(f"edit path must be relative to the workload: {relative}")
    target = (root / candidate).resolve()
    if target != root and root not in target.parents:
        raise VariantError(f"edit path escapes the variant root: {relative}")
    if spec.editable_paths and not any(
        fnmatch(relative, pattern) for pattern in spec.editable_paths
    ):
        raise VariantError(
            f"edit path {relative} is not in variants.editable_paths "
            f"({', '.join(spec.editable_paths)})"
        )
    return target


def materialize_variant(
    campaign_dir: str | Path,
    source_dir: str | Path,
    edits: list[VariantEdit],
    spec: VariantSpec | None = None,
    *,
    parent: str | None = None,
    hypothesis: str = "",
    rationale: str = "",
) -> VariantRecord:
    """Copy the workload, apply the edits, and return the record."""
    spec = spec or VariantSpec()
    source = Path(source_dir).resolve()
    if not source.is_dir():
        raise VariantError(f"workload source directory does not exist: {source}")

    variant_id = f"var_{uuid.uuid4().hex[:8]}"
    root = variants_root(campaign_dir) / variant_id
    root.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, root, ignore=_ignore, symlinks=False)
    resolved_root = root.resolve()

    edited: list[str] = []
    for edit in edits:
        target = resolve_edit_path(resolved_root, edit.path, spec)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(edit.content)
        edited.append(edit.path)

    return VariantRecord(
        variant_id=variant_id,
        root=str(resolved_root),
        source_dir=str(source),
        parent=parent,
        edited_paths=edited,
        hypothesis=hypothesis,
        rationale=rationale,
    )


def smoke_check(record: VariantRecord, spec: VariantSpec) -> VariantRecord:
    """Run the configured smoke command inside the variant.

    A variant that cannot start wastes the whole round: every trial in it
    fails the same way, and the campaign learns nothing. The smoke check is
    what makes an agent-written variant safe to spend a round on. With no
    command configured, ``smoke_ok`` stays ``None`` — unknown, not passed.
    """
    if not spec.smoke_command:
        return record
    try:
        completed = subprocess.run(  # noqa: S603  # user-configured smoke command
            spec.smoke_command,
            cwd=record.root,
            capture_output=True,
            text=True,
            timeout=spec.smoke_timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        record.smoke_ok = False
        record.smoke_output = f"smoke check timed out after {spec.smoke_timeout_seconds}s"
        return record
    except OSError as exc:
        record.smoke_ok = False
        record.smoke_output = f"smoke check could not run: {exc}"
        return record
    record.smoke_ok = completed.returncode == 0
    record.smoke_output = ((completed.stdout or "") + (completed.stderr or ""))[-4000:]
    return record


def discard_variant(record: VariantRecord) -> VariantRecord:
    """Delete a variant's directory, and say whether the directory is gone.

    Used for a variant that failed its smoke check. The smoke command runs
    inside the copy and can leave a subprocess still writing there (``uv``
    populating ``.venv``), so the first delete can fail on a directory that
    is empty a moment later; it is retried once. Whatever is left is recorded
    on the returned record instead of being swallowed: a leftover directory
    is a variant the campaign believes it discarded and did not.
    """
    root = Path(record.root)
    error = ""
    for _ in range(2):
        if not root.exists():
            break
        try:
            shutil.rmtree(root)
        except OSError as exc:
            error = str(exc)
    gone = not root.exists()
    return record.model_copy(update={"discarded": gone, "discard_error": "" if gone else error})


def _ignore(directory: str, names: list[str]) -> set[str]:
    return {name for name in names if name in SKIP_DIRS}
