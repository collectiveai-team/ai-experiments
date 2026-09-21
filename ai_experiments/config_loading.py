"""Turning a YAML file on disk into one of this package's models.

Two doors, and which one a caller uses is a statement about who wrote the file.
A file a human or an agent typed goes through `load_config`, where an unknown key
is an error worth reporting by name. A file iax itself wrote goes through
`load_stored`, which forgives the keys older versions used to emit -- a run
directory has to keep replaying after a field is retired.

The models themselves live in `ai_experiments.schemas`; only the loading lives
here, so neither file has to grow to hold the other.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

from ai_experiments.schema_errors import describe

#: Keys older versions of iax defined, defaulted, documented -- and never read.
#: They are rejected in a hand-written file and dropped from a stored one (#14).
REMOVED_MONITOR_KEYS = ("checks", "no_event_after_minutes")


class ConfigModel(BaseModel):
    """Base for every model a human or an agent writes by hand.

    An unknown key is an error, not a comment: silently dropping ``monitor:``
    for ``monitoring:`` leaves the author sure they configured something they
    did not. Models that only describe *stored* state stay permissive, so an
    older file keeps loading after a field is added.
    """

    model_config = ConfigDict(extra="forbid")


ConfigT = TypeVar("ConfigT", bound=ConfigModel)


def load_config(model: type[ConfigT], path: str | Path) -> ConfigT:
    """Load a hand-written YAML config, reporting a bad key by name."""
    data = _mapping(path)
    return _build(model, data, path)


def load_stored(model: type[ConfigT], path: str | Path) -> ConfigT:
    """Load a config iax itself wrote, tolerating keys older versions emitted.

    A run directory written before a field was removed still has to replay,
    so the stored path drops those keys instead of refusing the file.
    """
    data = _mapping(path)
    monitoring = data.get("monitoring")
    if isinstance(monitoring, dict):
        for key in REMOVED_MONITOR_KEYS:
            monitoring.pop(key, None)
    return _build(model, data, path)


def _mapping(  # ast-grep-ignore: no-dict-return-annotation
    path: str | Path,
) -> dict[str, Any]:
    """Read a YAML file that has to be a mapping of fields.

    A raw dict is the point: this runs *before* the model exists, and the
    caller's job is to hand these fields to the model that gives them a shape.
    """
    with Path(path).open() as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a YAML mapping of fields")
    return data


def _build(model: type[ConfigT], data: dict[str, Any], path: str | Path) -> ConfigT:
    """Construct the model, restating a validation failure in the file's terms."""
    try:
        return model(**data)
    except ValidationError as exc:
        raise ValueError(describe(model, exc)) from exc
