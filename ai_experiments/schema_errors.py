"""Turn a pydantic ``ValidationError`` over a config file into one readable line.

An agent writes the goal and manifest files this package reads (#14). When it
misspells a key, the useful answer is the key it should have written, not a
nested pydantic dump. So every config entry point renders its errors here:

```
monitor: unknown field; did you mean 'monitoring'?
budget.max_trails: unknown field; known fields: gpu_hour_rate, max_gpu_hours, ...
```
"""

from __future__ import annotations

import difflib
from typing import Any, get_args

from pydantic import BaseModel, ValidationError

#: Below this ratio a "did you mean" is a guess, not a correction.
_SIMILAR_ENOUGH = 0.6


def describe(model: type[BaseModel], error: ValidationError) -> str:
    """Render every problem in `error` as one semicolon-separated line."""
    return "; ".join(_describe_one(model, item) for item in error.errors())


def _describe_one(model: type[BaseModel], item: dict[str, Any]) -> str:
    location = item["loc"]
    where = ".".join(str(part) for part in location) or "<root>"
    if item["type"] != "extra_forbidden":
        # pydantic prefixes a raised ValueError with "Value error, "; the
        # author needs the message, not the exception class.
        message = item["msg"].removeprefix("Value error, ")
        return f"{where}: {message}"
    owner = _model_at(model, location[:-1])
    return f"{where}: unknown field{_hint(str(location[-1]), owner)}"


def _hint(key: str, owner: type[BaseModel] | None) -> str:
    if owner is None:
        return ""
    known = sorted(owner.model_fields)
    close = difflib.get_close_matches(key, known, n=1, cutoff=_SIMILAR_ENOUGH)
    if close:
        return f"; did you mean '{close[0]}'?"
    return f"; known fields: {', '.join(known)}"


def _model_at(model: type[BaseModel], location: tuple[Any, ...]) -> Any:
    """Walk a field path to the model that owns the last key, if we can.

    A path through a list index or a discriminated union tag is not a field
    name; those return None, and the caller drops the hint rather than guess.
    """
    current: Any = model
    for part in location:
        if not isinstance(part, str):
            return None
        field = current.model_fields.get(part)
        if field is None:
            return None
        current = _model_in(field.annotation)
        if current is None:
            return None
    return current


def _model_in(annotation: Any) -> Any:
    """Find the model inside `X | None`, `list[X]`, `dict[str, X]`, ..."""
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    for argument in get_args(annotation):
        found = _model_in(argument)
        if found is not None:
            return found
    return None
