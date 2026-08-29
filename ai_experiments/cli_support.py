"""One error and exit-code convention for the whole `iax` CLI.

An agent drives this CLI in a loop. It cannot branch on a python traceback,
and it cannot tell a typo'd id from a healthy empty result when both exit 0.
So every command that can fail raises :class:`IaxError`, and one handler turns
it into either a human line on stderr or a JSON object on stdout:

```json
{"error": "unknown run 'run_nope'", "code": "not_found"}
```

Exit codes are part of the CLI contract (see CONVENTIONS.md §6):

| code | meaning |
|---|---|
| 0 | success |
| 1 | the thing asked for does not exist (`not_found`) |
| 2 | the input is invalid — bad YAML, bad params (`invalid_input`) |
| 3 | the execution backend could not be reached (`backend_unavailable`) |
"""

from __future__ import annotations

import json
from typing import Any, NoReturn

import typer

EXIT_OK = 0
EXIT_NOT_FOUND = 1
EXIT_INVALID_INPUT = 2
EXIT_BACKEND_UNAVAILABLE = 3

_EXIT_FOR_CODE = {
    "not_found": EXIT_NOT_FOUND,
    "invalid_input": EXIT_INVALID_INPUT,
    "backend_unavailable": EXIT_BACKEND_UNAVAILABLE,
}


class IaxError(Exception):
    """A failure the caller is expected to handle, not a bug in the harness."""

    def __init__(
        self,
        message: str,
        code: str = "error",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details or {}

    @property
    def exit_code(self) -> int:
        return _EXIT_FOR_CODE.get(self.code, EXIT_NOT_FOUND)

    def payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"error": self.message, "code": self.code}
        if self.details:
            payload["details"] = self.details
        return payload


def not_found(what: str, identifier: str, hint: str = "") -> NoReturn:
    message = f"unknown {what} '{identifier}'"
    if hint:
        message = f"{message}; {hint}"
    raise IaxError(message, code="not_found", details={what: identifier})


def invalid_input(message: str, details: dict[str, Any] | None = None) -> NoReturn:
    raise IaxError(message, code="invalid_input", details=details)


def backend_unavailable(message: str) -> NoReturn:
    raise IaxError(message, code="backend_unavailable")


def report(error: IaxError, json_mode: bool) -> None:
    """Print an error in the mode the caller asked for, then exit."""
    if json_mode:
        typer.echo(json.dumps(error.payload(), indent=2))
    else:
        typer.echo(f"Error: {error.message}", err=True)
    raise typer.Exit(code=error.exit_code)
