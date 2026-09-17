"""Notifications for the daemon: campaign finishes, kills, escalations.

Three sinks, all best-effort (a failing sink never breaks the daemon):

- **Webhook** (`IAX_NOTIFY_WEBHOOK` or ``--notify-webhook``): POSTs JSON.
  Slack incoming webhooks work out of the box (payload carries ``text``);
  any other receiver gets the full structured payload too.
- **Command** (`IAX_NOTIFY_COMMAND` or ``--notify-command``): runs a shell
  command with the JSON payload on stdin — the hook for email, PagerDuty,
  ``terminal-notifier``, or anything else.
- **Log**: every notification is always appended to
  ``<runs>/_notifications.jsonl`` so nothing is lost when no sink is set.
"""

from __future__ import annotations

import contextlib
import json
import shlex
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from ai_experiments.schemas import utc_now
from ai_experiments.settings import get_settings

WEBHOOK_TIMEOUT = 10
COMMAND_TIMEOUT = 30
_ALLOWED_WEBHOOK_SCHEMES = ("http://", "https://")


class Notifier:
    def __init__(
        self,
        runs_root: str | Path,
        webhook_url: str | None = None,
        command: str | None = None,
    ) -> None:
        self.runs_root = Path(runs_root)
        self.webhook_url = webhook_url or get_settings().notify_webhook
        self.command = command or get_settings().notify_command

    def send(self, title: str, message: str, **details: Any) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "timestamp": utc_now().isoformat(),
            "title": title,
            "message": message,
            "text": f"{title}: {message}",  # Slack-compatible field
            **details,
        }
        self._log(payload)
        if self.webhook_url:
            self._post_webhook(payload)
        if self.command:
            self._run_command(payload)
        return payload

    def _log(self, payload: dict[str, Any]) -> None:
        try:
            self.runs_root.mkdir(parents=True, exist_ok=True)
            with (self.runs_root / "_notifications.jsonl").open("a") as fh:
                fh.write(json.dumps(payload) + "\n")
        except OSError:
            pass

    def _post_webhook(self, payload: dict[str, Any]) -> None:
        assert self.webhook_url is not None  # noqa: S101  # type narrowing, not a runtime check
        if not self.webhook_url.startswith(_ALLOWED_WEBHOOK_SCHEMES):
            # A misconfigured scheme (e.g. file://) must not raise -- this sink is best-effort --
            # but it also must not vanish silently, so record why it was skipped.
            self._log(
                {
                    **payload,
                    "notify_sink_error": f"webhook scheme not allowed: {self.webhook_url!r}",
                }
            )
            return
        request = urllib.request.Request(  # noqa: S310  # scheme allowlisted immediately above
            self.webhook_url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        # Best-effort sink: a failing webhook must never break the daemon. Log this at
        # debug level once the house logger exists.
        with contextlib.suppress(urllib.error.URLError, OSError):
            urllib.request.urlopen(  # noqa: S310  # scheme allowlisted immediately above
                request, timeout=WEBHOOK_TIMEOUT
            )

    def _run_command(self, payload: dict[str, Any]) -> None:
        assert self.command is not None  # noqa: S101  # type narrowing, not a runtime check
        with contextlib.suppress(OSError, subprocess.TimeoutExpired):
            subprocess.run(  # noqa: S603  # user-configured notify command, this is the product
                shlex.split(self.command),
                input=json.dumps(payload),
                text=True,
                capture_output=True,
                timeout=COMMAND_TIMEOUT,
                check=False,
            )


def read_notifications(runs_root: str | Path, tail: int | None = None) -> list[dict[str, Any]]:
    path = Path(runs_root) / "_notifications.jsonl"
    if not path.exists():
        return []
    lines = path.read_text().splitlines()
    if tail is not None:
        lines = lines[-tail:]
    return [json.loads(line) for line in lines if line.strip()]
