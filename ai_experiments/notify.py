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

import json
import os
import shlex
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from ai_experiments.schemas import utc_now

WEBHOOK_TIMEOUT = 10
COMMAND_TIMEOUT = 30


class Notifier:
    def __init__(
        self,
        runs_root: str | Path,
        webhook_url: str | None = None,
        command: str | None = None,
    ) -> None:
        self.runs_root = Path(runs_root)
        self.webhook_url = webhook_url or os.environ.get("IAX_NOTIFY_WEBHOOK")
        self.command = command or os.environ.get("IAX_NOTIFY_COMMAND")

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
        assert self.webhook_url is not None
        request = urllib.request.Request(
            self.webhook_url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            urllib.request.urlopen(request, timeout=WEBHOOK_TIMEOUT)  # noqa: S310
        except (urllib.error.URLError, OSError):
            pass

    def _run_command(self, payload: dict[str, Any]) -> None:
        assert self.command is not None
        try:
            subprocess.run(
                shlex.split(self.command),
                input=json.dumps(payload),
                text=True,
                capture_output=True,
                timeout=COMMAND_TIMEOUT,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass


def read_notifications(
    runs_root: str | Path, tail: int | None = None
) -> list[dict[str, Any]]:
    path = Path(runs_root) / "_notifications.jsonl"
    if not path.exists():
        return []
    lines = path.read_text().splitlines()
    if tail is not None:
        lines = lines[-tail:]
    return [json.loads(line) for line in lines if line.strip()]
