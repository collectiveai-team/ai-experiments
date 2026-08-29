"""Proof that a daemon is alive, and a warning when none is (#22).

A healthy daemon tick prints nothing, so a working daemon and a dead one look
identical from the outside. Worse, a campaign started without a daemon never
advances and says nothing about why. So every tick stamps a file in the run
store, and the commands a user runs while wondering -- `iax runs`,
`iax campaign list`, `iax campaign status` -- read that stamp and say when the
last tick was.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path

from pydantic import BaseModel, Field

from ai_experiments.schemas import utc_now
from ai_experiments.store.filesystem import atomic_write_text

#: Underscore-prefixed, like `_campaigns`: shares the run store, is not a run.
DAEMON_DIR = "_daemon"

#: A tick older than this means the daemon is gone, restarting, or wedged.
#: Comfortably above the 30s default interval, below a user's patience.
STALE_AFTER = timedelta(minutes=10)


class Heartbeat(BaseModel):
    timestamp: datetime = Field(default_factory=utc_now)
    pid: int = Field(default_factory=os.getpid)
    interval_seconds: int | None = None
    ticks: int = 0
    runs_checked: int = 0
    campaigns_advanced: int = 0

    @property
    def age(self) -> timedelta:
        return utc_now() - self.timestamp

    def is_stale(self, stale_after: timedelta = STALE_AFTER) -> bool:
        return self.age > stale_after


def heartbeat_path(root: str | Path) -> Path:
    return Path(root) / DAEMON_DIR / "heartbeat.json"


def write_heartbeat(root: str | Path, beat: Heartbeat) -> None:
    path = heartbeat_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(beat.model_dump(mode="json"), indent=2))


def read_heartbeat(root: str | Path) -> Heartbeat | None:
    """The last tick recorded in this store, or None if no daemon ever ticked."""
    path = heartbeat_path(root)
    try:
        return Heartbeat(**json.loads(path.read_text()))
    except (OSError, ValueError):
        return None


def daemon_warning(
    root: str | Path, stale_after: timedelta = STALE_AFTER
) -> str | None:
    """One line to print when work is waiting on a daemon that is not ticking.

    Callers pass this only when the store actually holds something a daemon
    would move; a quiet store with no daemon is not a problem.
    """
    beat = read_heartbeat(root)
    if beat is None:
        return (
            "No daemon has ticked in this run store. Campaigns advance and runs "
            "are monitored only while `iax daemon` is running."
        )
    if beat.is_stale(stale_after):
        return (
            f"No daemon tick for {_humanize(beat.age)} "
            f"(last: {beat.timestamp.isoformat()}, pid {beat.pid}). "
            "Start one with `iax daemon`."
        )
    return None


def _humanize(age: timedelta) -> str:
    minutes = int(age.total_seconds() // 60)
    if minutes < 60:
        return f"{minutes}m"
    hours, minutes = divmod(minutes, 60)
    if hours < 24:
        return f"{hours}h{minutes:02d}m"
    return f"{hours // 24}d{hours % 24}h"
