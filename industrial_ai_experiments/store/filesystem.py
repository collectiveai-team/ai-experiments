from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Iterable

from industrial_ai_experiments.schemas import (
    ExperimentManifest,
    RunEvent,
    RunHandle,
    RunStatus,
    utc_now,
)


class FilesystemRunStore:
    """Filesystem-backed run state used by schedulers and agents."""

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root or os.environ.get("IAX_RUNS_DIR", "outputs/experiments/runs"))

    def create_run(self, manifest: ExperimentManifest) -> tuple[str, Path]:
        run_id = f"run_{uuid.uuid4().hex[:12]}"
        run_dir = self.root / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "manifest.yaml").write_text(manifest.to_yaml())
        (run_dir / "events.jsonl").touch()
        return run_id, run_dir

    def run_dir(self, run_id: str) -> Path:
        return self.root / run_id

    def status_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "status.json"

    def write_handle(self, handle: RunHandle) -> None:
        status = RunStatus(
            run_id=handle.run_id,
            backend=handle.backend,
            status=handle.status,
            status_uri=handle.status_uri,
            run_dir=handle.run_dir,
            external_id=handle.external_id,
            submitted_at=handle.submitted_at,
        )
        self.write_status(status)

    def write_status(self, status: RunStatus) -> None:
        self.status_path(status.run_id).write_text(
            json.dumps(status.model_dump(mode="json"), indent=2)
        )

    def read_status(self, run_id: str) -> RunStatus:
        path = self.status_path(run_id)
        if not path.exists():
            return RunStatus(
                run_id=run_id,
                backend="local",
                status="unknown",
                status_uri=str(path),
                run_dir=str(self.run_dir(run_id)),
                error="status file not found",
            )
        return RunStatus(**json.loads(path.read_text()))

    def update_status(self, run_id: str, **updates: object) -> RunStatus:
        status = self.read_status(run_id)
        data = status.model_dump()
        if isinstance(updates.get("details"), dict):
            updates["details"] = {
                **status.details,
                **updates["details"],  # type: ignore[index]
            }
        data.update(updates)
        data["updated_at"] = utc_now()
        updated = RunStatus(**data)
        self.write_status(updated)
        return updated

    def append_event(self, run_id: str, event: RunEvent) -> None:
        events_path = self.run_dir(run_id) / "events.jsonl"
        with events_path.open("a") as fh:
            fh.write(json.dumps(event.model_dump(mode="json")) + "\n")

    def read_events(self, run_id: str, tail: int | None = None) -> list[RunEvent]:
        events_path = self.run_dir(run_id) / "events.jsonl"
        if not events_path.exists():
            return []
        lines = events_path.read_text().splitlines()
        if tail is not None:
            lines = lines[-tail:]
        return [RunEvent(**json.loads(line)) for line in lines if line.strip()]

    def list_runs(self) -> Iterable[str]:
        if not self.root.exists():
            return []
        return (path.name for path in self.root.iterdir() if path.is_dir())
