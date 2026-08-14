from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from ai_experiments.schemas import (
    ExperimentManifest,
    MetricPoint,
    RunEvent,
    RunHandle,
    RunStatus,
    utc_now,
)

#: Marks a ``RunStatus`` the store synthesized because the real file was
#: missing or unreadable. Such a status describes the *store's* inability to
#: answer, not the run — persisting it would fabricate history, so
#: :meth:`FilesystemRunStore.update_status` refuses to write on top of one.
SYNTHETIC_STATUS_KEY = "_synthetic"


def atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` so no reader ever observes a partial file.

    A plain ``write_text`` truncates first, so any concurrent reader (or any
    crash) can see a half-written document. Writing to a sibling temp file and
    ``os.replace``-ing it onto the target is atomic on POSIX and Windows, so
    readers see either the old file or the new one.

    Note this makes writes *indivisible*; it does not make read-modify-write
    sequences *serializable*. Concurrent updaters still race — see
    ``.scratch/proposals/concurrent-status-writes.md``.
    """
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(text)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


class FilesystemRunStore:
    """Filesystem-backed run state used by schedulers and agents."""

    def __init__(
        self, root: str | Path | None = None, capture_repro: bool = True
    ) -> None:
        self.root = Path(
            root or os.environ.get("IAX_RUNS_DIR", "outputs/experiments/runs")
        )
        self.capture_repro = capture_repro

    def create_run(self, manifest: ExperimentManifest) -> tuple[str, Path]:
        run_id = f"run_{uuid.uuid4().hex[:12]}"
        run_dir = self.root / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "manifest.yaml").write_text(manifest.to_yaml())
        (run_dir / "events.jsonl").touch()
        (run_dir / "artifacts").mkdir()
        if self.capture_repro:
            from ai_experiments.repro import capture_repro

            try:
                capture_repro(run_dir, manifest.workload.working_dir)
            except Exception:
                pass  # reproducibility capture must never block a submit
        return run_id, run_dir

    def run_dir(self, run_id: str) -> Path:
        return self.root / run_id

    def status_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "status.json"

    def write_handle(self, handle: RunHandle) -> None:
        """Establish a run's initial status. Only valid for a fresh run.

        A handle carries no ``details``, so writing one over an existing status
        would silently discard whatever a caller had already recorded there
        (this is how the MLflow linkage was lost on the Ray path). Establishing
        status is a create, never an update: use :meth:`update_status` to
        change an existing run.
        """
        path = self.status_path(handle.run_id)
        if path.exists():
            raise RuntimeError(
                f"refusing to overwrite an existing status for {handle.run_id}: "
                "write_handle establishes a run's initial status; "
                "use update_status to modify an existing one"
            )
        self.write_status(
            RunStatus(
                run_id=handle.run_id,
                backend=handle.backend,
                status=handle.status,
                status_uri=handle.status_uri,
                run_dir=handle.run_dir,
                external_id=handle.external_id,
                submitted_at=handle.submitted_at,
            )
        )

    def write_status(self, status: RunStatus) -> None:
        atomic_write_text(
            self.status_path(status.run_id),
            json.dumps(status.model_dump(mode="json"), indent=2),
        )

    def _synthetic_status(self, run_id: str, error: str) -> RunStatus:
        """Stand-in for a status the store cannot read. Never persisted."""
        return RunStatus(
            run_id=run_id,
            backend="local",
            status="unknown",
            status_uri=str(self.status_path(run_id)),
            run_dir=str(self.run_dir(run_id)),
            error=error,
            details={SYNTHETIC_STATUS_KEY: True},
        )

    def read_status(self, run_id: str) -> RunStatus:
        """Current status, or a synthetic ``unknown`` when it cannot be read.

        A truncated or otherwise unparsable file is quarantined the same way a
        missing one is: one corrupt run must not take down the daemon
        supervising every other run.
        """
        path = self.status_path(run_id)
        if not path.exists():
            return self._synthetic_status(run_id, "status file not found")
        try:
            return RunStatus(**json.loads(path.read_text()))
        except Exception as exc:
            return self._synthetic_status(run_id, f"status file corrupt: {exc}")

    def update_status(self, run_id: str, **updates: object) -> RunStatus:
        status = self.read_status(run_id)
        if status.details.get(SYNTHETIC_STATUS_KEY):
            # The read did not describe the run, so this update has no base to
            # merge onto. Writing it would fabricate a status and, for a
            # corrupt file, destroy the evidence of what went wrong.
            raise RuntimeError(f"cannot update status for {run_id}: {status.error}")
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

    def metrics_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "metrics.jsonl"

    def append_metric(self, run_id: str, point: MetricPoint) -> None:
        with self.metrics_path(run_id).open("a") as fh:
            fh.write(json.dumps(point.model_dump(mode="json")) + "\n")

    def write_metrics(self, run_id: str, points: list[MetricPoint]) -> None:
        lines = [json.dumps(point.model_dump(mode="json")) for point in points]
        atomic_write_text(
            self.metrics_path(run_id), "\n".join(lines) + ("\n" if lines else "")
        )

    def read_metrics(self, run_id: str, tail: int | None = None) -> list[MetricPoint]:
        path = self.metrics_path(run_id)
        if not path.exists():
            return []
        lines = path.read_text().splitlines()
        if tail is not None:
            lines = lines[-tail:]
        return [MetricPoint(**json.loads(line)) for line in lines if line.strip()]

    def artifacts_dir(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "artifacts"

    def list_artifacts(self, run_id: str) -> list[dict[str, object]]:
        """Relative path, size, and mtime for every file under artifacts/."""
        root = self.artifacts_dir(run_id)
        if not root.exists():
            return []
        entries: list[dict[str, object]] = []
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            stat = path.stat()
            entries.append(
                {
                    "path": str(path.relative_to(root)),
                    "size_bytes": stat.st_size,
                    "modified_at": datetime.fromtimestamp(
                        stat.st_mtime, tz=timezone.utc
                    ).isoformat(),
                }
            )
        return entries

    def read_manifest(self, run_id: str) -> ExperimentManifest | None:
        path = self.run_dir(run_id) / "manifest.yaml"
        if not path.exists():
            return None
        return ExperimentManifest.from_yaml(path)

    def list_runs(self) -> Iterable[str]:
        if not self.root.exists():
            return []
        # Underscore-prefixed dirs (_campaigns, _escalations) share the root
        # but are not runs.
        return (
            path.name
            for path in self.root.iterdir()
            if path.is_dir() and not path.name.startswith("_")
        )
