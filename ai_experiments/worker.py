from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from pathlib import Path

from ai_experiments.monitoring.rules import event_from_log_line
from ai_experiments.schemas import ExperimentManifest, RunEvent, utc_now
from ai_experiments.store import FilesystemRunStore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--runs-dir", required=True)
    args = parser.parse_args()

    store = FilesystemRunStore(args.runs_dir)
    run_dir = store.run_dir(args.run_id)
    manifest = ExperimentManifest.from_yaml(run_dir / "manifest.yaml")

    command = [*shlex.split(manifest.workload.entrypoint), *manifest.workload.args]
    env = os.environ.copy()
    env.update(manifest.workload.env)
    env["IAX_RUN_ID"] = args.run_id
    env["IAX_RUN_DIR"] = str(run_dir)

    store.update_status(args.run_id, status="running", started_at=utc_now())
    store.append_event(args.run_id, RunEvent(message="workload started", details={"command": command}))

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=Path(manifest.workload.working_dir).resolve(),
        env=env,
    )
    store.update_status(args.run_id, details={"workload_pid": process.pid})

    assert process.stdout is not None
    for line in process.stdout:
        store.append_event(args.run_id, event_from_log_line(line))

    exit_code = process.wait()
    if exit_code == 0:
        store.update_status(args.run_id, status="completed", exit_code=exit_code, completed_at=utc_now())
        store.append_event(args.run_id, RunEvent(message="workload completed"))
    else:
        store.update_status(
            args.run_id,
            status="failed",
            exit_code=exit_code,
            completed_at=utc_now(),
            error=f"workload exited with code {exit_code}",
        )
        store.append_event(
            args.run_id,
            RunEvent(
                level="error",
                message="workload failed",
                details={"exit_code": exit_code},
            ),
        )


if __name__ == "__main__":
    main()
