"""Run a workload's phases in order, with the evaluator kept separate.

A trial is not one process. The trainer is the code an agent may rewrite; the
evaluator is the code that judges it. Running them as one process would put
the number and the thing being measured in the same hands, so they run as two,
and only the second one may declare a result.
"""

from __future__ import annotations

import argparse
import sys
import traceback

from ai_experiments.config_loading import load_stored
from ai_experiments.schemas import (
    ACTIVE_RUN_STATES,
    ExperimentManifest,
    RunEvent,
)
from ai_experiments.store import FilesystemRunStore
from ai_experiments.worker import _Supervisor, report_supervisor_failure


def run_phases(store: FilesystemRunStore, run_id: str) -> int:
    """Run every declared phase in order; stop at the first failure."""
    # load_stored, not ExperimentManifest.from_yaml: this reads a manifest
    # iax itself persisted, so it has to tolerate keys an older iax emitted
    # and later removed, the same tolerance the supervisor already relies on.
    manifest = load_stored(ExperimentManifest, store.run_dir(run_id) / "manifest.yaml")
    # A run's initial status is established by the backend's submit() before
    # this module is ever spawned; it is not this function's job to invent
    # one. If it is missing, the run dir is broken, and `_Supervisor` raising
    # out of `_update_status` is what lets `phases.main()` report that
    # loudly through `report_supervisor_failure` -- not a fabricated record.
    # The two phases hand off through this directory and not through the cwd:
    # once the trainer runs from a variant copy, they no longer share one.
    work_dir = store.run_dir(run_id) / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    phases = manifest.workload.phases()
    for index, (phase, command) in enumerate(phases):
        if store.cancel_requested(run_id):
            # Checked here, not only inside the running supervisor: a
            # cancellation that arrives in the gap between two phases has no
            # process to signal, so without this the next phase starts
            # anyway. The backend that requested the cancellation already
            # wrote the `cancelled` status (LocalBackend.cancel does, right
            # after calling request_cancel); overwriting it here would be
            # fabricating a second answer to the same question.
            store.append_event(
                run_id,
                RunEvent(
                    level="warning",
                    message="remaining phases skipped: cancellation requested",
                    details={"phase": phase},
                ),
            )
            return 1
        store.append_event(run_id, RunEvent(message="phase started", details={"phase": phase}))
        supervisor = _Supervisor(store, run_id, phase=phase, final=index == len(phases) - 1)
        exit_code = supervisor.run(command=command, work_dir=work_dir)
        if exit_code != 0:
            return exit_code
        if index != len(phases) - 1:
            # `_Supervisor` has its own, independent notion of cancelled
            # (`_cancelled`, set by its SIGTERM handler) that never touches
            # the marker `cancel_requested` above reads -- a bare SIGTERM to
            # this process sets it without `store.request_cancel` ever being
            # called. A non-final phase that ran normally leaves the status
            # `running` (that is what the `final` gating is for), so a
            # terminal status here means something ended the run and the
            # phase we are about to start must not run on top of it.
            status = store.read_status(run_id)
            if status.status not in ACTIVE_RUN_STATES:
                skipped_phase, _ = phases[index + 1]
                store.append_event(
                    run_id,
                    RunEvent(
                        level="warning",
                        message="remaining phases skipped: run already ended",
                        details={"status": status.status, "phase": skipped_phase},
                    ),
                )
                return 1
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--runs-dir", required=True)
    args = parser.parse_args()

    store = FilesystemRunStore(args.runs_dir)
    try:
        raise SystemExit(run_phases(store, args.run_id))
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)  # keep the evidence in worker.log
        report_supervisor_failure(store, args.run_id, exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
