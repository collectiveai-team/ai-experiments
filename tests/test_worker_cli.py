"""Characterization test for `ai_experiments.worker`'s argv contract.

`worker.py` is spawned as a subprocess by :class:`LocalBackend.submit`
(`ai_experiments/backends/local.py:53-62`), never imported and called directly.
Its argv is therefore a wire protocol between processes: nothing type-checks
it, and nothing but a live invocation of the module would notice a renamed or
dropped flag. This test invokes the module exactly the way the backend does --
same interpreter, same module path, same two flags, same flag names -- and
asserts on the run store's observable output (status.json, exit code), not on
any parser internals. It must pass unmodified across the argparse -> Typer
rewrite of `worker.py`; that is what makes it a characterization test rather
than a test of the rewrite.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from ai_experiments.schemas import ExperimentManifest, WorkloadSpec
from ai_experiments.store import FilesystemRunStore

REPO_ROOT = Path(__file__).resolve().parents[1]


def _submit(tmp_path: Path, entrypoint: str) -> tuple[FilesystemRunStore, str]:
    """Create a run the same way `LocalBackend.submit` does, minus the spawn."""
    store = FilesystemRunStore(tmp_path / "runs", capture_repro=False)
    manifest = ExperimentManifest(
        experiment="worker-cli-characterization",
        backend="local",
        workload=WorkloadSpec(entrypoint=entrypoint, working_dir=str(tmp_path)),
    )
    run_id, run_dir = store.create_run(manifest)
    from ai_experiments.schemas import RunHandle

    store.write_handle(
        RunHandle(
            run_id=run_id,
            backend="local",
            status="submitted",
            status_uri=str(store.status_path(run_id)),
            run_dir=str(run_dir),
        )
    )
    return store, run_id


def test_worker_cli_accepts_the_backend_argv_and_completes_the_run(tmp_path):
    """`--run-id`/`--runs-dir` are the wire protocol backends/local.py relies on."""
    workload = tmp_path / "workload.py"
    workload.write_text("print('hello from worker cli test')\n")
    store, run_id = _submit(tmp_path, f"{sys.executable} {workload}")

    argv = [
        sys.executable,
        "-m",
        "ai_experiments.worker",
        "--run-id",
        run_id,
        "--runs-dir",
        str(store.root),
    ]
    result = subprocess.run(  # noqa: S603  # fixed argv, our own worker module
        argv,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    status = json.loads(store.status_path(run_id).read_text())
    assert status["status"] == "completed"
    assert status["exit_code"] == 0


def test_worker_cli_rejects_an_unknown_flag(tmp_path):
    """A bad flag must fail as a parser error, not as some unrelated runtime failure.

    The run is set up for real via `_submit` first, so the accepted-flag path
    would genuinely succeed; the only thing that can make this invocation fail
    is the unrecognized option. Without a real run backing `--run-id`, a
    parser that silently ignored `--bogus-flag` would still exit non-zero (a
    `FileNotFoundError` reading the nonexistent manifest.yaml) and this test
    would pass for the wrong reason.
    """
    workload = tmp_path / "workload.py"
    workload.write_text("print('should never run')\n")
    store, run_id = _submit(tmp_path, f"{sys.executable} {workload}")

    argv = [
        sys.executable,
        "-m",
        "ai_experiments.worker",
        "--run-id",
        run_id,
        "--runs-dir",
        str(store.root),
        "--bogus-flag",
        "x",
    ]
    result = subprocess.run(  # noqa: S603  # fixed argv, our own worker module
        argv,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
        # Typer renders usage errors through Rich, which wraps them to the terminal width it
        # infers from COLUMNS. An exported COLUMNS in the environment running pytest would split
        # the message across lines of the error panel and break the substring assertion below,
        # so pin a width wide enough that it never wraps.
        env={**os.environ, "COLUMNS": "200"},
    )

    assert result.returncode == 2, result.stdout + result.stderr
    assert "No such option: --bogus-flag" in result.stderr


def test_worker_cli_rejects_an_empty_runs_dir(tmp_path):
    """An empty `--runs-dir` must fail loudly, not silently redefine where runs live.

    `argparse` delivered `""` as a falsy `str`, so `FilesystemRunStore("")` fell
    back to its default runs directory. A bare `Path`-typed Typer option turns
    `""` into `Path(".")` instead, which is truthy and points at the process's
    cwd -- a different, equally silent behavior. Neither is acceptable, so this
    pins the loud rejection instead.
    """
    argv = [
        sys.executable,
        "-m",
        "ai_experiments.worker",
        "--run-id",
        "does-not-matter",
        "--runs-dir",
        "",
    ]
    result = subprocess.run(  # noqa: S603  # fixed argv, our own worker module
        argv,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 2, result.stdout + result.stderr
    assert "--runs-dir" in result.stderr
