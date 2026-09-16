# tests/test_worker_phases.py
"""The training phase cannot declare a result. Only the evaluator can."""

from __future__ import annotations

import sys
import textwrap

from ai_experiments.schemas import ExperimentManifest, RunHandle, WorkloadSpec
from ai_experiments.store import FilesystemRunStore
from ai_experiments.worker import _Supervisor


def _run(tmp_path, body: str, phase: str):
    script = tmp_path / "workload.py"
    script.write_text(textwrap.dedent(body))
    store = FilesystemRunStore(tmp_path / "runs")
    manifest = ExperimentManifest(
        experiment="e",
        workload=WorkloadSpec(
            entrypoint=sys.executable,
            args=[str(script)],
            working_dir=str(tmp_path),
        ),
    )
    run_id, run_dir = store.create_run(manifest)
    # _Supervisor.run() updates status as it goes, and update_status refuses
    # to fabricate a status for a run that was never handed one -- every real
    # caller goes through a backend's submit(), which writes this first.
    store.write_handle(
        RunHandle(
            run_id=run_id,
            backend="local",
            status="submitted",
            status_uri=str(store.status_path(run_id)),
            run_dir=str(run_dir),
        )
    )
    _Supervisor(store, run_id, phase=phase).run()
    return store, run_id


CHEATS = """
    print('IAX_METRIC {"step": 0, "loss": 0.5}')
    print('IAX_RESULT {"test_acc": 0.99}')
    """


def test_the_train_phase_cannot_declare_a_result(tmp_path):
    store, run_id = _run(tmp_path, CHEATS, phase="train")

    assert store.read_results(run_id) == []
    assert [p.values["loss"] for p in store.read_metrics(run_id)] == [0.5]


def test_a_result_from_the_train_phase_is_recorded_as_a_warning(tmp_path):
    store, run_id = _run(tmp_path, CHEATS, phase="train")

    warnings = [e for e in store.read_events(run_id) if e.level == "warning"]
    assert any("discarded" in e.message for e in warnings)


def test_the_evaluate_phase_declares_the_result(tmp_path):
    store, run_id = _run(tmp_path, CHEATS, phase="evaluate")

    assert [r.values for r in store.read_results(run_id)] == [{"test_acc": 0.99}]


def test_a_leaked_iax_data_test_does_not_reach_the_train_phase(tmp_path, monkeypatch):
    """`DataSpec.env_for` only ever adds keys, so it cannot take one away --
    `_Supervisor.run` has to scrub the *inherited* environment before adding
    them back, or a value the parent process already set (`iax daemon` runs
    with whatever environment the operator started it in) reaches the train
    phase untouched, letting a trainer read the held-out set directly.
    """
    monkeypatch.setenv("IAX_DATA_TEST", "s3://bucket/held-out.parquet")
    store, run_id = _run(
        tmp_path,
        """
        import os
        saw_test = 1 if "IAX_DATA_TEST" in os.environ else 0
        print('IAX_METRIC {"step": 0, "saw_test": %d}' % saw_test)
        """,
        phase="train",
    )

    assert [m.values["saw_test"] for m in store.read_metrics(run_id)] == [0.0]
