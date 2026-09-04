"""What reaches `events.jsonl`, and at what level.

The event stream is what `iax logs` prints, what the monitoring rules read,
and what an agent reviewing a round is shown. Noise in it is not cosmetic:
it is the evidence, and a bad level is a false alarm.
"""

from __future__ import annotations

import sys
import textwrap
import time

import pytest

from ai_experiments.backends.local import LocalBackend
from ai_experiments.monitoring.rules import event_from_log_line
from ai_experiments.schemas import ExperimentManifest, WorkloadSpec
from ai_experiments.store import FilesystemRunStore

TERMINAL = {"completed", "failed", "cancelled"}


def _run(tmp_path, script: str) -> tuple[FilesystemRunStore, str]:
    path = tmp_path / "workload.py"
    path.write_text(textwrap.dedent(script))
    store = FilesystemRunStore(tmp_path / "runs")
    run_id = (
        LocalBackend(store)
        .submit(
            ExperimentManifest(
                experiment="log-noise",
                backend="local",
                workload=WorkloadSpec(
                    entrypoint=f"{sys.executable} {path}", working_dir=str(tmp_path)
                ),
            )
        )
        .run_id
    )
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        status = store.read_status(run_id)
        if status.status in TERMINAL:
            assert status.status == "completed", status.error
            return store, run_id
        time.sleep(0.05)
    pytest.fail(f"run {run_id} never finished: {store.read_status(run_id).status}")


def test_a_progress_bar_is_one_event_not_one_per_refresh(tmp_path):
    r"""A tqdm bar refreshes with "\r". A terminal overwrites; so do we.

    Reading with universal newlines made every refresh its own line and so its
    own event: one bar became 5,006 lines in `events.jsonl`.
    """
    store, run_id = _run(
        tmp_path,
        """
        import sys
        for i in range(500):
            sys.stdout.write(f"\\rstep {i}/500")
            sys.stdout.flush()
        sys.stdout.write("\\n")
        print("done", flush=True)
        """,
    )

    messages = [event.message for event in store.read_events(run_id)]

    assert len(messages) < 10, messages
    assert "step 499/500" in messages  # the final state of the bar, kept
    assert "step 0/500" not in messages  # every refresh before it, overwritten


def test_a_metric_reported_mid_bar_is_still_a_metric(tmp_path):
    r"""Progress bars and IAX_METRIC lines share one stream.

    A workload that prints a bar without a trailing newline, then a metric,
    must not lose the metric to the overwrite.
    """
    store, run_id = _run(
        tmp_path,
        """
        import json, sys
        for i in range(50):
            sys.stdout.write(f"\\rstep {i}/50")
        print("\\nIAX_METRIC " + json.dumps({"step": 1, "loss": 0.25}), flush=True)
        """,
    )

    points = store.read_metrics(run_id)

    assert [point.values["loss"] for point in points] == [0.25]


def test_a_blank_line_is_not_an_event(tmp_path):
    store, run_id = _run(
        tmp_path,
        """
        print("first")
        print("")
        print("   ")
        print("last")
        """,
    )

    messages = [event.message for event in store.read_events(run_id)]

    assert "first" in messages and "last" in messages
    assert "" not in messages


@pytest.mark.parametrize(
    "line",
    [
        "train_error=0.02",
        "reconstruction_error 0.117",
        "epoch 3 | error_rate=0.04",
        "step 12 val_error: 0.31",
    ],
)
def test_a_metric_that_measures_error_is_not_an_error(line):
    """ "error" is ordinary metric vocabulary, and a substring match said
    every epoch of a healthy run was a failure."""
    assert event_from_log_line(line).level == "info"


@pytest.mark.parametrize(
    "line",
    [
        "Traceback (most recent call last):",
        "RuntimeError: out of memory",
        "ERROR: could not open the dataset",
        "error: no such file",
        "[ERROR] the loader gave up",
        "ValueError: lr must be positive",
    ],
)
def test_a_real_failure_is_still_an_error(line):
    assert event_from_log_line(line).level == "error"


def test_tail_reads_only_the_tail(tmp_path):
    """Every monitoring tick reads the last few events of an unbounded file."""
    store = FilesystemRunStore(tmp_path / "runs")
    manifest = ExperimentManifest(
        experiment="tail", workload=WorkloadSpec(entrypoint="true")
    )
    run_id, run_dir = store.create_run(manifest)
    (run_dir / "events.jsonl").write_text(
        "".join(
            '{"level": "info", "message": "line %d"}\n' % index
            for index in range(5_000)
        )
    )

    events = store.read_events(run_id, tail=3)

    assert [event.message for event in events] == [
        "line 4997",
        "line 4998",
        "line 4999",
    ]
