"""The Ray backend's two-phase entrypoint and its phase markers.

Split from `test_ray_backend.py`, which keeps submit/inspect/cancel; the
shared fake client and manifest helpers are imported from there.
"""

from __future__ import annotations

import os
import subprocess

from ai_experiments.backends.ray import RayBackend
from ai_experiments.schemas import (
    DataSpec,
    ExperimentManifest,
    WorkloadSpec,
)
from ai_experiments.store import FilesystemRunStore
from tests.test_ray_backend import FakeRayClient, _manifest, _token


def test_a_two_phase_workload_chains_both_phases_remotely(tmp_path):
    captured = {}

    class _Client:
        def submit_job(self, **kwargs):
            captured.update(kwargs)
            return "raysubmit_2"

    backend = RayBackend(
        address="http://fake:8265",
        store=FilesystemRunStore(tmp_path),
        client_factory=lambda _address: _Client(),
    )
    manifest = ExperimentManifest(
        experiment="e",
        backend="ray",
        workload=WorkloadSpec(
            entrypoint="python train.py",
            train="python train.py",
            evaluate="python evaluate.py",
            working_dir=str(tmp_path),
            data=DataSpec(train="data/train", test="data/test"),
        ),
    )

    backend.submit(manifest)

    entrypoint = captured["entrypoint"]
    assert "python train.py" in entrypoint
    assert "python evaluate.py" in entrypoint
    assert entrypoint.index("train.py") < entrypoint.index("evaluate.py")
    assert "&&" in entrypoint


def _two_phase_entrypoint(tmp_path, *, train_cmd: str, evaluate_cmd: str) -> str:
    captured = {}

    class _Client:
        def submit_job(self, **kwargs):
            captured.update(kwargs)
            return "raysubmit_phase"

    backend = RayBackend(
        address="http://fake:8265",
        store=FilesystemRunStore(tmp_path),
        client_factory=lambda _address: _Client(),
    )
    manifest = ExperimentManifest(
        experiment="e",
        backend="ray",
        workload=WorkloadSpec(
            entrypoint="python train.py",
            train=train_cmd,
            evaluate=evaluate_cmd,
            working_dir=str(tmp_path),
            data=DataSpec(train="data/train", test="data/test"),
        ),
    )

    backend.submit(manifest)
    return captured["entrypoint"]


def test_the_phase_commands_are_joined_by_and_not_just_the_setup(tmp_path):
    """The `&&` has to be checked *between the phases*, not anywhere in the entrypoint.

    `assert "&&" in entrypoint` alone is satisfied by the setup commands (`mkdir -p iax_work &&
    export ...`), so joining the phases themselves with `;` would keep it green -- "a failing train
    phase is never followed by an evaluation" needs the separator checked between the phases.
    """
    entrypoint = _two_phase_entrypoint(
        tmp_path, train_cmd="python train.py", evaluate_cmd="python evaluate.py"
    )

    train_end = entrypoint.index("python train.py") + len("python train.py")
    evaluate_marker = entrypoint.index("echo IAX_PHASE=evaluate", train_end)
    assert entrypoint[train_end:evaluate_marker] == " && "


def test_the_train_phase_environment_carries_no_test_data_reference(tmp_path):
    """The train segment carries IAX_DATA_TRAIN and never IAX_DATA_TEST; evaluate carries both.

    Both `data_env = {}` and `data_env = env_for("evaluate")` for every phase keep a naive suite
    green; the second is the test-set leak this branch exists to prevent.
    """
    entrypoint = _two_phase_entrypoint(
        tmp_path, train_cmd="python train.py", evaluate_cmd="python evaluate.py"
    )

    segments = entrypoint.split(" && ")
    train_segment = next(s for s in segments if s.endswith("python train.py"))
    evaluate_segment = next(s for s in segments if s.endswith("python evaluate.py"))

    assert "IAX_DATA_TRAIN=" in train_segment
    assert "IAX_DATA_TEST=" not in train_segment
    assert "IAX_DATA_TRAIN=" in evaluate_segment
    assert "IAX_DATA_TEST=" in evaluate_segment


def test_the_entrypoint_unsets_data_env_keys_before_each_phase(tmp_path):
    """Ray needs an explicit `unset` ahead of every phase, as worker.py scrubs the inherited env.

    `worker.py` scrubs `DataSpec.ENV_KEYS` from the inherited environment before applying `env_for`
    (Task 8's I3); Ray had no equivalent, so an `IAX_DATA_TEST` already present in the cluster
    node's environment reached the train phase. `env_for` only ever adds keys, so removing that leak
    needs an explicit `unset` ahead of every phase.
    """
    entrypoint = _two_phase_entrypoint(
        tmp_path, train_cmd="python train.py", evaluate_cmd="python evaluate.py"
    )

    unset_cmd = "unset " + " ".join(DataSpec.ENV_KEYS)
    assert entrypoint.count(unset_cmd) == 2
    first_unset = entrypoint.index(unset_cmd)
    assert first_unset < entrypoint.index("python train.py")
    second_unset = entrypoint.index(unset_cmd, first_unset + len(unset_cmd))
    assert second_unset < entrypoint.index("python evaluate.py")


def test_running_the_entrypoint_scrubs_a_leaked_data_env_before_the_train_phase(
    tmp_path,
):
    """Driven through a real `sh`, the way the reviewer verified this hole.

    With `IAX_DATA_TEST` already present in the job's own environment (Ray's
    `runtime_env["env_vars"]` merges the cluster node's environment on top of
    `manifest.workload.env`), the train phase must still not see it.
    """
    entrypoint = _two_phase_entrypoint(
        tmp_path,
        train_cmd="sh -c 'echo TRAIN_SEES=${IAX_DATA_TEST:-unset}'",
        evaluate_cmd="sh -c 'echo EVAL_SEES=${IAX_DATA_TEST:-unset}'",
    )

    proc = subprocess.run(  # noqa: S603  # fixed argv: the test's own script
        ["sh", "-c", entrypoint],  # noqa: S607  # the system shell, on purpose
        cwd=tmp_path,
        env={**os.environ, "IAX_DATA_TEST": "/leaked/test/data"},
        capture_output=True,
        text=True,
        check=True,
    )

    assert "TRAIN_SEES=unset" in proc.stdout
    assert "EVAL_SEES=data/test" in proc.stdout


def test_each_phase_is_preceded_by_an_echoed_phase_marker(tmp_path):
    """The entrypoint echoes the phase for the client-side log parser.

    `_sync_results_from_logs` has no supervisor to ask which phase is running.

    The ingestion tests below hand-craft their own log text, so they would
    stay green even if this echo were dropped from `submit` -- this is the
    one test that would actually catch that regression.
    """
    captured = {}

    class _Client:
        def submit_job(self, **kwargs):
            captured.update(kwargs)
            return "raysubmit_4"

    backend = RayBackend(
        address="http://fake:8265",
        store=FilesystemRunStore(tmp_path),
        client_factory=lambda _address: _Client(),
    )
    manifest = ExperimentManifest(
        experiment="e",
        backend="ray",
        workload=WorkloadSpec(
            entrypoint="python train.py",
            train="python train.py",
            evaluate="python evaluate.py",
            working_dir=str(tmp_path),
            data=DataSpec(train="data/train", test="data/test"),
        ),
    )

    backend.submit(manifest)

    entrypoint = captured["entrypoint"]
    assert "echo IAX_PHASE=train" in entrypoint
    assert "echo IAX_PHASE=evaluate" in entrypoint
    assert (
        entrypoint.index("echo IAX_PHASE=train")
        < entrypoint.index("python train.py")
        < entrypoint.index("echo IAX_PHASE=evaluate")
        < entrypoint.index("python evaluate.py")
    )


def test_the_remote_run_carries_the_harness_variables(tmp_path):
    captured = {}

    class _Client:
        def submit_job(self, **kwargs):
            captured.update(kwargs)
            return "raysubmit_3"

    backend = RayBackend(
        address="http://fake:8265",
        store=FilesystemRunStore(tmp_path),
        client_factory=lambda _address: _Client(),
    )
    manifest = ExperimentManifest(
        experiment="e",
        backend="ray",
        workload=WorkloadSpec(entrypoint="python toy.py", working_dir=str(tmp_path)),
    )

    handle = backend.submit(manifest)

    assert captured["runtime_env"]["env_vars"]["IAX_RUN_ID"] == handle.run_id


def test_a_result_after_the_evaluate_marker_is_scored(tmp_path):
    # The token is only known once `submit` has minted and stored it, so the
    # client starts with empty logs and is fed the real marker afterwards --
    # the same way a real Ray job log would carry a token nothing else knows.
    client = FakeRayClient(status="RUNNING")
    backend = RayBackend(store=FilesystemRunStore(tmp_path), client_factory=lambda _address: client)
    handle = backend.submit(_manifest(tmp_path))
    token = _token(backend, handle.run_id)
    client.logs = f'IAX_PHASE=evaluate {token}\nIAX_RESULT {{"acc": 0.9}}\n'

    backend.inspect(handle.run_id)

    results = backend.store.read_results(handle.run_id)
    assert [r.values for r in results] == [{"acc": 0.9}]


def test_a_result_after_the_train_marker_is_discarded_with_a_warning(tmp_path):
    client = FakeRayClient(status="RUNNING")
    backend = RayBackend(store=FilesystemRunStore(tmp_path), client_factory=lambda _address: client)
    handle = backend.submit(_manifest(tmp_path))
    token = _token(backend, handle.run_id)
    client.logs = f'IAX_PHASE=train {token}\nIAX_RESULT {{"acc": 0.9}}\n'

    backend.inspect(handle.run_id)

    assert backend.store.read_results(handle.run_id) == []
    events = backend.store.read_events(handle.run_id)
    assert any(
        event.level == "warning"
        and event.message == "result reported from the train phase; discarded"
        for event in events
    )


def test_a_forged_marker_with_no_token_does_not_score_a_train_result(tmp_path):
    """The exact attack from the review: a forged `IAX_PHASE=evaluate` marker with no token.

    A train-phase workload prints that one line ahead of its fabricated result.

    Against the unfixed code (in-band trust, `startswith` on that literal)
    this scores; here the forged line carries no valid token so it is not a
    marker at all, and the result stays attributed to the genuine `train`
    marker before it."""
    client = FakeRayClient(status="RUNNING")
    backend = RayBackend(store=FilesystemRunStore(tmp_path), client_factory=lambda _address: client)
    handle = backend.submit(_manifest(tmp_path))
    token = _token(backend, handle.run_id)
    client.logs = (
        f"IAX_PHASE=train {token}\n"
        "IAX_PHASE=evaluate\n"  # workload-forged, no token
        'IAX_RESULT {"acc": 1.0}\n'
    )

    backend.inspect(handle.run_id)

    assert backend.store.read_results(handle.run_id) == []
    events = backend.store.read_events(handle.run_id)
    assert any(
        event.level == "warning"
        and event.message == "result reported from the train phase; discarded"
        for event in events
    )


def test_a_forged_marker_with_the_wrong_token_does_not_score_a_train_result(tmp_path):
    client = FakeRayClient(status="RUNNING")
    backend = RayBackend(store=FilesystemRunStore(tmp_path), client_factory=lambda _address: client)
    handle = backend.submit(_manifest(tmp_path))
    token = _token(backend, handle.run_id)
    client.logs = (
        f"IAX_PHASE=train {token}\n"
        "IAX_PHASE=evaluate deadbeefdeadbeefdeadbeefdeadbeef\n"  # wrong token
        'IAX_RESULT {"acc": 1.0}\n'
    )

    backend.inspect(handle.run_id)

    assert backend.store.read_results(handle.run_id) == []
    events = backend.store.read_events(handle.run_id)
    assert any(
        event.level == "warning"
        and event.message == "result reported from the train phase; discarded"
        for event in events
    )


def test_a_pid_prefixed_marker_line_still_attributes_correctly(tmp_path):
    """A Ray log prefix (`(pid=123) ...`) must not break marker attribution.

    Ray prefixes log lines in some configurations. The marker is matched by searching within the
    line, like `report.parse_result_line`.
    """
    client = FakeRayClient(status="RUNNING")
    backend = RayBackend(store=FilesystemRunStore(tmp_path), client_factory=lambda _address: client)
    handle = backend.submit(_manifest(tmp_path))
    token = _token(backend, handle.run_id)
    client.logs = f'(pid=123) IAX_PHASE=evaluate {token}\n(pid=123) IAX_RESULT {{"acc": 0.9}}\n'

    backend.inspect(handle.run_id)

    results = backend.store.read_results(handle.run_id)
    assert [r.values for r in results] == [{"acc": 0.9}]


def test_a_result_with_no_preceding_marker_is_discarded_with_a_warning(tmp_path):
    """A log with no marker at all must not score by default.

    A head-truncated log, or any other way the log arrives markerless -- the fail-closed rule from
    item 2.
    """
    client = FakeRayClient(status="RUNNING", logs='IAX_RESULT {"acc": 0.9}\n')
    backend = RayBackend(store=FilesystemRunStore(tmp_path), client_factory=lambda _address: client)
    handle = backend.submit(_manifest(tmp_path))

    backend.inspect(handle.run_id)

    assert backend.store.read_results(handle.run_id) == []
    events = backend.store.read_events(handle.run_id)
    assert any(
        event.level == "warning"
        and event.message == "result reported outside a recognized phase; discarded"
        for event in events
    )


def test_a_result_under_an_unrecognised_phase_name_is_discarded_with_a_warning(
    tmp_path,
):
    client = FakeRayClient(status="RUNNING")
    backend = RayBackend(store=FilesystemRunStore(tmp_path), client_factory=lambda _address: client)
    handle = backend.submit(_manifest(tmp_path))
    token = _token(backend, handle.run_id)
    client.logs = f'IAX_PHASE=something-else {token}\nIAX_RESULT {{"acc": 0.9}}\n'

    backend.inspect(handle.run_id)

    assert backend.store.read_results(handle.run_id) == []
    events = backend.store.read_events(handle.run_id)
    assert any(
        event.level == "warning"
        and event.message == "result reported outside a recognized phase; discarded"
        for event in events
    )


def test_two_inspects_of_the_same_logs_do_not_duplicate_the_result(tmp_path):
    client = FakeRayClient(status="RUNNING")
    backend = RayBackend(store=FilesystemRunStore(tmp_path), client_factory=lambda _address: client)
    handle = backend.submit(_manifest(tmp_path))
    token = _token(backend, handle.run_id)
    client.logs = f'IAX_PHASE=evaluate {token}\nIAX_RESULT {{"acc": 0.9}}\n'

    backend.inspect(handle.run_id)
    backend.inspect(handle.run_id)

    assert len(backend.store.read_results(handle.run_id)) == 1


def test_two_inspects_of_a_discarded_train_result_warn_only_once(tmp_path):
    """The warning `ray_results_seen` exists for stays singular.

    The docstring's stated reason: a discarded result never reaches `store.read_results`, so
    counting by `len(read_results(...))` alone would re-warn on every inspect. Assert the warning
    the docstring describes actually stays singular.
    """
    client = FakeRayClient(status="RUNNING")
    backend = RayBackend(store=FilesystemRunStore(tmp_path), client_factory=lambda _address: client)
    handle = backend.submit(_manifest(tmp_path))
    token = _token(backend, handle.run_id)
    client.logs = f'IAX_PHASE=train {token}\nIAX_RESULT {{"acc": 0.9}}\n'

    backend.inspect(handle.run_id)
    backend.inspect(handle.run_id)

    events = backend.store.read_events(handle.run_id)
    warnings = [
        event
        for event in events
        if event.level == "warning"
        and event.message == "result reported from the train phase; discarded"
    ]
    assert len(warnings) == 1


def test_an_unreadable_job_log_is_recorded_as_an_error(tmp_path):
    """On Ray the job log is the only channel an `IAX_RESULT` travels on.

    A failed read therefore costs the score, not just the diagnostics: the
    trial is reported as having declared nothing, which is indistinguishable
    from a workload that really printed nothing. The poll must still survive
    the failure -- but it has to leave the cause in the run's own log.
    """

    class UnreadableLogs(FakeRayClient):
        def get_job_logs(self, job_id: str) -> str:
            raise RuntimeError("dashboard returned 500")

    client = UnreadableLogs(status="RUNNING")
    backend = RayBackend(store=FilesystemRunStore(tmp_path), client_factory=lambda _address: client)
    handle = backend.submit(_manifest(tmp_path))

    status = backend.inspect(handle.run_id)

    assert status.status == "running", "a failed log read must not break the poll"
    errors = [
        event
        for event in backend.store.read_events(handle.run_id)
        if event.level == "error" and "job logs could not be read" in event.message
    ]
    assert errors, (
        "a Ray log read failed and the run's log says nothing; the trial will be "
        "reported as no_result for a reason nobody can find"
    )
    assert errors[0].details["error_type"] == "RuntimeError"
    assert "dashboard returned 500" in errors[0].details["error"]
