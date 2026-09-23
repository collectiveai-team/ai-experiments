from __future__ import annotations

import pytest
from pydantic import ValidationError

from ai_experiments.schemas import DataSpec, ExperimentManifest, WorkloadSpec
from ai_experiments.store import FilesystemRunStore


def _manifest(**overrides: object) -> ExperimentManifest:
    data: dict[str, object] = {
        "experiment": "smoke",
        "backend": "ray",
        "workload": WorkloadSpec(entrypoint="python train.py"),
    }
    data.update(overrides)
    return ExperimentManifest.model_validate(data)


def test_backend_address_round_trips_yaml(tmp_path):
    manifest = _manifest(backend_address=" https://ray.example.com ")
    path = tmp_path / "manifest.yaml"
    path.write_text(manifest.to_yaml())

    restored = ExperimentManifest.from_yaml(path)

    assert restored.backend_address == "https://ray.example.com"


def test_backend_address_is_persisted_in_run_manifest(tmp_path):
    store = FilesystemRunStore(tmp_path / "runs")
    manifest = _manifest(backend_address="https://ray.example.com")

    run_id, _run_dir = store.create_run(manifest)

    persisted = store.read_manifest(run_id)
    assert persisted is not None
    assert persisted.backend_address == "https://ray.example.com"


def test_missing_run_manifest_falls_back_to_environment_resolution(tmp_path):
    store = FilesystemRunStore(tmp_path / "runs")

    assert store.read_manifest("run_missing") is None


def test_list_runs_skips_internal_directories(tmp_path):
    store = FilesystemRunStore(tmp_path / "runs")
    run_id, _ = store.create_run(_manifest(backend="local", backend_address=None))
    (store.root / "_campaigns").mkdir()
    (store.root / "_escalations").mkdir()

    assert list(store.list_runs()) == [run_id]


@pytest.mark.parametrize("backend_address", ["", "   ", "ray://cluster"])
def test_backend_address_rejects_malformed_values(backend_address):
    with pytest.raises(ValidationError):
        _manifest(backend_address=backend_address)


def test_local_manifest_does_not_need_backend_address():
    manifest = _manifest(backend="local", backend_address=None)

    assert manifest.backend == "local"
    assert manifest.backend_address is None


def test_error_tail_keeps_the_last_thing_a_workload_said():
    """A planner learns from the error, not from the exit code."""
    from ai_experiments.failures import error_tail

    tail = error_tail(
        [
            "Traceback (most recent call last):\n",
            "  File train.py, line 10\n",
            "RuntimeError: out of memory: needs 25.2 GB\n",
        ]
    )

    assert tail.endswith("RuntimeError: out of memory: needs 25.2 GB")
    assert "Traceback" in tail


def test_error_tail_is_bounded_because_workload_output_is_untrusted():
    from ai_experiments.failures import ERROR_TAIL_CHARS, error_tail

    tail = error_tail(["x" * 5000])

    assert len(tail) <= ERROR_TAIL_CHARS
    assert tail.endswith("…")


def test_error_tail_of_a_silent_workload_is_empty():
    from ai_experiments.failures import error_tail

    assert error_tail(["\n", "   \n"]) == ""


def test_failure_message_trims_a_ray_job_log_to_the_part_that_explains_it():
    """Ray hands back up to 20,000 characters; the planner reads this field."""
    from ai_experiments.failures import ERROR_TAIL_CHARS, failure_message

    message = failure_message(
        "Ray job failed",
        "Job entrypoint command failed with exit code 1, last available logs:\n"
        + "noise\n" * 500
        + "RuntimeError: out of memory: needs 25.2 GB\n",
    )

    assert message.startswith("Ray job failed: ")
    assert message.endswith("RuntimeError: out of memory: needs 25.2 GB")
    assert len(message) <= ERROR_TAIL_CHARS + len("Ray job failed: ")


def test_failure_message_without_output_is_just_the_prefix():
    from ai_experiments.failures import failure_message

    assert failure_message("Ray job failed", "") == "Ray job failed"


def test_a_single_entrypoint_workload_runs_as_the_evaluate_phase():
    workload = WorkloadSpec(entrypoint="python toy.py")

    assert workload.phases() == [("evaluate", "python toy.py")]


def test_a_two_phase_workload_runs_train_then_evaluate():
    workload = WorkloadSpec(
        entrypoint="python train.py",
        train="python train.py",
        evaluate="python evaluate.py",
    )

    assert workload.phases() == [
        ("train", "python train.py"),
        ("evaluate", "python evaluate.py"),
    ]


def test_a_phase_dropped_after_construction_is_not_run_silently():
    workload = WorkloadSpec(
        entrypoint="python x.py", train="python train.py", evaluate="python evaluate.py"
    )

    mutated = workload.model_copy(update={"evaluate": None})

    with pytest.raises(ValueError, match="both are required"):
        mutated.phases()


def test_the_train_phase_never_receives_the_test_reference():
    data = DataSpec(train="data/train", val="data/val", test="data/test")

    env = data.env_for("train")

    assert env == {"IAX_DATA_TRAIN": "data/train", "IAX_DATA_VAL": "data/val"}
    assert "IAX_DATA_TEST" not in env


def test_the_evaluate_phase_receives_the_test_reference():
    data = DataSpec(train="data/train", val="data/val", test="data/test")

    assert data.env_for("evaluate")["IAX_DATA_TEST"] == "data/test"


def test_declaring_evaluate_without_train_is_rejected():
    with pytest.raises(ValueError, match="train"):
        WorkloadSpec(entrypoint="python x.py", evaluate="python evaluate.py")


def test_declaring_train_without_evaluate_is_rejected():
    with pytest.raises(ValueError, match="evaluate"):
        WorkloadSpec(entrypoint="python x.py", train="python train.py")


def test_declaring_test_data_without_an_evaluate_phase_is_rejected():
    with pytest.raises(ValueError, match="evaluate"):
        WorkloadSpec(entrypoint="python x.py", data=DataSpec(test="data/test"))
