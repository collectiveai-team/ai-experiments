"""Shared in-memory fakes for the backend and mlflow ports (CES-64), exposed as fixtures.

``FakeBackend``/``_goal`` and ``FakeMlflowModule`` used to be defined once (in
``test_orchestrator.py`` and ``test_tracking.py`` respectively) and reached from other test
modules via a bare ``from test_orchestrator import ...`` / ``from test_tracking import ...``.
That import only resolves because pytest injects the test rootdir into ``sys.path`` at
collection time; a static checker such as pyrefly has no such injection and reports
``missing-import``. Defining the fakes here means every consumer gets them the normal pytest
way -- as fixtures -- with exactly one definition of each.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from ai_experiments.backends.base import ExperimentBackend
from ai_experiments.monitoring.rules import diagnose_run
from ai_experiments.schemas import (
    BudgetSpec,
    DiagnosisReport,
    ExperimentManifest,
    GoalSpec,
    MetricPoint,
    ObjectiveSpec,
    RunEvent,
    RunHandle,
    RunStatus,
    StrategySpec,
    WorkloadSpec,
    utc_now,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from ai_experiments.store import FilesystemRunStore


class FakeBackend(ExperimentBackend):
    """Runs 'complete' instantly.

    The objective is a deterministic function of the submitted params, recorded as a
    metric on inspect.
    """

    def __init__(
        self,
        store: FilesystemRunStore,
        objective_fn: Callable[[dict], float] | None = None,
    ) -> None:
        self.store = store
        self.objective_fn = objective_fn or (lambda p: (p["x"] - 2.0) ** 2)
        self.submitted: list[ExperimentManifest] = []
        self.cancelled: list[str] = []

    def submit(self, manifest: ExperimentManifest) -> RunHandle:
        run_id, run_dir = self.store.create_run(manifest)
        self.submitted.append(manifest)
        handle = RunHandle(
            run_id=run_id,
            backend="local",
            status="submitted",
            status_uri=str(self.store.status_path(run_id)),
            run_dir=str(run_dir),
        )
        self.store.write_handle(handle)
        return handle

    def inspect(self, run_id: str) -> RunStatus:
        status = self.store.read_status(run_id)
        if status.status in {"submitted", "running"}:
            manifest = self.store.read_manifest(run_id)
            assert manifest is not None
            params = manifest.metadata["params"]
            value = self.objective_fn(params)
            self.store.append_metric(run_id, MetricPoint(step=1, values={"loss": value}))
            status = self.store.update_status(run_id, status="completed", completed_at=utc_now())
        return status

    def logs(self, run_id: str, tail: int = 200) -> list[RunEvent]:
        return []

    def cancel(self, run_id: str) -> None:
        self.cancelled.append(run_id)
        self.store.update_status(run_id, status="cancelled", completed_at=utc_now())

    def diagnose(self, run_id: str) -> DiagnosisReport:
        return diagnose_run(self.store, run_id)


def _goal(**overrides: object) -> GoalSpec:
    data: dict = {
        "goal": "minimize (x-2)^2",
        "name": "quadratic",
        "objective": ObjectiveSpec(metric="loss", mode="min"),
        "search_space": {"x": {"type": "uniform", "low": -5.0, "high": 5.0}},
        "workload": WorkloadSpec(entrypoint="python toy.py"),
        "budget": BudgetSpec(max_trials=6, max_parallel=2),
        "strategy": StrategySpec(name="adaptive", seed=3),
    }
    data.update(overrides)
    return GoalSpec(**data)


class FakeMlflowClient:
    def __init__(self, tracking_uri=None):
        self.tracking_uri = tracking_uri
        self.experiments: dict[str, str] = {}
        self.runs: dict[str, dict] = {}
        self.counter = 0

    def get_experiment_by_name(self, name):
        if name in self.experiments:
            return SimpleNamespace(experiment_id=self.experiments[name])
        return None

    def create_experiment(self, name):
        self.experiments[name] = f"exp_{len(self.experiments)}"
        return self.experiments[name]

    def create_run(self, experiment_id, tags):
        self.counter += 1
        run_id = f"mlf_{self.counter}"
        self.runs[run_id] = {
            "experiment_id": experiment_id,
            "tags": dict(tags),
            "params": {},
            "metrics": [],
            "artifacts": [],
            "status": "RUNNING",
        }
        return SimpleNamespace(info=SimpleNamespace(run_id=run_id))

    def log_param(self, run_id, key, value):
        self.runs[run_id]["params"][key] = value

    def log_metric(self, run_id, key, value, timestamp=None, step=None):
        self.runs[run_id]["metrics"].append((key, value, step))

    def log_artifacts(self, run_id, local_dir):
        self.runs[run_id]["artifacts"].append(local_dir)

    def set_terminated(self, run_id, status):
        self.runs[run_id]["status"] = status


class FakeMlflowModule:
    """Hands every MlflowClient() the same backing store, like a real server."""

    def __init__(self):
        self.last_client = None

    def MlflowClient(self, tracking_uri=None):
        if self.last_client is None:
            self.last_client = FakeMlflowClient(tracking_uri)
        return self.last_client

    def get_tracking_uri(self):
        return "file:///fake-mlruns"


@pytest.fixture
def fake_backend_factory() -> type[FakeBackend]:
    """Return the in-memory ``ExperimentBackend`` fake: call as ``fake_backend_factory(store)``."""
    return FakeBackend


@pytest.fixture
def goal_factory() -> Callable[..., GoalSpec]:
    """Return a ``GoalSpec`` builder with defaults, overridable: ``goal_factory(x=y)``."""
    return _goal


@pytest.fixture
def fake_mlflow_module_factory() -> type[FakeMlflowModule]:
    """Return the fake ``mlflow`` module, as a constructor: ``fake_mlflow_module_factory()``."""
    return FakeMlflowModule
