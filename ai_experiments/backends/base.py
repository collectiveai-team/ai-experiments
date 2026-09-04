from __future__ import annotations

from abc import ABC, abstractmethod

from ai_experiments.schemas import (
    DiagnosisReport,
    ExperimentManifest,
    RunEvent,
    RunHandle,
    RunStatus,
)


class ExperimentBackend(ABC):
    @abstractmethod
    def submit(self, manifest: ExperimentManifest) -> RunHandle:
        raise NotImplementedError

    @abstractmethod
    def inspect(self, run_id: str) -> RunStatus:
        raise NotImplementedError

    @abstractmethod
    def logs(self, run_id: str, tail: int = 200) -> list[RunEvent]:
        raise NotImplementedError

    @abstractmethod
    def cancel(self, run_id: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def diagnose(self, run_id: str) -> DiagnosisReport:
        raise NotImplementedError

    def reap(self, run_id: str) -> dict[str, object]:
        """Clean up whatever a run left behind when its supervisor died.

        Called instead of :meth:`cancel` when there is no supervisor left to
        ask. Backends whose runs cannot outlive their supervision (Ray tracks
        the job itself) need do nothing.
        """
        return {"outcome": "unsupported"}
