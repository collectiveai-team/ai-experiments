"""Walk-forward con gap, y las métricas que se reportan.

Los cortes son expansivos: cada fold entrena con todo el pasado y evalúa el
bloque siguiente. Entre el fin del train y el inicio del test hay un hueco del
tamaño del horizonte —sin ese hueco, las etiquetas del final del train miran
eventos que caen dentro del test, y la métrica sale inflada.

Con eventos tan agrupados como los de este dataset, un bloque de test puede
quedarse sin ningún positivo. Ese fold no tiene PR-AUC definido: se reporta como
``None`` y no entra en el promedio.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

MIN_TRAIN = 30


@dataclass
class FoldResult:
    index: int
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    n_train: int
    n_test: int
    positives: int
    pr_auc: float | None
    auroc: float | None
    recall_at_p50: float | None
    baseline: float | None

    def as_metric(self) -> dict[str, float]:
        return {
            "step": self.index,
            "fold_pr_auc": self.pr_auc,
            "fold_auroc": self.auroc,
            "fold_positives": self.positives,
            "fold_n_test": self.n_test,
        }


def walk_forward_splits(
    as_of: pd.DatetimeIndex, n_folds: int, horizon_days: int = 15
) -> list[tuple[np.ndarray, np.ndarray]]:
    n = len(as_of)
    if n == 0 or n_folds < 1:
        return []
    block = n // (n_folds + 1)
    if block == 0:
        return []

    gap_days = pd.Timedelta(days=horizon_days)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for fold in range(n_folds):
        train_stop = block * (fold + 1)
        test_start = train_stop
        while test_start < n and (as_of[test_start] - as_of[train_stop - 1]) <= gap_days:
            test_start += 1
        test_stop = min(test_start + block, n)
        if train_stop < MIN_TRAIN or test_start >= test_stop:
            continue
        splits.append((np.arange(train_stop), np.arange(test_start, test_stop)))
    return splits


def _recall_at_precision(y_true: np.ndarray, scores: np.ndarray, floor: float = 0.5) -> float:
    precision, recall, _ = precision_recall_curve(y_true, scores)
    eligible = recall[precision >= floor]
    return float(eligible.max()) if eligible.size else 0.0


def evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    as_of: pd.DatetimeIndex,
    make_model: Callable[[], object],
    n_folds: int = 5,
    horizon_days: int = 15,
) -> tuple[list[FoldResult], dict[str, float]]:
    folds: list[FoldResult] = []
    for i, (train_idx, test_idx) in enumerate(walk_forward_splits(as_of, n_folds, horizon_days)):
        y_train = y.iloc[train_idx].to_numpy()
        y_test = y.iloc[test_idx].to_numpy()
        positives = int(y_test.sum())

        base = FoldResult(
            index=i,
            train_end=as_of[train_idx[-1]],
            test_start=as_of[test_idx[0]],
            test_end=as_of[test_idx[-1]],
            n_train=len(train_idx),
            n_test=len(test_idx),
            positives=positives,
            pr_auc=None,
            auroc=None,
            recall_at_p50=None,
            baseline=None,
        )
        if positives == 0 or len(np.unique(y_train)) < 2:
            folds.append(base)
            continue

        model = make_model()
        model.fit(X.iloc[train_idx], y_train)
        scores = model.predict_proba(X.iloc[test_idx])[:, 1]

        base.pr_auc = float(average_precision_score(y_test, scores))
        base.auroc = float(roc_auc_score(y_test, scores)) if len(np.unique(y_test)) > 1 else None
        base.recall_at_p50 = _recall_at_precision(y_test, scores)
        base.baseline = float(y_test.mean())
        folds.append(base)

    scored = [f for f in folds if f.pr_auc is not None]
    summary = {
        "pr_auc": float(np.mean([f.pr_auc for f in scored])) if scored else 0.0,
        "pr_auc_std": float(np.std([f.pr_auc for f in scored])) if scored else 0.0,
        "auroc": float(np.mean([f.auroc for f in scored if f.auroc is not None])) if scored else 0.0,
        "recall_at_p50": float(np.mean([f.recall_at_p50 for f in scored])) if scored else 0.0,
        "baseline_pr_auc": float(np.mean([f.baseline for f in scored])) if scored else 0.0,
        "valid_folds": len(scored),
    }
    return folds, summary


def folds_to_frame(folds: list[FoldResult]) -> pd.DataFrame:
    return pd.DataFrame([asdict(f) for f in folds])
