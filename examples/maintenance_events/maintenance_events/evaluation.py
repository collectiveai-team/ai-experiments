"""Walk-forward con gap, y las métricas que se reportan.

Los cortes son expansivos: cada fold entrena con todo el pasado y evalúa el
bloque siguiente. Entre el fin del train y el inicio del test hay un hueco del
tamaño del horizonte —sin ese hueco, las etiquetas del final del train miran
eventos que caen dentro del test, y la métrica sale inflada.

Con eventos tan agrupados como los de este dataset, un bloque de test puede
quedarse sin ningún positivo. Ese fold no tiene PR-AUC definido: se reporta como
``None`` y no entra en el resumen.

El objetivo **no** sale del promedio de los folds. Medido sobre estos datos, la
dispersión del lift entre folds crece casi exactamente como la raíz de su
cantidad —bloques de test más chicos dan folds más ruidosos—, así que subir
folds no compra poder: el mínimo detectable queda clavado en ~0.115 de 12 a 20
folds. El promedio de folds es un estimador que tira información: resume ~800
ventanas en 17 números, cada uno un PR-AUC de muestra chica.

Lo que se reporta es un único PR-AUC sobre todas las predicciones out-of-fold
juntas, y su incertidumbre sale de remuestrear esas ventanas. El remuestreo es
**por bloques contiguos**: las ventanas van día a día y la etiqueta mira 15 días
adelante, así que dos vecinas son casi la misma observación y remuestrearlas de
a una devolvería un error estándar más chico que el real.

Lo que este intervalo cubre y lo que no: mide cuánta suerte hay en *qué ventanas
tocaron*, con el modelo fijo. No incluye la varianza de volver a entrenar, así
que es la respuesta a "¿este config le gana a la tasa base sobre estos datos?" y
no a "¿cuánto va a rendir el mes que viene?".
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)

MIN_TRAIN = 30

#: El largo de bloque por defecto del remuestreo: el horizonte de la etiqueta,
#: que es la distancia a la que dos ventanas dejan de mirar los mismos días.
HORIZON_DAYS_DEFAULT = 15


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

    def as_metric(self) -> dict[str, float | None]:
        """El fold como observación del objetivo.

        ``objective.aggregate: mean`` promedia las observaciones y calcula el
        error estándar, así que cada fold llega con la clave objetivo y su
        baseline juntas: el lift se arma dentro de una observación, no entre
        dos.

        Un fold sin positivas no tiene PR-AUC definido y no emite ninguna de
        las dos claves. Mandar un cero arrastraría el promedio por un fold que
        no midió nada; el arnés saltea la observación incompleta.
        """
        values: dict[str, float | None] = {
            "step": self.index,
            "fold_auroc": self.auroc,
            "fold_positives": self.positives,
            "fold_n_test": self.n_test,
        }
        if self.pr_auc is not None and self.baseline is not None:
            values["pr_auc"] = self.pr_auc
            values["baseline_pr_auc"] = self.baseline
        return values


@dataclass
class OutOfFold:
    """Cada ventana evaluada, con la predicción del modelo que no la vio.

    En orden temporal, que es lo que hace que los bloques del remuestreo sean
    tramos de tiempo y no índices sueltos.
    """

    as_of: pd.DatetimeIndex
    y: np.ndarray
    scores: np.ndarray

    def __len__(self) -> int:
        return len(self.y)

    @property
    def positives(self) -> int:
        return int(self.y.sum())


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
        while (
            test_start < n and (as_of[test_start] - as_of[train_stop - 1]) <= gap_days
        ):
            test_start += 1
        test_stop = min(test_start + block, n)
        if train_stop < MIN_TRAIN or test_start >= test_stop:
            continue
        splits.append((np.arange(train_stop), np.arange(test_start, test_stop)))
    return splits


def _recall_at_precision(
    y_true: np.ndarray, scores: np.ndarray, floor: float = 0.5
) -> float:
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
    oof_index: list[np.ndarray] = []
    oof_scores: list[np.ndarray] = []
    for i, (train_idx, test_idx) in enumerate(
        walk_forward_splits(as_of, n_folds, horizon_days)
    ):
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
        base.auroc = (
            float(roc_auc_score(y_test, scores)) if len(np.unique(y_test)) > 1 else None
        )
        base.recall_at_p50 = _recall_at_precision(y_test, scores)
        base.baseline = float(y_test.mean())
        oof_index.append(test_idx)
        oof_scores.append(scores)
        folds.append(base)

    scored = [f for f in folds if f.pr_auc is not None]
    # Nada acá se llama `pr_auc` ni `baseline_pr_auc`: el resumen es una línea
    # más de stdout, y si repitiera las claves objetivo sería una observación
    # extra — el promedio contado dos veces y un error estándar más chico que
    # el real. Lo que resume ya lo calcula el arnés.
    summary = {
        "mean_pr_auc": float(np.mean([f.pr_auc for f in scored])) if scored else 0.0,
        "pr_auc_std": float(np.std([f.pr_auc for f in scored], ddof=1))
        if len(scored) > 1
        else 0.0,
        "auroc": float(np.mean([f.auroc for f in scored if f.auroc is not None]))
        if scored
        else 0.0,
        "recall_at_p50": float(np.mean([f.recall_at_p50 for f in scored]))
        if scored
        else 0.0,
        "mean_baseline": float(np.mean([f.baseline for f in scored]))
        if scored
        else 0.0,
        "valid_folds": len(scored),
    }
    if oof_index:
        order = np.concatenate(oof_index)
        pooled = OutOfFold(
            as_of=as_of[order],
            y=y.iloc[order].to_numpy(),
            scores=np.concatenate(oof_scores),
        )
    else:
        pooled = OutOfFold(as_of=as_of[:0], y=np.empty(0), scores=np.empty(0))
    summary["pooled_windows"] = len(pooled)
    summary["pooled_positives"] = pooled.positives
    return folds, summary, pooled


def moving_blocks(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """``n`` índices remuestreados de a tramos contiguos de largo ``block``.

    Los arranques se sortean uniformemente y el tramo se toma circular, así
    que ninguna posición está sub-representada por vivir cerca de un borde.
    """
    if n <= 0 or block <= 0:
        return np.empty(0, dtype=int)
    block = min(block, n)
    starts = rng.integers(0, n, size=-(-n // block))
    drawn = (starts[:, None] + np.arange(block)[None, :]) % n
    return drawn.reshape(-1)[:n]


def block_bootstrap(
    oof: OutOfFold,
    n_resamples: int = 500,
    block: int = HORIZON_DAYS_DEFAULT,
    seed: int = 0,
) -> list[tuple[float, float]]:
    """``(pr_auc, tasa_base)`` por réplica, medidos sobre la misma remuestra.

    La tasa base se recalcula dentro de cada réplica a propósito: tomarla de la
    muestra original produciría un lift que ninguna réplica alcanzó, el mismo
    error que ``baseline_metric`` evita entre folds.

    Una réplica que no tiene las dos clases no define PR-AUC y se descarta en
    vez de puntuar cero. Con datos tan desbalanceados que eso pase seguido, lo
    que queda es un conteo de réplicas más bajo, que es información honesta
    sobre lo poco que hay para medir.
    """
    n = len(oof)
    if n == 0:
        return []
    rng = np.random.default_rng(seed)
    out: list[tuple[float, float]] = []
    for _ in range(n_resamples):
        idx = moving_blocks(n, block, rng)
        y_r = oof.y[idx]
        positives = int(y_r.sum())
        if positives == 0 or positives == len(y_r):
            continue
        out.append(
            (
                float(average_precision_score(y_r, oof.scores[idx])),
                float(y_r.mean()),
            )
        )
    return out


def folds_to_frame(folds: list[FoldResult]) -> pd.DataFrame:
    return pd.DataFrame([asdict(f) for f in folds])
