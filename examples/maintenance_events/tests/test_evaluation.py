import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier

from maintenance_events.evaluation import (
    block_bootstrap,
    evaluate,
    moving_blocks,
    walk_forward_splits,
)


def _as_of(n=400):
    return pd.DatetimeIndex(
        pd.date_range("2024-01-01", periods=n, freq="1D"), name="as_of"
    )


def test_every_fold_respects_the_horizon_gap():
    as_of = _as_of()
    for train_idx, test_idx in walk_forward_splits(as_of, n_folds=4, horizon_days=15):
        gap = (as_of[test_idx[0]] - as_of[train_idx[-1]]).days
        assert gap > 15


def test_training_sets_expand():
    as_of = _as_of()
    sizes = [
        len(tr) for tr, _ in walk_forward_splits(as_of, n_folds=4, horizon_days=15)
    ]
    assert sizes == sorted(sizes)
    assert len(set(sizes)) == len(sizes)


def test_test_blocks_do_not_overlap():
    as_of = _as_of()
    seen: set[int] = set()
    for _, test_idx in walk_forward_splits(as_of, n_folds=4, horizon_days=15):
        assert not seen & set(test_idx.tolist())
        seen |= set(test_idx.tolist())


def test_no_test_index_ever_appears_in_its_own_training_set():
    as_of = _as_of()
    for train_idx, test_idx in walk_forward_splits(as_of, n_folds=4, horizon_days=15):
        assert not set(train_idx.tolist()) & set(test_idx.tolist())


def test_fold_without_positives_is_reported_as_none_and_excluded():
    as_of = _as_of(200)
    X = pd.DataFrame({"f": np.arange(200, dtype=float)}, index=as_of)
    y = pd.Series(np.zeros(200, dtype=int), index=as_of)
    y.iloc[:40] = 1  # positivos solo al principio: los folds tardíos quedan vacíos
    folds, summary, _ = evaluate(
        X, y, as_of, lambda: DummyClassifier(strategy="prior"), n_folds=3
    )
    assert any(f.pr_auc is None for f in folds)
    assert summary["valid_folds"] == sum(f.pr_auc is not None for f in folds)


def test_summary_reports_the_baseline_next_to_the_objective():
    as_of = _as_of(300)
    rng = np.random.default_rng(0)
    y = pd.Series((rng.random(300) < 0.25).astype(int), index=as_of)
    X = pd.DataFrame({"f": y.to_numpy() + rng.normal(0, 0.1, 300)}, index=as_of)
    _, summary, _ = evaluate(
        X, y, as_of, lambda: DummyClassifier(strategy="prior"), n_folds=3
    )
    assert 0.0 <= summary["mean_baseline"] <= 1.0
    assert 0.0 <= summary["mean_pr_auc"] <= 1.0


# --- la incertidumbre se mide sobre las predicciones, no sobre las medias ---


def _scored(n=300, seed=0):
    as_of = _as_of(n)
    rng = np.random.default_rng(seed)
    y = pd.Series((rng.random(n) < 0.3).astype(int), index=as_of)
    X = pd.DataFrame({"f": y.to_numpy() + rng.normal(0, 0.5, n)}, index=as_of)
    return X, y, as_of


def test_every_tested_window_keeps_its_prediction():
    """El estimador nuevo puntúa una sola vez sobre todas las predicciones
    out-of-fold. Promediar el PR-AUC de cada bloque tira la mitad de la
    información: son medias de muestras chicas, no una medición."""
    X, y, as_of = _scored()
    folds, _, oof = evaluate(
        X, y, as_of, lambda: DummyClassifier(strategy="stratified"), n_folds=4
    )

    tested = sum(f.n_test for f in folds if f.pr_auc is not None)
    assert len(oof.y) == tested
    assert len(oof.scores) == tested
    assert list(oof.as_of) == sorted(oof.as_of)


def test_a_replicate_pairs_its_score_with_its_own_base_rate():
    """La tasa base se mide dentro de la réplica. Tomarla de la muestra
    original daría un lift que ninguna réplica alcanzó — el mismo error que
    `baseline_metric` evita entre folds."""
    X, y, as_of = _scored()
    _, _, oof = evaluate(
        X, y, as_of, lambda: DummyClassifier(strategy="stratified"), n_folds=4
    )

    reps = block_bootstrap(oof, n_resamples=50, block=15, seed=1)

    assert len(reps) == 50
    for pr_auc, baseline in reps:
        assert 0.0 <= pr_auc <= 1.0
        assert 0.0 < baseline < 1.0


def test_overlapping_windows_widen_the_interval():
    """La propiedad que justifica el bootstrap por bloques.

    Las ventanas van día a día y su etiqueta mira 15 días adelante, así que
    una racha de días vecinos comparte casi la misma etiqueta. Remuestrear de a
    una los trata como independientes y devuelve un error estándar más chico
    que el real: confianza inventada por ignorar la correlación.

    El fixture tiene esa correlación a propósito —las positivas vienen en
    rachas de 20 días, como los eventos reales—. Con etiquetas iid la
    corrección no tiene nada que corregir y la propiedad no se sostiene, que
    es exactamente lo que el bootstrap por bloques dice.
    """
    from maintenance_events.evaluation import OutOfFold

    n = 600
    as_of = _as_of(n)
    rng = np.random.default_rng(5)
    runs = np.repeat(rng.random(n // 20) < 0.4, 20)
    y = runs.astype(int)
    scores = y * 0.6 + rng.normal(0, 0.4, n)
    oof = OutOfFold(as_of=as_of, y=y, scores=scores)

    def se(block):
        lifts = [
            pr - base
            for pr, base in block_bootstrap(oof, n_resamples=300, block=block, seed=3)
        ]
        return float(np.std(lifts, ddof=1))

    assert se(20) > 1.5 * se(1)


def test_a_block_draw_keeps_its_neighbours():
    """Un bloque es un tramo contiguo: eso es lo que conserva la
    correlación que el remuestreo de a una destruye."""
    idx = moving_blocks(n=100, block=10, rng=np.random.default_rng(0))

    assert len(idx) == 100
    runs = [idx[i : i + 10] for i in range(0, 100, 10)]
    for run in runs:
        assert list(run) == list(range(run[0], run[0] + 10))


def test_a_replicate_without_both_classes_is_dropped_not_scored():
    """Un PR-AUC sobre una réplica sin positivas no está definido. Mandar un
    cero sería una observación que baja el promedio sin haber medido nada."""
    as_of = _as_of(120)
    y = pd.Series(np.zeros(120, dtype=int), index=as_of)
    y.iloc[:8] = 1
    X = pd.DataFrame({"f": y.to_numpy() * 1.0}, index=as_of)
    from maintenance_events.evaluation import OutOfFold

    oof = OutOfFold(as_of=as_of, y=y.to_numpy(), scores=X["f"].to_numpy())
    reps = block_bootstrap(oof, n_resamples=40, block=20, seed=0)

    assert len(reps) < 40
    assert all(0.0 < base < 1.0 for _, base in reps)
