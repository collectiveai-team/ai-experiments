import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier

from maintenance_events.evaluation import evaluate, walk_forward_splits


def _as_of(n=400):
    return pd.DatetimeIndex(pd.date_range("2024-01-01", periods=n, freq="1D"), name="as_of")


def test_every_fold_respects_the_horizon_gap():
    as_of = _as_of()
    for train_idx, test_idx in walk_forward_splits(as_of, n_folds=4, horizon_days=15):
        gap = (as_of[test_idx[0]] - as_of[train_idx[-1]]).days
        assert gap > 15


def test_training_sets_expand():
    as_of = _as_of()
    sizes = [len(tr) for tr, _ in walk_forward_splits(as_of, n_folds=4, horizon_days=15)]
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
    folds, summary = evaluate(X, y, as_of, lambda: DummyClassifier(strategy="prior"), n_folds=3)
    assert any(f.pr_auc is None for f in folds)
    assert summary["valid_folds"] == sum(f.pr_auc is not None for f in folds)


def test_summary_reports_the_baseline_next_to_the_objective():
    as_of = _as_of(300)
    rng = np.random.default_rng(0)
    y = pd.Series((rng.random(300) < 0.25).astype(int), index=as_of)
    X = pd.DataFrame({"f": y.to_numpy() + rng.normal(0, 0.1, 300)}, index=as_of)
    _, summary = evaluate(X, y, as_of, lambda: DummyClassifier(strategy="prior"), n_folds=3)
    assert 0.0 <= summary["baseline_pr_auc"] <= 1.0
    assert 0.0 <= summary["pr_auc"] <= 1.0
