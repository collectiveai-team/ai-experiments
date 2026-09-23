"""Los tres modelos del ejemplo, detrás de una fábrica común.

``dummy`` predice la tasa base: es el piso contra el que se leen los otros dos.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

if TYPE_CHECKING:
    from collections.abc import Callable

MODELS = ("hist_gb", "logreg", "dummy")


def make_model(
    name: str,
    learning_rate: float = 0.1,
    max_leaf_nodes: int = 31,
    min_samples_leaf: int = 10,
    l2: float = 0.0,
    max_bins: int = 255,
    class_weight: str | None = "balanced",
    seed: int = 0,
) -> Callable[[], object]:
    if name == "hist_gb":
        return lambda: HistGradientBoostingClassifier(
            learning_rate=learning_rate,
            max_leaf_nodes=max_leaf_nodes,
            min_samples_leaf=min_samples_leaf,
            l2_regularization=l2,
            max_bins=max_bins,
            class_weight=class_weight,
            random_state=seed,
        )
    if name == "logreg":
        return lambda: make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight=class_weight, random_state=seed),
        )
    if name == "dummy":
        return lambda: DummyClassifier(strategy="prior")
    raise ValueError(f"modelo desconocido: {name!r}; opciones: {MODELS}")
