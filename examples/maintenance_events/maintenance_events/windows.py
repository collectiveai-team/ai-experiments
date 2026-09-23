"""Ventanas diarias etiquetadas.

El modelo corre todos los días: cada ``as_of`` es una medianoche, mira hacia
atrás ``window_days`` y pregunta si hay una limpieza en los próximos 15 días.

La invariante que sostiene todo: el historial es **semiabierto por la derecha**,
``[as_of - window_days, as_of)``. Ninguna fila con timestamp ``>= as_of`` puede
entrar en las features, y ``days_since_last_event`` solo mira eventos anteriores
a ``as_of``. Es la información que realmente existe esa mañana.

La segunda invariante es de cobertura. Las señales van de 2020 a 2024 pero los
registros de limpieza no: el de MAPRO empieza en 2022-08 y el log de operarios
en 2023-11. Fuera del período de su fuente, "no hay evento" significa "no hay
registro", no "no pasó nada". Etiquetar esas ventanas como negativas mete años
de falsos negativos y, peor, hace que ``label_source`` se compare sobre
cantidades distintas de basura. Por eso ``build_windows`` recorta los ``as_of``
a las fechas donde el horizonte entero cae dentro de la cobertura de la fuente.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

HORIZON_DAYS = 15
STRIDE_DAYS = 1

_SOURCES = {
    "mapro": {"mapro", "both"},
    "operator": {"operator", "both"},
    "union": {"mapro", "operator", "both"},
}


@dataclass(frozen=True)
class Window:
    as_of: pd.Timestamp
    history: pd.DataFrame
    label: int
    days_since_last_event: float


def select_events(events: pd.DataFrame, label_source: str) -> pd.Series:
    try:
        keep = _SOURCES[label_source]
    except KeyError:
        raise ValueError(
            f"label_source inválido: {label_source!r}; opciones: {sorted(_SOURCES)}"
        ) from None
    return events.loc[events["source"].isin(keep), "date"].sort_values().reset_index(drop=True)


def label_coverage(
    events: pd.DataFrame, label_source: str
) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """El período en el que la fuente elegida realmente registró.

    Se deriva del primer y último evento observado: es lo único que el
    artefacto sabe sobre cuándo el registro estuvo activo. Es conservador —
    puede recortar de más si el registro arrancó antes de su primera limpieza—
    pero nunca inventa negativos.
    """
    dates = select_events(events, label_source)
    if dates.empty:
        return None
    return pd.Timestamp(dates.iloc[0]), pd.Timestamp(dates.iloc[-1])


def build_windows(
    signals: pd.DataFrame,
    event_dates: pd.Series,
    window_days: int,
    horizon_days: int = HORIZON_DAYS,
    stride_days: int = STRIDE_DAYS,
    coverage: tuple[pd.Timestamp, pd.Timestamp] | None = None,
) -> list[Window]:
    if signals.empty:
        return []
    events = pd.to_datetime(pd.Series(event_dates)).sort_values().to_numpy()

    first = signals.index.min().normalize() + pd.Timedelta(days=window_days)
    last = signals.index.max().normalize() - pd.Timedelta(days=horizon_days)
    if coverage is not None:
        # el horizonte entero tiene que caer dentro del período registrado
        first = max(first, pd.Timestamp(coverage[0]).normalize())
        last = min(
            last,
            pd.Timestamp(coverage[1]).normalize() - pd.Timedelta(days=horizon_days),
        )
    if last < first:
        return []

    horizon = pd.Timedelta(days=horizon_days)
    span = pd.Timedelta(days=window_days)
    windows: list[Window] = []
    for as_of in pd.date_range(first, last, freq=f"{stride_days}D"):
        history = signals.loc[(signals.index >= as_of - span) & (signals.index < as_of)]
        if history.empty:
            continue
        upcoming = events[
            (events > as_of.to_datetime64()) & (events <= (as_of + horizon).to_datetime64())
        ]
        past = events[events < as_of.to_datetime64()]
        elapsed = (as_of - pd.Timestamp(past[-1])).days if past.size else np.nan
        windows.append(
            Window(
                as_of=as_of,
                history=history,
                label=int(upcoming.size > 0),
                days_since_last_event=float(elapsed),
            )
        )
    return windows
