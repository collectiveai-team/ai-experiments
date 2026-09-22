"""Agregados por ventana.

Cada ventana se resume en un vector de features: estadísticos y tendencia por
señal, sobre la ventana completa y sobre las sub-ventanas recientes de 7 y 30
días, más dos derivadas físicas —el salto térmico del agua de circulación y el
vacío normalizado por temperatura ambiente— que son el mecanismo por el que el
ensuciamiento se hace visible.

Los tramos con la unidad parada se excluyen: promediar temperaturas de un
condensador apagado inventa señal donde no la hay.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from maintenance_events.windows import Window

SUB_WINDOWS = (7, 30)
_POWER = "active_power"


def _column_stats(frame: pd.DataFrame) -> tuple[np.ndarray, ...]:
    """Los siete estadísticos de todas las columnas de una vez.

    Hacerlo columna por columna con la API de pandas cuesta una llamada a
    ``Series.quantile`` por señal y por sub-ventana: decenas de miles por
    trial, y era dos tercios del tiempo de la featurización. Sobre el bloque
    numpy salen los mismos números en una pasada.

    Se respetan las convenciones de pandas: ``std`` con ``ddof=1`` y cuantiles
    con interpolación lineal.
    """
    values = frame.to_numpy(dtype=float)
    hours = (frame.index - frame.index[0]).total_seconds().to_numpy() / 3600.0

    with warnings.catch_warnings():
        # Una columna enteramente NaN es un dato legítimo acá (señal caída ese
        # tramo): el aviso de numpy no aporta y el NaN resultante es correcto.
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(values, axis=0)
        std = np.nanstd(values, axis=0, ddof=1)
        minimum = np.nanmin(values, axis=0)
        maximum = np.nanmax(values, axis=0)
        p10, p90 = np.nanpercentile(values, [10.0, 90.0], axis=0)

        mask = np.isfinite(values)
        xs = np.where(mask, hours[:, None], np.nan)
        ys = np.where(mask, values, np.nan)
        dx = xs - np.nanmean(xs, axis=0)
        dy = ys - np.nanmean(ys, axis=0)
        numerator = np.nansum(dx * dy, axis=0)
        denominator = np.nansum(dx * dx, axis=0)

    usable = (mask.sum(axis=0) >= 2) & (denominator > 0)
    slope = np.where(usable, numerator / np.where(usable, denominator, 1.0), np.nan)
    return mean, std, minimum, maximum, p10, p90, slope


_STAT_NAMES = ("mean", "std", "min", "max", "p10", "p90", "slope")


def _summarize(frame: pd.DataFrame, suffix: str = "") -> dict[str, float]:
    if frame.empty:
        return {
            f"{c}_{s}{suffix}": float("nan") for c in frame.columns for s in _STAT_NAMES
        }
    stats = _column_stats(frame)
    return {
        f"{column}_{name}{suffix}": float(values[i])
        for i, column in enumerate(frame.columns)
        for name, values in zip(_STAT_NAMES, stats)
    }


def _with_derivatives(history: pd.DataFrame) -> pd.DataFrame:
    enriched = history.copy()
    if {"cw_outlet_east_temp", "cw_inlet_temp"} <= set(history.columns):
        enriched["delta_t_east"] = (
            history["cw_outlet_east_temp"] - history["cw_inlet_temp"]
        )
    if {"cw_outlet_west_temp", "cw_inlet_temp"} <= set(history.columns):
        enriched["delta_t_west"] = (
            history["cw_outlet_west_temp"] - history["cw_inlet_temp"]
        )
    if {"condenser_vacuum_east", "ambient_temp"} <= set(history.columns):
        ambient = history["ambient_temp"].replace(0.0, np.nan)
        enriched["vacuum_per_ambient"] = history["condenser_vacuum_east"] / ambient
    return enriched


def _online(history: pd.DataFrame, threshold: float) -> pd.DataFrame:
    if _POWER not in history.columns:
        return history
    return history[history[_POWER] > threshold]


def build_feature_table(
    windows: list[Window],
    offline_power_threshold: float = 1.0,
    min_valid_fraction: float = 0.5,
) -> tuple[pd.DataFrame, pd.Series, pd.DatetimeIndex]:
    rows: list[dict[str, float]] = []
    labels: list[int] = []
    stamps: list[pd.Timestamp] = []

    for window in windows:
        online = _online(window.history, offline_power_threshold)
        if (
            len(window.history) == 0
            or len(online) / len(window.history) < min_valid_fraction
        ):
            continue
        enriched = _with_derivatives(online)

        features = _summarize(enriched)
        for days in SUB_WINDOWS:
            recent = enriched.loc[
                enriched.index >= window.as_of - pd.Timedelta(days=days)
            ]
            if recent.empty:
                recent = enriched
            features.update(_summarize(recent, suffix=f"_{days}d"))
        features["days_since_last_event"] = window.days_since_last_event
        features["valid_fraction"] = len(online) / len(window.history)

        rows.append(features)
        labels.append(window.label)
        stamps.append(window.as_of)

    index = pd.DatetimeIndex(stamps, name="as_of")
    if not rows:
        return (
            pd.DataFrame(index=index),
            pd.Series(dtype=int, index=index, name="label"),
            index,
        )
    X = pd.DataFrame(rows, index=index).replace([np.inf, -np.inf], np.nan)
    y = pd.Series(labels, index=index, name="label")
    return X, y, index
