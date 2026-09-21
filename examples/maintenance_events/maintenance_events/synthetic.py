"""Un dataset sintético con la misma forma que el real.

Existe para que los tests y ``--self-test`` corran sin haber bajado nada: hay
señal de verdad —el salto térmico crece al ensuciarse y se reinicia con cada
limpieza— así que el pipeline completo se puede ejercitar de punta a punta.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

COLUMNS = [
    "lp_steam_vacuum",
    "condenser_inlet_temp",
    "hotwell_level",
    "cw_inlet_temp",
    "cw_outlet_east_temp",
    "cw_outlet_west_temp",
    "ip_bypass_valve",
    "lp_bypass_valve",
    "condenser_vacuum_east",
    "condenser_vacuum_west",
    "ambient_temp",
    "active_power",
    "makeup_pct",
    "lube_oil_temp_front",
    "lube_oil_temp_rear",
    "gen_active_power",
]


def synthetic_dataset(seed: int = 0, days: int = 540) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2022-01-01", periods=days * 96, freq="15min", name="timestamp")
    hours = np.arange(len(index)) / 4.0

    event_days = np.sort(rng.choice(np.arange(60, days - 20), size=days // 18, replace=False))
    event_dates = pd.to_datetime(index[0].normalize() + pd.to_timedelta(event_days, unit="D"))

    fouling = np.zeros(len(index))
    last = 0
    for day in event_days:
        end = int(day * 96)
        fouling[last:end] = np.linspace(0, 1, end - last)
        last = end
    fouling[last:] = np.linspace(0, 1, len(index) - last)

    season = 10 * np.sin(2 * np.pi * hours / (24 * 365)) + 3 * np.sin(2 * np.pi * hours / 24)
    ambient = 18 + season + rng.normal(0, 1.0, len(index))
    inlet = ambient - 4 + rng.normal(0, 0.4, len(index))
    delta_t = 12 + 6 * fouling + rng.normal(0, 0.6, len(index))

    frame = pd.DataFrame(index=index)
    frame["ambient_temp"] = ambient
    frame["cw_inlet_temp"] = inlet
    frame["cw_outlet_east_temp"] = inlet + delta_t
    frame["cw_outlet_west_temp"] = inlet + delta_t + rng.normal(0, 0.3, len(index))
    frame["condenser_vacuum_east"] = -0.88 + 0.05 * fouling + rng.normal(0, 0.004, len(index))
    frame["condenser_vacuum_west"] = frame["condenser_vacuum_east"] + rng.normal(0, 0.002, len(index))
    frame["lp_steam_vacuum"] = frame["condenser_vacuum_east"] + rng.normal(0, 0.003, len(index))
    frame["condenser_inlet_temp"] = 40 + 2 * fouling + rng.normal(0, 0.5, len(index))
    frame["active_power"] = np.where(rng.random(len(index)) < 0.02, 0.0, 80 - 4 * fouling)
    frame["gen_active_power"] = frame["active_power"] * 0.98
    frame["hotwell_level"] = 450 + rng.normal(0, 5, len(index))
    frame["makeup_pct"] = 1.5 + 0.5 * fouling + rng.normal(0, 0.1, len(index))
    frame["ip_bypass_valve"] = rng.normal(0, 1, len(index))
    frame["lp_bypass_valve"] = rng.normal(0, 1, len(index))
    frame["lube_oil_temp_front"] = 45 + rng.normal(0, 0.5, len(index))
    frame["lube_oil_temp_rear"] = 46 + rng.normal(0, 0.5, len(index))

    sources = rng.choice(["mapro", "operator", "both"], size=len(event_dates))
    events = pd.DataFrame(
        {
            "date": event_dates,
            "source": sources,
            "duration_hours": rng.integers(1, 5, len(event_dates)).astype(float),
            "buckets_left": rng.integers(1, 6, len(event_dates)).astype(float),
            "buckets_right": rng.integers(1, 6, len(event_dates)).astype(float),
            "scheduled": rng.random(len(event_dates)) < 0.5,
        }
    )
    return frame[COLUMNS], events
