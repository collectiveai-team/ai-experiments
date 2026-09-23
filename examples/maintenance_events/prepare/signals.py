"""Ensamblado de los trimestrales en una sola tabla ancha, ya anonimizada."""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

from prepare.sppa import parse_sppa_file

TAG_RENAMES: dict[str, str] = {
    "13MAC01CP001__XQ01": "lp_steam_vacuum",
    "13MAG01CT002__XQ01": "condenser_inlet_temp",
    "13MAG10FL901__XQ01": "hotwell_level",
    "13PAB10CT001__XQ01": "cw_inlet_temp",
    "13PAB10CT002__XQ01": "cw_outlet_east_temp",
    "13PAB10CT003__XQ01": "cw_outlet_west_temp",
    "13MAN10CG101H__XQ01": "ip_bypass_valve",
    "13MAN30CG101H__XQ01": "lp_bypass_valve",
    "13PAB10CP001__XQ01": "condenser_vacuum_east",
    "13PAB10CP002__XQ01": "condenser_vacuum_west",
    "25MBL11CT002__XQ01": "ambient_temp",
    "25MBY10CE901__XQ01": "active_power",
    "13CJA00DP100__OUT_2": "makeup_pct",
    "13MAV13CT001__XQ01": "lube_oil_temp_front",
    "13MAV13CT002__XQ01": "lube_oil_temp_rear",
    "13MKA01CE003__XQ01": "gen_active_power",
}

# GRUPO4 repite tres señales con otro sufijo. No se publican como señales
# propias: tapan huecos de la homónima del grupo principal. En el corpus real
# el hueco es casi total -las dos de vacío vienen 99 % vacías en su grupo-, así
# que en la práctica GRUPO4 es la fuente y no el parche.
FALLBACK_TAGS: dict[str, str] = {
    "13PAB10CP001__XQ01__OUT": "condenser_vacuum_east",
    "13PAB10CP002__XQ01__OUT": "condenser_vacuum_west",
    "25MBL11CT002__209442__OUT": "ambient_temp",
}

QUARTER_FILE = re.compile(r"^(?P<quarter>\d{4}Q\d)GRUPO(?P<group>\d)\.csv$")


def load_signals(raw_signals_dir: Path) -> pd.DataFrame:
    by_quarter: dict[str, list[pd.DataFrame]] = defaultdict(list)
    for path in sorted(Path(raw_signals_dir).glob("*.csv")):
        match = QUARTER_FILE.match(path.name)
        if not match:
            continue
        by_quarter[match.group("quarter")].append(parse_sppa_file(path))

    quarters = [_merge_quarter(by_quarter[quarter]) for quarter in sorted(by_quarter)]
    if not quarters:
        raise FileNotFoundError(f"{raw_signals_dir}: ningún trimestral GRUPO*.csv")

    signals = pd.concat(quarters).sort_index()
    signals = signals[~signals.index.duplicated(keep="first")]
    return signals.reindex(columns=list(TAG_RENAMES.values()))


def _merge_quarter(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Un trimestre: los grupos lado a lado, renombrados, con los tags de respaldo aplicados."""
    wide = pd.concat(frames, axis=1)
    wide = wide.loc[:, ~wide.columns.duplicated()]
    renamed = wide.rename(columns=TAG_RENAMES)
    for raw_tag, target in FALLBACK_TAGS.items():
        if raw_tag not in wide.columns:
            continue
        if target in renamed.columns:
            renamed[target] = renamed[target].fillna(wide[raw_tag])
        else:
            # El grupo principal no aportó la señal en este trimestre: la
            # de GRUPO4 es todo lo que hay. Perderla sería peor que usarla.
            renamed[target] = wide[raw_tag]
    keep = [name for name in TAG_RENAMES.values() if name in renamed.columns]
    return renamed[keep]
