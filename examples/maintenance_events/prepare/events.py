"""Las dos fuentes de eventos de limpieza, normalizadas y unidas.

El CSV MAPRO son los partes con valorización; el xlsx es la bitácora que llena
quien limpia. Se solapan parcialmente. Toda columna que identifique personas,
la unidad generadora o montos en pesos se descarta acá, no aguas abajo.

La bitácora no tiene una cabecera usable: son dos filas con celdas combinadas,
y las posiciones de los encabezados no coinciden con las de los datos. Los
baldes derechos se cargan en la columna 3 aunque el rótulo ``Derecho`` esté en
la 4, que está vacía en las 50 filas. Por eso se lee por posición, con las
posiciones fijadas por un test.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from pathlib import Path

CLEANING_CODE = "Lim"
OPERATOR_COLUMNS = ["buckets_left", "buckets_right", "scheduled"]

# Posiciones en la bitácora. La 0 (``Limpió``) y la 5 (``Observaciones``) llevan
# nombres de personas y no se leen nunca.
COL_DATE = 1
COL_BUCKETS_LEFT = 2
COL_BUCKETS_RIGHT = 3
COL_MAPRO = 7


def read_mapro_events(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, sep=";", encoding="latin-1")
    cleaning = raw[raw["cod_tipo_evento"].str.strip() == CLEANING_CODE]
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(cleaning["fecha"], format="%d/%m/%Y %H:%M").dt.normalize(),
            "duration_hours": pd.to_numeric(cleaning["Suma de duracion_hs"], errors="coerce"),
        }
    )
    return frame.sort_values("date").reset_index(drop=True)


def read_operator_log(path: Path) -> pd.DataFrame:
    """Lee la bitácora. La columna de nombres (``Limpió``) nunca se devuelve."""
    raw = pd.read_excel(path, sheet_name=0, header=None)
    rows = []
    for _, row in raw.iterrows():
        date = pd.to_datetime(row.iloc[COL_DATE], errors="coerce")
        if pd.isna(date):
            continue
        rows.append(
            {
                "date": date.normalize(),
                "buckets_left": pd.to_numeric(row.iloc[COL_BUCKETS_LEFT], errors="coerce"),
                "buckets_right": pd.to_numeric(row.iloc[COL_BUCKETS_RIGHT], errors="coerce"),
                "scheduled": str(row.iloc[COL_MAPRO]).strip().upper() == "X",
            }
        )
    return (
        pd.DataFrame(rows, columns=["date", *OPERATOR_COLUMNS])
        .sort_values("date")
        .reset_index(drop=True)
    )


def merge_events(mapro: pd.DataFrame, operator: pd.DataFrame) -> pd.DataFrame:
    mapro_daily = mapro.groupby("date", as_index=False)["duration_hours"].sum()
    operator_daily = operator.groupby("date", as_index=False).agg(
        buckets_left=("buckets_left", "sum"),
        buckets_right=("buckets_right", "sum"),
        scheduled=("scheduled", "any"),
    )
    merged = mapro_daily.merge(operator_daily, on="date", how="outer", indicator=True)
    merged["source"] = (
        merged["_merge"]
        .map({"left_only": "mapro", "right_only": "operator", "both": "both"})
        .astype("string")
    )
    merged = merged.drop(columns="_merge")
    # Los eventos que solo están en MAPRO no traen dato de programada: ``pd.NA``,
    # no ``False``, porque no saberlo y saber que no lo era no son lo mismo.
    merged["scheduled"] = merged["scheduled"].astype("boolean")
    ordered = ["date", "source", "duration_hours", *OPERATOR_COLUMNS]
    return merged[ordered].sort_values("date").reset_index(drop=True)
