"""Parser de los exports de intervalo analógico de SPPA-T3000.

Formato: separador ``;``, codificación latin-1, una cabecera que declara
``TagN -> <tag real>`` y filas cuya primera columna es un intervalo
``AAAA/MM/DD HH:MM:SS,mmm - AAAA/MM/DD HH:MM:SS,mmm``. Se toma el inicio del
intervalo como timestamp.

El corpus real trae **dos locales de export**, correlacionados en todo:

===========  ==================  ==================  ====================
locale       fila de columnas    separador decimal   milisegundos
===========  ==================  ==================  ====================
castellano   ``;Tiempo;``        coma                ``HH:MM:SS,mmm``
inglés       ``;Time;``          punto               ``HH:MM:SS.mmm``
===========  ==================  ==================  ====================

La palabra de la fila de columnas decide el separador decimal del archivo. No
hay separador de miles en ningún archivo del corpus, así que el parser es
estricto: cualquier otra forma numérica rompe con archivo y línea en vez de
mutilar el valor en silencio.

Un ``?`` antepuesto a un valor es el flag de mala calidad del DCS (se
corresponde con ``QF 0.0`` en la cabecera). Se lee como ``NaN``: ausencia
explícita, que es lo que el remuestreo aguas abajo ya sabe manejar.

Las columnas de datos **no son contiguas**: la fila ``;Tiempo;...;Tag1;Tag2;;;Tag3``
es la única fuente de verdad sobre en qué posición cae cada tag.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

TAG_ROW = re.compile(r"^;;Tag(\d+);([^;]+);")
COLUMN_ROW = re.compile(r"^;(Tiempo|Time);")
TAG_CELL = re.compile(r"^Tag(\d+)$")
INTERVAL = re.compile(r"^;(\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2})[.,](\d{3}) - ")
ENCODING = "latin-1"

#: Palabra de la fila de columnas -> separador decimal de los datos.
DECIMAL_BY_KEYWORD = {"Tiempo": ",", "Time": "."}


class SppaParseError(Exception):
    """El archivo no tiene la forma esperada de un export SPPA."""


def _normalize_tag(raw: str) -> str:
    """``13MAC01CP001||XQ01`` -> ``13MAC01CP001__XQ01``."""
    return raw.strip().replace("||", "__").replace("|", "__")


def _to_float(cell: str, decimal: str) -> float:
    cell = cell.strip()
    if cell.startswith("?"):  # flag de mala calidad del DCS
        return float("nan")
    if not cell:
        return float("nan")
    if decimal == ",":
        cell = cell.replace(",", ".")
    return float(cell)


def parse_sppa_file(path: Path) -> pd.DataFrame:
    lines = Path(path).read_text(encoding=ENCODING).splitlines()

    tags: dict[int, str] = {}
    for line in lines:
        match = TAG_ROW.match(line)
        if match:
            tags[int(match.group(1))] = _normalize_tag(match.group(2))
    if not tags:
        raise SppaParseError(f"{path}: ninguna definición de tag en la cabecera")

    columns: dict[int, int] = {}
    decimal = ""
    for line in lines:
        header = COLUMN_ROW.match(line)
        if not header:
            continue
        for position, cell in enumerate(line.split(";")):
            cell_match = TAG_CELL.match(cell.strip())
            if cell_match:
                columns[position] = int(cell_match.group(1))
        if columns:
            decimal = DECIMAL_BY_KEYWORD[header.group(1)]
            break
    if not columns:
        raise SppaParseError(
            f"{path}: no se encontró la fila de columnas ';Tiempo;...;TagN;'"
        )

    unknown = sorted(set(columns.values()) - set(tags))
    if unknown:
        raise SppaParseError(
            f"{path}: la fila de columnas nombra Tag{unknown} sin definición en la cabecera"
        )

    order = sorted(columns.items())
    names = [tags[number] for _, number in order]
    timestamps: list[pd.Timestamp] = []
    rows: list[list[float]] = []
    for lineno, line in enumerate(lines, start=1):
        match = INTERVAL.match(line)
        if not match:
            continue
        cells = line.split(";")
        try:
            values = [_to_float(cells[position], decimal) for position, _ in order]
        except (IndexError, ValueError) as exc:
            raise SppaParseError(f"{path}:{lineno}: fila ilegible: {exc}") from exc
        timestamps.append(pd.Timestamp(f"{match.group(1)}.{match.group(2)}"))
        rows.append(values)

    if not rows:
        raise SppaParseError(f"{path}: ninguna fila de datos")

    frame = pd.DataFrame(
        rows, columns=names, index=pd.DatetimeIndex(timestamps, name="timestamp")
    )
    return frame.sort_index()
