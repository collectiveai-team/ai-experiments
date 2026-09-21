"""Construye el artefacto publicable a partir de los datos crudos de planta.

Corre offline, una sola vez, y produce el zip que se sube a Drive. Quien usa el
ejemplo nunca ejecuta este script: solo consume su salida.

    uv run --extra prepare python -m prepare.build_dataset \
        --raw-data ~/Projects/collectiveai/cepu-cc25/resources/data \
        --out dist/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from datetime import date
from pathlib import Path

import pandas as pd

from prepare.events import merge_events, read_mapro_events, read_operator_log
from prepare.signals import load_signals

MAPRO_FILE = "eventos condensador CC(in).csv"
OPERATOR_FILE = "Limpieza condensador.xlsx"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dataset_card(signals: pd.DataFrame, events: pd.DataFrame, version: str) -> str:
    coverage = (signals.notna().mean() * 100).round(1).to_dict()
    rows = "\n".join(f"  - `{name}`: {pct}% de cobertura" for name, pct in coverage.items())
    return f"""# maintenance-events {version}

Señales de proceso de un condensador de central térmica y el registro de sus
limpiezas. Publicado para servir de workload de referencia del arnés `iax`.

## Contenido

- `signals.parquet`: {len(signals):,} filas a 15 minutos,
  {signals.index.min():%Y-%m-%d} → {signals.index.max():%Y-%m-%d},
  {len(signals.columns)} señales.
{rows}
- `events.csv`: {len(events)} limpiezas,
  {events['date'].min():%Y-%m-%d} → {events['date'].max():%Y-%m-%d}.
  `source` indica de qué registro salió cada una:
  {events['source'].value_counts().to_dict()}.

Los huecos son `NaN` y no están rellenados. Un `?` en el export original marca
mala calidad declarada por el DCS y se leyó como ausencia, no como valor.

## Anonimización

Se eliminaron los identificadores de tag y de planta, los nombres de los
operarios, el texto libre de observaciones y motivos, y los montos. Las señales
llevan nombres funcionales en inglés.

**Timestamps y valores son reales.** Quien conozca el sector puede reconocer la
unidad por el patrón de paradas: este dataset es privado y el link no se comparte.

## Tarea

Dado el historial hasta el día `t`, predecir si hay una limpieza en
`(t, t + 15 días]`. El modelo corre una vez por día.
"""


def build(raw_data_dir: Path, out_dir: Path, version: str = "v1") -> Path:
    raw_data_dir = Path(raw_data_dir)
    dataset_dir = Path(out_dir) / f"maintenance-events-{version}"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    signals = load_signals(raw_data_dir / "raw" / "raw_signals")
    events_dir = raw_data_dir / "raw" / "cleaning-events"
    events = merge_events(
        read_mapro_events(events_dir / MAPRO_FILE),
        read_operator_log(events_dir / OPERATOR_FILE),
    )

    signals_path = dataset_dir / "signals.parquet"
    events_path = dataset_dir / "events.csv"
    signals.to_parquet(signals_path, index=True)
    events.to_csv(events_path, index=False)

    manifest = {
        "name": "maintenance-events",
        "version": version,
        "built_on": date.today().isoformat(),
        "sha256": {p.name: _sha256(p) for p in (signals_path, events_path)},
        "signals": {
            "columns": list(signals.columns),
            "freq": "15min",
            "start": signals.index.min().isoformat(),
            "end": signals.index.max().isoformat(),
            "coverage": {c: round(float(signals[c].notna().mean()), 4) for c in signals.columns},
        },
        "counts": {
            "signal_rows": int(len(signals)),
            "events": int(len(events)),
            "events_by_source": {str(k): int(v) for k, v in events["source"].value_counts().items()},
        },
    }
    (dataset_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    (dataset_dir / "dataset_card.md").write_text(_dataset_card(signals, events, version))

    archive = Path(out_dir) / f"maintenance-events-{version}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in ("signals.parquet", "events.csv", "manifest.json", "dataset_card.md"):
            zf.write(dataset_dir / name, f"maintenance-events-{version}/{name}")
    return archive


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-data", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("dist"))
    parser.add_argument("--version", default="v1")
    args = parser.parse_args()
    archive = build(args.raw_data, args.out, args.version)
    print(f"escrito: {archive}")


if __name__ == "__main__":
    main()
