"""Resolución, descarga, verificación y carga del dataset maintenance-events.

El link de Drive es privado: sale de ``MAINTENANCE_EVENTS_URL`` o de
``dataset.local.toml``, nunca del repositorio.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import zipfile
from pathlib import Path

import pandas as pd

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CACHE = Path.home() / ".cache" / "maintenance-events"
ENV_URL = "MAINTENANCE_EVENTS_URL"
ENV_CACHE = "MAINTENANCE_EVENTS_CACHE"
# Solo para tests: fuerza dónde se busca dataset.local.toml, para que un link
# configurado en la máquina del desarrollador no cambie el resultado.
ENV_PROJECT_DIR = "MAINTENANCE_EVENTS_PROJECT_DIR"
FILES = ("signals.parquet", "events.csv", "manifest.json")

_INSTRUCTIONS = f"""No hay dataset configurado.

Elegí una:
  1. export {ENV_URL}=<link de Drive o ruta a maintenance-events-v1.zip>
  2. copiá dataset.example.toml a dataset.local.toml y completá `url` o `local_path`

Para correr sin dataset: `python -m maintenance_events.train --self-test`."""


class DatasetNotConfigured(Exception):
    """No se puede resolver o verificar el dataset."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cache_dir() -> Path:
    return Path(os.environ.get(ENV_CACHE, DEFAULT_CACHE))


def resolve_source(project_dir: Path | None = None) -> str:
    from_env = os.environ.get(ENV_URL)
    if from_env:
        return from_env
    root = project_dir or Path(os.environ.get(ENV_PROJECT_DIR) or PROJECT_DIR)
    config = root / "dataset.local.toml"
    if config.exists():
        data = tomllib.loads(config.read_text())
        source = data.get("url") or data.get("local_path")
        if source:
            return str(source)
    raise DatasetNotConfigured(_INSTRUCTIONS)


def _verify(dataset_dir: Path) -> None:
    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.exists():
        raise DatasetNotConfigured(f"{dataset_dir}: falta manifest.json")
    expected = json.loads(manifest_path.read_text())["sha256"]
    for name, digest in expected.items():
        actual = sha256(dataset_dir / name)
        if actual != digest:
            raise DatasetNotConfigured(
                f"{dataset_dir / name}: sha256 no coincide con el manifest "
                f"(esperado {digest[:12]}…, obtenido {actual[:12]}…). "
                "Borrá el cache y volvé a bajarlo."
            )


def _fetch(source: str, archive: Path) -> None:
    local = Path(source)
    if local.exists():
        shutil.copy2(local, archive)
        return
    import gdown

    if gdown.download(url=source, output=str(archive), quiet=False, fuzzy=True) is None:
        raise DatasetNotConfigured(f"no se pudo bajar el dataset desde {source}")


def ensure_dataset(version: str = "v1") -> Path:
    root = cache_dir()
    dataset_dir = root / f"maintenance-events-{version}"
    if all((dataset_dir / name).exists() for name in FILES):
        _verify(dataset_dir)
        return dataset_dir

    root.mkdir(parents=True, exist_ok=True)
    archive = root / f"maintenance-events-{version}.zip"
    _fetch(resolve_source(), archive)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(root)
    _verify(dataset_dir)
    return dataset_dir


def load(version: str = "v1") -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset_dir = ensure_dataset(version)
    signals = pd.read_parquet(dataset_dir / "signals.parquet")
    signals.index = pd.DatetimeIndex(signals.index, name="timestamp")
    events = pd.read_csv(dataset_dir / "events.csv", parse_dates=["date"])
    return signals.sort_index(), events.sort_values("date").reset_index(drop=True)
