import json
import shutil
import zipfile

import pandas as pd
import pytest
from maintenance_events import dataset as ds


def _make_artifact(tmp_path):
    inner = tmp_path / "maintenance-events-v1"
    inner.mkdir()
    signals = pd.DataFrame(
        {"ambient_temp": [1.0, 2.0, 3.0]},
        index=pd.date_range("2024-01-01", periods=3, freq="15min", name="timestamp"),
    )
    signals.to_parquet(inner / "signals.parquet")
    events = pd.DataFrame({"date": ["2024-01-02"], "source": ["mapro"]})
    events.to_csv(inner / "events.csv", index=False)
    manifest = {
        "version": "v1",
        "sha256": {
            "signals.parquet": ds.sha256(inner / "signals.parquet"),
            "events.csv": ds.sha256(inner / "events.csv"),
        },
    }
    (inner / "manifest.json").write_text(json.dumps(manifest))
    archive = tmp_path / "maintenance-events-v1.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for name in ("signals.parquet", "events.csv", "manifest.json"):
            zf.write(inner / name, f"maintenance-events-v1/{name}")
    shutil.rmtree(inner)
    return archive


def test_missing_configuration_explains_what_to_do(tmp_path, monkeypatch):
    monkeypatch.delenv("MAINTENANCE_EVENTS_URL", raising=False)
    with pytest.raises(ds.DatasetNotConfigured, match=r"dataset\.local\.toml"):
        ds.resolve_source(project_dir=tmp_path)


def test_env_var_wins_over_file(tmp_path, monkeypatch):
    (tmp_path / "dataset.local.toml").write_text('url = "from-file"\n')
    monkeypatch.setenv("MAINTENANCE_EVENTS_URL", "from-env")
    assert ds.resolve_source(project_dir=tmp_path) == "from-env"


def test_local_archive_is_extracted_and_verified(tmp_path, monkeypatch):
    archive = _make_artifact(tmp_path)
    monkeypatch.setenv("MAINTENANCE_EVENTS_URL", str(archive))
    monkeypatch.setenv("MAINTENANCE_EVENTS_CACHE", str(tmp_path / "cache"))
    signals, events = ds.load()
    assert list(signals.columns) == ["ambient_temp"]
    assert events["date"].iloc[0] == pd.Timestamp("2024-01-02")


def test_second_load_does_not_refetch(tmp_path, monkeypatch):
    """El cache verificado tiene que evitar la descarga, no repetirla."""
    archive = _make_artifact(tmp_path)
    cache = tmp_path / "cache"
    monkeypatch.setenv("MAINTENANCE_EVENTS_URL", str(archive))
    monkeypatch.setenv("MAINTENANCE_EVENTS_CACHE", str(cache))
    ds.ensure_dataset()
    archive.unlink()  # si volviera a bajar, fallaría
    assert ds.ensure_dataset().name == "maintenance-events-v1"


def test_corrupt_file_is_rejected(tmp_path, monkeypatch):
    archive = _make_artifact(tmp_path)
    cache = tmp_path / "cache"
    monkeypatch.setenv("MAINTENANCE_EVENTS_URL", str(archive))
    monkeypatch.setenv("MAINTENANCE_EVENTS_CACHE", str(cache))
    ds.ensure_dataset()
    (cache / "maintenance-events-v1" / "events.csv").write_text("corrompido\n")
    with pytest.raises(ds.DatasetNotConfigured, match="sha256"):
        ds.ensure_dataset()
