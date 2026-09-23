import json
import zipfile

import pandas as pd
from prepare.build_dataset import build
from prepare.signals import TAG_RENAMES, load_signals


def _write_sppa(path, tags, start, rows):
    header = [
        ";;;;SPPA-T3000;;;;;;;;;;;;;;;;",
        "Reporte de Intervalo Analógico;;;;;;;;;;;;;;;;;;;;",
    ]
    for i, tag in enumerate(tags, start=1):
        header.append(f";;Tag{i};{tag.replace('__', '||')};;;;DESC;;;u;;;;;;;0,0;;;1,0")
    # Siete ';' después de 'Tiempo' dejan el primer Tag en la posición 8,
    # igual que en los exports reales; las filas de datos usan el mismo offset.
    header.append(";Tiempo;;;;;;;" + ";".join(f"Tag{i}" for i in range(1, len(tags) + 1)) + ";")
    lines = list(header)
    ts = pd.Timestamp(start)
    for r in range(rows):
        left = ts.strftime("%Y/%m/%d %H:%M:%S,000")
        right = (ts + pd.Timedelta(minutes=15)).strftime("%Y/%m/%d %H:%M:%S,000")
        values = ";".join(f"{r + i},5" for i in range(len(tags)))
        lines.append(f";{left} - {right};;;;;;;{values};;;;;;;;;;;")
        ts += pd.Timedelta(minutes=15)
    path.write_text("\n".join(lines), encoding="latin-1")


def _write_events(events_dir, operator_name="alguien"):
    (events_dir / "eventos condensador CC(in).csv").write_text(
        "cod_tipo_evento;Suma de duracion_hs;fecha;Suma de hora_inicio;nemo_grupo;"
        "Suma de periodo;Suma de valorizacion_ars;motivo\n"
        "Lim;3;02/10/2023 0:00;2;LDCUTV15;202310;5998;Limpieza\n",
        encoding="latin-1",
    )
    pd.DataFrame(
        [[operator_name, pd.Timestamp("2023-10-01"), 5, 3, None, None, None, "X", None]]
    ).to_excel(events_dir / "Limpieza condensador.xlsx", index=False, header=False)


def _make_raw(tmp_path, rows=96, operator_name="alguien"):
    raw_data = tmp_path / "data"
    signals_dir = raw_data / "raw" / "raw_signals"
    events_dir = raw_data / "raw" / "cleaning-events"
    signals_dir.mkdir(parents=True)
    events_dir.mkdir(parents=True)
    _write_sppa(signals_dir / "2023Q4GRUPO1.csv", ["13MAC01CP001__XQ01"], "2023-10-01", rows)
    _write_events(events_dir, operator_name)
    return raw_data


def test_load_signals_renames_and_concatenates(tmp_path):
    raw = tmp_path / "raw_signals"
    raw.mkdir()
    _write_sppa(raw / "2023Q4GRUPO1.csv", ["13MAC01CP001__XQ01"], "2023-10-01", 4)
    _write_sppa(raw / "2023Q4GRUPO2.csv", ["25MBL11CT002__XQ01"], "2023-10-01", 4)
    signals = load_signals(raw)
    assert list(signals.columns) == list(TAG_RENAMES.values())
    assert signals["lp_steam_vacuum"].notna().sum() == 4
    assert signals["ambient_temp"].notna().sum() == 4
    assert signals["makeup_pct"].isna().all()


def test_grupo4_fallback_fills_the_main_signal(tmp_path):
    """Las dos de vacío vienen casi vacías en su grupo: GRUPO4 es la fuente real."""
    raw = tmp_path / "raw_signals"
    raw.mkdir()
    _write_sppa(raw / "2023Q4GRUPO2.csv", ["13PAB10CP001__XQ01"], "2023-10-01", 4)
    _write_sppa(raw / "2023Q4GRUPO4.csv", ["13PAB10CP001__XQ01__OUT"], "2023-10-01", 4)
    signals = load_signals(raw)
    assert signals["condenser_vacuum_east"].notna().sum() == 4
    assert "13PAB10CP001__XQ01__OUT" not in signals.columns


def test_grupo4_fallback_is_used_when_the_main_group_is_absent(tmp_path):
    raw = tmp_path / "raw_signals"
    raw.mkdir()
    _write_sppa(raw / "2023Q4GRUPO4.csv", ["25MBL11CT002__209442__OUT"], "2023-10-01", 4)
    signals = load_signals(raw)
    assert signals["ambient_temp"].notna().sum() == 4


def test_build_writes_a_verifiable_artifact(tmp_path):
    raw_data = _make_raw(tmp_path)
    out = tmp_path / "out"
    archive = build(raw_data, out, version="v1")

    assert archive.exists()
    with zipfile.ZipFile(archive) as zf:
        names = set(zf.namelist())
    assert "maintenance-events-v1/signals.parquet" in names
    assert "maintenance-events-v1/events.csv" in names

    manifest = json.loads((out / "maintenance-events-v1" / "manifest.json").read_text())
    assert manifest["version"] == "v1"
    assert set(manifest["sha256"]) == {"signals.parquet", "events.csv"}
    assert manifest["counts"]["events"] == 2


def test_build_never_leaks_operator_names(tmp_path):
    raw_data = _make_raw(tmp_path, operator_name="Sarmiento/Liberal")
    out = tmp_path / "out"
    build(raw_data, out, version="v1")
    events = pd.read_csv(out / "maintenance-events-v1" / "events.csv")
    assert set(events.columns) == {
        "date",
        "source",
        "duration_hours",
        "buckets_left",
        "buckets_right",
        "scheduled",
    }
    blob = (out / "maintenance-events-v1" / "events.csv").read_text()
    assert "Sarmiento" not in blob
    assert "LDCUTV15" not in blob
    assert "5998" not in blob


def test_build_is_deterministic(tmp_path):
    raw_data = _make_raw(tmp_path, operator_name="alguien")
    first = json.loads(
        (
            build(raw_data, tmp_path / "a").parent / "maintenance-events-v1" / "manifest.json"
        ).read_text()
    )
    second = json.loads(
        (
            build(raw_data, tmp_path / "b").parent / "maintenance-events-v1" / "manifest.json"
        ).read_text()
    )
    assert first["sha256"] == second["sha256"]
