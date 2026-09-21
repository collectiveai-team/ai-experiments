from pathlib import Path

import pandas as pd

from prepare.events import merge_events, read_mapro_events

FIXTURE = Path(__file__).parent / "fixtures" / "eventos_sample.csv"


def test_only_cleaning_events_are_kept():
    df = read_mapro_events(FIXTURE)
    assert len(df) == 3
    assert df["date"].tolist() == [
        pd.Timestamp("2022-08-17"),
        pd.Timestamp("2022-12-21"),
        pd.Timestamp("2023-11-01"),
    ]


def test_sensitive_columns_are_dropped():
    df = read_mapro_events(FIXTURE)
    assert set(df.columns) == {"date", "duration_hours"}


def test_merge_marks_the_overlap_as_both():
    mapro = pd.DataFrame(
        {"date": [pd.Timestamp("2023-11-01"), pd.Timestamp("2022-08-17")], "duration_hours": [1.0, 3.0]}
    )
    operator = pd.DataFrame(
        {
            "date": [pd.Timestamp("2023-11-01"), pd.Timestamp("2024-06-02")],
            "buckets_left": [5.0, 2.0],
            "buckets_right": [3.0, 2.0],
            "scheduled": [True, False],
        }
    )
    merged = merge_events(mapro, operator)
    assert merged["date"].is_monotonic_increasing
    assert merged.set_index("date")["source"].to_dict() == {
        pd.Timestamp("2022-08-17"): "mapro",
        pd.Timestamp("2023-11-01"): "both",
        pd.Timestamp("2024-06-02"): "operator",
    }


def test_merge_keeps_one_row_per_date():
    mapro = pd.DataFrame({"date": [pd.Timestamp("2024-01-05")], "duration_hours": [2.0]})
    operator = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-05"), pd.Timestamp("2024-01-05")],
            "buckets_left": [1.0, 3.0],
            "buckets_right": [1.0, 2.0],
            "scheduled": [True, True],
        }
    )
    merged = merge_events(mapro, operator)
    assert len(merged) == 1
    assert merged["buckets_left"].iloc[0] == 4.0


def _write_operator_log(path):
    """Reproduce la forma real de la bitácora: dos filas de encabezado con celdas
    combinadas, ``Derecho`` rotulado en la columna 4 pero cargado en la 3."""
    import openpyxl

    book = openpyxl.Workbook()
    sheet = book.active
    sheet.append(["Limpió", "Fecha", "Cantidad de baldes", None, None, "Observaciones", None, "Mapro", "No programada"])
    sheet.append([None, None, "Izquierdo", None, "Derecho", None, None, "X", None])
    sheet.append(["Sarmiento/Liberal", "2023-11-01", 5, 3.0, None, None, None, "X", None])
    sheet.append(["Carmona/Viola", "2023-12-25", 4, 2.0, None, "Urfalino", None, None, "x"])
    sheet.append(["Stroscio", "2024-01-08", 2, 1.0, None, None, None, "x", None])
    book.save(path)
    return path


def test_operator_log_reads_buckets_from_the_loaded_columns(tmp_path):
    from prepare.events import read_operator_log

    df = read_operator_log(_write_operator_log(tmp_path / "log.xlsx"))
    assert len(df) == 3
    assert df["buckets_left"].tolist() == [5.0, 4.0, 2.0]
    assert df["buckets_right"].tolist() == [3.0, 2.0, 1.0]


def test_operator_log_scheduled_flag_is_case_insensitive(tmp_path):
    from prepare.events import read_operator_log

    df = read_operator_log(_write_operator_log(tmp_path / "log.xlsx"))
    assert df["scheduled"].tolist() == [True, False, True]


def test_operator_log_never_returns_names(tmp_path):
    from prepare.events import read_operator_log

    df = read_operator_log(_write_operator_log(tmp_path / "log.xlsx"))
    assert set(df.columns) == {"date", "buckets_left", "buckets_right", "scheduled"}


def test_merge_leaves_scheduled_unknown_when_only_mapro_saw_the_event():
    mapro = pd.DataFrame({"date": [pd.Timestamp("2022-08-17")], "duration_hours": [3.0]})
    operator = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-06-02")],
            "buckets_left": [2.0],
            "buckets_right": [2.0],
            "scheduled": [False],
        }
    )
    merged = merge_events(mapro, operator).set_index("date")
    assert merged["scheduled"].dtype == "boolean"
    assert pd.isna(merged.loc[pd.Timestamp("2022-08-17"), "scheduled"])
    assert merged.loc[pd.Timestamp("2024-06-02"), "scheduled"] is False or not merged.loc[pd.Timestamp("2024-06-02"), "scheduled"]
