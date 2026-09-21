from pathlib import Path

import pandas as pd
import pytest

from prepare.sppa import SppaParseError, parse_sppa_file

FIXTURE = Path(__file__).parent / "fixtures" / "sppa_sample.csv"


def test_columns_use_original_tag_names():
    df = parse_sppa_file(FIXTURE)
    assert list(df.columns) == ["13MAC01CP001__XQ01", "13PAB10CT001__XQ01"]


def test_timestamp_is_interval_start():
    df = parse_sppa_file(FIXTURE)
    assert df.index[0] == pd.Timestamp("2023-10-01 00:00:00")
    assert df.index[1] == pd.Timestamp("2023-10-01 00:15:00")


def test_decimal_comma_is_parsed():
    df = parse_sppa_file(FIXTURE)
    assert df["13MAC01CP001__XQ01"].iloc[0] == pytest.approx(-0.8477378)
    assert df["13PAB10CT001__XQ01"].iloc[0] == pytest.approx(16.40001)


def test_missing_value_becomes_nan():
    df = parse_sppa_file(FIXTURE)
    assert pd.isna(df["13PAB10CT001__XQ01"].iloc[1])


def test_file_without_tag_header_raises(tmp_path):
    broken = tmp_path / "broken.csv"
    broken.write_text(";;;;SPPA-T3000;;;;\n;Tiempo;;;;;;Tag1;\n", encoding="latin-1")
    with pytest.raises(SppaParseError, match="ninguna definición de tag"):
        parse_sppa_file(broken)


def test_quality_flagged_value_becomes_nan():
    """``?`` antepuesto es el flag de mala calidad del DCS: ausencia, no valor."""
    df = parse_sppa_file(FIXTURE)
    assert pd.isna(df["13PAB10CT001__XQ01"].iloc[2])


FIXTURE_EN = Path(__file__).parent / "fixtures" / "sppa_sample_en.csv"


def test_english_export_columns_use_original_tag_names():
    df = parse_sppa_file(FIXTURE_EN)
    assert list(df.columns) == ["13PAB10CP001__XQ01__OUT", "25MBL11CT002__209442__OUT"]


def test_english_export_timestamp_is_interval_start():
    df = parse_sppa_file(FIXTURE_EN)
    assert df.index[0] == pd.Timestamp("2020-01-01 00:00:00")
    assert df.index[2] == pd.Timestamp("2020-01-01 00:30:00")


def test_english_export_uses_decimal_point():
    df = parse_sppa_file(FIXTURE_EN)
    assert df["13PAB10CP001__XQ01__OUT"].iloc[1] == pytest.approx(1.25)
    assert df["13PAB10CP001__XQ01__OUT"].iloc[2] == pytest.approx(-0.5)
    assert df["25MBL11CT002__209442__OUT"].iloc[0] == pytest.approx(26.886179)


def test_english_export_quality_flag_becomes_nan():
    df = parse_sppa_file(FIXTURE_EN)
    assert pd.isna(df["13PAB10CP001__XQ01__OUT"].iloc[0])
    assert pd.isna(df["25MBL11CT002__209442__OUT"].iloc[1])


def test_file_without_column_row_raises(tmp_path):
    broken = tmp_path / "broken.csv"
    broken.write_text(
        ";;;;SPPA-T3000;;;;\n;;Tag1;13MAC01CP001||XQ01;;;;\n", encoding="latin-1"
    )
    with pytest.raises(SppaParseError, match="no se encontró la fila de columnas"):
        parse_sppa_file(broken)


def test_unparseable_number_raises_with_file_and_line(tmp_path):
    """Un separador inesperado tiene que romper ruidosamente, no mangling silencioso."""
    broken = tmp_path / "broken.csv"
    broken.write_text(
        ";;;;SPPA-T3000;;;;\n"
        ";;Tag1;13MAC01CP001||XQ01;;;;\n"
        ";Tiempo;;;;;;;Tag1;\n"
        ";2023/10/01 00:00:00,000 - 2023/10/01 00:15:00,000;;;;;;;1.234,5;\n",
        encoding="latin-1",
    )
    with pytest.raises(SppaParseError, match=r"broken\.csv:4: fila ilegible"):
        parse_sppa_file(broken)
