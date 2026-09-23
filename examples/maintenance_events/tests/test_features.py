import numpy as np
import pandas as pd
import pytest
from maintenance_events.features import build_feature_table
from maintenance_events.windows import Window


def _history(hours=24 * 60, power=50.0, start="2024-01-01"):
    index = pd.date_range(start, periods=hours, freq="1h", name="timestamp")
    return pd.DataFrame(
        {
            "cw_inlet_temp": np.linspace(10.0, 20.0, hours),
            "cw_outlet_east_temp": np.linspace(25.0, 40.0, hours),
            "cw_outlet_west_temp": np.linspace(25.0, 40.0, hours),
            "condenser_vacuum_east": np.linspace(-0.9, -0.8, hours),
            "condenser_vacuum_west": np.linspace(-0.9, -0.8, hours),
            "ambient_temp": np.linspace(15.0, 30.0, hours),
            "active_power": np.full(hours, power),
        },
        index=index,
    )


def _window(history, label=0, days_since=3.0):
    as_of = history.index.max() + pd.Timedelta(hours=1)
    return Window(as_of=as_of, history=history, label=label, days_since_last_event=days_since)


def test_feature_table_is_numeric_and_aligned():
    windows = [
        _window(_history(), label=1),
        _window(_history(start="2024-03-01"), label=0),
    ]
    X, y, as_of = build_feature_table(windows)
    assert len(X) == len(y) == len(as_of) == 2
    assert y.tolist() == [1, 0]
    assert X.select_dtypes(exclude="number").empty


def test_physical_derivatives_are_present():
    X, _, _ = build_feature_table([_window(_history())])
    assert "delta_t_east_mean" in X.columns
    assert "vacuum_per_ambient_mean" in X.columns
    assert "days_since_last_event" in X.columns


def test_slope_captures_a_rising_trend():
    X, _, _ = build_feature_table([_window(_history())])
    assert X["cw_outlet_east_temp_slope"].iloc[0] > 0


def test_offline_rows_are_excluded_from_the_aggregates():
    history = _history()
    half = len(history) // 2
    history.iloc[:half, history.columns.get_loc("active_power")] = 0.0
    history.iloc[:half, history.columns.get_loc("ambient_temp")] = -999.0
    X, _, _ = build_feature_table([_window(history)], offline_power_threshold=1.0)
    assert X["ambient_temp_mean"].iloc[0] > 0


def test_window_with_too_few_valid_rows_is_dropped():
    history = _history()
    history["active_power"] = 0.0
    X, y, as_of = build_feature_table([_window(history)], offline_power_threshold=1.0)
    assert X.empty
    assert y.empty
    assert len(as_of) == 0


def test_sub_windows_produce_their_own_columns():
    X, _, _ = build_feature_table([_window(_history())])
    assert "ambient_temp_mean_7d" in X.columns
    assert "ambient_temp_mean_30d" in X.columns


def test_features_never_see_rows_at_or_after_as_of():
    """La sub-ventana se recorta sobre el historial, que ya es semiabierto."""
    history = _history()
    window = _window(history)
    X, _, _ = build_feature_table([window])
    assert history.index.max() < window.as_of
    assert np.isfinite(X["ambient_temp_max_7d"].iloc[0])


def test_vectorized_stats_match_the_pandas_reference():
    """La versión vectorizada existe solo por velocidad: los números no cambian."""
    from maintenance_events.features import _summarize

    rng = np.random.default_rng(0)
    index = pd.date_range("2024-01-01", periods=500, freq="1h", name="timestamp")
    frame = pd.DataFrame(
        {
            "a": rng.normal(size=500),
            "b": rng.normal(size=500),
            "c": np.full(500, np.nan),
        },
        index=index,
    )
    frame.iloc[10:40, 0] = np.nan  # patrón de faltantes distinto por columna

    got = _summarize(frame)
    for column in frame.columns:
        series = frame[column]
        assert got[f"{column}_mean"] == pytest.approx(series.mean(), nan_ok=True)
        assert got[f"{column}_std"] == pytest.approx(series.std(), nan_ok=True)
        assert got[f"{column}_min"] == pytest.approx(series.min(), nan_ok=True)
        assert got[f"{column}_max"] == pytest.approx(series.max(), nan_ok=True)
        assert got[f"{column}_p10"] == pytest.approx(series.quantile(0.10), nan_ok=True)
        assert got[f"{column}_p90"] == pytest.approx(series.quantile(0.90), nan_ok=True)

    # la pendiente, contra el ajuste de grado 1 de numpy sobre los válidos
    for column in ("a", "b"):
        clean = frame[column].dropna()
        hours = (clean.index - clean.index[0]).total_seconds().to_numpy() / 3600.0
        assert got[f"{column}_slope"] == pytest.approx(np.polyfit(hours, clean.to_numpy(), 1)[0])
    assert np.isnan(got["c_slope"])
