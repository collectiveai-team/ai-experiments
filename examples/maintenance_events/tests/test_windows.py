import numpy as np
import pandas as pd
import pytest
from maintenance_events.windows import build_windows, select_events

FREQ = "1h"


def _signals(days=120):
    index = pd.date_range("2024-01-01", periods=days * 24, freq=FREQ, name="timestamp")
    return pd.DataFrame({"ambient_temp": np.arange(len(index), dtype=float)}, index=index)


def test_label_is_one_when_an_event_falls_on_the_horizon_edge():
    signals = _signals()
    as_of = pd.Timestamp("2024-02-15")
    windows = build_windows(signals, pd.Series([as_of + pd.Timedelta(days=15)]), window_days=30)
    hit = [w for w in windows if w.as_of == as_of]
    assert hit
    assert hit[0].label == 1


def test_label_is_zero_one_day_past_the_horizon():
    signals = _signals()
    as_of = pd.Timestamp("2024-02-15")
    windows = build_windows(signals, pd.Series([as_of + pd.Timedelta(days=16)]), window_days=30)
    hit = [w for w in windows if w.as_of == as_of]
    assert hit
    assert hit[0].label == 0


def test_an_event_exactly_at_as_of_does_not_label_its_own_window():
    signals = _signals()
    as_of = pd.Timestamp("2024-02-15")
    windows = build_windows(signals, pd.Series([as_of]), window_days=30)
    hit = [w for w in windows if w.as_of == as_of]
    assert hit
    assert hit[0].label == 0


def test_history_never_reaches_as_of():
    signals = _signals()
    windows = build_windows(signals, pd.Series([], dtype="datetime64[ns]"), window_days=30)
    for window in windows:
        assert window.history.index.max() < window.as_of
        assert window.history.index.min() >= window.as_of - pd.Timedelta(days=30)


def test_days_since_last_event_ignores_the_future():
    signals = _signals()
    events = pd.Series([pd.Timestamp("2024-02-10"), pd.Timestamp("2024-03-01")])
    windows = build_windows(signals, events, window_days=30)
    hit = next(w for w in windows if w.as_of == pd.Timestamp("2024-02-15"))
    assert hit.days_since_last_event == 5.0


def test_days_since_last_event_is_nan_before_any_event():
    signals = _signals()
    windows = build_windows(signals, pd.Series([pd.Timestamp("2024-04-01")]), window_days=30)
    assert np.isnan(windows[0].days_since_last_event)


def test_windows_start_after_a_full_history_and_stop_before_the_horizon():
    signals = _signals(days=120)
    windows = build_windows(signals, pd.Series([], dtype="datetime64[ns]"), window_days=30)
    assert windows[0].as_of == pd.Timestamp("2024-01-31")
    assert windows[-1].as_of <= signals.index.max().normalize() - pd.Timedelta(days=15)
    assert (windows[1].as_of - windows[0].as_of) == pd.Timedelta(days=1)


@pytest.mark.parametrize(
    ("source", "expected"),
    [("mapro", 2), ("operator", 2), ("union", 3)],
)
def test_select_events_filters_by_source(source, expected):
    events = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
            "source": ["mapro", "both", "operator"],
        }
    )
    assert len(select_events(events, source)) == expected


def test_select_events_rejects_an_unknown_source():
    events = pd.DataFrame({"date": pd.to_datetime(["2024-01-01"]), "source": ["mapro"]})
    with pytest.raises(ValueError, match="label_source inválido"):
        select_events(events, "todos")


def test_label_coverage_is_the_span_of_the_chosen_source():
    """Cada registro cubre un período distinto.

    Mapro 2022-08 a 2024-07, el log de operarios 2023-11 a 2024-12. Fuera de su período, 'no hay
    evento' significa 'no hay registro', no 'no pasó nada'.
    """
    from maintenance_events.windows import label_coverage

    events = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2022-08-17", "2024-07-13", "2023-11-01", "2024-12-09", "2024-01-01"]
            ),
            "source": ["mapro", "mapro", "operator", "operator", "both"],
        }
    )
    assert label_coverage(events, "mapro") == (
        pd.Timestamp("2022-08-17"),
        pd.Timestamp("2024-07-13"),
    )
    assert label_coverage(events, "operator") == (
        pd.Timestamp("2023-11-01"),
        pd.Timestamp("2024-12-09"),
    )
    assert label_coverage(events, "union") == (
        pd.Timestamp("2022-08-17"),
        pd.Timestamp("2024-12-09"),
    )


def test_windows_outside_the_label_coverage_are_dropped():
    """Cada fuente de etiquetas se compara sobre su propio período de cobertura.

    Sin esto, elegir --label-source operator agrega mil ventanas etiquetadas negativas por falta de
    registro, y la campaña compara fuentes sobre cantidades distintas de basura.
    """
    index = pd.date_range("2024-01-01", "2024-12-31", freq="1h", name="timestamp")
    signals = pd.DataFrame({"active_power": 100.0}, index=index)
    coverage = (pd.Timestamp("2024-06-01"), pd.Timestamp("2024-09-01"))
    event_dates = pd.Series(pd.to_datetime(["2024-07-10"]))

    windows = build_windows(signals, event_dates, window_days=30, coverage=coverage)

    as_of = [w.as_of for w in windows]
    assert min(as_of) >= coverage[0]
    # el horizonte completo tiene que caer dentro de la cobertura
    assert max(as_of) + pd.Timedelta(days=15) <= coverage[1]


def test_coverage_none_keeps_every_window():
    index = pd.date_range("2024-01-01", "2024-06-30", freq="1h", name="timestamp")
    signals = pd.DataFrame({"active_power": 100.0}, index=index)
    event_dates = pd.Series(pd.to_datetime(["2024-03-10"]))
    assert len(build_windows(signals, event_dates, window_days=30, coverage=None)) > 100
