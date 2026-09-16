from __future__ import annotations

import json

from ai_experiments.report import parse_metric_line, parse_result_line, report_result


def test_parses_step_and_values():
    parsed = parse_metric_line('IAX_METRIC {"step": 3, "loss": 0.5, "acc": 0.9}')

    assert parsed == {"step": 3, "values": {"loss": 0.5, "acc": 0.9}}


def test_parses_without_step():
    parsed = parse_metric_line('IAX_METRIC {"loss": 1.25}')

    assert parsed == {"step": None, "values": {"loss": 1.25}}


def test_ignores_non_metric_lines():
    assert parse_metric_line("epoch 3: loss=0.5") is None
    assert parse_metric_line("IAX_METRIC not-json") is None
    assert parse_metric_line('IAX_METRIC ["not", "a", "dict"]') is None


def test_handles_prefixed_output():
    parsed = parse_metric_line('[worker-1] IAX_METRIC {"step": 1, "loss": 2.0}')

    assert parsed is not None
    assert parsed["values"] == {"loss": 2.0}


def test_preserves_non_finite_values():
    line = "IAX_METRIC " + json.dumps({"step": 9, "loss": float("nan")})
    parsed = parse_metric_line(line)

    assert parsed is not None
    assert parsed["values"]["loss"] != parsed["values"]["loss"]  # NaN


def test_skips_non_numeric_values():
    parsed = parse_metric_line('IAX_METRIC {"step": 1, "phase": "warmup", "loss": 0.1}')

    assert parsed is not None
    assert parsed["values"] == {"loss": 0.1}


def test_parses_a_result_line():
    parsed = parse_result_line('IAX_RESULT {"test_acc": 0.91}')

    assert parsed == {"test_acc": 0.91}


def test_a_metric_line_is_not_a_result():
    assert parse_result_line('IAX_METRIC {"loss": 0.5}') is None


def test_a_result_line_is_not_a_metric():
    assert parse_metric_line('IAX_RESULT {"test_acc": 0.91}') is None


def test_a_result_ignores_step_because_it_has_no_progress():
    parsed = parse_result_line('IAX_RESULT {"step": 7, "test_acc": 0.91}')

    assert parsed == {"test_acc": 0.91}


def test_a_result_with_no_numeric_values_is_not_a_result():
    assert parse_result_line('IAX_RESULT {"note": "done"}') is None


def test_report_result_prints_the_contract_line(capsys):
    report_result(test_acc=0.91)

    assert capsys.readouterr().out.strip() == 'IAX_RESULT {"test_acc": 0.91}'


def test_handles_prefixed_result_output():
    parsed = parse_result_line('[worker-1] IAX_RESULT {"test_acc": 0.91}')

    assert parsed is not None
    assert parsed == {"test_acc": 0.91}
