"""Workload-side metric and result reporting.

Workloads import this module (or just print the line themselves) to report
training metrics and evaluation results to the harness::

    from ai_experiments.report import report_metric, report_result

    report_metric(step=epoch, loss=loss, val_loss=val_loss)
    report_result(test_acc=0.95)

Two channels carry data: the metric channel ``IAX_METRIC {json}`` carries
progress points (step + values) during training; the result channel
``IAX_RESULT {json}`` carries the final evaluation result (values only,
no step). Only the result channel is scored.

Both channels work identically for the local backend (the worker tails stdout)
and for Ray clusters with no shared filesystem (the harness extracts the lines
from the job logs). Workloads need no other dependency on the harness;
printing the lines directly is a supported contract.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any

METRIC_PREFIX = "IAX_METRIC "
RESULT_PREFIX = "IAX_RESULT "


def _coerce_numeric_values(payload: dict) -> dict[str, float]:
    """Extract numeric values from a payload dict, coercing types and handling non-finite values.

    Skips boolean values, converts int/float to float, and parses string
    representations of non-finite floats (nan, inf, -inf, infinity, -infinity).
    """
    values: dict[str, float] = {}
    for key, value in payload.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            values[str(key)] = float(value)
        elif isinstance(value, str):
            lowered = value.lower()
            if lowered in {"nan", "inf", "-inf", "infinity", "-infinity"}:
                values[str(key)] = float(lowered.replace("infinity", "inf"))
    return values


def artifacts_dir() -> Path | None:
    """Directory where the workload should write checkpoints/plots/models.

    Set by the harness (``IAX_ARTIFACTS_DIR``); files written here are listed
    by ``iax artifacts <run_id>`` and downloadable from the dashboard. Returns
    None when running outside the harness (local backend sets it; on remote
    Ray clusters artifacts stay on the cluster's own storage).
    """
    value = os.environ.get("IAX_ARTIFACTS_DIR")
    if not value:
        return None
    path = Path(value)
    path.mkdir(parents=True, exist_ok=True)
    return path


def report_metric(step: int | None = None, **values: float) -> None:
    """Print one metrics observation for the harness to collect."""
    payload: dict[str, Any] = dict(values)
    if step is not None:
        payload["step"] = step
    sys.stdout.write(METRIC_PREFIX + json.dumps(payload) + "\n")
    sys.stdout.flush()


def report_result(**values: float) -> None:
    """Print the one evaluation result the harness will score.

    Only the ``evaluate`` phase may call this. Progress belongs in
    :func:`report_metric`, which never scores.
    """
    sys.stdout.write(RESULT_PREFIX + json.dumps(dict(values)) + "\n")
    sys.stdout.flush()


def parse_metric_line(line: str) -> dict[str, Any] | None:
    """Parse an ``IAX_METRIC {...}`` stdout line into step + numeric values.

    Returns ``{"step": int | None, "values": {name: float}}`` or None when the
    line is not a metric line or carries no usable values. Non-finite floats
    (nan/inf) are preserved — detecting them is the monitor's job.
    """
    stripped = line.strip()
    idx = stripped.find(METRIC_PREFIX.strip())
    if idx == -1:
        return None
    raw = stripped[idx + len(METRIC_PREFIX.strip()) :].strip()
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None

    step: int | None = None
    step_raw = payload.pop("step", None)
    if isinstance(step_raw, (int, float)) and not isinstance(step_raw, bool):
        step = int(step_raw)

    values = _coerce_numeric_values(payload)
    if not values and step is None:
        return None
    return {"step": step, "values": values}


def parse_result_line(line: str) -> dict[str, float] | None:
    """Parse an ``IAX_RESULT {...}`` line into numeric values, or None.

    A result has no step: it is the answer, not a point on a curve.
    """
    stripped = line.strip()
    idx = stripped.find(RESULT_PREFIX.strip())
    if idx == -1:
        return None
    raw = stripped[idx + len(RESULT_PREFIX.strip()) :].strip()
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    payload.pop("step", None)

    values = _coerce_numeric_values(payload)
    return values or None


def is_finite(value: float) -> bool:
    return math.isfinite(value)
