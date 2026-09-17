"""The house logger is the only logging entry point (CES-74, CES-45).

Two deviations from the brief's literal text, both forced by real structlog/stdlib-logging
behaviour rather than by choice -- see the task-7 report for the full repro:

1. ``get_logger()`` is configured with ``wrapper_class=structlog.make_filtering_bound_logger``
   (the canonical, unmodified CES-74 snippet), which returns a ``structlog._native`` filtering
   logger -- never a ``structlog.stdlib.BoundLogger``. The isinstance check below targets
   ``structlog.typing.BindableLogger``, the runtime-checkable protocol that actually describes
   "a bound structlog logger", instead of the concrete stdlib class the original text named.
2. ``logging.basicConfig()`` is a documented no-op once the root logger already has a handler
   -- and pytest's own log-capture plugin always installs one, for every test, whether or not
   that test requests ``caplog``. Left as the snippet has it (no ``force=True``), the very first
   ``get_logger()`` call in the whole process finds pytest's handler already there, silently
   skips configuring its own, and every test that binds sys.stdout per-test (``capsys``) reads
   back an empty capture. ``force=True`` is the third upstream snippet defect (see the report);
   the autouse fixture below only resets this module's own "configured once" cache so the
   (now-forced) reconfiguration reruns for every test, independent of execution order (CES-111
   randomizes it).
"""

import pytest
import structlog

from ai_experiments.core import logger as logger_module
from ai_experiments.core.logger import get_logger


@pytest.fixture(autouse=True)
def _reset_logger_configuration():
    logger_module._configured = False
    yield
    logger_module._configured = False


def test_get_logger_returns_a_bound_structlog_logger():
    log = get_logger("ai_experiments.daemon")
    assert isinstance(log, structlog.typing.BindableLogger)


def test_events_are_emitted_as_key_value_pairs(capsys):
    get_logger("ai_experiments.daemon").info("run_finished", run_id="abc123")
    captured = capsys.readouterr()
    assert "run_finished" in captured.out
    assert "abc123" in captured.out


def test_bind_carries_context_into_the_event(capsys):
    get_logger("ai_experiments.daemon").bind(campaign_id="c1").info("tick")
    assert "c1" in capsys.readouterr().out
