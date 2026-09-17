"""The house logger is the only logging entry point (CES-74, CES-45).

Two notes on how these tests probe the (unmodified) snippet's behaviour:

1. ``get_logger()`` is configured with ``wrapper_class=structlog.make_filtering_bound_logger``
   (the canonical, unmodified CES-74 snippet), which never returns a
   ``structlog.stdlib.BoundLogger`` -- ``get_logger()`` itself always returns the lazy
   ``BoundLoggerLazyProxy``, and even after ``.bind()`` the actual bound logger is a
   ``structlog._native`` filtering logger, a sibling class to ``structlog.stdlib.BoundLogger``,
   never a subclass. A bare ``isinstance(..., structlog.typing.BindableLogger)`` check would pass
   even for the *unconfigured* proxy and prove nothing about the house configuration;
   ``isinstance(..., structlog.typing.FilteringBoundLogger)`` is also unusable here (it returns
   False for this object). The test below instead binds first and asserts on
   ``type(...).__module__``, which only the configured filtering-bound-logger stack produces.
2. This module's tests read through ``caplog`` rather than ``capsys``. stdout is the wrong seam:
   ``logging.basicConfig()`` is a documented no-op once the root logger already has a handler, and
   pytest's own log-capture plugin always installs one on root for *every* test (whether or not
   that test requests ``caplog``) -- confirmed by inspecting ``logging.root.handlers`` from inside a
   test. Forcing reconfiguration past pytest's handler (``force=True``) was tried and reverted: it
   is a production behaviour change (tearing out *any* handler already on the root logger) made only
   to satisfy a test-capture requirement, and the very next consumer of this library -- any host
   application that has already configured its own logging -- would have its handlers silently
   removed on the first ``get_logger()`` call. ``caplog`` reads the log records the structlog
   processor chain actually produced, which is what these tests are about, and needs no such
   snippet change.

The autouse fixture below still resets the module's own "configured once" cache before/after each
test -- a module-level lazy singleton genuinely needs this given CES-111's test-order
randomization, independent of the capsys/caplog question above.
"""

import pytest

from ai_experiments.core import logger as logger_module
from ai_experiments.core.logger import get_logger


@pytest.fixture(autouse=True)
def _reset_logger_configuration():
    logger_module._configured = False
    yield
    logger_module._configured = False


def test_get_logger_returns_a_bound_structlog_logger():
    # No public seam distinguishes the configured filtering logger from an unconfigured
    # BoundLoggerLazyProxy: `isinstance(log, BindableLogger)` holds for both, and
    # `FilteringBoundLogger` matches neither. Asserting on structlog's private module path is
    # the only discriminating check available, so a structlog release that restructures
    # `_native` will break this test without the logger itself being wrong.
    log = get_logger("ai_experiments.daemon").bind()
    assert type(log).__module__ == "structlog._native"


def test_events_are_emitted_as_key_value_pairs(caplog):
    with caplog.at_level("INFO"):
        get_logger("ai_experiments.daemon").info("run_finished", run_id="abc123")
    assert "run_finished" in caplog.text
    assert "abc123" in caplog.text


def test_bind_carries_context_into_the_event(caplog):
    with caplog.at_level("INFO"):
        get_logger("ai_experiments.daemon").bind(campaign_id="c1").info("tick")
    assert "c1" in caplog.text
