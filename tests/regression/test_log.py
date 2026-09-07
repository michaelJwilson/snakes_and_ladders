"""The run logger against a fake clock (issue #311).

Every record must carry the elapsed minutes and the shared phase, a
once-only message must be emitted once, a second configuration must not
double the handlers, and nothing may reach the root logger. The clock is
injected, so the format is pinned exactly rather than approximately after a
sleep.
"""

from __future__ import annotations

import io
import logging

import pytest
from snakes_and_ladders.log import RunLogger, get_logger, phase


def _logger(name: str, stream: io.StringIO, ticks: list[float]) -> RunLogger:
    times = iter(ticks)
    return get_logger(name, start_time=0.0, stream=stream, clock=lambda: next(times))


@pytest.mark.structural
def test_every_line_carries_the_elapsed_minutes_the_phase_and_the_call_site() -> None:
    stream = io.StringIO()
    logger = _logger("sal.test.format", stream, [90.0, 150.0])
    logger.info("outside any phase")
    with phase("render"):
        logger.warning("inside %s", "one")
    first, second = stream.getvalue().splitlines()
    assert " - 1.50m - INFO - sal.test.format." in first
    assert first.endswith(" - outside any phase")
    assert " - 2.50m - WARNING (render) - " in second
    assert (
        "test_every_line_carries_the_elapsed_minutes_the_phase_and_the_call_site:"
        in second
    )
    assert second.endswith(" - inside one")


@pytest.mark.structural
def test_the_phase_is_shared_across_loggers_and_restored_on_exit() -> None:
    first_stream, second_stream = io.StringIO(), io.StringIO()
    first = _logger("sal.test.shared.a", first_stream, [0.0] * 4)
    second = _logger("sal.test.shared.b", second_stream, [0.0] * 4)
    with phase("outer"):
        assert first.runtime_phase == "outer"
        with phase("inner"):
            second.info("deep")
        first.info("back out")
    second.info("after")
    assert "(inner)" in second_stream.getvalue().splitlines()[0]
    assert "(outer)" in first_stream.getvalue()
    assert "(" not in second_stream.getvalue().splitlines()[1].split(" - ")[2]
    assert first.runtime_phase is None


@pytest.mark.edge_case
def test_once_only_messages_are_emitted_once_per_message() -> None:
    stream = io.StringIO()
    logger = _logger("sal.test.once", stream, [0.0] * 10)
    for _ in range(3):
        logger.warning_once("same thing")
    logger.warning_once("a different thing")
    logger.info_once("same thing")
    lines = stream.getvalue().splitlines()
    assert len(lines) == 3
    assert lines[0].endswith("same thing")
    assert lines[1].endswith("a different thing")
    assert " - INFO - " in lines[2]


@pytest.mark.structural
def test_reconfiguring_leaves_one_handler_and_nothing_propagates() -> None:
    root_stream = io.StringIO()
    root_handler = logging.StreamHandler(root_stream)
    logging.getLogger().addHandler(root_handler)
    try:
        stream = io.StringIO()
        logger = _logger("sal.test.handlers", stream, [0.0] * 4)
        logger = _logger("sal.test.handlers", stream, [0.0] * 4)
        logger.info("once")
        assert len(logger.handlers) == 1
        assert not logger.propagate
        assert stream.getvalue().count("once") == 1
        assert root_stream.getvalue() == ""
    finally:
        logging.getLogger().removeHandler(root_handler)


@pytest.mark.edge_case
def test_a_pre_existing_plain_logger_is_refused() -> None:
    logging.setLoggerClass(logging.Logger)
    logging.getLogger("sal.test.plain")
    with pytest.raises(TypeError, match="already exists as a plain Logger"):
        get_logger("sal.test.plain", start_time=0.0, stream=io.StringIO())
