"""The run logger: elapsed time and the current phase on every line, and warnings that say it once (issue #311).

A long run -- a release build, a notebook re-execution, a search over
hundreds of candidates -- is read back from its log, and a line that does not
say how far into the run it fell, or what the run was doing, is a line
someone has to reconstruct. Every record here carries the elapsed minutes
since a start time the entry point chose, and the *runtime phase* the run
is in, which is shared across every logger in the process because a phase is
a property of the run and not of the module that happened to log.

Libraries emit, entry points configure. A module under ``sim``, ``opt`` or
``search`` calls ``logging.getLogger(__name__)`` and nothing else; an entry
point -- a QA script, the notebook checker, the release build -- calls
:func:`get_logger` once with its start time, which installs the handler and
the format. That is the standard library's own division of labour
(Ramalho, *Fluent Python*), and it is what keeps the test suite's output clean.

The format is the ticket's::

    2026-09-07 16:20:01 - 1.50m - INFO (render) - snakes_and_ladders.qa.build.render:120 - wrote sim_tree.pdf
"""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import IO, Any

FORMAT = (
    "%(asctime)s - %(runtime)s - %(levelname)-4s%(runtime_phase_str)s - "
    "%(name)s.%(funcName)s:%(lineno)d - %(message)s"
)
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class RunLogger(logging.Logger):
    """A logger whose phase is shared by every logger in the process, with once-only messages."""

    _runtime_phase: str | None = None

    def __init__(self, name: str, level: int = logging.NOTSET) -> None:
        super().__init__(name, level)
        self._seen: set[tuple[int, str]] = set()

    @property
    def runtime_phase(self) -> str | None:
        """The phase the run is in, or ``None`` between phases. Class state, by design."""
        return RunLogger._runtime_phase

    @runtime_phase.setter
    def runtime_phase(self, value: str | None) -> None:
        RunLogger._runtime_phase = value

    def _once(self, level: int, msg: str, *args: object, **kwargs: Any) -> None:
        key = (level, msg)
        if key in self._seen:
            return
        self._seen.add(key)
        kwargs.setdefault("stacklevel", 3)
        self.log(level, msg, *args, **kwargs)

    def warning_once(self, msg: str, *args: object, **kwargs: Any) -> None:
        """``warning`` the first time this message is seen on this logger, silence after."""
        self._once(logging.WARNING, msg, *args, **kwargs)

    def info_once(self, msg: str, *args: object, **kwargs: Any) -> None:
        """``info`` the first time this message is seen on this logger, silence after."""
        self._once(logging.INFO, msg, *args, **kwargs)


class PhaseFilter(logging.Filter):
    """Stamp the shared phase onto every record, as `` (phase)`` or nothing."""

    def filter(self, record: logging.LogRecord) -> bool:
        phase = RunLogger._runtime_phase
        record.runtime_phase_str = f" ({phase})" if phase else ""
        return True


class RuntimeFormatter(logging.Formatter):
    """Format with the elapsed minutes since ``start_time``, read from ``clock``."""

    def __init__(
        self,
        fmt: str = FORMAT,
        datefmt: str = DATE_FORMAT,
        *,
        start_time: float,
        clock: Callable[[], float] = time.time,
    ) -> None:
        super().__init__(fmt, datefmt)
        self.start_time = start_time
        self.clock = clock

    def format(self, record: logging.LogRecord) -> str:
        record.runtime = f"{(self.clock() - self.start_time) / 60.0:.2f}m"
        if not hasattr(record, "runtime_phase_str"):
            record.runtime_phase_str = ""
        return super().format(record)


def get_logger(
    name: str,
    *,
    start_time: float,
    level: int = logging.INFO,
    stream: IO[str] | None = None,
    clock: Callable[[], float] = time.time,
) -> RunLogger:
    """The run logger for ``name``, configured to write to ``stream`` and nowhere else.

    Installs :class:`RunLogger` as the logger class, so the logger returned
    (and every logger created afterwards) carries the shared phase and the
    once-only methods. Existing handlers and filters on the logger are
    replaced, so calling this twice leaves one handler, not two; propagation
    is off, so the root logger's handlers do not print every line a second
    time.

    Parameters
    ----------
    name : str
        Logger name, ``__name__`` at an entry point.
    start_time : float
        The run's start, as ``time.time()`` at the entry point; every record
        reports its distance from it in minutes.
    level : int
        Threshold for this logger.
    stream : IO[str] | None
        Where records go; ``None`` is ``sys.stderr``, leaving stdout to the
        paths and stems the QA scripts print for the build to consume.
    clock : Callable[[], float]
        The clock the elapsed time is read from. Injected so a test can pin
        the format without sleeping.

    Raises
    ------
    TypeError
        If a plain :class:`logging.Logger` already exists under ``name``:
        the standard library will not change a logger's class after the
        fact, so the entry point has to be the first to ask for it.
    """
    logging.setLoggerClass(RunLogger)
    logger = logging.getLogger(name)
    if not isinstance(logger, RunLogger):
        msg = f"logger {name!r} already exists as a plain Logger; configure it before the first getLogger"
        raise TypeError(msg)
    logger.setLevel(level)
    logger.propagate = False
    logger.handlers.clear()
    logger.filters.clear()
    logger.addFilter(PhaseFilter())
    handler = logging.StreamHandler(sys.stderr if stream is None else stream)
    handler.setFormatter(RuntimeFormatter(start_time=start_time, clock=clock))
    logger.addHandler(handler)
    return logger


@contextmanager
def phase(name: str) -> Iterator[None]:
    """Set the shared runtime phase for the block, restoring the previous one on exit."""
    previous = RunLogger._runtime_phase
    RunLogger._runtime_phase = name
    try:
        yield
    finally:
        RunLogger._runtime_phase = previous


__all__ = [
    "DATE_FORMAT",
    "FORMAT",
    "PhaseFilter",
    "RunLogger",
    "RuntimeFormatter",
    "get_logger",
    "phase",
]
