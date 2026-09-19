"""Deprecated: moved to :mod:`snakes_and_ladders.sample.schedule` (issue #777).

Importing this module emits one :class:`DeprecationWarning` and re-exports
every public name the new one defines; the objects are the same objects,
which ``tests/regression/sample/test_sample_deprecated_paths.py`` asserts
with ``is``. The shim is removed in the release after 0.3.0, whose number
issue #522 names.
"""

from __future__ import annotations

import warnings

from snakes_and_ladders.sample.schedule import (
    AdaptedLadder,
    ConstantTempSchedule,
    CosineTempSchedule,
    ExponentialTempSchedule,
    FeedbackLadder,
    LinearTempSchedule,
    TempSchedule,
    adapt_ladder,
    adapt_ladder_by_round_trips,
    temperatures,
)

warnings.warn(
    "snakes_and_ladders.opt.schedule moved to snakes_and_ladders.sample.schedule "
    "(issue #777); this shim is removed in the release after 0.3.0.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "AdaptedLadder",
    "ConstantTempSchedule",
    "CosineTempSchedule",
    "ExponentialTempSchedule",
    "FeedbackLadder",
    "LinearTempSchedule",
    "TempSchedule",
    "adapt_ladder",
    "adapt_ladder_by_round_trips",
    "temperatures",
]
