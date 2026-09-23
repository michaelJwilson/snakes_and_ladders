"""Runtime goals: the package against an external framework's time, one run (issue #972).

A goal test times the package's call with :func:`median_seconds` and the
framework's with its adapter, which reports the seconds its script measured
around the framework's own call, each the median of :data:`REPEATS` runs on
one host in one session. :func:`assert_meets` fails when the package's median
exceeds the framework's times :data:`GOAL_RATIO`, and says by how much.

The goal is the framework's runtime and nothing looser: a ratio of one. A
goal test carries the `goal` marker and runs in a step of CI's `validation`
job that reports and does not block, since a goal fails until it is met.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence

import numpy as np

#: Runs whose median each side's figure is.
REPEATS = 5

#: The package's median over the framework's, at most.
GOAL_RATIO = 1.0


def median_seconds(call: Callable[[], object], repeats: int = REPEATS) -> float:
    """The median wall seconds of ``call()`` over ``repeats`` runs."""
    seconds = []
    for _ in range(repeats):
        start = time.perf_counter()
        call()
        seconds.append(time.perf_counter() - start)
    return float(np.median(seconds))


def assert_meets(ours: float, theirs: Sequence[float], what: str) -> None:
    """Fail when ``ours`` exceeds the median of ``theirs`` times the goal ratio."""
    goal = float(np.median(theirs)) * GOAL_RATIO
    assert ours <= goal, (
        f"{what}: the package takes {ours * 1e3:.2f} ms against the goal of "
        f"{goal * 1e3:.2f} ms, {ours / goal:.2f}x the external framework"
    )
