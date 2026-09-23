"""Runtime goals: the package against an external framework's measured time (issue #972).

A :class:`Goal` is an external framework's runtime on a declared fixture,
measured once on the 4-core reference host and written here as a number, so
a goal test needs the package alone: it times the package's call with
:func:`median_seconds` and :func:`assert_meets` fails when the median exceeds
the goal times :data:`GOAL_RATIO`, saying by how much. The framework is not
installed to run it; it is installed to re-measure the number, by the
benchmark pair in `tests/benchmarks/`, and the row records when and where it
was measured.

A :class:`MemoryGoal` is the same for the resident memory a call adds at its
peak (issue #987). The package's figure is read by
:func:`snakes_and_ladders.validation.runner.package`, in a fresh interpreter
as the framework's was, and :func:`assert_fits` fails when it exceeds the
goal.

A goal test carries the `goal` marker and runs in a step of CI's
`validation` job that reports and does not block, since a goal fails until it
is met. Measured on a different host, the ratio moves with the hardware.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

#: Runs whose median the package's figure is.
REPEATS = 5

#: The package's median over the framework's runtime, at most.
GOAL_RATIO = 1.0


@dataclass(frozen=True)
class Goal:
    """One external framework's runtime on one declared fixture."""

    #: The framework, as `snakes_and_ladders.validation.FRAMEWORKS` names it.
    framework: str
    #: The fixture and the call timed, in words a reader can rebuild.
    what: str
    #: The framework's runtime, in seconds.
    seconds: float
    #: When, where and by which ticket it was measured.
    measured: str


def median_seconds(call: Callable[[], object], repeats: int = REPEATS) -> float:
    """The median wall seconds of ``call()`` over ``repeats`` runs."""
    seconds = []
    for _ in range(repeats):
        start = time.perf_counter()
        call()
        seconds.append(time.perf_counter() - start)
    return float(np.median(seconds))


def assert_meets(ours: float, goal: Goal) -> None:
    """Fail when ``ours`` exceeds the goal's runtime times the goal ratio."""
    limit = goal.seconds * GOAL_RATIO
    assert ours <= limit, (
        f"{goal.what}: the package takes {ours * 1e3:.2f} ms against "
        f"{goal.framework}'s {limit * 1e3:.2f} ms ({goal.measured}), "
        f"{ours / limit:.2f}x"
    )


@dataclass(frozen=True)
class MemoryGoal:
    """One external framework's peak added resident memory on one declared fixture."""

    #: The framework, as `snakes_and_ladders.validation.FRAMEWORKS` names it.
    framework: str
    #: The fixture and the call measured, in words a reader can rebuild.
    what: str
    #: The framework's peak added resident bytes.
    peak_bytes: int
    #: When, where and by which ticket it was measured.
    measured: str


def assert_fits(ours: int, goal: MemoryGoal) -> None:
    """Fail when ``ours`` exceeds the goal's peak bytes times the goal ratio."""
    limit = goal.peak_bytes * GOAL_RATIO
    assert ours <= limit, (
        f"{goal.what}: the package peaks at {ours / 1e6:.1f} MB against "
        f"{goal.framework}'s {limit / 1e6:.1f} MB ({goal.measured}), "
        f"{ours / limit:.2f}x"
    )
