"""Runtime goals: the package against an external framework's measured time (issue #972).

A :class:`Goal` is a framework's runtime on a declared fixture, measured once
on the 4-core reference host and written here, so a goal test needs the
package alone: :func:`median_seconds` times the package and
:func:`assert_meets` fails above the goal times :data:`GOAL_RATIO`, 0.55. The
benchmark pair in `tests/benchmarks/` re-measures it. A :class:`MemoryGoal`
bounds the peak added resident memory (issue #987), read in a fresh
interpreter by :func:`snakes_and_ladders.validation.runner.package`
(:func:`assert_fits`). Goal tests carry `goal` and run in CI's `validation`
job without blocking; the ratio moves with the hardware.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
from snakes_and_ladders.validation.runner import package

#: Runs whose median the package's figure is.
REPEATS = 5

#: The package's figure over the framework's, at most: about half, so a goal
#: is met near twice the framework's speed or half its memory, not at parity
#: (issue #986; 0.55 rather than 0.5, the margin the issue's owner set).
GOAL_RATIO = 0.55


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


def median_package(
    call: str, inputs: Mapping[str, np.ndarray], read: str = "seconds", repeats: int = 3
) -> float:
    """The median over ``repeats`` fresh-interpreter runs of the package's ``call`` (issue #1010).

    ``read``: ``"seconds"``, ``"peak_bytes"`` (none reads 0) or a scalar output's name.
    """
    values = []
    for _ in range(repeats):
        run = package(call, inputs)
        if read == "seconds":
            values.append(run.seconds)
        elif read == "peak_bytes":
            values.append(float(run.peak_bytes))
        else:
            values.append(float(run.outputs[read]))
    return float(np.median(values))


def assert_meets(ours: float, goal: Goal) -> None:
    """Fail when ``ours`` exceeds the goal's runtime times the goal ratio."""
    limit = goal.seconds * GOAL_RATIO
    assert ours <= limit, (
        f"{goal.what}: the package takes {ours * 1e3:.2f} ms against "
        f"{goal.framework}'s {goal.seconds * 1e3:.2f} ms ({goal.measured}), "
        f"{ours / goal.seconds:.2f}x its figure where {GOAL_RATIO}x is the goal"
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
        f"{goal.framework}'s {goal.peak_bytes / 1e6:.1f} MB ({goal.measured}), "
        f"{ours / goal.peak_bytes:.2f}x its figure where {GOAL_RATIO}x is the goal"
    )
