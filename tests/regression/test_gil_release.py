"""The two compiled kernels #1059 found holding the GIL, released (issue #604's pattern).

`oxisal.ragged_posteriors` and `oxisal.leapfrog_trajectory` touched no
Python object between reading their arguments and returning, and held the
GIL throughout, so the thread pool (`sal.parallel`, `pool="threads"`) ran
them one at a time. Each now runs inside `py.detach`. The referee is the
calling thread: a kernel holding the GIL stops it for the whole call, so the
longest pause between two of its clock reads while four worker threads run
the kernel is at least one call's length; released, the pause is the
scheduler's.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

import numpy as np
import pytest
from sal import oxisal
from sal.sample.declared import declared_energy
from sal.validation.gaussian import GaussianTarget, diagonal_precision

#: Worker threads, as #604 measured.
THREADS = 4


def _ragged_call() -> Callable[[], None]:
    """A ragged forward-backward over 200 segments of 2,000 positions at four states."""
    rng = np.random.default_rng(1059)
    lengths = np.full(200, 2_000, dtype=np.int64)
    values = rng.normal(size=(int(lengths.sum()), 4))
    initial = np.log(np.full(4, 0.25))
    transition = np.log(np.full((4, 4), 0.25))

    def call() -> None:
        oxisal.ragged_posteriors(
            values,
            lengths,
            initial,
            transition,
            np.empty_like(values),
            np.empty((4, 4)),
            np.empty(lengths.size),
        )

    return call


def _leapfrog_call() -> Callable[[], object]:
    """A 2,000-step leapfrog trajectory on a 20,000-dimensional diagonal Gaussian."""
    dimension = 20_000
    declared = declared_energy(GaussianTarget(diagonal_precision(dimension)))
    assert declared is not None
    family, parameters = declared
    rng = np.random.default_rng(1059)
    theta, momentum = rng.normal(size=dimension), rng.normal(size=dimension)

    def call() -> object:
        return oxisal.leapfrog_trajectory(
            family, parameters, theta, momentum, 0.01, 2_000
        )

    return call


KERNELS = {"ragged_posteriors": _ragged_call, "leapfrog_trajectory": _leapfrog_call}


def _alone(call: Callable[[], object]) -> float:
    """The fastest of three calls in the calling thread: one call's length."""
    walls = []
    for _ in range(3):
        started = time.perf_counter()
        call()
        walls.append(time.perf_counter() - started)
    return min(walls)


@pytest.mark.backend
@pytest.mark.smoke
@pytest.mark.parametrize("kernel", sorted(KERNELS))
def test_four_threads_on_the_kernel_leave_the_calling_thread_running(
    kernel: str,
) -> None:
    call = KERNELS[kernel]()
    alone = _alone(call)
    workers = [threading.Thread(target=call) for _ in range(THREADS)]
    last = time.perf_counter()
    pause = 0.0
    for worker in workers:
        worker.start()
    while any(worker.is_alive() for worker in workers):
        now = time.perf_counter()
        pause, last = max(pause, now - last), now
    for worker in workers:
        worker.join()
    assert pause < 0.5 * alone, (kernel, pause, alone)
