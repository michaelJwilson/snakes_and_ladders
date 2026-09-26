"""What padding costs, measured rather than asserted (issue #666)."""

from __future__ import annotations

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture
from sal.likelihood.ragged import posteriors
from sal.ragged import Ragged

STATES = 4


def _density(lengths: tuple[int, ...], seed: int = 7) -> Ragged:
    rng = np.random.default_rng(seed)
    return Ragged(np.log(rng.random((sum(lengths), STATES))), lengths)


@pytest.mark.benchmark(group="ragged-posteriors")
@pytest.mark.parametrize(
    "lengths",
    [(500,) * 64, (30,) * 63 + (2000,)],
    ids=["even", "one-long-among-short"],
)
def test_ragged_posteriors_bench(
    benchmark: BenchmarkFixture, lengths: tuple[int, ...]
) -> None:
    """The compiled path pads nothing, so its cost follows the total only.

    The two shapes are chosen to separate the two costs: even lengths waste no
    padding, and one long segment among short ones is 97% padding in a block.
    """
    density = _density(lengths)
    initial = np.log(np.full(STATES, 1.0 / STATES))
    transition = np.log(np.full((STATES, STATES), 1.0 / STATES))
    benchmark(posteriors, density, initial, transition)


@pytest.mark.benchmark(group="ragged-switched")
@pytest.mark.parametrize("route", ["switch", "stack"])
def test_switched_transition_bench(benchmark: BenchmarkFixture, route: str) -> None:
    """A per-site switch (issue #1082): the kernel's per-step build, against the stack.

    `stack` is what a caller did without the switch: materialize one
    `(K, K)` kernel per step and run the oracle's per-step form, one segment
    at a time. 64 segments of 500 at four states.
    """
    from sal.backend import Backend

    lengths = (500,) * 64
    density = _density(lengths)
    initial = np.log(np.full(STATES, 1.0 / STATES))
    transition = np.log(np.random.default_rng(3).dirichlet(np.ones(STATES), STATES))
    switch = np.random.default_rng(4).uniform(size=sum(lengths))
    backend = Backend.RUST if route == "switch" else Backend.PYTHON
    benchmark(posteriors, density, initial, transition, switch=switch, backend=backend)
