"""What padding costs, measured rather than asserted (issue #666)."""

from __future__ import annotations

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture
from sal.likelihood.rust.ragged import posteriors
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
