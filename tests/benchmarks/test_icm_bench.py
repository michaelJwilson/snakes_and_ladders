"""The floored descent at release size: the compiled kernel against its Python oracle (issue #1055).

Correctness is pinned in `tests/regression/search/test_icm.py`. Here the
instance is `spatio_only/release`, 5,041 sites at ten states, and the floor
is 1% of the sites; both backends run the same descent from the same draws,
so the pair's ratio is the kernel's and nothing else. The whole call is timed
--- draws, validation, sweeps and the energy --- since that is what a caller
pays, and the kernel alone beside it, so the enclosing call's share is read
off the two.
"""

from __future__ import annotations

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture
from snakes_and_ladders.backend import Backend
from snakes_and_ladders.search.ground_state import Rung
from snakes_and_ladders.search.icm import icm_sweeps_checked, iterated_conditional_modes
from snakes_and_ladders.search.potts_starts import spatio_rung
from snakes_and_ladders.sim.fixtures import fixture
from snakes_and_ladders.sim.potts import site_field

#: 1% of 5,041 sites.
MIN_SITES = 50
#: Sweeps allowed; index order stops at the first clean sweep before this.
MAX_SWEEPS = 200


def _rung() -> Rung:
    return spatio_rung(fixture("spatio_only", "release").params, "release")


@pytest.mark.parametrize("backend", [Backend.NUMBA, Backend.PYTHON], ids=str)
def test_floored_descent_benchmark(
    benchmark: BenchmarkFixture, backend: Backend
) -> None:
    rung = _rung()

    def descend() -> float:
        return iterated_conditional_modes(
            rung.graph,
            rung.field,
            rung.n_states,
            np.random.default_rng(0),
            max_sweeps=MAX_SWEEPS,
            min_sites=MIN_SITES,
            backend=backend,
        ).energy

    value = benchmark.pedantic(descend, rounds=5, iterations=1, warmup_rounds=1)  # type: ignore[no-untyped-call]

    assert np.isfinite(value)


def test_floored_kernel_benchmark(benchmark: BenchmarkFixture) -> None:
    # The compiled sweeps alone, from the start and draws the descent above
    # reads: the difference from its numba median is the enclosing call.
    rung = _rung()
    rng = np.random.default_rng(0)
    start = rng.integers(0, rung.n_states, size=rung.n_nodes)
    draws = rng.random(MAX_SWEEPS * rung.n_nodes)
    field = site_field(rung.field, rung.n_nodes)
    offsets, neighbours, couplings = rung.graph.compressed_adjacency()
    orders = np.empty((0, rung.n_nodes), dtype=np.int64)

    def sweep() -> int:
        return icm_sweeps_checked(
            start.copy(),
            field,
            offsets,
            neighbours,
            couplings,
            orders,
            draws,
            MAX_SWEEPS,
            True,
            MIN_SITES,
        )

    sweeps = benchmark.pedantic(sweep, rounds=5, iterations=1, warmup_rounds=1)  # type: ignore[no-untyped-call]

    assert 1 <= sweeps <= MAX_SWEEPS
