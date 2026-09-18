"""What the spectral bound costs beside the bounds it sits between.

Correctness is pinned in `tests/regression/search/test_spectral.py` and
`tests/regression/sim/test_spectrum.py`, per the repository's division of
labour between the two directories.

Three rows at each size so the table reads as a comparison rather than a
number: the spectral bound (one eigenpair), the exact cut where it applies
(the referee, and the answer the bound is a bound on), and the dual bound of
issue #696 at its default sweeps (the other lower bound that admits either
coupling sign). 128 and 256 are the stress sizes and run at the release gate.
"""

from __future__ import annotations

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture
from snakes_and_ladders.search import maxflow_rust
from snakes_and_ladders.search.spectral import spectral_bound, spectral_start
from snakes_and_ladders.search.tightening import dual_bound
from snakes_and_ladders.sim.graph import BoundaryCondition, PottsGraph, lattice_graph

EXTENTS = [
    16,
    32,
    64,
    pytest.param(128, marks=pytest.mark.release),
    pytest.param(256, marks=pytest.mark.release),
]


def _problem(extent: int) -> tuple[PottsGraph, np.ndarray]:
    graph = lattice_graph((extent, extent), BoundaryCondition.OPEN, 0.6)
    rng = np.random.default_rng(extent)
    return graph, rng.normal(size=(graph.n_nodes, 2))


@pytest.mark.parametrize("extent", EXTENTS)
def test_spectral_bound_benchmark(benchmark: BenchmarkFixture, extent: int) -> None:
    graph, field_values = _problem(extent)

    result = benchmark(spectral_bound, graph, field_values)

    assert np.isfinite(result.bound)


@pytest.mark.parametrize("extent", EXTENTS)
def test_spectral_start_benchmark(benchmark: BenchmarkFixture, extent: int) -> None:
    graph, field_values = _problem(extent)

    start = benchmark(spectral_start, graph, field_values)

    assert start.shape == (graph.n_nodes,)


@pytest.mark.parametrize("extent", EXTENTS)
def test_exact_cut_beside_the_bound_benchmark(
    benchmark: BenchmarkFixture, extent: int
) -> None:
    # The referee's own cost, so the bound's price is read against the price
    # of the answer where the answer is available.
    graph, field_values = _problem(extent)

    _, energy = benchmark(maxflow_rust.ising_ground_state, graph, field_values)

    assert np.isfinite(energy)


@pytest.mark.parametrize(
    "extent", [16, 32, pytest.param(64, marks=pytest.mark.release)]
)
def test_dual_bound_beside_the_spectral_bound_benchmark(
    benchmark: BenchmarkFixture, extent: int
) -> None:
    graph, field_values = _problem(extent)

    certificate = benchmark(dual_bound, graph, field_values, iterations=50)

    assert np.isfinite(certificate.bound)
