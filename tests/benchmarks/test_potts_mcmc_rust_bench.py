"""The Rust single-site sweep against the Python oracle, at the sizes that
decide whether the port is kept.

Correctness is pinned in `tests/regression/search/test_potts_mcmc_rust.py`,
distributionally against exhaustive enumeration; this file only measures.

The cells rise in node count because that is the axis the cost is linear in:
issue #232 measured 100 sweeps taking 1.05 s at 1024 nodes against 0.064 s at
64, with the constant set by interpreter overhead rather than arithmetic. A
single cell would report a ratio; three report whether it holds as the problem
grows, which is what the pruning port turned out not to do.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture
from snakes_and_ladders.search import potts_mcmc_rust
from snakes_and_ladders.search.potts_mcmc import PottsMove, sample_potts
from snakes_and_ladders.sim.graph import BoundaryCondition, lattice_graph

SWEEPS = 100
FIELD = np.zeros(2)

#: Periodic, so every node has the same degree and the cost is linear in nodes
#: with no boundary term. 32x32 is the size #232 profiled.
EXTENTS = [8, 32]


@pytest.mark.parametrize("extent", EXTENTS, ids=lambda e: f"{e}x{e}")
def test_python_single_site_sweep(benchmark: BenchmarkFixture, extent: int) -> None:
    """The oracle, and the baseline the ratio is against."""
    graph = lattice_graph((extent, extent), BoundaryCondition.PERIODIC, 0.4)

    chain = benchmark(sample_potts, graph, FIELD, PottsMove.SINGLE_SITE, 1, SWEEPS)

    assert chain.states.shape == (SWEEPS, graph.n_nodes)


@pytest.mark.parametrize("extent", EXTENTS, ids=lambda e: f"{e}x{e}")
def test_rust_single_site_sweep(benchmark: BenchmarkFixture, extent: int) -> None:
    """The port, on the same problem.

    Shape and finiteness only. The two backends draw different chains by
    construction -- `f64::exp` and NumPy's differ by a unit in the last place,
    and `searchsorted` is a threshold -- so an equality assertion here would be
    asserting something false. What they must agree on is the distribution,
    which the regression suite checks against enumeration.
    """
    graph = lattice_graph((extent, extent), BoundaryCondition.PERIODIC, 0.4)

    chain = benchmark(
        potts_mcmc_rust.sample_potts,
        graph,
        FIELD,
        np.random.default_rng(1),
        SWEEPS,
    )

    assert chain.states.shape == (SWEEPS, graph.n_nodes)
    assert math.isfinite(float(chain.states.sum()))
