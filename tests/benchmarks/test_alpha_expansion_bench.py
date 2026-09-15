"""Benchmarks for alpha expansion and the single-site baseline it beats.

Correctness is pinned in `tests/regression/search/`, per the repo's division
of labor between the two directories.

Both are run on the same problems, because the interesting quantity is not
either time alone but what each buys: `search/CLAUDE.md` requires a budget be
counted in work rather than seconds, and here the work is minimum cuts (one
per label per cycle) against sweeps.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture
from snakes_and_ladders.search.alpha_expansion import (
    _expansion_network,
    _site_field,
    alpha_expansion,
    iterated_conditional_modes,
)
from snakes_and_ladders.search.backend import Backend
from snakes_and_ladders.sim.graph import BoundaryCondition, PottsGraph, lattice_graph

sys.setrecursionlimit(50_000)


def _problem(extent: int, n_states: int) -> tuple[PottsGraph, np.ndarray]:
    graph = lattice_graph((extent, extent), BoundaryCondition.OPEN, 1.2)
    rng = np.random.default_rng(extent * 10 + n_states)
    return graph, rng.normal(size=(graph.n_nodes, n_states))


@pytest.mark.parametrize("backend", [Backend.PYTHON, Backend.RUST], ids=str)
@pytest.mark.parametrize("n_states", [3, 5])
@pytest.mark.parametrize("extent", [8, 16])
def test_alpha_expansion_benchmark(
    benchmark: BenchmarkFixture, extent: int, n_states: int, backend: Backend
) -> None:
    # Cost is one minimum cut per label per cycle, so it scales in the label
    # count as well as the lattice -- which is the trade against single-site
    # descent, whose sweep is independent of how many labels there are.
    #
    # The two backends run the same expansions on the same networks and
    # differ only in which solver cuts them, so the ratio here is the Rust
    # minimum cut against the Python one *as a caller pays for it*, network
    # construction included. The kernel alone is timed by Criterion in
    # `benches/`, which is the pair `DEV.md` step 3 asks for (#528).
    graph, field_values = _problem(extent, n_states)

    result = benchmark(alpha_expansion, graph, field_values, n_states, backend=backend)

    assert result.cycles >= 1


@pytest.mark.parametrize("backend", [Backend.PYTHON, Backend.NUMBA], ids=str)
@pytest.mark.parametrize("n_states", [3, 5])
@pytest.mark.parametrize("extent", [8, 16])
def test_single_site_descent_benchmark(
    benchmark: BenchmarkFixture, extent: int, n_states: int, backend: Backend
) -> None:
    # The Python sweep is the oracle and the `numba` kernel the default since
    # #264; the two return the same labelling bitwise, so this measures cost
    # alone -- 7x at 32x32 with three labels when it landed.
    graph, field_values = _problem(extent, n_states)
    _, energy = benchmark(
        iterated_conditional_modes,
        graph,
        field_values,
        n_states,
        np.random.default_rng(0),
        backend=backend,
    )
    assert np.isfinite(energy)


@pytest.mark.parametrize("n_states", [3, 5])
@pytest.mark.parametrize("extent", [8, 16, 32])
def test_the_expansion_network_build_benchmark(
    benchmark: BenchmarkFixture, extent: int, n_states: int
) -> None:
    """Building the network, apart from solving it (issue #598).

    The half of :func:`~snakes_and_ladders.search.alpha_expansion.expand` the
    Rust cut kernel does not touch, timed on its own so the split between
    building and solving is a number rather than a profile reading.
    """
    graph, values = _problem(extent, n_states)
    field = _site_field(graph, values)
    rng = np.random.default_rng(598)
    labelling = rng.integers(0, n_states, size=graph.n_nodes)

    built = benchmark(_expansion_network, graph, field, labelling, 0)

    assert built.n_nodes >= graph.n_nodes + 2
