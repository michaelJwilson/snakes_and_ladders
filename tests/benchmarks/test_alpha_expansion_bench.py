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
    alpha_expansion,
    iterated_conditional_modes,
)
from snakes_and_ladders.search.backend import Backend
from snakes_and_ladders.sim.graph import BoundaryCondition, lattice_graph

sys.setrecursionlimit(50_000)


def _problem(extent: int, n_states: int) -> tuple[object, np.ndarray]:
    graph = lattice_graph((extent, extent), BoundaryCondition.OPEN, 1.2)
    rng = np.random.default_rng(extent * 10 + n_states)
    return graph, rng.normal(size=(graph.n_nodes, n_states))


@pytest.mark.parametrize("n_states", [3, 5])
@pytest.mark.parametrize("extent", [8, 16])
def test_alpha_expansion_benchmark(
    benchmark: BenchmarkFixture, extent: int, n_states: int
) -> None:
    # Cost is one minimum cut per label per cycle, so it scales in the label
    # count as well as the lattice -- which is the trade against single-site
    # descent, whose sweep is independent of how many labels there are.
    graph, field_values = _problem(extent, n_states)

    result = benchmark(alpha_expansion, graph, field_values, n_states)

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
