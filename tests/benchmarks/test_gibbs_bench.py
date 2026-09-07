"""The generic sweep against the Potts kernels it generalizes.

Python over factor tables against the Python single-site sweep and the Rust
one, on the same lattice: the ratio is the price of a sampler that knows
nothing about what its variables mean, and ``STATUS.md`` records it. The
correctness of every sweep is pinned in ``tests/regression/search/test_gibbs.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture
from snakes_and_ladders.opt.schedule import Constant
from snakes_and_ladders.search.backend import Backend
from snakes_and_ladders.search.gibbs import _Indexed, gibbs_sweep
from snakes_and_ladders.search.potts_mcmc import anneal_potts
from snakes_and_ladders.sim.factor_graph import from_potts
from snakes_and_ladders.sim.graph import BoundaryCondition, lattice_graph

SHAPE = (8, 8)
FIELD = np.array([0.3, -0.2, 0.1])
SWEEPS = 20


def test_generic_gibbs_sweep_benchmark(benchmark: BenchmarkFixture) -> None:
    """Twenty generic sweeps of an 8x8 three-state lattice."""
    graph = lattice_graph(SHAPE, BoundaryCondition.PERIODIC, 0.5)
    indexed = _Indexed(from_potts(graph, FIELD))
    rng = np.random.default_rng(1)
    state = rng.integers(0, 3, size=graph.n_nodes)

    def run() -> np.ndarray:
        for _ in range(SWEEPS):
            gibbs_sweep(indexed, state, rng)
        return state

    result = benchmark(run)
    assert result.shape == (graph.n_nodes,)


@pytest.mark.parametrize(
    "backend", [Backend.PYTHON, Backend.RUST], ids=["python", "rust"]
)
def test_potts_single_site_sweep_benchmark(
    benchmark: BenchmarkFixture, backend: Backend
) -> None:
    """The same twenty sweeps by the specialised Potts kernels, at a constant temperature."""
    graph = lattice_graph(SHAPE, BoundaryCondition.PERIODIC, 0.5)

    annealed = benchmark(
        anneal_potts,
        graph,
        FIELD,
        Constant(1.0, SWEEPS),
        np.random.default_rng(1),
        backend=backend,
    )
    assert annealed.labelling.shape == (graph.n_nodes,)
