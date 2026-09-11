"""The generic sweep against the Potts kernels it generalizes.

Python over factor tables against the Python single-site sweep and the Rust
one, on the same lattice: the ratio is the price of a sampler that knows
nothing about what its variables mean, and ``STATUS.md`` records it. The
correctness of every sweep is pinned in ``tests/regression/search/test_gibbs.py``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture
from snakes_and_ladders.opt.schedule import Constant
from snakes_and_ladders.search.gibbs import _Indexed, gibbs_sweep
from snakes_and_ladders.search.potts_mcmc import anneal_potts
from snakes_and_ladders.sim.factor_graph import from_potts
from snakes_and_ladders.sim.graph import BoundaryCondition, lattice_graph

# The Rust backend arrives with the #287 audit; without it this benchmark has
# nothing to compare and skips rather than fails.
Backend = pytest.importorskip("snakes_and_ladders.search.backend").Backend

SHAPE = (8, 8)
DENSITY_SHAPE = (32, 32)
FIELD = np.array([0.3, -0.2, 0.1])
SWEEPS = 20


@pytest.mark.parametrize(
    "backend", [Backend.PYTHON, Backend.NUMBA], ids=["python", "numba"]
)
def test_generic_gibbs_sweep_benchmark(
    benchmark: BenchmarkFixture, backend: Any
) -> None:
    """Twenty generic sweeps of an 8x8 three-state lattice, per backend.

    The ratio is what issue #561 bought, and the two are the same chain:
    ``tests/regression/search/test_gibbs.py`` pins the compiled sweep against
    the NumPy one bitwise. One sweep runs before the timing so that the edge
    layout and the kernel's compilation, each paid once per graph and per
    process, fall outside it.
    """
    graph = lattice_graph(SHAPE, BoundaryCondition.PERIODIC, 0.5)
    indexed = _Indexed(from_potts(graph, FIELD))
    rng = np.random.default_rng(1)
    state = rng.integers(0, 3, size=graph.n_nodes)
    gibbs_sweep(indexed, state, rng, backend=backend)

    def run() -> np.ndarray:
        for _ in range(SWEEPS):
            gibbs_sweep(indexed, state, rng, backend=backend)
        return state

    result = benchmark(run)
    assert result.shape == (graph.n_nodes,)


@pytest.mark.parametrize(
    "backend", [Backend.PYTHON, Backend.NUMBA], ids=["python", "numba"]
)
def test_generic_log_density_benchmark(
    benchmark: BenchmarkFixture, backend: Any
) -> None:
    """One log-density of a 32x32 three-state lattice, per backend.

    The size the #561 profile was taken at, where the dictionary implementation
    carries 47.8% of a compiled run over 3,072 factors. One call runs before
    the timing so that the edge layout and the kernel's compilation fall
    outside it, and its value is what the timed calls are asserted against;
    ``tests/regression/search/test_gibbs.py`` pins the compiled density against
    the dictionary one bitwise.
    """
    graph = lattice_graph(DENSITY_SHAPE, BoundaryCondition.PERIODIC, 0.5)
    indexed = _Indexed(from_potts(graph, FIELD))
    state = np.random.default_rng(1).integers(0, 3, size=graph.n_nodes)
    expected = indexed.log_density(state, backend)

    value = benchmark(indexed.log_density, state, backend)

    assert value == expected


@pytest.mark.parametrize(
    "backend", [Backend.PYTHON, Backend.RUST], ids=["python", "rust"]
)
def test_potts_single_site_sweep_benchmark(
    benchmark: BenchmarkFixture, backend: Any
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
