"""Benchmarks for the three Potts move sets.

Correctness is pinned in `tests/regression/search/`, per the repo's division
of labor between the two directories.

What these measure is cost per sweep, which is only half of what decides
whether a move set is worth having: `search/CLAUDE.md` requires a budget be
counted in work rather than seconds, and the other half --- how many sweeps
buy one independent sample --- is the autocorrelation measurement in the
regression suite. A cluster update that were twice the cost per sweep and
three times faster to decorrelate would still win, and neither number says so
alone.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture
from snakes_and_ladders.search.potts_mcmc import PottsMove, sample_potts
from snakes_and_ladders.sim.graph import BoundaryCondition, lattice_graph

# The exact q-state Potts transition on a square lattice, where single-site
# updates slow critically and the cluster algorithms are meant to earn their
# complexity.
TRANSITION = math.log(1.0 + math.sqrt(3.0))
FIELD = np.zeros(3)


@pytest.mark.parametrize("move", list(PottsMove))
@pytest.mark.parametrize("extent", [8, 16])
def test_potts_sweep_benchmark(
    benchmark: BenchmarkFixture, move: PottsMove, extent: int
) -> None:
    graph = lattice_graph((extent, extent), BoundaryCondition.OPEN, TRANSITION)

    chain = benchmark(
        sample_potts, graph, FIELD, move, np.random.default_rng(7), 50, 10
    )

    assert chain.states.shape == (50, graph.n_nodes)
