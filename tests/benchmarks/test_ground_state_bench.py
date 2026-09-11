"""Benchmarks for the entries issue #551 added: the swap, and the annealed cluster moves.

Correctness is pinned in `tests/regression/search/test_ground_state.py`, per
the division of labor between the two directories.

The swap and the expansion are benchmarked on the same problems, because the
quantity that matters is not either time alone but what each buys per cut: a
swap cycle is ``q (q - 1) / 2`` cuts over the sites carrying two labels
against the expansion's ``q`` cuts over the whole lattice, so "cheaper per
move" and "cheaper per cycle" are different claims and only a measurement
separates them.

The annealed cluster moves are benchmarked against the annealed heat bath at
**equal sweeps**, which is deliberately *not* the unit the comparison is run
in: the gap between a Wolff sweep's time here and a heat-bath sweep's is the
same gap that makes equal sweeps the wrong budget, and reading it directly is
what justifies the site-visit unit.
"""

from __future__ import annotations

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture
from snakes_and_ladders.opt.schedule import Exponential
from snakes_and_ladders.search.alpha_expansion import alpha_beta_swap, alpha_expansion
from snakes_and_ladders.search.backend import Backend
from snakes_and_ladders.search.potts_mcmc import PottsMove, anneal_potts
from snakes_and_ladders.sim.graph import BoundaryCondition, triangular_lattice_graph
from snakes_and_ladders.sim.potts import spots_field

#: The schedule the comparison anneals on, at a length a benchmark affords.
STEPS = 20


def _problem(extent: int, n_states: int) -> tuple[object, np.ndarray]:
    """`potts_spots`' construction at a benchmark size: triangular, lognormal sizes."""
    graph = triangular_lattice_graph((extent, extent), BoundaryCondition.OPEN, 0.7)
    rng = np.random.default_rng(extent * 10 + n_states)
    sizes = np.exp(rng.normal(0.0, 0.6, size=graph.n_nodes))
    return graph, spots_field(np.linspace(-0.9, 0.9, n_states), sizes)


@pytest.mark.parametrize("n_states", [3, 10])
@pytest.mark.parametrize("extent", [8, 16])
def test_alpha_beta_swap_benchmark(
    benchmark: BenchmarkFixture, extent: int, n_states: int
) -> None:
    # The swap's network covers only the sites carrying the two labels and
    # needs no auxiliary node, so its per-cut cost falls as the labels spread
    # -- and its cycle cost rises quadratically in the label count. Both
    # effects are in this number.
    graph, field = _problem(extent, n_states)

    result = benchmark(alpha_beta_swap, graph, field, n_states, backend=Backend.RUST)

    assert result.cycles >= 1


@pytest.mark.parametrize("n_states", [3, 10])
@pytest.mark.parametrize("extent", [8, 16])
def test_alpha_expansion_on_the_spots_construction_benchmark(
    benchmark: BenchmarkFixture, extent: int, n_states: int
) -> None:
    # The same problems, so the two move sets are read against each other
    # rather than against separate instances.
    graph, field = _problem(extent, n_states)

    result = benchmark(alpha_expansion, graph, field, n_states, backend=Backend.RUST)

    assert result.cycles >= 1


@pytest.mark.parametrize("move", list(PottsMove), ids=str)
@pytest.mark.parametrize("extent", [8, 16])
def test_annealed_move_set_benchmark(
    benchmark: BenchmarkFixture, extent: int, move: PottsMove
) -> None:
    # Equal sweeps, which is the unit the comparison refuses: a Wolff sweep
    # flips one cluster and a heat-bath sweep touches every site, so the times
    # here are not comparable work and the ratio between them is the finding.
    graph, field = _problem(extent, 10)

    result = benchmark(
        anneal_potts,
        graph,
        field,
        Exponential(2.0, 0.05, STEPS),
        np.random.default_rng(551),
        move=move,
    )

    assert result.n_sweeps == STEPS
