"""The Wolff step with its adjacency converted per step against once per chain (issue #919).

Correctness --- the same stream either way --- is pinned in
`tests/regression/sample/test_potts_mcmc.py`. The instance is the ticket's: a
50x50 open lattice, q = 3, the critical coupling at q = 3 and a
field tilted across the labels, the same 200 steps per round from one warmed
state and one stream.
"""

from __future__ import annotations

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture
from snakes_and_ladders.sample import potts_mcmc
from snakes_and_ladders.sim.graph import BoundaryCondition, lattice_graph
from snakes_and_ladders.sim.potts import critical_coupling, site_field

STEPS = 200


@pytest.mark.parametrize("held", [False, True], ids=["per-step", "once"])
@pytest.mark.parametrize("beta", [0.5, 1.0, 2.0])
def test_wolff_adjacency_benchmark(
    benchmark: BenchmarkFixture, beta: float, held: bool
) -> None:
    graph = lattice_graph((50, 50), BoundaryCondition.OPEN, critical_coupling(3))
    offsets, neighbours, couplings = graph.compressed_adjacency()
    rows = site_field(np.array([0.1, 0.0, -0.1]), graph.n_nodes)
    lists = potts_mcmc.adjacency_lists(offsets, neighbours, couplings) if held else None
    rng = np.random.default_rng(919)
    state = rng.integers(0, 3, graph.n_nodes)
    for _ in range(STEPS):
        potts_mcmc.wolff_sweep(
            state, rows, offsets, neighbours, couplings, rng, beta=beta, lists=lists
        )

    warmed = state.copy()

    def run() -> int:
        # The same 200 steps from the same warmed state and stream every
        # round, so the two variants time one chain.
        chain = warmed.copy()
        stream = np.random.default_rng(1)
        return sum(
            potts_mcmc.wolff_sweep(
                chain,
                rows,
                offsets,
                neighbours,
                couplings,
                stream,
                beta=beta,
                lists=lists,
            )
            for _ in range(STEPS)
        )

    assert benchmark(run) >= STEPS
