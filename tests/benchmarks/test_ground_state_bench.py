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
from snakes_and_ladders.backend import Backend
from snakes_and_ladders.sample.potts_mcmc import PottsMove, anneal_potts
from snakes_and_ladders.sample.schedule import ExponentialTempSchedule
from snakes_and_ladders.sandbox import maxflow_declined
from snakes_and_ladders.sandbox.maxflow_declined import DeclinedKernel
from snakes_and_ladders.search import alpha_expansion as expansion_module
from snakes_and_ladders.search.alpha_expansion import alpha_beta_swap, alpha_expansion
from snakes_and_ladders.search.maxflow import FlowNetwork, MinCut
from snakes_and_ladders.sim.graph import BoundaryCondition, triangular_lattice_graph
from snakes_and_ladders.sim.potts import spatio_only_field

#: The schedule the comparison anneals on, at a length a benchmark affords.
STEPS = 20


def _problem(extent: int, n_states: int) -> tuple[object, np.ndarray]:
    """`spatio_only`' construction at a benchmark size: triangular, lognormal sizes."""
    graph = triangular_lattice_graph((extent, extent), BoundaryCondition.OPEN, 0.7)
    rng = np.random.default_rng(extent * 10 + n_states)
    sizes = np.exp(rng.normal(0.0, 0.6, size=graph.n_nodes))
    return graph, spatio_only_field(np.linspace(-0.9, 0.9, n_states), sizes)


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

    # The oracle sweep explicitly, for the reason
    # `test_potts_mcmc_bench.py` gives: this compares move sets, not
    # backends (issue #599).
    result = benchmark(
        anneal_potts,
        graph,
        field,
        ExponentialTempSchedule(2.0, 0.05, STEPS),
        np.random.default_rng(551),
        move=move,
        backend=Backend.PYTHON,
    )

    assert result.n_sweeps == STEPS


# --- every max-flow kernel as the inner solver (issue #715) -------------------

#: The package kernel beside the three declined ones; the declined rows skip
#: unless the extension carries the `sandbox` feature.
KERNELS = ["boykov-kolmogorov", *(str(kernel) for kernel in DeclinedKernel)]


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("n_states", [3, 10])
@pytest.mark.parametrize(
    "extent", [8, 16, 32, pytest.param(64, marks=pytest.mark.release)]
)
def test_alpha_expansion_by_kernel_benchmark(
    benchmark: BenchmarkFixture,
    monkeypatch: pytest.MonkeyPatch,
    extent: int,
    n_states: int,
    kernel: str,
) -> None:
    # The effect-size row of issue #715: what a Potts caller pays is `q`
    # cuts per sweep, so the kernel was decided here as well as on the cut.
    # The package seam carries one kernel, so a declined one is routed in by
    # replacing the cut the expansion calls.
    if kernel != "boykov-kolmogorov":
        if not maxflow_declined.AVAILABLE:
            pytest.skip("the extension was built without the sandbox feature")
        declined = DeclinedKernel(kernel)

        def cut(network: FlowNetwork, source: int, sink: int) -> MinCut:
            return maxflow_declined.min_cut(network, source, sink, declined)

        monkeypatch.setattr(expansion_module, "min_cut", cut)
    graph, field = _problem(extent, n_states)

    result = benchmark(alpha_expansion, graph, field, n_states, backend=Backend.RUST)

    assert result.cycles >= 1
