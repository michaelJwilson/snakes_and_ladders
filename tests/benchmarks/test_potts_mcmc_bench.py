"""Benchmarks for the five Potts move sets.

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

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture
from sal.backend import Backend
from sal.sample.potts_mcmc import PottsMove, sample_potts
from sal.sim.graph import BoundaryCondition, lattice_graph
from sal.sim.potts import critical_coupling

# The exact q-state Potts transition on a square lattice, where single-site
# updates slow critically and the cluster algorithms are meant to earn their
# complexity.
TRANSITION = critical_coupling(3)
FIELD = np.zeros(3)


@pytest.mark.parametrize("move", list(PottsMove))
@pytest.mark.parametrize("extent", [8, 16])
def test_potts_sweep_benchmark(
    benchmark: BenchmarkFixture, move: PottsMove, extent: int
) -> None:
    graph = lattice_graph((extent, extent), BoundaryCondition.OPEN, TRANSITION)

    # The oracle sweep explicitly, though it is no longer the default: this
    # compares *move sets*, and a Rust single-site against two Python cluster
    # moves would be a comparison of backends (issue #599).
    chain = benchmark(
        sample_potts,
        graph,
        FIELD,
        move,
        np.random.default_rng(7),
        50,
        10,
        backend=Backend.PYTHON,
    )

    assert chain.states.shape == (50, graph.n_nodes)


@pytest.mark.parametrize(
    "move", [PottsMove.SWENDSEN_WANG, PottsMove.SWENDSEN_WANG_HEAT_BATH], ids=str
)
def test_release_cluster_pass_benchmark(
    benchmark: BenchmarkFixture, move: PottsMove
) -> None:
    # One pass at `spatio_tiling/release` (5,041 sites, q = 10), the size the
    # anneal comparison of issue #1142 runs at: the uniform proposal on its
    # compiled pass against the heat bath on NumPy and the compiled union-find.
    from sal.sample.potts_mcmc import sweeps
    from sal.search.potts_starts import tiling_rung
    from sal.sim.fixtures import fixture

    rung = tiling_rung(fixture("spatio_tiling", "release").params, "release")
    rng = np.random.default_rng(1142)
    state = rng.integers(0, rung.n_states, size=rung.n_nodes)

    def one_pass() -> None:
        if move is PottsMove.SWENDSEN_WANG:
            sweeps.swendsen_wang_sweep(
                state, rung.graph, rung.field, rng, None, 1.0, backend=Backend.RUST
            )
        else:
            sweeps.swendsen_wang_heat_bath_sweep(
                state, rung.graph, rung.field, rng, 1.0, backend=Backend.RUST
            )

    benchmark(one_pass)

    assert state.shape == (rung.n_nodes,)


#: Tempering steps a benchmark call runs.
STEPS = 10


@pytest.mark.parametrize(
    "move",
    [
        PottsMove.SINGLE_SITE,
        PottsMove.SWENDSEN_WANG,
        PottsMove.WOLFF,
        PottsMove.SWENDSEN_WANG_HEAT_BATH,
        PottsMove.WOLFF_HEAT_BATH,
    ],
    ids=str,
)
def test_release_tempering_steps_benchmark(
    benchmark: BenchmarkFixture, move: PottsMove
) -> None:
    # `STEPS` `parallel_tempering` steps at `spatio_tiling/release` on the
    # ladder and cluster backend `potts_tempering.run_tempering` runs (issue
    # #1156), each six replica moves, the energies and five exchange
    # proposals; ten steps so the per-call setup is not what is timed.
    from sal.sandbox.potts_tempering import N_REPLICAS, parallel_tempering
    from sal.search.ground_state import (
        _COMPILED_CLUSTERS,
        ANNEAL_END,
        ANNEAL_START,
    )
    from sal.search.potts_starts import tiling_rung
    from sal.sim.fixtures import fixture

    rung = tiling_rung(fixture("spatio_tiling", "release").params, "release")
    ladder = tuple(np.geomspace(ANNEAL_END, ANNEAL_START, N_REPLICAS).tolist())
    cluster_backend = Backend.RUST if move in _COMPILED_CLUSTERS else Backend.PYTHON

    run = benchmark(
        lambda: parallel_tempering(
            rung.graph,
            rung.field,
            ladder,
            np.random.default_rng(1156),
            STEPS,
            move=move,
            cluster_backend=cluster_backend,
        )
    )

    assert run.states.shape == (STEPS, N_REPLICAS, rung.n_nodes)


def test_release_rung_moves_tempering_steps_benchmark(
    benchmark: BenchmarkFixture,
) -> None:
    # `STEPS` steps of the ladder `rung_moves` picks at `spatio_tiling/release`
    # (issue #1158): heat-bath Swendsen-Wang on the two hottest rungs, it and a
    # single-site sweep on the four below, on the compiled cluster route
    # `potts_tempering.run_tempering` takes; beside the single-move rows above.
    from sal.sandbox.potts_tempering import (
        N_REPLICAS,
        parallel_tempering,
        rung_moves,
        tempering_ladder,
    )
    from sal.search.potts_starts import tiling_rung
    from sal.sim.fixtures import fixture

    rung = tiling_rung(fixture("spatio_tiling", "release").params, "release")
    ladder = tempering_ladder()
    moves = rung_moves(rung.graph, rung.field, ladder)

    run = benchmark(
        lambda: parallel_tempering(
            rung.graph,
            rung.field,
            ladder,
            np.random.default_rng(1158),
            STEPS,
            move=moves,
            cluster_backend=Backend.RUST,
        )
    )

    assert run.states.shape == (STEPS, N_REPLICAS, rung.n_nodes)
