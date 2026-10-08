"""Niedermayer's mixing limits under a strong field, moved with the move to the sandbox (issues #1314, #1365)."""

from __future__ import annotations

import numpy as np
import pytest
from sal.backend import Backend
from sal.opt.termination import Stop
from sal.sample.potts_mcmc import (
    PottsMove,
)
from sal.sample.potts_mcmc.chains import RHAT_THRESHOLD
from sal.sandbox import potts_moves
from sal.sandbox.potts_moves import SandboxMove
from sal.sim.graph import BoundaryCondition, lattice_graph
from sal.sim.potts import site_field

from tests.regression.sample.test_potts_mixing import (
    POOLS,
    SEED,
    TEMPERATURE,
)


@pytest.mark.oracle
def test_the_frozen_niedermayer_chain_ends_not_mixing_and_its_starts_disagree() -> None:
    # #1314's instance at 320 steps per start. Declared: the ordered chain
    # ends NOT_MIXING, and R-hat exceeds RHAT_THRESHOLD on every observable.
    graph = lattice_graph((64, 64), BoundaryCondition.PERIODIC, 1.0)
    field = np.random.default_rng(1314).normal(0.0, 1.0, (graph.n_nodes, 3))
    starts = potts_moves.sample_potts_starts(
        graph,
        field,
        SandboxMove.NIEDERMAYER,
        np.random.default_rng(SEED),
        320,
        temperature=TEMPERATURE,
    )
    ordered = starts.chains[0]

    assert ordered.termination.reason is Stop.NOT_MIXING
    # Acceptance is not a declared claim here: a bound read off #1314's three
    # seeds (0.0% to 0.3%) failed at this seed's 0.31%, and a bound widened
    # to admit it would be fitted. The seed is fixed, so the count is pinned.
    assert ordered.acceptance == 1 / 320, ordered.acceptance
    assert ordered.largest_cluster_share > 0.99, ordered.largest_cluster_share
    assert starts.termination.reason is Stop.NOT_MIXING
    assert np.all(starts.rhat > RHAT_THRESHOLD), starts.rhat


@pytest.mark.analytic
def test_a_rigid_niedermayer_pair_flips_every_step_by_hand() -> None:
    # Two sites, q = 2, J = 1e3, no field: the bond is certain and the
    # cluster has no boundary, so every Niedermayer transposition is
    # accepted and swaps both labels. By hand: acceptance 1, share 1, mean cluster 2, the
    # occupancy alternating and the energy constant at -1e3.
    graph = lattice_graph((2,), BoundaryCondition.OPEN, 1e3)
    chain = potts_moves.sample_potts(
        graph,
        np.zeros(2),
        SandboxMove.NIEDERMAYER,
        np.random.default_rng(SEED),
        6,
        start=np.zeros(2, dtype=np.int64),
    )

    assert chain.acceptance == 1.0
    assert chain.largest_cluster_share == 1.0
    assert chain.mean_cluster_size == 2.0
    np.testing.assert_array_equal(chain.states[:, 0], [1, 0, 1, 0, 1, 0])
    assert chain.ess[0] == 0.0
    assert chain.termination.reason is Stop.NOT_MIXING


@pytest.mark.oracle
def test_recording_the_diagnostics_draws_no_random_number() -> None:
    # The chain is the bare kernel loop bitwise: the uniform start, then one
    # sweep per step on the same generator, with nothing drawn between.
    graph = lattice_graph((4, 4), BoundaryCondition.PERIODIC, 1.0)
    field = np.random.default_rng(1314).normal(0.0, 1.0, (16, 3))
    for move in (SandboxMove.NIEDERMAYER, PottsMove.SINGLE_SITE, PottsMove.WOLFF):
        chain = potts_moves.sample_potts(
            graph, field, move, np.random.default_rng(SEED), 30, 5, 2
        )
        rng = np.random.default_rng(SEED)
        rows = site_field(field, graph.n_nodes)
        state = rng.integers(0, 3, size=graph.n_nodes)
        offsets, neighbours, couplings = graph.compressed_adjacency()
        advance = potts_moves.sweep_for(
            move, graph, rows, offsets, neighbours, couplings, Backend.RUST
        )
        expected = []
        for step in range(-10, 60):
            advance(state, rng, 1.0)
            if step >= 0 and (step + 1) % 2 == 0:
                expected.append(state.copy())

        np.testing.assert_array_equal(chain.states, np.array(expected), err_msg=move)


@pytest.mark.oracle
@pytest.mark.backend
def test_thread_pool_starts_are_the_serial_run_bitwise() -> None:
    graph = lattice_graph((4, 4), BoundaryCondition.PERIODIC, 1.0)
    field = np.random.default_rng(1314).normal(0.0, 1.0, (16, 3))
    runs = [
        potts_moves.sample_potts_starts(
            graph,
            field,
            SandboxMove.NIEDERMAYER,
            np.random.default_rng(SEED),
            40,
            equilibration_sweeps=5,
            workers=workers,
            pool=pool,
        )
        for workers, pool in POOLS
    ]

    for serial, threaded in zip(runs[0].chains, runs[1].chains, strict=True):
        np.testing.assert_array_equal(serial.states, threaded.states)
        assert serial.acceptance == threaded.acceptance
    np.testing.assert_array_equal(runs[0].rhat, runs[1].rhat)
