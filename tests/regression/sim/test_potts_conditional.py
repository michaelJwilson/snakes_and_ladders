"""The heat-bath conditional three sweeps share, and the draws it must reproduce.

`sim.potts.heat_bath_log_weights` is the expression
`sample.potts_mcmc.sweeps.single_site_sweep`, `sim.potts._simulate_gibbs` and the
Rust kernel each wrote out before issue #277. The loops around it stay
separate --- one chain in time, many chains at once, one compiled --- so what
is checked here is the expression and the draws, not a merged sweep.

The bar is **bitwise**, not a tolerance. An extraction that reassociated the
coupling sum would move every committed chain and every autocorrelation
`STATUS.md` pins, and would do it silently; the pinned configurations below
are the ones the sweeps drew before the extraction, at the seeds that drew
them (the standard #228, #266 and #571 were held to).
"""

from __future__ import annotations

import numpy as np
import pytest
from snakes_and_ladders.sample.potts_mcmc import PottsMove, sample_potts
from snakes_and_ladders.sim import fixtures
from snakes_and_ladders.sim.graph import lattice_graph
from snakes_and_ladders.sim.potts import (
    heat_bath_log_weights,
    simulate_potts,
    site_field,
)

#: `potts_lattice/ci`'s instance, which the pinned draws below were taken on
#: --- read from the registry rather than rebuilt from its literals (issue
#: #622). The graph is edge for edge the one the draws were taken on, and the
#: field is the declared array, so nothing below moves.
PARAMS = fixtures.fixture("potts_lattice", "ci").params
LATTICE = lattice_graph(PARAMS.shape, PARAMS.boundary, PARAMS.coupling)
FIELD = PARAMS.field

#: The first four configurations each sweep drew **before** the conditional
#: was extracted, at the seeds named in the calls below.
GIBBS_DRAWS = np.array(
    [
        [2, 1, 0, 2, 1, 1, 2, 1, 1],
        [1, 1, 0, 1, 2, 0, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 1, 2, 2, 1, 0, 0],
    ]
)
SINGLE_SITE_DRAWS = np.array(
    [
        [1, 0, 1, 0, 1, 1, 1, 1, 1],
        [0, 0, 1, 1, 1, 2, 0, 1, 0],
        [2, 1, 1, 2, 2, 2, 0, 0, 0],
        [0, 0, 2, 0, 0, 0, 1, 2, 2],
    ]
)


def _restated(
    field_row: np.ndarray,
    neighbour_states: np.ndarray,
    couplings: list[float],
    beta: float,
) -> np.ndarray:
    """``beta * (h_ik + sum_j J_ij [k = s_j])``, written from the definition.

    Independent of the implementation: a term per state per neighbour rather
    than an accumulation into the field row.
    """
    n_states = field_row.shape[0]
    return np.array(
        [
            beta
            * (
                field_row[state]
                + sum(
                    float(coupling) * (state == neighbour_state)
                    for neighbour_state, coupling in zip(
                        neighbour_states, couplings, strict=True
                    )
                )
            )
            for state in range(n_states)
        ]
    )


#: One site's row of a compressed adjacency: the whole of a two-node
#: neighbourhood is `state[neighbours[start:stop]]`, so a test names the
#: bounds the sweeps pass.
def _row(state: np.ndarray, neighbours: list[int]) -> np.ndarray:
    """The states at ``neighbours``, in order, for the restatement."""
    return np.asarray([state[..., neighbour] for neighbour in neighbours]).T


@pytest.mark.oracle
@pytest.mark.parametrize("beta", [1.0, 0.4, 2.5])
def test_the_conditional_is_the_field_plus_its_neighbours_couplings(
    beta: float,
) -> None:
    rng = np.random.default_rng(4)
    field_row = rng.normal(size=4)
    couplings = rng.normal(size=5).tolist()
    neighbours = rng.integers(0, 9, size=5).tolist()
    state = rng.integers(0, 4, size=9)

    realized = heat_bath_log_weights(
        field_row, state, neighbours, couplings, 0, len(neighbours), beta
    )

    # A sum of at most five terms in the same order either way, so the
    # restatement reproduces it exactly rather than within a tolerance.
    assert np.array_equal(
        realized, _restated(field_row, _row(state, neighbours), couplings, beta)
    )


@pytest.mark.oracle
def test_the_block_form_is_the_single_chain_form_on_every_row() -> None:
    # The two indexings the one expression carries: scalar for a chain
    # stepped in time, advanced for chains stepped together. A difference
    # between them would put the simulator and the sampler on different
    # models while both kept sampling something.
    rng = np.random.default_rng(5)
    field_row = rng.normal(size=3)
    couplings = rng.normal(size=4).tolist()
    neighbours = rng.integers(0, 7, size=4).tolist()
    block = rng.integers(0, 3, size=(6, 7))

    realized = heat_bath_log_weights(field_row, block, neighbours, couplings, 0, 4)

    for chain in range(block.shape[0]):
        assert np.array_equal(
            realized[chain],
            heat_bath_log_weights(field_row, block[chain], neighbours, couplings, 0, 4),
        )


@pytest.mark.smoke
def test_an_isolated_site_carries_its_field_alone() -> None:
    # Degree zero is the boundary an empty row `offsets[i] == offsets[i + 1]`
    # reaches on a graph with an isolated node, and an empty sum must leave
    # the field untouched rather than raise.
    field_row = np.array([0.5, -0.25, 1.0])
    state = np.zeros(4, dtype=np.int64)

    assert np.array_equal(
        heat_bath_log_weights(field_row, state, [], [], 0, 0), field_row
    )
    assert np.array_equal(
        heat_bath_log_weights(field_row, state[None, :], [], [], 0, 0),
        field_row[None, :],
    )


@pytest.mark.analytic
def test_tempering_scales_the_whole_conditional() -> None:
    # Temperature is model scaling (`potts_mcmc.tempered`): beta multiplies
    # the field and the couplings together, so the conditional at beta is
    # the conditional at 1 times beta. At beta = 1 it is the identity, which
    # is why the multiplication is skipped rather than applied.
    rng = np.random.default_rng(6)
    field_row = rng.normal(size=3)
    couplings = rng.normal(size=4).tolist()
    neighbours = rng.integers(0, 8, size=4).tolist()
    state = rng.integers(0, 3, size=8)

    untempered = heat_bath_log_weights(field_row, state, neighbours, couplings, 0, 4)

    assert np.array_equal(
        heat_bath_log_weights(field_row, state, neighbours, couplings, 0, 4, 1.0),
        untempered,
    )
    assert np.array_equal(
        heat_bath_log_weights(field_row, state, neighbours, couplings, 0, 4, 0.25),
        untempered * 0.25,
    )


@pytest.mark.smoke
def test_the_vectorized_simulator_draws_what_it_drew_before_the_extraction() -> None:
    realized = simulate_potts(
        LATTICE, FIELD, np.random.default_rng(8), 12, burn_in=7
    ).configurations

    assert np.array_equal(realized[:4], GIBBS_DRAWS)


@pytest.mark.smoke
def test_the_sequential_sampler_draws_what_it_drew_before_the_extraction() -> None:
    rows = site_field(FIELD, LATTICE.n_nodes)

    chain = sample_potts(
        LATTICE, rows, PottsMove.SINGLE_SITE, np.random.default_rng(3), 20, 5
    )

    assert np.array_equal(chain.states[:4], SINGLE_SITE_DRAWS)
