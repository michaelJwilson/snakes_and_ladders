"""The one Potts energy, against the oracle it replaced four loops with.

`sim.potts.energies` is what `sample.potts_mcmc`, `search.alpha_expansion`
and `search.maxflow` each scored a labelling with before issue #277.
`likelihood.potts.log_weights` is deliberately **not** merged into it: it is
the independent answer this is refereed by, and an implementation that
computes its own referee is no longer refereed.

The two sum the edge terms in different orders --- one gather and a dot
product here, a term per edge there --- so they agree to a relative `1e-12`
and not bitwise (issue #341). The four instances are the fixtures the
consolidation was validated on: a chain, a lattice, a glass with couplings of
both signs, and a lattice whose field varies by site.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose
from snakes_and_ladders.fixtures import load_params
from snakes_and_ladders.likelihood.potts import log_weights
from snakes_and_ladders.sim.fixtures import FIXTURES_DIR, fixture
from snakes_and_ladders.sim.graph import BoundaryCondition, PottsGraph, lattice_graph
from snakes_and_ladders.sim.potts import energies
from snakes_and_ladders.sim.potts_chain import PottsParams

#: What two exact routes over the same weights may differ by: the last bits
#: of a `float64` reduction, which belong to the host and not to the tree.
_EXACT = 1e-12


def _instances() -> dict[str, tuple[PottsGraph, np.ndarray, int]]:
    """The declared instances, as ``(graph, field, n_states)``."""
    chain = load_params(FIXTURES_DIR / "potts_chain/ci.yaml", PottsParams)
    lattice = fixture("potts_lattice", "ci").params
    glass_params = fixture("planted_glass", "ci").params
    (frustration,) = glass_params.glass_frustrations
    glass = glass_params.glass(frustration, np.random.default_rng(glass_params.seed))
    spatio = fixture("spatio_only", "ci").params
    return {
        "potts_chain": (
            lattice_graph(
                (chain.chain_length,), BoundaryCondition.OPEN, chain.coupling
            ),
            chain.field,
            chain.n_states,
        ),
        "potts_lattice": (
            lattice_graph(lattice.shape, lattice.boundary, lattice.coupling),
            lattice.field,
            lattice.n_states,
        ),
        "planted_glass": (glass.graph, np.zeros(2), 2),
        "spatio_only": (spatio.graph, spatio.field, spatio.n_classes),
    }


INSTANCES = _instances()


@pytest.mark.oracle
@pytest.mark.parametrize("name", sorted(INSTANCES))
def test_the_energy_is_the_negated_log_weight_of_the_configuration(name: str) -> None:
    graph, field, n_states = INSTANCES[name]
    states = np.random.default_rng(11).integers(0, n_states, size=(64, graph.n_nodes))

    assert_allclose(
        energies(graph, field, states),
        -log_weights(graph, field, states),
        rtol=_EXACT,
        atol=0.0,
    )


@pytest.mark.oracle
@pytest.mark.parametrize("name", sorted(INSTANCES))
def test_one_labelling_is_the_block_of_one(name: str) -> None:
    # The reduction the single-labelling callers take: `state[None]` and
    # element zero. A block scored row by row and a block scored at once
    # must agree bitwise, or `sim.potts.energy` and the sampler's
    # `energies` are two functions again.
    graph, field, n_states = INSTANCES[name]
    states = np.random.default_rng(12).integers(0, n_states, size=(8, graph.n_nodes))

    block = energies(graph, field, states)

    assert np.array_equal(
        block, np.array([energies(graph, field, state[None])[0] for state in states])
    )


@pytest.mark.smoke
def test_a_configuration_of_the_wrong_width_is_refused() -> None:
    # Broadcast rather than refused, this returns energies for a different
    # model -- the failure `log_weights` refuses the same way.
    graph = lattice_graph((4,), BoundaryCondition.OPEN, 0.5)

    with pytest.raises(ValueError, match="columns for a graph of 4 nodes"):
        energies(graph, np.zeros(2), np.zeros((3, 5), dtype=np.int64))
