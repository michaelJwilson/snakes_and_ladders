"""One shape draws every problem that has data, and the draw is the bits it always was (issue #829).

Two claims. The registry is complete: every model `sim.fixtures.PARAMS` names
is simulated or named with the reason it is not, and no model is both. And the
seam re-spells nothing: for each simulated model, `SIMULATORS[model](params,
default_rng(params.seed))` is **bitwise** the call its problem has always
drawn with, on the `ci` fixture --- so a caller who reaches a draw through the
seam gets the fixture's own data.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest
from snakes_and_ladders.sim.emission_mixture import simulate_emission_mixture
from snakes_and_ladders.sim.fixtures import PARAMS, fixture
from snakes_and_ladders.sim.graph import lattice_graph
from snakes_and_ladders.sim.hmm import simulate_sequences
from snakes_and_ladders.sim.mixture import simulate_mixture
from snakes_and_ladders.sim.potts import simulate_potts
from snakes_and_ladders.sim.potts_chain import simulate_chains
from snakes_and_ladders.sim.simulator import (
    NOT_SIMULATED,
    SIMULATORS,
    Simulator,
    simulate_tree,
)
from snakes_and_ladders.sim.spatio_sequential import simulate_spatio_sequential

#: One declared problem per simulated model, at the tier every test can afford.
PROBLEM_OF: dict[str, str] = {
    "jukes-cantor": "tree_search",
    "potts-chain": "potts_chain",
    "potts-lattice": "potts_lattice",
    "spatio-only": "spatio_only",
    "hidden-markov": "hmm",
    "gaussian-mixture": "mixture",
    "emission-mixture": "emission_mixture",
    "spatio-sequential": "spatio_sequential",
}


@pytest.mark.critical
@pytest.mark.smoke
def test_every_declared_model_is_simulated_or_says_why_not() -> None:
    assert set(SIMULATORS) | set(NOT_SIMULATED) == set(PARAMS)
    assert not set(SIMULATORS) & set(NOT_SIMULATED)
    assert all(reason.strip() for reason in NOT_SIMULATED.values())
    assert set(PROBLEM_OF) == set(SIMULATORS)


@pytest.mark.smoke
@pytest.mark.parametrize("model", sorted(SIMULATORS))
def test_every_simulator_has_the_one_shape(model: str) -> None:
    simulator = SIMULATORS[model]
    assert isinstance(simulator, Simulator)
    names = list(inspect.signature(simulator).parameters)
    assert names[0] == "params", names
    assert names[1] == "rng", names


def _equal(first: object, second: object) -> None:
    """Bitwise equality of two dataset records, field by field."""
    if isinstance(first, np.ndarray):
        np.testing.assert_array_equal(first, second)
        return
    assert type(first) is type(second)
    for name, value in vars(first).items():
        other = getattr(second, name)
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(value, other, err_msg=name)
        elif isinstance(value, dict):
            assert value.keys() == other.keys(), name
            for key in value:
                np.testing.assert_array_equal(
                    np.asarray(value[key]), np.asarray(other[key])
                )
        elif hasattr(value, "named_parameters"):
            for key, tensor in value.named_parameters().items():
                assert (tensor == other.named_parameters()[key]).all(), (name, key)
        else:
            assert value == other, name


@pytest.mark.oracle
@pytest.mark.parametrize("model", sorted(SIMULATORS))
def test_the_seam_draws_what_the_problems_own_call_draws(model: str) -> None:
    # The referee is the call each problem has always drawn with, on the ci
    # fixture, from a generator seeded as the fixture seeds it; the seam must
    # return the same bits, or a fixture reached through it is another fixture.
    params = fixture(PROBLEM_OF[model], "ci").params
    # The coupled model's params carry no seed: its fixture is drawn by the
    # caller's generator, so any one seed serves both sides here.
    seed = getattr(params, "seed", 0)
    through = SIMULATORS[model](params, np.random.default_rng(seed))
    rng = np.random.default_rng(seed)
    direct: object
    if model == "jukes-cantor":
        direct = simulate_tree(params, rng)
    elif model == "potts-chain":
        direct = simulate_chains(params)
    elif model == "potts-lattice":
        graph = lattice_graph(params.shape, params.boundary, params.coupling)
        direct = simulate_potts(
            graph, params.field, rng, params.n_samples, params.burn_in
        )
    elif model == "spatio-only":
        direct = simulate_potts(
            params.graph, params.field, rng, params.n_samples, params.burn_in
        )
    elif model == "hidden-markov":
        direct = simulate_sequences(params)
    elif model == "gaussian-mixture":
        direct = simulate_mixture(params)
    elif model == "emission-mixture":
        direct = simulate_emission_mixture(params)
    else:
        direct = simulate_spatio_sequential(params, rng)

    _equal(through, direct)
