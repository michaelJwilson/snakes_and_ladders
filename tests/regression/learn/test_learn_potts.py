"""The Potts environment: its physics, its oracle, and the two gauges in it.

The environment is only usable as a reference instance if its reward is the
energy difference it claims. So the ``O(1)`` local update is pinned
against a full re-evaluation of ``E``, the terminal condition against an
exhaustive scan of the neighbourhood, and the optimum against enumeration of
every configuration -- brute force in all three cases, which is what root
``CLAUDE.md`` asks an expected value to be pinned to.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from numpy.testing import assert_allclose
from sal.fixtures import load_params
from sal.learn.policy import LinearPolicy
from sal.learn.potts import (
    PottsEnvironment,
    enumerate_configurations,
    optimum,
)
from sal.learn.rollout import greedy_rollout
from sal.sim.potts_chain import PottsParams

from tests._fixtures import FIXTURES_DIR
from tests.regression.learn.conftest import FIELD, potts_environment

FIXTURE = FIXTURES_DIR / "potts_chain/ci.yaml"


# --- the model ------------------------------------------------------------


@pytest.mark.oracle
def test_energy_matches_its_definition_term_by_term() -> None:
    # E(s) = J * (agreeing adjacent pairs) + sum of the field at each site,
    # written out here independently of the implementation's loop.
    environment = potts_environment()
    state = (2, 2, 0, 1)
    expected = 0.75 * 1 + (FIELD[2] + FIELD[2] + FIELD[0] + FIELD[1])
    assert_allclose(environment.log_weight(state), expected, atol=1e-12)


@pytest.mark.oracle
def test_the_local_reward_equals_a_full_energy_difference() -> None:
    # The step function updates two bonds and one site rather than
    # re-evaluating E. Every state, every action: at this size the claim can
    # be checked exhaustively instead of sampled.
    environment = potts_environment()
    for state in enumerate_configurations(3, 4):
        base = environment.log_weight(state)
        for action in environment.actions(state):
            successor, reward = environment.step(state, action)
            assert_allclose(
                reward, environment.log_weight(successor) - base, atol=1e-12
            )


@pytest.mark.analytic
def test_the_features_span_the_reward_exactly() -> None:
    # delta_energy = J * agreement_delta + field_delta, which is why the
    # greedy searcher is inside the policy class. If this ever stopped
    # holding, `greedy_weights` would silently stop being greedy.
    environment = potts_environment()
    weights = environment.greedy_weights()
    for state in itertools.islice(enumerate_configurations(3, 4), 20):
        actions = environment.actions(state)
        scores = environment.features(state, actions) @ weights
        rewards = [environment.step(state, action)[1] for action in actions]
        assert_allclose(scores, rewards, atol=1e-12)


@pytest.mark.oracle
def test_the_neighbourhood_has_one_flip_per_site_and_alternative_state() -> None:
    environment = potts_environment(chain_length=5)
    state = (0, 1, 2, 0, 1)
    actions = environment.actions(state)
    assert len(actions) == 5 * (3 - 1)
    assert len(set(actions)) == len(actions)
    assert all(value != state[site] for site, value in actions)


@pytest.mark.smoke
def test_a_terminal_state_is_one_no_flip_improves() -> None:
    environment = potts_environment()
    for state in enumerate_configurations(3, 4):
        improvable = any(
            environment.step(state, action)[1] > 0.0
            for action in environment.actions(state)
        )
        assert environment.is_terminal(state) is not improvable


# --- the gauge ------------------------------------------------------------


@pytest.mark.analytic
def test_shifting_the_field_leaves_every_reward_unchanged() -> None:
    # h and h + c are one model; a shift moves every energy by L * c, so every
    # difference, and the environment, is untouched by canonicalization.
    base = potts_environment()
    shifted = PottsEnvironment(0.75, FIELD + 1.7, 4)
    for state in itertools.islice(enumerate_configurations(3, 4), 25):
        for action in base.actions(state):
            assert_allclose(
                base.step(state, action)[1], shifted.step(state, action)[1], atol=1e-12
            )


# --- the oracle -----------------------------------------------------------


@pytest.mark.oracle
def test_enumeration_produces_every_configuration_exactly_once() -> None:
    produced = list(enumerate_configurations(3, 4))
    assert len(produced) == 3**4
    assert len(set(produced)) == len(produced)


@pytest.mark.oracle
def test_the_optimum_is_the_best_of_every_configuration() -> None:
    environment = potts_environment()
    state, energy = optimum(environment)
    energies = [
        environment.log_weight(candidate)
        for candidate in enumerate_configurations(3, 4)
    ]
    assert_allclose(energy, max(energies), atol=1e-12)
    assert_allclose(environment.log_weight(state), energy, atol=1e-12)
    # The optimum must be a fixed point of the search, or "reached the
    # optimum" and "stopped improving" would be different events.
    assert environment.is_terminal(state)


@pytest.mark.smoke
def test_the_environment_is_hard_enough_to_be_worth_searching() -> None:
    # Measured: greedy stalls below the optimum from 16 of 81 starts. A fixture
    # greedy always solves makes comparisons vacuous (#128: a 41.6 log-unit
    # lead). Asserted as "some start stalls"; the count moves with tie-breaking.
    environment = potts_environment()
    best = optimum(environment)[1]
    stalled = sum(
        abs(
            environment.log_weight(
                greedy_rollout(environment, start=start, max_steps=50).states[-1]
            )
            - best
        )
        > 1e-9
        for start in enumerate_configurations(3, 4)
    )
    assert stalled > 0


# --- the greedy searcher is a member of the policy class ------------------


@pytest.mark.oracle
def test_the_greedy_weights_reproduce_the_greedy_searcher() -> None:
    # Not a convenience: it is what makes "the agent beat hill climbing" a
    # statement about learning rather than about two unrelated algorithms.
    environment = potts_environment()
    policy = LinearPolicy(2)
    policy.set_weights(environment.greedy_weights() * 50.0)
    for start in itertools.islice(enumerate_configurations(3, 4), 15):
        expected = greedy_rollout(environment, start=start, max_steps=20)
        state = start
        taken: list[tuple[int, int]] = []
        while not environment.is_terminal(state) and len(taken) < 20:
            actions = environment.actions(state)
            index = policy.greedy(environment.features(state, actions))
            taken.append(actions[index])
            state, _ = environment.step(state, actions[index])
        assert tuple(taken) == expected.actions


# --- construction from the shared fixture ---------------------------------


@pytest.mark.smoke
def test_the_fixture_yaml_builds_the_same_environment() -> None:
    # One model, two roles: the yaml that supplies `sal.opt`'s reference
    # objective read as a search problem instead of a fitting problem.
    params = load_params(FIXTURE, PottsParams)
    environment = PottsEnvironment.from_params(params)
    assert environment.n_states == params.n_states
    assert environment.chain_length == params.chain_length
    assert_allclose(environment.greedy_weights(), [params.coupling, 1.0], atol=1e-12)


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("coupling", "field", "chain_length", "message"),
    [
        (0.5, np.array([0.1]), 4, "field must be 1-D"),
        (0.5, np.zeros((2, 2)), 4, "field must be 1-D"),
        (0.5, FIELD, 1, "chain_length must be >= 2"),
    ],
)
def test_an_unusable_environment_is_rejected(
    coupling: float, field: np.ndarray, chain_length: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        PottsEnvironment(coupling, field, chain_length)


@pytest.mark.smoke
def test_reset_draws_a_configuration_of_the_right_shape() -> None:
    environment = potts_environment(chain_length=6)
    state = environment.reset(np.random.default_rng(0))
    assert len(state) == 6
    assert all(0 <= value < 3 for value in state)
    assert state == environment.reset(np.random.default_rng(0))


@pytest.mark.smoke
def test_features_have_one_row_per_action() -> None:
    environment = potts_environment()
    state = (0, 1, 2, 0)
    actions = environment.actions(state)
    features = environment.features(state, actions)
    assert features.shape == (len(actions), environment.n_features())
    assert features.dtype == np.float64


@pytest.mark.oracle
def test_the_vectorized_features_are_the_scalar_deltas_exactly() -> None:
    # `features` is one NumPy pass since #264; `_deltas` is the scalar oracle.
    # Exact: integer counts and one subtraction; 0.0 over 200 random states.
    environment = PottsEnvironment(0.75, np.array([0.4, -0.1, -0.3]), chain_length=5)
    rng = np.random.default_rng(0)

    for _ in range(50):
        state = environment.reset(rng)
        actions = environment.actions(state)
        fast = environment.features(state, actions)
        slow = np.array([environment._deltas(state, action) for action in actions])

        assert fast.shape == (len(actions), 2)
        assert np.array_equal(fast, slow)
