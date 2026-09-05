"""The Potts landscape: its physics, its oracle, and the two gauges in it.

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
import torch
from numpy.testing import assert_allclose
from phylo.learn.policy import LinearPolicy
from phylo.learn.potts import (
    PottsLandscape,
    enumerate_configurations,
    optimum,
)
from phylo.learn.rollout import greedy_rollout
from phylo.opt.potts import load_potts_params

from tests._fixtures import FIXTURES_DIR

FIELD = np.array([0.4, -0.1, -0.3])
FIXTURE = FIXTURES_DIR / "potts_params.yaml"


def _landscape(chain_length: int = 4) -> PottsLandscape:
    return PottsLandscape(coupling=0.75, field=FIELD, chain_length=chain_length)


# --- the model ------------------------------------------------------------


@pytest.mark.oracle
def test_energy_matches_its_definition_term_by_term() -> None:
    # E(s) = J * (agreeing adjacent pairs) + sum of the field at each site,
    # written out here independently of the implementation's loop.
    landscape = _landscape()
    state = (2, 2, 0, 1)
    expected = 0.75 * 1 + (FIELD[2] + FIELD[2] + FIELD[0] + FIELD[1])
    assert_allclose(landscape.energy(state), expected, atol=1e-12)


@pytest.mark.oracle
def test_the_local_reward_equals_a_full_energy_difference() -> None:
    # The step function updates two bonds and one site rather than
    # re-evaluating E. Every state, every action: at this size the claim can
    # be checked exhaustively instead of sampled.
    landscape = _landscape()
    for state in enumerate_configurations(3, 4):
        base = landscape.energy(state)
        for action in landscape.actions(state):
            successor, reward = landscape.step(state, action)
            assert_allclose(reward, landscape.energy(successor) - base, atol=1e-12)


@pytest.mark.mathematical
def test_the_features_span_the_reward_exactly() -> None:
    # delta_energy = J * agreement_delta + field_delta, which is why the
    # greedy searcher is inside the policy class. If this ever stopped
    # holding, `greedy_weights` would silently stop being greedy.
    landscape = _landscape()
    weights = landscape.greedy_weights()
    for state in itertools.islice(enumerate_configurations(3, 4), 20):
        actions = landscape.actions(state)
        scores = landscape.features(state, actions) @ weights
        rewards = [landscape.step(state, action)[1] for action in actions]
        assert_allclose(scores.numpy(), rewards, atol=1e-12)


@pytest.mark.oracle
def test_the_neighbourhood_has_one_flip_per_site_and_alternative_state() -> None:
    landscape = _landscape(chain_length=5)
    state = (0, 1, 2, 0, 1)
    actions = landscape.actions(state)
    assert len(actions) == 5 * (3 - 1)
    assert len(set(actions)) == len(actions)
    assert all(value != state[site] for site, value in actions)


@pytest.mark.structural
def test_a_terminal_state_is_one_no_flip_improves() -> None:
    landscape = _landscape()
    for state in enumerate_configurations(3, 4):
        improvable = any(
            landscape.step(state, action)[1] > 0.0
            for action in landscape.actions(state)
        )
        assert landscape.is_terminal(state) is not improvable


# --- the gauge ------------------------------------------------------------


@pytest.mark.mathematical
def test_shifting_the_field_leaves_every_reward_unchanged() -> None:
    # h and h + c are the same model, and `phylo.opt.potts` has to fix that
    # gauge because a fitted field would otherwise have no value. Here it
    # costs nothing: a shift moves every configuration's energy by L * c, so
    # every *difference* is untouched. Recorded because the fixture is
    # canonicalized on load and a reader is entitled to know whether the
    # landscape depended on it. It does not.
    base = _landscape()
    shifted = PottsLandscape(0.75, FIELD + 1.7, 4)
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
    landscape = _landscape()
    state, energy = optimum(landscape)
    energies = [
        landscape.energy(candidate) for candidate in enumerate_configurations(3, 4)
    ]
    assert_allclose(energy, max(energies), atol=1e-12)
    assert_allclose(landscape.energy(state), energy, atol=1e-12)
    # The optimum must be a fixed point of the search, or "reached the
    # optimum" and "stopped improving" would be different events.
    assert landscape.is_terminal(state)


@pytest.mark.structural
def test_the_landscape_is_hard_enough_to_be_worth_searching() -> None:
    # Measured: greedy hill climbing stalls below the global optimum from 16
    # of the 81 starting configurations. A landscape greedy always solved
    # would make every comparison against it vacuous -- the trap issue #128
    # found in the 6-taxon tree fixture, where the optimum led the runner-up
    # by 41.6 log units and both move sets reached it every time. Asserting
    # only that *some* start stalls: the count is the measurement, and
    # pinning it would break on any harmless change to tie-breaking.
    landscape = _landscape()
    best = optimum(landscape)[1]
    stalled = sum(
        abs(landscape.energy(greedy_rollout(landscape, start, 50).states[-1]) - best)
        > 1e-9
        for start in enumerate_configurations(3, 4)
    )
    assert stalled > 0


# --- the greedy searcher is a member of the policy class ------------------


@pytest.mark.oracle
def test_the_greedy_weights_reproduce_the_greedy_searcher() -> None:
    # Not a convenience: it is what makes "the agent beat hill climbing" a
    # statement about learning rather than about two unrelated algorithms.
    landscape = _landscape()
    policy = LinearPolicy(2)
    policy.set_weights(landscape.greedy_weights() * 50.0)
    for start in itertools.islice(enumerate_configurations(3, 4), 15):
        expected = greedy_rollout(landscape, start, max_steps=20)
        state = start
        taken: list[tuple[int, int]] = []
        while not landscape.is_terminal(state) and len(taken) < 20:
            actions = landscape.actions(state)
            index = policy.greedy(landscape.features(state, actions))
            taken.append(actions[index])
            state, _ = landscape.step(state, actions[index])
        assert tuple(taken) == expected.actions


# --- construction from the shared fixture ---------------------------------


@pytest.mark.structural
def test_the_fixture_yaml_builds_the_same_landscape() -> None:
    # One model, two roles: the yaml that supplies `phylo.opt`'s reference
    # objective read as a search problem instead of a fitting problem.
    params = load_potts_params(FIXTURE)
    landscape = PottsLandscape.from_params(params)
    assert landscape.n_states == params.n_states
    assert landscape.chain_length == params.chain_length
    assert_allclose(
        landscape.greedy_weights().numpy(), [params.coupling, 1.0], atol=1e-12
    )


@pytest.mark.edge_case
@pytest.mark.parametrize(
    ("coupling", "field", "chain_length", "message"),
    [
        (0.5, np.array([0.1]), 4, "field must be 1-D"),
        (0.5, np.zeros((2, 2)), 4, "field must be 1-D"),
        (0.5, FIELD, 1, "chain_length must be >= 2"),
    ],
)
def test_an_unusable_landscape_is_rejected(
    coupling: float, field: np.ndarray, chain_length: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        PottsLandscape(coupling, field, chain_length)


@pytest.mark.structural
def test_reset_draws_a_configuration_of_the_right_shape() -> None:
    landscape = _landscape(chain_length=6)
    state = landscape.reset(np.random.default_rng(0))
    assert len(state) == 6
    assert all(0 <= value < 3 for value in state)
    assert state == landscape.reset(np.random.default_rng(0))


@pytest.mark.structural
def test_features_have_one_row_per_action() -> None:
    landscape = _landscape()
    state = (0, 1, 2, 0)
    actions = landscape.actions(state)
    features = landscape.features(state, actions)
    assert features.shape == (len(actions), landscape.n_features())
    assert features.dtype == torch.float64
