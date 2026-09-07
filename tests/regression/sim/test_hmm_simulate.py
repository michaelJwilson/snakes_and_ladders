"""Regression tests for the HMM data generator.

Validated against brute-force enumeration and the transition matrix's own
stationary distribution -- never against ``phylo.opt``'s fitted likelihood,
per ``sim/CLAUDE.md``'s "validate against the analytic result" rule. The
generator draws a hidden path and an observation sequence jointly; every
check here is a property of that joint distribution, computed independently
of the code under test.
"""

from __future__ import annotations

from dataclasses import replace
from itertools import product
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose
from phylo.sim.hmm import load_hmm_params, simulate_sequences

from tests._fixtures import FIXTURES_DIR

FIXTURE = FIXTURES_DIR / "hmm_params.yaml"

# Brute force is O(n_states ** length); 3 ** 7 = 2187 paths, matching the
# scale tests/regression/opt/test_opt_hmm.py already uses for the same
# fixture, and the recursion under test is length-agnostic so a longer
# window would not exercise anything new.
_ENUMERATION_LENGTH = 7

# A single-position marginal draws only n_sequences = 600 samples, against
# 9000 (n_sequences * sequence_length) for the whole-sequence checks the
# fixture's own tolerance is sized for. Worst-case standard error at p = 0.5
# is sqrt(0.25 / 600) = 0.0204; three of those is 0.061.
_PER_POSITION_TOLERANCE = 0.06

# Long enough for the fixture's transition matrix to mix to its stationary
# distribution: measured to agree to 1e-5 by step 30 (eigenvalue decay of
# the second-largest |lambda|), so a 400-step chain's occupancy carries
# negligible transient bias from the non-stationary start.
_MIXING_LENGTH = 400


def _enumerate_paths(
    initial: np.ndarray, transition: np.ndarray, length: int
) -> tuple[list[tuple[int, ...]], np.ndarray]:
    """Every hidden path of ``length`` steps, with its exact prior probability."""
    n_states = initial.shape[0]
    paths = list(product(range(n_states), repeat=length))
    probabilities = np.empty(len(paths))
    for index, path in enumerate(paths):
        probability = initial[path[0]]
        for t in range(1, length):
            probability *= transition[path[t - 1], path[t]]
        probabilities[index] = probability
    return paths, probabilities


@pytest.mark.structural
def test_simulated_dataset_has_the_declared_shape_and_alphabet() -> None:
    params = load_hmm_params(FIXTURE)
    dataset = simulate_sequences(params)
    assert dataset.observations.shape == (params.n_sequences, params.sequence_length)
    assert dataset.states.shape == (params.n_sequences, params.sequence_length)
    assert set(np.unique(dataset.observations)) <= set(range(params.n_symbols))
    assert set(np.unique(dataset.states)) <= set(range(params.n_states))


@pytest.mark.structural
def test_simulation_is_reproducible_from_the_seed() -> None:
    params = load_hmm_params(FIXTURE)
    first = simulate_sequences(params)
    second = simulate_sequences(params)
    assert np.array_equal(first.observations, second.observations)
    assert np.array_equal(first.states, second.states)


@pytest.mark.oracle
@pytest.mark.simulated_truth
def test_simulated_symbol_frequencies_match_the_analytic_marginal() -> None:
    # The stationary-free marginal is exact: average the emission rows over
    # the hidden-state distribution at each step, which is the initial
    # distribution pushed through the transition matrix. Computed in closed
    # form, so this pins the simulator against the model rather than against
    # a second simulation.
    params = load_hmm_params(FIXTURE)
    dataset = simulate_sequences(params)

    state = params.initial
    expected = np.zeros(params.n_symbols)
    for _ in range(params.sequence_length):
        expected += state @ params.emission
        state = state @ params.transition
    expected /= params.sequence_length

    observed = (
        np.bincount(dataset.observations.ravel(), minlength=params.n_symbols)
        / dataset.observations.size
    )
    assert_allclose(observed, expected, atol=params.tolerance)


@pytest.mark.oracle
@pytest.mark.simulated_truth
def test_realized_state_occupancy_matches_the_stationary_distribution() -> None:
    # The exact per-position marginals below do not pin this: at the
    # fixture's own sequence_length = 15, the chain has not mixed away from
    # its non-stationary initial distribution (the length-15 time-averaged
    # marginal is [0.377, 0.423, 0.2], not the stationary [0.356, 0.444,
    # 0.2] -- a real transient, not sampling noise). Occupancy is drawn at
    # _MIXING_LENGTH instead, long enough for that transient to have decayed,
    # so what remains to check is the stationary distribution itself: the
    # left eigenvector of the transition matrix A at eigenvalue 1.
    params = load_hmm_params(FIXTURE)
    dataset = simulate_sequences(replace(params, sequence_length=_MIXING_LENGTH))

    eigenvalues, eigenvectors = np.linalg.eig(params.transition.T)
    stationary_index = int(np.argmin(np.abs(eigenvalues - 1.0)))
    stationary = np.real(eigenvectors[:, stationary_index])
    stationary /= stationary.sum()

    occupancy = (
        np.bincount(dataset.states.ravel(), minlength=params.n_states)
        / dataset.states.size
    )
    assert_allclose(occupancy, stationary, atol=params.tolerance)


@pytest.mark.oracle
def test_realized_state_marginals_match_brute_force_enumeration() -> None:
    # The exact marginal p(state_t = s) at each of the first _ENUMERATION_LENGTH
    # positions, summed over every one of n_states ** length hidden paths --
    # independent of the transition matrix's stationary distribution, and
    # exact at any length rather than only in the long-run limit the
    # occupancy check above relies on.
    params = load_hmm_params(FIXTURE)
    length = _ENUMERATION_LENGTH
    paths, probabilities = _enumerate_paths(params.initial, params.transition, length)

    expected = np.zeros((length, params.n_states))
    for path, probability in zip(paths, probabilities, strict=True):
        for t, state in enumerate(path):
            expected[t, state] += probability

    dataset = simulate_sequences(params)
    prefix = dataset.states[:, :length]
    for t in range(length):
        observed = (
            np.bincount(prefix[:, t], minlength=params.n_states) / prefix.shape[0]
        )
        assert_allclose(observed, expected[t], atol=_PER_POSITION_TOLERANCE)


@pytest.mark.oracle
def test_marginal_emission_distribution_matches_brute_force_enumeration() -> None:
    # The exact marginal emission distribution at each position, from the
    # same path enumeration as the state-marginal check, pushed through the
    # emission matrix -- the two-step (state, then symbol) counterpart of
    # the whole-sequence-averaged check above, at single-position
    # resolution.
    params = load_hmm_params(FIXTURE)
    length = _ENUMERATION_LENGTH
    paths, probabilities = _enumerate_paths(params.initial, params.transition, length)

    expected = np.zeros((length, params.n_symbols))
    for path, probability in zip(paths, probabilities, strict=True):
        for t, state in enumerate(path):
            expected[t] += probability * params.emission[state]

    dataset = simulate_sequences(params)
    prefix = dataset.observations[:, :length]
    for t in range(length):
        observed = (
            np.bincount(prefix[:, t], minlength=params.n_symbols) / prefix.shape[0]
        )
        assert_allclose(observed, expected[t], atol=_PER_POSITION_TOLERANCE)


@pytest.mark.oracle
def test_realized_path_posterior_matches_brute_force_enumeration() -> None:
    # The exact posterior p(state_t = s | observations) for one realized
    # observation sequence, computed by brute-force enumeration over every
    # hidden path consistent with the model (no forward-backward recursion
    # involved). Estimated from the simulator's own draws by self-normalized
    # importance sampling: every simulated path is reweighted by the
    # likelihood it assigns the *target* sequence's emissions -- a quantity
    # computable from the path and the (known) emission matrix alone, not
    # from what that path actually emitted -- which is an unbiased estimator
    # of the posterior in the number of paths sampled (Koller & Friedman,
    # ch. 12). The check is at one interior position, where both the
    # forward and backward halves of the recursion matter.
    params = load_hmm_params(FIXTURE)
    length = _ENUMERATION_LENGTH
    position = length // 2
    dataset = simulate_sequences(params)
    target = dataset.observations[0, :length]

    paths, prior = _enumerate_paths(params.initial, params.transition, length)
    likelihood = np.array(
        [
            np.prod([params.emission[state, target[t]] for t, state in enumerate(path)])
            for path in paths
        ]
    )
    joint = prior * likelihood
    exact_posterior = np.zeros(params.n_states)
    for path, weight in zip(paths, joint, strict=True):
        exact_posterior[path[position]] += weight
    exact_posterior /= joint.sum()

    sampled_paths = dataset.states[:, :length]
    weights = np.ones(sampled_paths.shape[0])
    for t in range(length):
        weights *= params.emission[sampled_paths[:, t], target[t]]
    estimated_posterior = np.array(
        [
            weights[sampled_paths[:, position] == state].sum()
            for state in range(params.n_states)
        ]
    )
    estimated_posterior /= weights.sum()

    # Self-normalized importance sampling with n_sequences proposals; the
    # tolerance is the fixture's own, widened by an effective-sample-size
    # factor since the emission-likelihood weights concentrate the sample
    # (measured effective sample size at this fixture's seed and size is
    # in the low hundreds, against 600 proposals).
    assert_allclose(estimated_posterior, exact_posterior, atol=5 * params.tolerance)


@pytest.mark.edge_case
@pytest.mark.parametrize(
    ("replace", "with_", "message"),
    [
        ("n_states: 3", "n_states: 1", "n_states must be >= 2"),
        ("n_symbols: 4", "n_symbols: 1", "n_symbols must be >= 2"),
        ("sequence_length: 15", "sequence_length: 1", "sequence_length must be >= 2"),
        ("initial: [0.5, 0.3, 0.2]", "initial: [0.5, 0.3]", "initial has shape"),
        ("initial: [0.5, 0.3, 0.2]", "initial: [0.5, 0.3, 0.9]", "initial rows sum to"),
    ],
)
def test_a_malformed_fixture_is_refused(
    replace: str, with_: str, message: str, tmp_path: Path
) -> None:
    path = tmp_path / "hmm.yaml"
    path.write_text(FIXTURE.read_text().replace(replace, with_))
    with pytest.raises(ValueError, match=message):
        load_hmm_params(path)


@pytest.mark.edge_case
def test_a_missing_field_is_refused(tmp_path: Path) -> None:
    text = "\n".join(
        line
        for line in FIXTURE.read_text().splitlines()
        if not line.startswith("seed:")
    )
    path = tmp_path / "hmm.yaml"
    path.write_text(text)
    with pytest.raises(ValueError, match="missing required field"):
        load_hmm_params(path)
