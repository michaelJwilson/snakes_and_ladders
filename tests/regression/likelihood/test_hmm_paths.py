"""The path enumeration, against the forward recursion and against algebra.

The enumeration is the oracle decoders are separated by, so it is pinned
against `opt.hmm`'s forward algorithm, which shares no code with it
(`test_message_passing.py::test_every_chain_evaluator_is_the_path_enumeration`,
#982), and here against hand-worked quantities. It also referees
`forward_backward.sample_path`, which has no module of its own (issue #734).
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from sal.emissions import CategoricalEmission
from sal.likelihood.forward_backward import forward_backward, sample_path
from sal.likelihood.hmm_paths import (
    MAX_ENUMERABLE_PATHS,
    emission_log_density,
    enumerate_hidden_paths,
    path_log_probability,
)
from sal.sample.statistics import chi_square_p_value
from sal.sim.canonical import AMBIGUOUS_OBSERVATIONS, ambiguous_hmm
from sal.sim.hmm import HmmParams

from tests._rows import every_row
from tests.regression.likelihood.conftest import CHAIN_CASES, random_hmm

#: Declared significance, as in `search/test_potts_mcmc.py`. Over 18 runs (three
#: instances x six seeds) the smallest p was 0.0077 joint, 0.0053 marginal.
SIGNIFICANCE = 0.001

#: Draws per instance. Every draw is independent --- a forward filter and a
#: backward sample is not a Markov chain --- so there is nothing to thin.
SAMPLED_DRAWS = 6000

#: Expected count a chi-square cell is not cut below, the floor
#: `search/test_gibbs.py` lumps paths at. A chain of `k ** T` paths puts most
#: of its cells under one expected count, and those cells decide the
#: statistic rather than the fit.
CELL_FLOOR = 25.0

#: Total variation between the sampled frequencies and the enumerated law,
#: over the whole path space rather than the lumped cells. Realized 0.0099,
#: 0.0443 and 0.0229 on the three instances; the bound is what 6,000 draws
#: over up to 81 paths supports, not a fitted value.
TOTAL_VARIATION = 0.06

#: Largest per-position deviation from the forward--backward marginal.
#: Realized 0.0164.
MARGINAL_DEVIATION = 0.03


@pytest.mark.analytic
def test_the_marginals_sum_to_one_and_are_a_valid_distribution() -> None:
    params = random_hmm(3, 3, 5, 11)
    observations = np.array([0, 2, 1, 1, 2])

    result = enumerate_hidden_paths(params, observations)

    assert result.posterior.shape == (5, 3)
    assert result.posterior.sum(axis=1) == pytest.approx(np.ones(5))
    assert float(result.posterior.min()) >= 0.0


@pytest.mark.oracle
def test_a_marginal_is_the_summed_joint_over_paths_through_that_state() -> None:
    # The definition, computed a second way: `P(state_t = s | y)` is the total
    # weight of paths passing through `s` at `t`, normalized. Written out here
    # rather than reusing the implementation's accumulation.
    params = random_hmm(2, 3, 4, 12)
    observations = np.array([1, 0, 2, 1])

    result = enumerate_hidden_paths(params, observations)

    site, state = 2, 1
    through = sum(
        np.exp(
            path_log_probability(params, np.array(path, dtype=np.int64), observations)
        )
        for path in itertools.product(range(2), repeat=4)
        if path[site] == state
    )
    assert result.posterior[site, state] == pytest.approx(
        through / np.exp(result.log_likelihood)
    )


def _lumped(law: np.ndarray, n_draws: int) -> list[list[int]]:
    """Paths in descending posterior order, cut wherever a cell expects :data:`CELL_FLOOR`."""
    cells: list[list[int]] = []
    current: list[int] = []
    mass = 0.0
    for index in np.argsort(-law):
        current.append(int(index))
        mass += float(law[index])
        if mass * n_draws >= CELL_FLOOR:
            cells.append(current)
            current, mass = [], 0.0
    cells[-1].extend(current)
    return cells


@pytest.mark.oracle
@pytest.mark.critical
def test_the_sampled_paths_are_drawn_from_the_enumerated_path_posterior() -> None:
    # The rung below (#734): the joint over paths refutes an exact sampler; a
    # per-site sampler passes marginals and fails the joint. 32, 81 and 64
    # paths, 6,000 draws each. Joint chi-square p 0.0127, 0.1378, 0.7805
    # (0.001 declared); marginal smallest p 0.0754, 0.0255, 0.0861, deviation
    # <= 0.0164 (0.03); TV 0.0099, 0.0443, 0.0229 (0.06). `forward_backward`
    # is the enumeration's to 2.3e-15, asserted.
    def check(n_states: int, n_symbols: int, length: int, seed: int) -> None:
        params = random_hmm(n_states, n_symbols, length, seed)
        observations = np.random.default_rng(seed).integers(0, n_symbols, size=length)
        enumerated = enumerate_hidden_paths(params, observations)

        paths = list(itertools.product(range(n_states), repeat=length))
        joint = np.array(
            [
                path_log_probability(
                    params, np.array(path, dtype=np.int64), observations
                )
                for path in paths
            ]
        )
        law = np.exp(joint - enumerated.log_likelihood)
        index = {path: position for position, path in enumerate(paths)}

        log_density = emission_log_density(params, observations)
        log_initial, log_transition = np.log(params.initial), np.log(params.transition)
        run = forward_backward(log_density, log_initial, log_transition)
        assert np.abs(run.posterior - enumerated.posterior).max() < 1e-14

        rng = np.random.default_rng(734)
        drawn = np.zeros(len(paths))
        marginal = np.zeros_like(enumerated.posterior)
        for _ in range(SAMPLED_DRAWS):
            path = sample_path(log_density, log_initial, log_transition, rng)
            drawn[index[tuple(int(state) for state in path)]] += 1
            marginal[np.arange(length), path] += 1

        cells = _lumped(law, SAMPLED_DRAWS)
        assert (
            chi_square_p_value(
                np.array([drawn[cell].sum() for cell in cells]),
                np.array([law[cell].sum() for cell in cells]) * SAMPLED_DRAWS,
            )
            > SIGNIFICANCE
        )
        assert 0.5 * np.abs(drawn / SAMPLED_DRAWS - law).sum() < TOTAL_VARIATION

        frequency = marginal / SAMPLED_DRAWS
        assert np.abs(frequency - run.posterior).max() < MARGINAL_DEVIATION
        for position in range(length):
            assert (
                chi_square_p_value(
                    marginal[position], SAMPLED_DRAWS * run.posterior[position]
                )
                > SIGNIFICANCE
            )

    every_row(CHAIN_CASES[:3], check)


@pytest.mark.oracle
def test_the_viterbi_path_is_the_maximum_of_the_enumerated_joints() -> None:
    params = random_hmm(3, 3, 4, 13)
    observations = np.array([2, 0, 1, 2])

    result = enumerate_hidden_paths(params, observations)

    highest = max(
        path_log_probability(params, np.array(path, dtype=np.int64), observations)
        for path in itertools.product(range(3), repeat=4)
    )
    assert result.viterbi_log_probability == pytest.approx(highest)
    assert result.viterbi_log_probability == pytest.approx(
        path_log_probability(params, result.viterbi, observations)
    )


@pytest.mark.analytic
def test_the_evidence_bounds_the_best_path_from_above() -> None:
    # `P(observations)` sums over every path and `P(viterbi, observations)` is
    # one term of that sum, so the second cannot exceed the first. A decoder
    # that returned a conditional where a joint belongs would break this.
    params = random_hmm(3, 2, 5, 14)
    observations = np.array([0, 1, 1, 0, 1])

    result = enumerate_hidden_paths(params, observations)

    assert result.viterbi_log_probability < result.log_likelihood


@pytest.mark.smoke
def test_a_deterministic_chain_makes_both_decoders_agree() -> None:
    # The degenerate case, and the reason a disagreeing fixture had to be
    # built: when one path carries essentially all the mass the two decoders
    # coincide, and every test built on such a fixture is blind to the
    # distinction.
    params = HmmParams(
        n_states=2,
        lengths=(5,) * 1,
        initial=np.array([0.999, 0.001]),
        transition=np.array([[0.999, 0.001], [0.001, 0.999]]),
        emissions=CategoricalEmission(np.array([[0.999, 0.001], [0.001, 0.999]])),
        seed=0,
        tolerance=1e-12,
    )

    result = enumerate_hidden_paths(params, np.array([0, 0, 0, 0, 0]))

    assert result.decoders_agree()
    assert list(result.viterbi) == [0, 0, 0, 0, 0]


@pytest.mark.oracle
def test_a_single_observation_is_decoded_by_the_prior_and_the_emission() -> None:
    # Length 1 has no transition, so both decoders reduce to
    # `argmax_s pi[s] B[s, y]` and the answer is arithmetic rather than a
    # recursion.
    params = random_hmm(3, 3, 1, 15)
    observations = np.array([1])

    result = enumerate_hidden_paths(params, observations)

    expected = int(np.argmax(params.initial * params.emission[:, 1]))
    assert list(result.viterbi) == [expected]
    assert list(result.posterior_path) == [expected]
    assert result.log_likelihood == pytest.approx(
        float(np.log((params.initial * params.emission[:, 1]).sum()))
    )


@pytest.mark.smoke
def test_the_ambiguous_fixture_is_within_the_cap() -> None:
    # The fixture the module exists for must be enumerable in the fast suite,
    # or the distinction it draws is untestable per pull request.
    params = ambiguous_hmm()

    assert params.n_states ** len(AMBIGUOUS_OBSERVATIONS) < MAX_ENUMERABLE_PATHS


@pytest.mark.smoke
def test_a_sequence_too_long_to_enumerate_is_refused() -> None:
    params = random_hmm(4, 2, 12, 16)

    # The wording is `sal.enumeration`'s, shared with every other
    # enumerator since issue #230; what is asserted here is that this caller
    # reaches it, and that the message names the size that was too large.
    with pytest.raises(
        ValueError, match=r"refusing to enumerate .*4\*\*12 hidden paths"
    ):
        enumerate_hidden_paths(params, np.zeros(12, dtype=np.int64))


@pytest.mark.smoke
def test_a_symbol_outside_the_alphabet_is_refused() -> None:
    params = random_hmm(2, 2, 3, 17)

    with pytest.raises(ValueError, match=r"observations must lie in \[0, 2\)"):
        enumerate_hidden_paths(params, np.array([0, 5, 1]))


@pytest.mark.smoke
def test_an_empty_observation_sequence_is_refused() -> None:
    params = random_hmm(2, 2, 3, 18)

    with pytest.raises(ValueError, match="must be non-empty"):
        enumerate_hidden_paths(params, np.array([], dtype=np.int64))
