"""The path enumeration, against the forward recursion and against algebra.

The enumeration is the oracle two decoders are separated by, so it cannot be
validated by a decoder. It is pinned instead against `phylo.opt.hmm`'s forward
algorithm --- which shares no code with it --- and against quantities that can
be worked out by hand.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
import torch
from phylo.likelihood.hmm_paths import (
    MAX_ENUMERABLE_PATHS,
    enumerate_hidden_paths,
    path_log_probability,
)
from phylo.opt.hmm import forward_log_likelihood
from phylo.sim.canonical import AMBIGUOUS_OBSERVATIONS, ambiguous_hmm
from phylo.sim.hmm import HmmParams


def _params(n_states: int, n_symbols: int, length: int, seed: int) -> HmmParams:
    rng = np.random.default_rng(seed)
    initial = rng.dirichlet(np.ones(n_states))
    transition = rng.dirichlet(np.ones(n_states), size=n_states)
    emission = rng.dirichlet(np.ones(n_symbols), size=n_states)
    return HmmParams(
        n_states=n_states,
        n_symbols=n_symbols,
        sequence_length=length,
        n_sequences=1,
        initial=initial,
        transition=transition,
        emission=emission,
        seed=seed,
        tolerance=1e-12,
    )


@pytest.mark.oracle
@pytest.mark.parametrize(
    ("n_states", "n_symbols", "length", "seed"),
    [(2, 2, 5, 1), (3, 2, 4, 2), (2, 4, 6, 3), (4, 3, 3, 4)],
)
def test_the_enumerated_evidence_matches_the_forward_recursion(
    n_states: int, n_symbols: int, length: int, seed: int
) -> None:
    # Summing over every path and the forward recursion compute the same
    # quantity by different routes, and neither shares code with the other.
    # Agreement to float64 is the check that the enumeration is summing the
    # model it claims to.
    params = _params(n_states, n_symbols, length, seed)
    observations = np.random.default_rng(seed).integers(0, n_symbols, size=length)

    enumerated = enumerate_hidden_paths(params, observations)
    forward = forward_log_likelihood(
        torch.from_numpy(observations[None, :]),
        torch.log(torch.from_numpy(params.initial)),
        torch.log(torch.from_numpy(params.transition)),
        torch.log(torch.from_numpy(params.emission)),
    )

    assert enumerated.log_likelihood == pytest.approx(float(forward), rel=1e-12)


@pytest.mark.mathematical
def test_the_marginals_sum_to_one_and_are_a_valid_distribution() -> None:
    params = _params(3, 3, 5, 11)
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
    params = _params(2, 3, 4, 12)
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


@pytest.mark.oracle
def test_the_viterbi_path_is_the_maximum_of_the_enumerated_joints() -> None:
    params = _params(3, 3, 4, 13)
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


@pytest.mark.mathematical
def test_the_evidence_bounds_the_best_path_from_above() -> None:
    # `P(observations)` sums over every path and `P(viterbi, observations)` is
    # one term of that sum, so the second cannot exceed the first. A decoder
    # that returned a conditional where a joint belongs would break this.
    params = _params(3, 2, 5, 14)
    observations = np.array([0, 1, 1, 0, 1])

    result = enumerate_hidden_paths(params, observations)

    assert result.viterbi_log_probability < result.log_likelihood


@pytest.mark.edge_case
def test_a_deterministic_chain_makes_both_decoders_agree() -> None:
    # The degenerate case, and the reason a disagreeing fixture had to be
    # built: when one path carries essentially all the mass the two decoders
    # coincide, and every test built on such a fixture is blind to the
    # distinction.
    params = HmmParams(
        n_states=2,
        n_symbols=2,
        sequence_length=5,
        n_sequences=1,
        initial=np.array([0.999, 0.001]),
        transition=np.array([[0.999, 0.001], [0.001, 0.999]]),
        emission=np.array([[0.999, 0.001], [0.001, 0.999]]),
        seed=0,
        tolerance=1e-12,
    )

    result = enumerate_hidden_paths(params, np.array([0, 0, 0, 0, 0]))

    assert result.decoders_agree()
    assert list(result.viterbi) == [0, 0, 0, 0, 0]


@pytest.mark.edge_case
@pytest.mark.oracle
def test_a_single_observation_is_decoded_by_the_prior_and_the_emission() -> None:
    # Length 1 has no transition, so both decoders reduce to
    # `argmax_s pi[s] B[s, y]` and the answer is arithmetic rather than a
    # recursion.
    params = _params(3, 3, 1, 15)
    observations = np.array([1])

    result = enumerate_hidden_paths(params, observations)

    expected = int(np.argmax(params.initial * params.emission[:, 1]))
    assert list(result.viterbi) == [expected]
    assert list(result.posterior_path) == [expected]
    assert result.log_likelihood == pytest.approx(
        float(np.log((params.initial * params.emission[:, 1]).sum()))
    )


@pytest.mark.structural
def test_the_ambiguous_fixture_is_within_the_cap() -> None:
    # The fixture the module exists for must be enumerable in the fast suite,
    # or the distinction it draws is untestable per pull request.
    params = ambiguous_hmm()

    assert params.n_states ** len(AMBIGUOUS_OBSERVATIONS) < MAX_ENUMERABLE_PATHS


@pytest.mark.edge_case
def test_a_sequence_too_long_to_enumerate_is_refused() -> None:
    params = _params(4, 2, 12, 16)

    # The wording is `phylo.enumeration`'s, shared with every other
    # enumerator since issue #230; what is asserted here is that this caller
    # reaches it, and that the message names the size that was too large.
    with pytest.raises(
        ValueError, match=r"refusing to enumerate .*4\*\*12 hidden paths"
    ):
        enumerate_hidden_paths(params, np.zeros(12, dtype=np.int64))


@pytest.mark.edge_case
def test_a_symbol_outside_the_alphabet_is_refused() -> None:
    params = _params(2, 2, 3, 17)

    with pytest.raises(ValueError, match=r"observations must lie in \[0, 2\)"):
        enumerate_hidden_paths(params, np.array([0, 5, 1]))


@pytest.mark.edge_case
def test_an_empty_observation_sequence_is_refused() -> None:
    params = _params(2, 2, 3, 18)

    with pytest.raises(ValueError, match="must be non-empty"):
        enumerate_hidden_paths(params, np.array([], dtype=np.int64))
