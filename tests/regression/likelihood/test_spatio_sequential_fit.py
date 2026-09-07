"""The E step, the field and the joint of the coupled model, against the enumeration oracle.

Forward--backward is pinned against the path enumeration on the plain HMM and
against the conditional state posterior of #300 on the coupled fixture; the
field is checked against its definition; the labelled joint against the
enumeration's per-labelling term; and the M-step identity of
the M-step identity of the textbook's coupled-model section against autograd through the forward recursion, which
shares no code with the posterior-weighted score (issue #306).
"""

from __future__ import annotations

from dataclasses import replace
from itertools import product

import numpy as np
import pytest
import torch
from snakes_and_ladders.emissions import CategoricalEmission, GaussianEmission
from snakes_and_ladders.likelihood.forward_backward import forward_backward, sample_path
from snakes_and_ladders.likelihood.hmm_paths import (
    emission_log_density,
    enumerate_hidden_paths,
)
from snakes_and_ladders.likelihood.spatio_sequential import (
    class_posteriors,
    conditional_state_posterior,
    enumerate_spatio_sequential,
    external_field,
    labelled_log_likelihood,
    log_joint_at,
    map_labelling,
    marginal_log_likelihood_torch,
)
from snakes_and_ladders.numerics import logsumexp
from snakes_and_ladders.search.statistics import chi_square_p_value
from snakes_and_ladders.sim.hmm import HmmParams
from snakes_and_ladders.sim.spatio_sequential import (
    canonical_spatio_sequential,
    gated_log_density,
    simulate_spatio_sequential,
)


def _hmm(n_states: int, n_symbols: int, length: int, seed: int) -> HmmParams:
    rng = np.random.default_rng(seed)
    return HmmParams(
        n_states=n_states,
        sequence_length=length,
        n_sequences=1,
        initial=rng.dirichlet(np.ones(n_states)),
        transition=rng.dirichlet(np.ones(n_states), size=n_states),
        emissions=CategoricalEmission(rng.dirichlet(np.ones(n_symbols), size=n_states)),
        seed=seed,
        tolerance=1e-12,
    )


@pytest.mark.oracle
@pytest.mark.parametrize(
    ("n_states", "n_symbols", "length", "seed"),
    [(2, 2, 5, 1), (3, 2, 4, 2), (2, 4, 6, 3), (4, 3, 3, 4)],
)
def test_forward_backward_is_the_path_enumeration(
    n_states: int, n_symbols: int, length: int, seed: int
) -> None:
    params = _hmm(n_states, n_symbols, length, seed)
    observations = np.random.default_rng(seed).integers(0, n_symbols, size=length)

    run = forward_backward(
        emission_log_density(params, observations),
        np.log(params.initial),
        np.log(params.transition),
    )
    enumerated = enumerate_hidden_paths(params, observations)

    assert abs(run.log_evidence - enumerated.log_likelihood) < 1e-12 * abs(
        enumerated.log_likelihood
    )
    np.testing.assert_allclose(
        run.posterior, enumerated.posterior, rtol=1e-11, atol=1e-13
    )
    np.testing.assert_allclose(run.pairwise.sum(axis=(1, 2)), 1.0, rtol=1e-12)
    np.testing.assert_allclose(run.pairwise.sum(axis=2), run.posterior[:-1], rtol=1e-11)
    np.testing.assert_allclose(run.pairwise.sum(axis=1), run.posterior[1:], rtol=1e-11)


@pytest.mark.oracle
def test_the_class_e_step_is_the_conditional_posterior_by_enumeration() -> None:
    params = canonical_spatio_sequential()
    data = simulate_spatio_sequential(params, np.random.default_rng(1))

    posteriors = class_posteriors(params, data.observations, data.labels)
    exact = conditional_state_posterior(params, data.observations, data.labels)

    np.testing.assert_allclose(posteriors.posterior, exact, rtol=1e-11, atol=1e-13)


@pytest.mark.oracle
def test_the_labelled_joint_is_the_enumeration_s_per_labelling_term() -> None:
    params = canonical_spatio_sequential()
    data = simulate_spatio_sequential(params, np.random.default_rng(2))
    exact = enumerate_spatio_sequential(params, data.observations)
    # log p(x, l) = logsumexp over every joint path of the written-out joint.
    paths = np.array(
        list(product(range(params.n_states), repeat=params.n_positions)), dtype=np.int64
    )
    terms = np.array(
        [
            log_joint_at(
                params, data.observations, data.labels, np.stack([first, second])
            )
            for first in paths
            for second in paths
        ]
    )
    expected = float(logsumexp(terms[None, :], axis=1)[0]) - exact.log_prior_normalizer

    realized = labelled_log_likelihood(params, data.observations, data.labels)

    assert abs(realized - exact.log_prior_normalizer - expected) < 1e-10 * abs(expected)


@pytest.mark.mathematical
def test_the_field_is_minus_the_posterior_expected_emission_score() -> None:
    params = canonical_spatio_sequential()
    data = simulate_spatio_sequential(params, np.random.default_rng(3))
    posterior = class_posteriors(params, data.observations, data.labels).posterior
    gated = gated_log_density(params, data.observations)

    field = external_field(params, data.observations, data.labels)

    for n in range(params.graph.n_nodes):
        for m in range(params.n_classes):
            expected = -sum(
                posterior[m, s, k] * gated[n, s, m, k]
                for s in range(params.n_positions)
                for k in range(params.n_states)
            )
            assert abs(field[n, m] - expected) < 1e-12 * max(1.0, abs(expected))


@pytest.mark.oracle
def test_the_m_step_identity_holds_through_autograd() -> None:
    # the M-step identity: the gradient of log p(x | l, theta) equals the
    # posterior-weighted gradient of the emission terms. The left side goes
    # through the forward recursion, the right through the E step.
    params = canonical_spatio_sequential()
    means = (np.array([0.0, 2.0]), np.array([-1.0, 1.0]))
    gaussian = replace(
        params,
        emissions=tuple(GaussianEmission(mean, np.ones(2), 1e-3) for mean in means),
    )
    data = simulate_spatio_sequential(
        gaussian, np.random.default_rng(4), labels=np.array([0, 0, 1, 1])
    )
    tracked = [
        torch.tensor(mean, dtype=torch.float64, requires_grad=True) for mean in means
    ]
    differentiable = replace(
        gaussian,
        emissions=tuple(GaussianEmission(mean, np.ones(2), 1e-3) for mean in tracked),
    )

    left = torch.autograd.grad(
        marginal_log_likelihood_torch(differentiable, data.observations, data.labels),
        tracked,
    )
    posterior = class_posteriors(
        differentiable, data.observations, data.labels
    ).posterior
    weighted = torch.zeros((), dtype=torch.float64)
    for m, family in enumerate(differentiable.emissions):
        members = np.flatnonzero(data.labels == m)
        scores = family.log_density(
            torch.as_tensor(data.observations[:, members], dtype=torch.float64)
        )
        weighted = weighted + (torch.as_tensor(posterior[m])[:, None, :] * scores).sum()
    right = torch.autograd.grad(weighted, tracked)

    for a, b in zip(left, right, strict=True):
        assert float((a - b).abs().max()) < 1e-10 * max(1.0, float(a.abs().max()))
        assert float(a.abs().max()) > 0.0


@pytest.mark.simulated_truth
def test_the_backward_sampler_draws_paths_from_the_posterior() -> None:
    params = _hmm(2, 2, 3, 7)
    observations = np.array([0, 1, 1])
    density = emission_log_density(params, observations)
    enumerated = enumerate_hidden_paths(params, observations)
    paths = list(product(range(2), repeat=3))
    expected = np.array(
        [
            np.exp(
                np.log(params.initial[path[0]])
                + density[0, path[0]]
                + sum(
                    np.log(params.transition[path[t - 1], path[t]])
                    + density[t, path[t]]
                    for t in range(1, 3)
                )
                - enumerated.log_likelihood
            )
            for path in paths
        ]
    )
    rng = np.random.default_rng(8)
    n_draws = 4000
    counts = np.zeros(len(paths))
    for _ in range(n_draws):
        drawn = sample_path(
            density, np.log(params.initial), np.log(params.transition), rng
        )
        counts[paths.index(tuple(int(v) for v in drawn))] += 1

    assert chi_square_p_value(counts, n_draws * expected) > 0.001


@pytest.mark.mathematical
def test_the_map_labelling_has_the_largest_labelled_joint() -> None:
    params = canonical_spatio_sequential()
    data = simulate_spatio_sequential(params, np.random.default_rng(5))

    best = map_labelling(params, data.observations)
    value = labelled_log_likelihood(params, data.observations, best)

    for labelling in product(range(params.n_classes), repeat=params.graph.n_nodes):
        other = labelled_log_likelihood(params, data.observations, np.array(labelling))
        assert other <= value + 1e-9


@pytest.mark.edge_case
def test_forward_backward_refuses_an_empty_chain_and_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="T >= 1"):
        forward_backward(np.zeros((0, 2)), np.zeros(2), np.zeros((2, 2)))
    with pytest.raises(ValueError, match="do not match"):
        forward_backward(np.zeros((3, 2)), np.zeros(3), np.zeros((2, 2)))
