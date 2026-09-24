"""The conjugate Gibbs sampler for a Gaussian mixture (issue #964).

Referees: the chain's allocation frequencies against the enumerated exact
posterior on four observations, by chi-square on a thinned chain, with the
exact posterior under another prior as the control the test must reject; and
the generating means and weights recovered after relabelling.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from snakes_and_ladders.sample.mixture_gibbs import (
    GaussianMixturePrior,
    exact_allocation_posterior,
    gibbs_gaussian_mixture,
)
from snakes_and_ladders.sample.relabel import permute_parameters, stephens
from snakes_and_ladders.sample.statistics import chi_square_p_value

#: The level a chi-square goodness-of-fit test rejects at, as the Potts
#: samplers' tests use it.
SIGNIFICANCE = 1e-3

OBSERVATIONS = np.array([-1.0, -0.7, 0.9, 1.4])
PRIOR = GaussianMixturePrior(
    concentration=1.0,
    location=np.array([0.0]),
    precision_scale=0.5,
    shape=3.0,
    rate=np.array([0.5]),
)


def _frequencies(allocations: np.ndarray, n_components: int) -> np.ndarray:
    """Counts of each labelled allocation, in the enumeration's lexicographic order."""
    index = np.zeros(allocations.shape[0], dtype=np.int64)
    for column in range(allocations.shape[1]):
        index = index * n_components + allocations[:, column]
    return np.bincount(index, minlength=n_components ** allocations.shape[1]).astype(
        np.float64
    )


@pytest.mark.critical
@pytest.mark.oracle
def test_the_allocations_are_drawn_from_the_exact_posterior() -> None:
    # 16 allocations of 4 observations into 2 components, each with its
    # closed-form posterior probability; 8,000 draws thinned by 5.
    _, exact = exact_allocation_posterior(OBSERVATIONS, 2, PRIOR)
    chain = gibbs_gaussian_mixture(
        OBSERVATIONS, 2, PRIOR, np.random.default_rng(964), 8_000, burn_in=200, thin=5
    )
    observed = _frequencies(chain.allocations, 2)
    assert chi_square_p_value(observed, exact * observed.sum()) > SIGNIFICANCE
    # The control: the exact posterior under a prior ten times as sure of
    # its centre is not what this chain drew.
    _, other = exact_allocation_posterior(
        OBSERVATIONS, 2, replace(PRIOR, precision_scale=5.0, rate=np.array([0.05]))
    )
    assert chi_square_p_value(observed, other * observed.sum()) < SIGNIFICANCE


@pytest.mark.oracle
def test_the_exact_posterior_is_symmetric_and_sums_to_one() -> None:
    allocations, exact = exact_allocation_posterior(OBSERVATIONS, 2, PRIOR)
    assert abs(exact.sum() - 1.0) < 1e-12
    # Relabelling every allocation is a permutation of the configurations
    # with the same probability: an exchangeable prior.
    index = np.zeros(len(allocations), dtype=np.int64)
    for column in range(allocations.shape[1]):
        index = index * 2 + (1 - allocations[:, column])
    np.testing.assert_allclose(exact[index], exact, rtol=1e-12)


@pytest.mark.end2end
def test_a_separated_mixture_is_recovered_after_relabelling() -> None:
    # Planted: three components at -4, 0 and 4, unit scale, weights 0.2,
    # 0.3 and 0.5, 300 draws. Started from a split at the sample's tertiles:
    # from uniform allocations the chain holds a local mode with one wide
    # component over two clusters for hundreds of sweeps. Relabelled by
    # STEPHENS, the posterior means lie within 0.25 of the generating means
    # and the weights within 0.03 of the draw's own shares (0.183, 0.250,
    # 0.567), which is what the posterior centres on.
    rng = np.random.default_rng([964, 1])
    means = np.array([-4.0, 0.0, 4.0])
    weights = np.array([0.2, 0.3, 0.5])
    labels = rng.choice(3, size=300, p=weights)
    observations = rng.normal(means[labels], 1.0)
    prior = GaussianMixturePrior.weakly_informative(observations, 3)
    tertiles = np.quantile(observations, [1 / 3, 2 / 3])
    chain = gibbs_gaussian_mixture(
        observations,
        3,
        prior,
        rng,
        1_000,
        burn_in=200,
        start=np.digitize(observations, tertiles),
    )
    relabelled = stephens(chain.probabilities)
    fitted = permute_parameters(chain.means[:, :, 0], relabelled.permutations).mean(
        axis=0
    )
    order = np.argsort(fitted)
    np.testing.assert_allclose(fitted[order], means, atol=0.25)
    share = permute_parameters(chain.weights, relabelled.permutations).mean(axis=0)
    shares = np.bincount(labels, minlength=3) / labels.size
    np.testing.assert_allclose(share[order], shares, atol=0.03)


@pytest.mark.smoke
def test_what_the_sampler_cannot_run_is_refused() -> None:
    with pytest.raises(ValueError, match="at least two"):
        gibbs_gaussian_mixture(OBSERVATIONS, 1, PRIOR, np.random.default_rng(0), 10)
    with pytest.raises(ValueError, match="thin"):
        gibbs_gaussian_mixture(
            OBSERVATIONS, 2, PRIOR, np.random.default_rng(0), 10, thin=0
        )
    with pytest.raises(ValueError, match="enumeration limit"):
        exact_allocation_posterior(np.zeros(20), 2, PRIOR)
