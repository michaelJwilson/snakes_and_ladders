"""A mixture objective over any emission family (issues #964, #1164).

Referees: the objective against the count-pair mixture's likelihood computed
by `scipy.stats` --- a negative-binomial total and a beta-binomial count of
successes out of it --- summed in NumPy with no line of the package's
`log_density`; its gradient against central differences; and the map from
``theta`` to the named parameters inverted exactly. A Gaussian, a binomial
and a categorical family, whose parameters are real, a probability and a row
of log-probabilities, against `scipy.stats` the same way (issue #1164).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from sal.emissions import (
    BinomialEmission,
    CategoricalEmission,
    CountPairEmission,
    GaussianEmission,
)
from sal.opt.emission_mixture import EmissionMixtureObjective
from scipy.special import logsumexp
from scipy.stats import betabinom, binom, nbinom, norm

TRUTH = CountPairEmission([6.0, 12.0], [20.0, 60.0], [2.0, 9.0], [8.0, 3.0], joint=True)


def _pairs(seed: int) -> np.ndarray:
    rng = np.random.default_rng([964, seed])
    labels = rng.choice(2, size=200, p=[0.4, 0.6])
    return np.asarray(TRUTH.sample(labels, rng))


@pytest.mark.critical
@pytest.mark.oracle
def test_the_objective_is_the_scipy_likelihood() -> None:
    pairs = _pairs(0)
    objective = EmissionMixtureObjective(pairs, TRUTH)
    theta = objective.initial() + torch.linspace(
        -0.3, 0.3, objective.n_parameters, dtype=torch.float64
    )
    named = {k: v.detach().numpy() for k, v in objective.constrain(theta).items()}
    total, successes = pairs[:, 0], pairs[:, 1]
    scores = np.stack(
        [
            nbinom.logpmf(
                total,
                named["dispersion"][k],
                named["dispersion"][k] / (named["dispersion"][k] + named["mean"][k]),
            )
            + betabinom.logpmf(successes, total, named["alpha"][k], named["beta"][k])
            for k in range(2)
        ],
        axis=1,
    )
    expected = -logsumexp(scores + named["log_weight"][None, :], axis=1).sum()
    assert abs(float(objective(theta)) - expected) <= 1e-10 * abs(expected)


@pytest.mark.critical
@pytest.mark.analytic
def test_the_gradient_is_the_central_difference_and_the_map_inverts() -> None:
    objective = EmissionMixtureObjective(_pairs(1), TRUTH)
    theta = objective.initial() + 0.1
    np.testing.assert_allclose(
        objective.theta_from(objective.constrain(theta)).numpy(),
        theta.numpy(),
        atol=1e-12,
    )
    point = theta.clone().requires_grad_(True)
    (gradient,) = torch.autograd.grad(objective(point), point)
    step = 1e-6
    for index in range(objective.n_parameters):
        shift = torch.zeros_like(theta)
        shift[index] = step
        central = (
            float(objective(theta + shift)) - float(objective(theta - shift))
        ) / (2 * step)
        assert abs(float(gradient[index]) - central) <= 1e-5 * max(1.0, abs(central)), (
            index
        )


@pytest.mark.smoke
def test_a_family_it_cannot_parameterize_is_refused() -> None:
    with pytest.raises(ValueError, match="at least two"):
        EmissionMixtureObjective(
            _pairs(2),
            CountPairEmission([6.0], [20.0], [2.0], [8.0], joint=True),
        )
    assert EmissionMixtureObjective(_pairs(2), TRUTH).n_parameters == 1 + 2 * 4


def _scipy_scores(family: object, observations: np.ndarray) -> np.ndarray:
    """Every observation's log-density under every component, by `scipy.stats` and NumPy."""
    if isinstance(family, GaussianEmission):
        named = {k: v.detach().numpy() for k, v in family.named_parameters().items()}
        return np.asarray(
            norm.logpdf(observations[:, None], named["mean"], named["scale"])
        )
    if isinstance(family, BinomialEmission):
        return np.asarray(
            binom.logpmf(
                observations[:, None],
                family.trials.numpy(),
                family.probability.detach().numpy(),
            )
        )
    assert isinstance(family, CategoricalEmission)
    return family.log_matrix.detach().numpy().T[observations.astype(np.int64)]


#: A family over each domain the count families do not reach: a real mean, a
#: probability, a row of log-probabilities.
DOMAIN_FAMILIES = {
    "gaussian": GaussianEmission([-1.5, 2.0], [0.5, 1.5], 1e-3),
    "binomial": BinomialEmission([40.0, 40.0], [0.2, 0.7]),
    "categorical": CategoricalEmission(np.array([[0.2, 0.5, 0.3], [0.6, 0.1, 0.3]])),
}


@pytest.mark.oracle
@pytest.mark.parametrize("name", list(DOMAIN_FAMILIES))
def test_a_family_over_any_domain_is_its_scipy_likelihood(name: str) -> None:
    # Issue #1164: the objective maps theta through the domains the family
    # declares, so a family with no positive parameter is fitted as one with.
    family = DOMAIN_FAMILIES[name]
    rng = np.random.default_rng([1164, len(name)])
    observations = np.asarray(family.sample(rng.choice(2, size=300), rng))
    objective = EmissionMixtureObjective(observations, family)
    theta = objective.initial() + torch.linspace(
        -0.3, 0.3, objective.n_parameters, dtype=torch.float64
    )
    named = objective.constrain(theta)
    log_weight = named["log_weight"].numpy()
    rebuilt = family.with_parameters(
        {key: value for key, value in named.items() if key != "log_weight"}
    )

    expected = -logsumexp(
        _scipy_scores(rebuilt, observations) + log_weight[None, :], axis=1
    ).sum()

    assert abs(float(objective(theta)) - expected) <= 1e-10 * abs(expected)
    np.testing.assert_allclose(
        objective.theta_from(named).numpy(), theta.numpy(), rtol=0, atol=1e-14
    )


@pytest.mark.critical
@pytest.mark.oracle
def test_a_count_family_s_theta_is_its_log_parameters_bitwise() -> None:
    # The layout the compiled route reads: after K - 1 free weights, K log
    # values per named parameter in the family's order, exactly as before
    # the families declared their domains.
    objective = EmissionMixtureObjective(_pairs(3), TRUTH)
    theta = objective.initial()
    blocks = [torch.log(value) for value in TRUTH.named_parameters().values()]

    assert torch.equal(theta[1:], torch.cat(blocks))
