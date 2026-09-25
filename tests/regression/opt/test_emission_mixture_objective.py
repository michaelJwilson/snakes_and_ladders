"""A mixture objective over any positive-parameter family (issue #964).

Referees: the objective against the count-pair mixture's likelihood computed
by `scipy.stats` --- a negative-binomial total and a beta-binomial count of
successes out of it --- summed in NumPy with no line of the package's
`log_density`; its gradient against central differences; and the map from
``theta`` to the named parameters inverted exactly.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pytest
import torch
from sal.emissions import CountPairEmission, EmissionFamily
from sal.opt.emission_mixture import EmissionMixtureObjective
from scipy.special import logsumexp
from scipy.stats import betabinom, nbinom

TRUTH = CountPairEmission([6.0, 12.0], [20.0, 60.0], [2.0, 9.0], [8.0, 3.0], joint=True)


def _build(named: Mapping[str, torch.Tensor]) -> EmissionFamily:
    return CountPairEmission(
        named["dispersion"], named["mean"], named["alpha"], named["beta"], joint=True
    )


def _pairs(seed: int) -> np.ndarray:
    rng = np.random.default_rng([964, seed])
    labels = rng.choice(2, size=200, p=[0.4, 0.6])
    return np.asarray(TRUTH.sample(labels, rng))


@pytest.mark.critical
@pytest.mark.oracle
def test_the_objective_is_the_scipy_likelihood() -> None:
    pairs = _pairs(0)
    objective = EmissionMixtureObjective(pairs, TRUTH, _build)
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
    objective = EmissionMixtureObjective(_pairs(1), TRUTH, _build)
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
            _build,
        )
    assert EmissionMixtureObjective(_pairs(2), TRUTH, _build).n_parameters == 1 + 2 * 4
