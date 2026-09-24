"""What a sampled chain is judged against: the mixture posterior, and the error bar.

Shared by HMC (issue #734), MALA and slice sampling (issue #756). The
reference is `likelihood.mixture_assignments.enumerate_mixture_assignments`:
the evidence sums all 4,096 assignments and the marginals are that sum's own,
integrated over the one free coordinate on a grid; no factorized form and no
second sampler. :func:`monte_carlo_sigmas` scales a deviation by the chain's
own error over the effective sample size, not the draw count, which would
understate it several times. The assertions are written once (issue #982).
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch
from snakes_and_ladders.emissions import GaussianEmission
from snakes_and_ladders.likelihood.mixture_assignments import (
    enumerate_mixture_assignments,
)
from snakes_and_ladders.opt.constrain import free_from_log_simplex, log_simplex
from snakes_and_ladders.opt.mixture import mixture_log_likelihood, responsibilities
from snakes_and_ladders.sample.expectation import Expectation, KalmanMean
from snakes_and_ladders.sample.hmc import WithGaussianPrior, effective_sample_size
from snakes_and_ladders.sim.mixture import MixtureParams, simulate_mixture

from tests._objective_checks import AnalyticGaussian

#: The correlated two-dimensional Gaussian every continuous sampler is first
#: judged on: its mean and covariance are closed forms.
GAUSSIAN = AnalyticGaussian([1.0, -2.0], [[2.0, 0.6], [0.6, 0.5]])

#: The enumerable mixture the chain is run on: twelve observations over two
#: Gaussian components 1.5 standard deviations apart, so ``2 ** 12 = 4,096``
#: assignments enumerate and no observation's component is obvious --- the
#: enumerated marginals below run from 0.003 to 0.999.
MIXTURE_SAMPLES = 12
MIXTURE_WEIGHTS = np.array([0.4, 0.6])
MIXTURE_MEAN = np.array([-1.5, 1.5])
MIXTURE_SCALE = np.array([1.0, 1.0])

#: The prior and quadrature window. Posterior mean 0.415, sd 0.660; over
#: half-widths 4.0 to 12.0 and 301 to 1,201 points the mean weight and a
#: marginal move by at most 1.4e-06, against the 2.0e-02 the chain is judged to.
MIXTURE_PRIOR_SCALE = 2.0
QUADRATURE_WINDOW = 9.0
QUADRATURE_POINTS = 451


class WeightPosterior:
    """The enumerable mixture with its components known and its weight free.

    One free coordinate, so the assignment posterior integrates on a line.
    """

    def __init__(self, observations: np.ndarray, components: GaussianEmission) -> None:
        self.values = torch.as_tensor(observations, dtype=torch.float64)
        self.components = components

    def initial(self) -> torch.Tensor:
        return torch.zeros(1, dtype=torch.float64)

    def constrain(self, theta: torch.Tensor) -> Mapping[str, torch.Tensor]:
        return {"log_weight": log_simplex(theta)}

    def theta_from(self, named: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return free_from_log_simplex(named["log_weight"])

    def __call__(self, theta: torch.Tensor) -> torch.Tensor:
        return -mixture_log_likelihood(self.values, log_simplex(theta), self.components)


def mixture_draw() -> tuple[np.ndarray, GaussianEmission]:
    """The twelve observations and the components that generated them."""
    components = GaussianEmission(MIXTURE_MEAN, MIXTURE_SCALE, 1e-12)
    params = MixtureParams(
        weights=MIXTURE_WEIGHTS,
        components=components,
        n_samples=MIXTURE_SAMPLES,
        seed=734,
        tolerance=1e-12,
    )
    return simulate_mixture(params).observations, components


def enumerated_quadrature(
    observations: np.ndarray, components: GaussianEmission
) -> tuple[float, np.ndarray]:
    """The posterior mean weight and marginal assignment probabilities, by quadrature.

    Every grid value is the enumeration over all 4,096 assignments.
    """
    grid = np.linspace(-QUADRATURE_WINDOW, QUADRATURE_WINDOW, QUADRATURE_POINTS)
    log_posterior = np.empty(grid.shape)
    marginal = np.empty((grid.shape[0], observations.shape[0]))
    weight = np.empty(grid.shape)
    for index, point in enumerate(grid):
        theta = torch.tensor([point], dtype=torch.float64)
        weights = torch.exp(log_simplex(theta)).numpy()
        enumerated = enumerate_mixture_assignments(weights, components, observations)
        log_posterior[index] = enumerated.log_evidence - 0.5 * point**2 / (
            MIXTURE_PRIOR_SCALE**2
        )
        marginal[index] = enumerated.responsibilities[:, 0]
        weight[index] = weights[0]

    density = np.exp(log_posterior - log_posterior.max())
    density /= np.trapezoid(density, grid)
    return (
        float(np.trapezoid(density * weight, grid)),
        np.array(
            [
                np.trapezoid(density * marginal[:, site], grid)
                for site in range(observations.shape[0])
            ]
        ),
    )


def weight_posterior() -> tuple[WithGaussianPrior, np.ndarray, GaussianEmission]:
    """The target, the observations behind it, and the components that made them."""
    observations, components = mixture_draw()
    return (
        WithGaussianPrior(
            WeightPosterior(observations, components), scale=MIXTURE_PRIOR_SCALE
        ),
        observations,
        components,
    )


def monte_carlo_sigmas(
    draws: torch.Tensor, mean: np.ndarray, variance: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """A chain's marginal mean and variance against the truth, in its own standard errors.

    Each error is over its own estimator's effective sample size, not the draw count.
    """
    centred = draws - draws.mean(dim=0)
    squares = centred * centred
    mean_error = draws.std(dim=0).numpy() / np.sqrt(
        effective_sample_size(draws).numpy()
    )
    variance_error = squares.std(dim=0).numpy() / np.sqrt(
        effective_sample_size(squares).numpy()
    )
    return (
        np.abs(draws.mean(dim=0).numpy() - mean) / mean_error,
        np.abs(squares.mean(dim=0).numpy() - variance) / variance_error,
    )


def assert_within_sigmas(
    draws: torch.Tensor,
    mean: np.ndarray,
    variance: np.ndarray,
    sigmas: float,
    context: object = None,
) -> None:
    """Fail when a marginal mean or variance is ``sigmas`` standard errors off."""
    mean_sigmas, variance_sigmas = monte_carlo_sigmas(draws, mean, variance)
    assert float(mean_sigmas.max()) < sigmas, (context, mean_sigmas)
    assert float(variance_sigmas.max()) < sigmas, (context, variance_sigmas)


def assert_recovers_assignment_posterior(
    theta: torch.Tensor,
    observations: np.ndarray,
    components: GaussianEmission,
    reference: tuple[float, np.ndarray],
    *,
    sigmas: float,
    size: float,
    context: object = None,
) -> None:
    """Fail when a chain's weight or assignment marginals miss the enumeration's.

    Tolerance: ``sigmas`` of the drawn spread over ``sqrt(size)``, size the caller's.
    """
    quadrature_weight, quadrature_marginal = reference
    log_weights = log_simplex(theta)
    drawn_weight = torch.exp(log_weights)[:, 0].numpy()
    drawn_marginal = np.stack(
        [
            responsibilities(
                torch.as_tensor(observations, dtype=torch.float64), row, components
            ).numpy()[:, 0]
            for row in log_weights
        ]
    )
    weight_tolerance = sigmas * float(drawn_weight.std()) / np.sqrt(size)
    marginal_tolerance = (
        sigmas * float(drawn_marginal.std(axis=0).max()) / np.sqrt(size)
    )
    marginal_gap = np.abs(drawn_marginal.mean(axis=0) - quadrature_marginal).max()

    assert abs(drawn_weight.mean() - quadrature_weight) < weight_tolerance, (
        context,
        drawn_weight.mean(),
        quadrature_weight,
    )
    assert marginal_gap < marginal_tolerance, (context, marginal_gap)


def draw_moments(draws: torch.Tensor) -> tuple[Expectation, Expectation]:
    """The first and second moments of stored draws, each by its AR(1) filter."""
    first, second = KalmanMean(), KalmanMean()
    for row in draws:
        first.update(row)
        second.update(row * row)
    return first.estimate(), second.estimate()


def assert_gaussian_moments(
    first: Expectation,
    second: Expectation,
    precision: np.ndarray,
    *,
    temperature: float = 1.0,
    bound: float = 4.5,
) -> None:
    """Fail when a centred diagonal Gaussian's moments are ``bound`` errors off.

    Mean against 0, second moment against ``temperature / precision`` (``KalmanMean``).
    """
    assert np.abs(first.mean / first.standard_error).max() < bound
    residual = (second.mean - temperature / precision) / second.standard_error
    assert np.abs(residual).max() < bound
