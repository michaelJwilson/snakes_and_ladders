"""What a sampled chain is judged against: the mixture posterior, and the error bar.

`DEV.md`'s Test Layout puts a fixture shared across modules in a top-level
underscore-prefixed module, imported rather than collected. Issue #734 wrote
this one for HMC; issue #756 gave it two more readers --- MALA and slice
sampling --- and a fixture copied per sampler is a fixture that stops being
the same fixture, so it moved here rather than being written a second time.

Everything a sampler is judged against comes from
`likelihood.mixture_assignments.enumerate_mixture_assignments`: the evidence
is the sum over all 4,096 whole assignments and the marginals are that sum's
own, integrated over the one free coordinate on a grid. No factorized form
and no second sampler appears anywhere in the reference.

:func:`monte_carlo_sigmas` sits beside it because the two are one statement.
A deviation from an exact reference is a finding only against the error the
chain itself carries, and that error is the draws' spread over the
*effective* sample size rather than over the draw count --- correlated draws
divided by their number give a standard error several times too small, and a
sampler passes a comparison it should fail. The assertions each sampler's
module makes against these references are written once below (issue #982).
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

#: The prior that makes the one free coordinate a posterior rather than the
#: improper flat-prior one, and the window and spacing the quadrature runs on.
#: The posterior's mean is 0.415 and its standard deviation 0.660, so the
#: window is 13 of them wide, and the quadrature is converged: over half-widths
#: 4.0 to 12.0 and 301 to 1,201 points the mean weight moves by at most
#: 1.4e-06 and a marginal by 1.4e-06, against the 2.0e-02 the chain is judged
#: to below.
MIXTURE_PRIOR_SCALE = 2.0
QUADRATURE_WINDOW = 9.0
QUADRATURE_POINTS = 451


class WeightPosterior:
    """The enumerable mixture with its components known and its weight free.

    One free coordinate, so the posterior integrates on a line and the
    enumeration supplies every value on it. The mixture's own objective carries
    six coordinates and no reference but a sampler; what is wanted from it here
    is the *assignment* posterior, which does not need the other five free.

    Parameters
    ----------
    observations : np.ndarray
        The draw, shape ``(n_samples,)``.
    components : GaussianEmission
        The known component densities.
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

    Every value on the grid comes from the enumeration: the evidence is the
    sum over all 4,096 assignments and the marginals are that sum's own, so
    the reference uses no factorized form anywhere.
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

    Parameters
    ----------
    draws : torch.Tensor
        One chain, shape ``(n, dimension)``.
    mean, variance : np.ndarray
        The closed-form marginal mean and variance, shape ``(dimension,)``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``|estimate - truth| / standard error`` for the mean and for the
        variance, per coordinate. Each standard error is the estimator's own
        spread over the square root of the effective sample size *of that
        estimator*: the draws for the mean, the squared deviations for the
        variance, which are a slower-mixing series and carry the larger error.
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

    ``reference`` is :func:`enumerated_quadrature`'s. The drawn marginals come
    from the factorized E step at each draw, and each tolerance is ``sigmas``
    of the drawn spread over ``sqrt(size)``: the draw count or the effective
    sample size, whichever the caller declares.
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

    Every coordinate's mean against 0 and its second moment against
    ``temperature / precision``, each in the standard error of its own
    filter (``KalmanMean``).
    """
    assert np.abs(first.mean / first.standard_error).max() < bound
    residual = (second.mean - temperature / precision) / second.standard_error
    assert np.abs(residual).max() < bound
