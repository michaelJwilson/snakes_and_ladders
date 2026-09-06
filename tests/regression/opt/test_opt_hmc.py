"""HMC, checked where it is exact before it is checked where it is statistical.

The integrator has two properties that are exact statements about arithmetic
and need no sampling at all: it is reversible, and its energy error is second
order in the step size. Those come first, because a distributional test says
the chain is wrong while these say which half is wrong.

Then the distribution, against two references that are not another sampler: a
Gaussian whose mean and covariance are analytic, and a real `Objective` whose
posterior is computed by grid quadrature.

The last test is the one that changed the module. A step size too large biases
the posterior *spread* downward while leaving the mean correct and the
acceptance rate looking healthy --- measured at 0.982 acceptance with the
standard deviation 12% low. Acceptance does not detect it; the energy error
does, which is why `HmcChain` carries it.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping

import numpy as np
import pytest
import torch
from phylo.opt.hmc import (
    WithGaussianPrior,
    hamiltonian,
    leapfrog,
    sample,
)
from phylo.opt.potts import PottsObjective, PottsParams, simulate_chains

from tests._scale import stress_only

EXACT = 1e-13


class Gaussian:
    """``-log N(mean, covariance)`` up to a constant: an analytic target."""

    def __init__(self, mean: list[float], covariance: list[list[float]]) -> None:
        self.mean = torch.tensor(mean, dtype=torch.float64)
        self.covariance = torch.tensor(covariance, dtype=torch.float64)
        self._precision = torch.linalg.inv(self.covariance)

    def initial(self) -> torch.Tensor:
        return torch.zeros_like(self.mean)

    def constrain(self, theta: torch.Tensor) -> Mapping[str, torch.Tensor]:
        return {"x": theta}

    def __call__(self, theta: torch.Tensor) -> torch.Tensor:
        deviation = theta - self.mean
        quadratic: torch.Tensor = 0.5 * deviation @ self._precision @ deviation
        return quadratic


GAUSSIAN = Gaussian([1.0, -2.0], [[2.0, 0.6], [0.6, 0.5]])


def _potts_posterior() -> WithGaussianPrior:
    """A real `Objective` whose `theta` is 2-D, so quadrature can referee it."""
    field = np.array([0.3, -0.3])
    field = field - np.log(np.exp(field).sum())
    params = PottsParams(
        n_states=2,
        chain_length=8,
        n_chains=300,
        coupling=0.75,
        field=field,
        seed=7,
    )
    return WithGaussianPrior(PottsObjective(simulate_chains(params), 2), scale=2.0)


# Computed by grid quadrature over the 2-D posterior and verified converged:
# identical to five decimals across window half-widths 3.0 down to 0.3 and
# spacings from 0.55 to 0.03 posterior standard deviations.
QUADRATURE_MEAN = (0.72051, -0.61289)
QUADRATURE_SD = (0.05497, 0.04524)


@pytest.mark.parametrize(("n_steps", "step_size"), [(20, 0.05), (50, 0.1), (100, 0.02)])
def test_the_integrator_is_reversible(n_steps: int, step_size: float) -> None:
    # Run forward, negate the momentum, run forward again: the exact statement
    # that makes the Metropolis proposal symmetric. Without it the acceptance
    # ratio is not the energy difference alone and the chain targets the wrong
    # distribution, which no amount of sampling would localize to here.
    theta = torch.tensor([0.4, 0.9], dtype=torch.float64)
    momentum = torch.tensor([-0.3, 1.1], dtype=torch.float64)

    forward, forward_momentum = leapfrog(GAUSSIAN, theta, momentum, step_size, n_steps)
    back, back_momentum = leapfrog(
        GAUSSIAN, forward, -forward_momentum, step_size, n_steps
    )

    assert float((back - theta).abs().max()) < EXACT
    assert float((-back_momentum - momentum).abs().max()) < EXACT


def test_the_energy_error_is_second_order_in_the_step_size() -> None:
    # Halving the step must quarter the error. The *trajectory length* is held
    # fixed and the step count scaled with it: varying the step at a fixed
    # step count moves the endpoint around the orbit instead, and the errors
    # then oscillate rather than converge -- which is what the first draft of
    # this test measured.
    theta = torch.tensor([0.4, 0.9], dtype=torch.float64)
    momentum = torch.tensor([-0.3, 1.1], dtype=torch.float64)
    reference = hamiltonian(GAUSSIAN, theta, momentum)

    errors = []
    for n_steps in (10, 20, 40, 80, 160):
        position, velocity = leapfrog(GAUSSIAN, theta, momentum, 1.0 / n_steps, n_steps)
        errors.append(abs(hamiltonian(GAUSSIAN, position, velocity) - reference))

    ratios = [before / after for before, after in itertools.pairwise(errors)]
    for ratio in ratios:
        assert ratio == pytest.approx(4.0, rel=0.02)


def test_the_chain_recovers_an_analytic_gaussian() -> None:
    # Mean and covariance in closed form, so nothing here rests on a second
    # sampler. The tolerance is set from the spread across eight independent
    # chains, measured at 0.030 and 0.041 on the two coordinates.
    chain = sample(
        GAUSSIAN, seed=11, n_samples=4000, step_size=0.25, n_steps=12, burn_in=400
    )

    mean = chain.theta.mean(dim=0)
    covariance = torch.cov(chain.theta.T)

    assert chain.acceptance_rate > 0.9
    np.testing.assert_allclose(mean.numpy(), GAUSSIAN.mean.numpy(), atol=0.12)
    np.testing.assert_allclose(
        covariance.numpy(), GAUSSIAN.covariance.numpy(), atol=0.2
    )


@stress_only(
    "the second moment needs pooled 2000-draw chains; a shorter run "
    "estimates the spread badly and would assert less than it claims"
)
def test_the_chain_matches_grid_quadrature_on_a_real_objective() -> None:
    # The Potts chain's `theta` is two-dimensional at two states, so the
    # posterior can be integrated on a grid and there is a reference that is
    # not a sampler. The step size is deliberately small: see the test below
    # for what a larger one does to the second moment.
    # Three independent chains, pooled. One chain of the same total length
    # estimates the *mean* fine and the *spread* badly: a single 2000-draw
    # chain came out 18% high, because the second moment needs far more
    # effective samples than the first and this posterior mixes slowly.
    # Pooling independent chains fixes that without lengthening any of them.
    posterior = _potts_posterior()

    draws = torch.cat(
        [
            sample(
                posterior,
                seed=200 + index,
                n_samples=700,
                step_size=0.01,
                n_steps=15,
                burn_in=100,
            ).theta
            for index in range(3)
        ]
    )

    np.testing.assert_allclose(
        draws.mean(dim=0).numpy(), np.array(QUADRATURE_MEAN), atol=0.005
    )
    np.testing.assert_allclose(
        draws.std(dim=0).numpy(), np.array(QUADRATURE_SD), rtol=0.10
    )


@stress_only(
    "the bias is in the spread, which needs the same chain lengths "
    "as the test above before it is measurable at all"
)
def test_a_step_size_that_diverges_biases_the_spread_not_the_mean() -> None:
    # The failure this module's `energy_error` exists for, and the reason
    # acceptance rate is not enough on its own. Measured against quadrature:
    #
    #   step   accept   max|dH|   sd / true
    #   0.050   0.799   9.3e+00   0.92, 0.93
    #   0.020   0.982   3.1e-01   0.88, 0.98
    #   0.005   1.000   8.1e-03   1.07, 1.04
    #   0.002   1.000   5.1e-03   1.01, 1.00
    #
    # At 0.020 the acceptance rate looks healthy and the standard deviation is
    # 12% low, because divergent trajectories are rejected preferentially in
    # the tails. What tracks the bias is the energy error.
    posterior = _potts_posterior()

    coarse = sample(
        posterior, seed=3, n_samples=400, step_size=0.05, n_steps=20, burn_in=50
    )
    fine = sample(
        posterior, seed=3, n_samples=400, step_size=0.005, n_steps=40, burn_in=50
    )

    assert float(coarse.energy_error.max()) > 100.0 * float(fine.energy_error.max())
    # The mean survives a step size that the spread does not.
    assert abs(float(coarse.theta.mean(dim=0)[0]) - QUADRATURE_MEAN[0]) < 0.01
    assert float(coarse.theta.std(dim=0)[0]) < float(fine.theta.std(dim=0)[0])


def test_the_posterior_is_wider_than_the_laplace_approximation_predicts() -> None:
    # The comparison this module exists to make. The Laplace standard error is
    # the curvature at the mode; the HMC one is a quantile of the posterior.
    # On this fixture they agree closely, which is the expected outcome for a
    # well-identified two-parameter model and is what makes a *disagreement*
    # elsewhere informative rather than ambiguous.
    posterior = _potts_posterior()
    mode = torch.tensor(QUADRATURE_MEAN, dtype=torch.float64)

    hessian = torch.autograd.functional.hessian(  # type: ignore[no-untyped-call]
        posterior.__call__, mode
    )
    laplace = torch.linalg.inv(hessian).diagonal().sqrt()

    np.testing.assert_allclose(laplace.numpy(), np.array(QUADRATURE_SD), rtol=0.15)


def test_a_zero_length_trajectory_is_refused() -> None:
    # It would propose the current point every time: acceptance rate 1, energy
    # error 0, and a chain that has not moved. Every diagnostic reports health.
    with pytest.raises(ValueError, match="looks healthy and samples nothing"):
        sample(GAUSSIAN, seed=1, n_samples=10, step_size=0.1, n_steps=0)


def test_a_non_positive_step_size_is_refused() -> None:
    with pytest.raises(ValueError, match="step_size must be positive"):
        sample(GAUSSIAN, seed=1, n_samples=10, step_size=0.0)


def test_a_non_positive_prior_scale_is_refused() -> None:
    with pytest.raises(ValueError, match="prior scale must be positive"):
        WithGaussianPrior(GAUSSIAN, scale=0.0)


def test_the_prior_is_added_to_the_objective_and_nothing_else() -> None:
    # The wrapper is what turns a likelihood into something with a posterior;
    # if it changed the likelihood term the chain would target a different
    # model silently.
    point = torch.tensor([0.3, -1.1], dtype=torch.float64)
    wrapped = WithGaussianPrior(GAUSSIAN, scale=2.0)

    expected = float(GAUSSIAN(point)) + float((point * point).sum()) / (2.0 * 4.0)

    assert float(wrapped(point)) == pytest.approx(expected, rel=EXACT)


def test_a_chain_is_reproducible_from_its_seed() -> None:
    first = sample(GAUSSIAN, seed=5, n_samples=50, step_size=0.2, n_steps=10)
    second = sample(GAUSSIAN, seed=5, n_samples=50, step_size=0.2, n_steps=10)

    assert torch.equal(first.theta, second.theta)
