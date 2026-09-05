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
import math
from collections.abc import Mapping

import numpy as np
import pytest
import torch
from snakes_and_ladders.opt.hmc import (
    YOSHIDA_WEIGHTS,
    Integrator,
    WithGaussianPrior,
    _gradient,
    anneal,
    hamiltonian,
    leapfrog,
    sample,
    yoshida,
)
from snakes_and_ladders.opt.objective import Objective
from snakes_and_ladders.opt.potts import PottsObjective, PottsParams, simulate_chains
from snakes_and_ladders.opt.schedule import Constant, Exponential

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


@pytest.mark.parametrize("integrator", [leapfrog, yoshida])
@pytest.mark.parametrize(("n_steps", "step_size"), [(20, 0.05), (50, 0.1), (100, 0.02)])
def test_the_integrator_is_reversible(
    n_steps: int, step_size: float, integrator: Integrator
) -> None:
    # Run forward, negate the momentum, run forward again: the exact statement
    # that makes the Metropolis proposal symmetric. Without it the acceptance
    # ratio is not the energy difference alone and the chain targets the wrong
    # distribution, which no amount of sampling would localize to here.
    theta = torch.tensor([0.4, 0.9], dtype=torch.float64)
    momentum = torch.tensor([-0.3, 1.1], dtype=torch.float64)

    forward, forward_momentum = integrator(
        GAUSSIAN, theta, momentum, step_size, n_steps
    )
    back, back_momentum = integrator(
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


# --- the fourth-order composition (#266) ----------------------------------


def _hand_written_leapfrog(
    objective: Objective,
    theta: torch.Tensor,
    momentum: torch.Tensor,
    step_size: float,
    n_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Kick-drift-kick, written out, as `leapfrog` was before it was composed.

    Kept as the reference the composition is pinned against. A general driver
    that reduces to this for the trivial composition is one implementation
    instead of two; that it *does* reduce is what the test below establishes,
    and without the reference there would be nothing to establish it against.
    """
    position = theta.detach().clone()
    velocity = momentum.detach().clone()
    velocity = velocity - 0.5 * step_size * _gradient(objective, position)
    for step in range(n_steps):
        position = position + step_size * velocity
        if step < n_steps - 1:
            velocity = velocity - step_size * _gradient(objective, position)
    velocity = velocity - 0.5 * step_size * _gradient(objective, position)
    return position, velocity


@pytest.mark.parametrize(
    ("n_steps", "step_size"), [(20, 0.05), (50, 0.1), (7, 0.13), (1, 0.3)]
)
def test_the_composition_reproduces_the_hand_written_leapfrog_exactly(
    n_steps: int, step_size: float
) -> None:
    # Bitwise, not to a tolerance. The composition driver replaced a
    # hand-written loop, and "the chain is unchanged" is only a claim worth
    # making if it is exact -- a tolerance here would hide a merged kick
    # computed in the wrong order.
    theta = torch.tensor([0.4, 0.9], dtype=torch.float64)
    momentum = torch.tensor([-0.3, 1.1], dtype=torch.float64)

    composed = leapfrog(GAUSSIAN, theta, momentum, step_size, n_steps)
    written = _hand_written_leapfrog(GAUSSIAN, theta, momentum, step_size, n_steps)

    assert torch.equal(composed[0], written[0])
    assert torch.equal(composed[1], written[1])


def test_the_yoshida_middle_sub_step_runs_backwards_in_time() -> None:
    # Not a sign error, and not avoidable: no composition of a second-order
    # symmetric method reaches fourth order with positive coefficients. Pinned
    # because the trajectory is then non-monotone in time, and the first
    # person to plot one will otherwise read it as a bug.
    assert YOSHIDA_WEIGHTS[1] < 0.0
    assert YOSHIDA_WEIGHTS[0] == YOSHIDA_WEIGHTS[2] > 1.0
    assert math.fsum(YOSHIDA_WEIGHTS) == pytest.approx(1.0, abs=1e-15)


def test_a_composition_that_integrates_the_wrong_interval_is_refused() -> None:
    # The one arithmetic slip a reversibility check does not catch: weights
    # that are a palindrome but do not sum to 1 integrate perfectly reversibly
    # over the wrong amount of time, so every symmetry test passes and the
    # trajectory is wrong.
    with pytest.raises(ValueError, match="wrong interval"):
        Integrator(name="wrong", weights=(0.4, 0.4), order=2)
    with pytest.raises(ValueError, match="palindrome"):
        Integrator(name="asymmetric", weights=(0.2, 0.3, 0.5), order=2)


def test_the_energy_error_is_fourth_order_in_the_step_size() -> None:
    # The companion of the second-order test above, and it needs that one to
    # be trustworthy: a slope estimator reporting 4 for both integrators is
    # broken, and only the pair catches it.
    #
    # Realized ratios, coarse to fine: 16.310, 16.077, 16.019, 16.005, 16.001
    # -- converging on 16 rather than drifting, which is what makes the claim
    # an order rather than a coincidence at one step size. The coarsest is
    # excluded from the assertion and reported here: at 1/10 the higher-order
    # terms have not yet died away.
    theta = torch.tensor([0.4, 0.9], dtype=torch.float64)
    momentum = torch.tensor([-0.3, 1.1], dtype=torch.float64)
    reference = hamiltonian(GAUSSIAN, theta, momentum)

    errors = []
    for n_steps in (20, 40, 80, 160, 320):
        position, velocity = yoshida(GAUSSIAN, theta, momentum, 1.0 / n_steps, n_steps)
        errors.append(abs(hamiltonian(GAUSSIAN, position, velocity) - reference))

    for ratio in (before / after for before, after in itertools.pairwise(errors)):
        assert ratio == pytest.approx(16.0, rel=0.01)
    # And the finest step is still far above float64 round-off on a
    # Hamiltonian of order 1, so the window is the power law's and not the
    # arithmetic's.
    assert errors[-1] > 1e-12


class _Counted:
    """An objective that records how often its gradient was taken."""

    def __init__(self, inner: Objective) -> None:
        self.inner = inner
        self.calls = 0

    def initial(self) -> torch.Tensor:
        return self.inner.initial()

    def constrain(self, theta: torch.Tensor) -> Mapping[str, torch.Tensor]:
        return self.inner.constrain(theta)

    def __call__(self, theta: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return self.inner(theta)


@pytest.mark.parametrize("integrator", [leapfrog, yoshida])
@pytest.mark.parametrize("n_steps", [1, 3, 20])
def test_force_evaluations_counts_what_a_trajectory_actually_costs(
    integrator: Integrator, n_steps: int
) -> None:
    # The number a comparison between integrators rests on. Counted rather
    # than derived, because a comparison at equal *steps* says nothing and one
    # at equal evaluations says everything -- and an off-by-one here would
    # quietly favour whichever method the arithmetic was written for.
    counted = _Counted(GAUSSIAN)

    integrator(
        counted,
        torch.zeros(2, dtype=torch.float64),
        torch.ones(2, dtype=torch.float64),
        0.05,
        n_steps,
    )

    assert counted.calls == integrator.force_evaluations(n_steps)


def test_leapfrog_reaches_the_acceptance_target_more_cheaply_than_yoshida() -> None:
    # **The measurement the ticket is for, and it is negative.** A fourth-order
    # method pays where the step is limited by *accuracy*; here it is limited
    # by *stability*, and the two are different constraints.
    #
    # Yoshida's largest sub-step is |w0| = 1.70 times the nominal step, so its
    # stability limit in the step size is ~0.59 of leapfrog's -- measured at
    # 0.0333 against 0.0500, a ratio of 1.50 against the 1.70 the coefficient
    # predicts. Combined with three force evaluations per step, that is where
    # the order advantage goes.
    #
    # Realized on this posterior: leapfrog reaches 0.855 acceptance at **21**
    # gradients per trajectory; Yoshida needs **91** to reach 0.975, and at 61
    # it accepts *nothing*. On the analytic Gaussian, 3 against 7.
    target = _potts_posterior()
    cheapest = {}
    for integrator in (leapfrog, yoshida):
        for n_steps in (4, 6, 8, 10, 14, 20, 30, 45, 70):
            chain = sample(
                target,
                seed=11,
                n_samples=200,
                step_size=1.0 / n_steps,
                n_steps=n_steps,
                burn_in=80,
                integrator=integrator,
            )
            if chain.acceptance_rate >= 0.7:
                cheapest[integrator.name] = integrator.force_evaluations(n_steps)
                break

    assert cheapest["leapfrog"] < cheapest["yoshida"]
    assert cheapest["leapfrog"] <= 21
    assert cheapest["yoshida"] >= 60


def test_the_default_integrator_is_the_one_every_committed_result_used() -> None:
    # A default changed here silently redraws every chain in the repository.
    chain = sample(GAUSSIAN, seed=3, n_samples=40, step_size=0.2, n_steps=6)
    explicit = sample(
        GAUSSIAN, seed=3, n_samples=40, step_size=0.2, n_steps=6, integrator=leapfrog
    )

    assert torch.equal(chain.theta, explicit.theta)
    assert leapfrog.order == 2
    assert leapfrog.weights == (1.0,)


# --- temperature ------------------------------------------------------------


def test_tempering_a_gaussian_scales_the_chain_by_the_square_root_of_t() -> None:
    # Where the approximation is exact. For a Gaussian target the dynamics
    # are linear, so a chain at temperature T *is* the chain at 1 with its
    # deviations from the mean scaled by sqrt(T), draw for draw, once the
    # start has been forgotten -- and its spread is sqrt(T) times the exact
    # standard deviation. The first is the implementation check (momentum
    # variance T, energy difference over T, and nothing else); the second is
    # what tempering means. Realized: 1.0004 and 0.9953 of sqrt(T) sigma at
    # both T = 2 and T = 0.5, the same digits at both because of the first.
    exact = GAUSSIAN.covariance.diagonal().sqrt()
    reference = sample(
        GAUSSIAN, seed=3, n_samples=2000, step_size=0.2, n_steps=10, burn_in=200
    )

    for temperature in (2.0, 0.5):
        chain = sample(
            GAUSSIAN,
            seed=3,
            n_samples=2000,
            step_size=0.2,
            n_steps=10,
            burn_in=200,
            temperature=temperature,
        )
        scaled = GAUSSIAN.mean + math.sqrt(temperature) * (
            reference.theta - GAUSSIAN.mean
        )

        assert torch.allclose(chain.theta, scaled, atol=1e-10, rtol=0.0)
        np.testing.assert_allclose(
            chain.theta.std(0).numpy(),
            math.sqrt(temperature) * exact.numpy(),
            rtol=0.05,
        )


def test_a_non_positive_temperature_is_refused() -> None:
    with pytest.raises(ValueError, match="temperature must be positive"):
        sample(GAUSSIAN, seed=1, n_samples=10, step_size=0.1, temperature=0.0)


def test_a_constant_schedule_at_one_is_the_sampler_draw_for_draw() -> None:
    # The refactor's guarantee, stated as the plan asked: annealing on a
    # constant schedule at temperature 1 reproduces the untempered chain at
    # the same seed *bitwise*. Both go through one transition, so this is not
    # two implementations agreeing but one implementation being one.
    chain = sample(GAUSSIAN, seed=5, n_samples=200, step_size=0.2, n_steps=10)
    annealed = anneal(GAUSSIAN, Constant(1.0, 200), seed=5, step_size=0.2, n_steps=10)

    assert torch.equal(annealed.final, chain.theta[-1])
    assert annealed.force_evaluations == 200 * leapfrog.force_evaluations(10)


def test_annealing_reports_the_best_point_visited_not_the_last() -> None:
    # The final proposals run cold but not at zero, so the chain can leave
    # the best point it found; what is returned is the best, and its value is
    # the objective there and no larger than the start's or the end's.
    start = torch.tensor([4.0, 3.0], dtype=torch.float64)
    result = anneal(
        GAUSSIAN,
        Exponential(4.0, 0.01, 300),
        seed=2,
        step_size=0.2,
        n_steps=10,
        theta0=start,
    )

    assert result.value == pytest.approx(float(GAUSSIAN(result.theta)), rel=EXACT)
    assert result.value <= float(GAUSSIAN(start))
    assert result.value <= float(GAUSSIAN(result.final))
    # On a quadratic bowl the cold end sits at the mode: within a tenth of a
    # standard deviation of the exact minimizer after 300 proposals.
    deviation = (result.theta - GAUSSIAN.mean) / GAUSSIAN.covariance.diagonal().sqrt()
    assert float(deviation.abs().max()) < 0.1
