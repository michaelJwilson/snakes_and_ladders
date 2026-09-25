"""HMC, checked where it is exact before it is checked where it is statistical.

Exact first: the integrator is reversible and its energy error second order,
which say which half is wrong. Then the distribution against non-samplers: an
analytic Gaussian, a real `Objective` by grid quadrature, and a mixture summed
over all 4,096 assignments (issue #734). A too-large step biases the spread
while acceptance looks healthy (0.982 accepted, sd 12% low); the energy
error detects it, which is why `HmcChain` carries it.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping

import numpy as np
import pytest
import torch
from sal.emissions import ParameterDomainError
from sal.likelihood.mixture_assignments import (
    enumerate_mixture_assignments,
)
from sal.opt.constrain import log_simplex
from sal.opt.objective import Objective
from sal.opt.potts import PottsObjective
from sal.sample import hmc
from sal.sample.hmc import (
    YOSHIDA_WEIGHTS,
    Integrator,
    WithGaussianPrior,
    anneal,
    gradient_at,
    hamiltonian,
    leapfrog,
    parallel_tempering,
    sample,
    yoshida,
)
from sal.sample.schedule import (
    ConstantTempSchedule,
    ExponentialTempSchedule,
)
from sal.sample.tempered import (
    round_trip_time,
    round_trips,
    up_fraction,
)
from sal.sim.potts_chain import PottsParams, simulate_chains

from tests._objective_checks import Counted
from tests._posteriors import (
    GAUSSIAN,
    MIXTURE_PRIOR_SCALE,
    assert_recovers_assignment_posterior,
    enumerated_quadrature,
    weight_posterior,
)
from tests._rows import every_row, every_value
from tests._scale import stress_only

EXACT = 1e-13


def _potts_posterior() -> WithGaussianPrior:
    """A real `Objective` whose `theta` is 2-D, so quadrature can referee it.

    Takes the squaring route at ``q = 2``, chain 8; the pins held there (#754).
    """
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


@pytest.mark.analytic
@pytest.mark.parametrize("integrator", [leapfrog, yoshida])
def test_the_integrator_is_reversible(integrator: Integrator) -> None:
    # Run forward, negate the momentum, run forward again: the exact statement
    # that makes the Metropolis proposal symmetric. Without it the acceptance
    # ratio is not the energy difference alone and the chain targets the wrong
    # distribution, which no sampling would localize to here.
    def check(n_steps: int, step_size: float) -> None:
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

    every_row([(20, 0.05), (50, 0.1), (100, 0.02)], check)


@pytest.mark.analytic
def test_the_energy_error_is_second_order_in_the_step_size() -> None:
    # Halving the step must quarter the error at fixed trajectory length; at a
    # fixed step count the errors oscillate around the orbit.
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


@pytest.mark.oracle
def test_the_chain_recovers_an_analytic_gaussian() -> None:
    # Mean and covariance in closed form, so nothing here rests on a second
    # sampler. The tolerance is set from the spread across eight independent
    # chains, measured at 0.030 and 0.041 on the two coordinates.
    chain = sample(
        GAUSSIAN,
        generator=torch.Generator().manual_seed(11),
        n_samples=4000,
        step_size=0.25,
        n_steps=12,
        burn_in=400,
    )

    mean = chain.theta.mean(dim=0)
    covariance = torch.cov(chain.theta.T)

    assert chain.acceptance_rate > 0.9
    np.testing.assert_allclose(mean.numpy(), GAUSSIAN.mean.numpy(), atol=0.12)
    np.testing.assert_allclose(
        covariance.numpy(), GAUSSIAN.covariance.numpy(), atol=0.2
    )


@pytest.mark.oracle
@stress_only(
    "the second moment needs pooled 2000-draw chains; a shorter run "
    "estimates the spread badly and would assert less than it claims"
)
def test_the_chain_matches_grid_quadrature_on_a_real_objective() -> None:
    # 2-D `theta`, so quadrature referees. Three pooled chains: one 2000-draw
    # chain put the spread 18% high, as the second moment mixes slowly.
    posterior = _potts_posterior()

    draws = torch.cat(
        [
            sample(
                posterior,
                generator=torch.Generator().manual_seed(200 + index),
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


@pytest.mark.analytic
@stress_only(
    "the bias is in the spread, which needs the same chain lengths "
    "as the test above before it is measurable at all"
)
def test_a_step_size_that_diverges_biases_the_spread_not_the_mean() -> None:
    # Against quadrature:
    #
    #   step   accept   max|dH|   sd / true
    #   0.050   0.799   9.3e+00   0.92, 0.93
    #   0.020   0.982   3.1e-01   0.88, 0.98
    #   0.005   1.000   8.1e-03   1.07, 1.04
    #   0.002   1.000   5.1e-03   1.01, 1.00
    #
    # At 0.020 acceptance looks healthy and sd is 12% low: the energy error tracks it.
    posterior = _potts_posterior()

    coarse = sample(
        posterior,
        generator=torch.Generator().manual_seed(3),
        n_samples=400,
        step_size=0.05,
        n_steps=20,
        burn_in=50,
    )
    fine = sample(
        posterior,
        generator=torch.Generator().manual_seed(3),
        n_samples=400,
        step_size=0.005,
        n_steps=40,
        burn_in=50,
    )

    assert float(coarse.energy_error.max()) > 100.0 * float(fine.energy_error.max())
    # The mean survives a step size that the spread does not.
    assert abs(float(coarse.theta.mean(dim=0)[0]) - QUADRATURE_MEAN[0]) < 0.01
    assert float(coarse.theta.std(dim=0)[0]) < float(fine.theta.std(dim=0)[0])


@pytest.mark.analytic
def test_the_posterior_is_wider_than_the_laplace_approximation_predicts() -> None:
    # The Laplace standard error is the curvature at the mode; the HMC one is
    # a quantile of the posterior. They agree on this fixture, the expected
    # outcome for a well-identified two-parameter model, which is what makes a
    # *disagreement* elsewhere informative.
    posterior = _potts_posterior()
    mode = torch.tensor(QUADRATURE_MEAN, dtype=torch.float64)

    hessian = torch.autograd.functional.hessian(  # type: ignore[no-untyped-call]
        posterior.__call__, mode
    )
    laplace = torch.linalg.inv(hessian).diagonal().sqrt()

    np.testing.assert_allclose(laplace.numpy(), np.array(QUADRATURE_SD), rtol=0.15)


@pytest.mark.smoke
def test_a_zero_length_trajectory_is_refused() -> None:
    # It would propose the current point every time: acceptance rate 1, energy
    # error 0, and a chain that has not moved. Every diagnostic reports health.
    with pytest.raises(ValueError, match="looks healthy and samples nothing"):
        sample(
            GAUSSIAN,
            generator=torch.Generator().manual_seed(1),
            n_samples=10,
            step_size=0.1,
            n_steps=0,
        )


@pytest.mark.smoke
def test_a_non_positive_step_size_is_refused() -> None:
    with pytest.raises(ValueError, match="step_size must be positive"):
        sample(
            GAUSSIAN,
            generator=torch.Generator().manual_seed(1),
            n_samples=10,
            step_size=0.0,
        )


@pytest.mark.smoke
def test_a_non_positive_prior_scale_is_refused() -> None:
    with pytest.raises(ValueError, match="prior scale must be positive"):
        WithGaussianPrior(GAUSSIAN, scale=0.0)


@pytest.mark.analytic
def test_the_prior_is_added_to_the_objective_and_nothing_else() -> None:
    # The wrapper turns a likelihood into something with a posterior; if it
    # changed the likelihood term the chain would silently target a different
    # model.
    point = torch.tensor([0.3, -1.1], dtype=torch.float64)
    wrapped = WithGaussianPrior(GAUSSIAN, scale=2.0)

    expected = float(GAUSSIAN(point)) + float((point * point).sum()) / (2.0 * 4.0)

    assert float(wrapped(point)) == pytest.approx(expected, rel=EXACT)


@pytest.mark.analytic
def test_the_prior_leaves_the_coordinates_it_is_stated_in_alone() -> None:
    # The prior is isotropic *in unconstrained coordinates*, so the wrapper
    # adds a term and changes no coordinate. An inverse of its own would put
    # an interval read at a sampled point in the wrong units.
    point = torch.tensor([0.3, -1.1], dtype=torch.float64)
    wrapped = WithGaussianPrior(GAUSSIAN, scale=2.0)

    assert torch.equal(wrapped.theta_from(wrapped.constrain(point)), point)


@pytest.mark.smoke
def test_a_chain_is_reproducible_from_generators_seeded_alike() -> None:
    first = sample(
        GAUSSIAN,
        generator=torch.Generator().manual_seed(5),
        n_samples=50,
        step_size=0.2,
        n_steps=10,
    )
    second = sample(
        GAUSSIAN,
        generator=torch.Generator().manual_seed(5),
        n_samples=50,
        step_size=0.2,
        n_steps=10,
    )

    assert torch.equal(first.theta, second.theta)


# --- the fourth-order composition (#266) ----------------------------------


def _hand_written_leapfrog(
    objective: Objective,
    theta: torch.Tensor,
    momentum: torch.Tensor,
    step_size: float,
    n_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Kick-drift-kick written out: the reference the composed `leapfrog` is pinned to."""
    position = theta.detach().clone()
    velocity = momentum.detach().clone()
    velocity = velocity - 0.5 * step_size * gradient_at(objective, position)
    for step in range(n_steps):
        position = position + step_size * velocity
        if step < n_steps - 1:
            velocity = velocity - step_size * gradient_at(objective, position)
    velocity = velocity - 0.5 * step_size * gradient_at(objective, position)
    return position, velocity


@pytest.mark.analytic
def test_the_composition_reproduces_the_hand_written_leapfrog_exactly() -> None:
    # Bitwise, not to a tolerance. The composition driver replaced a
    # hand-written loop, and a tolerance here would hide a merged kick
    # computed in the wrong order.
    def check(n_steps: int, step_size: float) -> None:
        theta = torch.tensor([0.4, 0.9], dtype=torch.float64)
        momentum = torch.tensor([-0.3, 1.1], dtype=torch.float64)

        composed = leapfrog(GAUSSIAN, theta, momentum, step_size, n_steps)
        written = _hand_written_leapfrog(GAUSSIAN, theta, momentum, step_size, n_steps)

        assert torch.equal(composed.position, written[0])
        assert torch.equal(composed.momentum, written[1])

    every_row([(20, 0.05), (50, 0.1), (7, 0.13), (1, 0.3)], check)


@pytest.mark.analytic
def test_the_yoshida_middle_sub_step_runs_backwards_in_time() -> None:
    # Not a sign error, and not avoidable: no composition of a second-order
    # symmetric method reaches fourth order with positive coefficients. Pinned
    # because the trajectory is then non-monotone in time, and the first
    # person to plot one will otherwise read it as a bug.
    assert YOSHIDA_WEIGHTS[1] < 0.0
    assert YOSHIDA_WEIGHTS[0] == YOSHIDA_WEIGHTS[2] > 1.0
    assert math.fsum(YOSHIDA_WEIGHTS) == pytest.approx(1.0, abs=1e-15)


@pytest.mark.smoke
def test_a_composition_that_integrates_the_wrong_interval_is_refused() -> None:
    # The one arithmetic slip a reversibility check does not catch: weights
    # that are a palindrome but do not sum to 1 integrate perfectly reversibly
    # over the wrong amount of time, so every symmetry test passes and the
    # trajectory is wrong.
    with pytest.raises(ValueError, match="wrong interval"):
        Integrator(name="wrong", weights=(0.4, 0.4), order=2)
    with pytest.raises(ValueError, match="palindrome"):
        Integrator(name="asymmetric", weights=(0.2, 0.3, 0.5), order=2)


@pytest.mark.analytic
def test_the_energy_error_is_fourth_order_in_the_step_size() -> None:
    # With the second-order test, catches a slope estimator reporting 4 for
    # both. Ratios 16.310, 16.077, 16.019, 16.005, 16.001; the coarsest (1/10)
    # is excluded from the assertion.
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


@pytest.mark.smoke
@pytest.mark.parametrize("integrator", [leapfrog, yoshida])
def test_force_evaluations_counts_what_a_trajectory_actually_costs(
    integrator: Integrator,
) -> None:
    # The number a comparison between integrators rests on. Counted rather
    # than derived: a comparison at equal *steps* says nothing, and an
    # off-by-one here would quietly favour one method.
    def check(n_steps: int) -> None:
        counted = Counted(GAUSSIAN)

        integrator(
            counted,
            torch.zeros(2, dtype=torch.float64),
            torch.ones(2, dtype=torch.float64),
            0.05,
            n_steps,
        )

        assert counted.calls == integrator.force_evaluations(n_steps)

    every_value([1, 3, 20], check)


@pytest.mark.analytic
def test_leapfrog_reaches_the_acceptance_target_more_cheaply_than_yoshida() -> None:
    # Negative: the step is limited by stability, not accuracy. Yoshida's
    # |w0| = 1.70 gives ~0.59 of leapfrog's limit (0.0333 against 0.0500).
    # Leapfrog: 0.855 acceptance at 21 gradients; Yoshida 0.975 at 91, nothing
    # at 61; analytic Gaussian 3 against 7.
    target = _potts_posterior()
    cheapest = {}
    for integrator in (leapfrog, yoshida):
        for n_steps in (4, 6, 8, 10, 14, 20, 30, 45, 70):
            chain = sample(
                target,
                generator=torch.Generator().manual_seed(11),
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


@pytest.mark.smoke
def test_the_default_integrator_is_the_one_every_committed_result_used() -> None:
    # A default changed here silently redraws every chain in the repository.
    chain = sample(
        GAUSSIAN,
        generator=torch.Generator().manual_seed(3),
        n_samples=40,
        step_size=0.2,
        n_steps=6,
    )
    explicit = sample(
        GAUSSIAN,
        generator=torch.Generator().manual_seed(3),
        n_samples=40,
        step_size=0.2,
        n_steps=6,
        integrator=leapfrog,
    )

    assert torch.equal(chain.theta, explicit.theta)
    assert leapfrog.order == 2
    assert leapfrog.weights == (1.0,)


# --- temperature ------------------------------------------------------------


@pytest.mark.smoke
def test_tempering_a_gaussian_scales_the_chain_by_the_square_root_of_t() -> None:
    # Gaussian dynamics are linear: a chain at T is the chain at 1 scaled by
    # sqrt(T), draw for draw. Realized 1.0004 and 0.9953 of sqrt(T) sigma at
    # T = 2 and 0.5.
    exact = GAUSSIAN.covariance.diagonal().sqrt()
    reference = sample(
        GAUSSIAN,
        generator=torch.Generator().manual_seed(3),
        n_samples=2000,
        step_size=0.2,
        n_steps=10,
        burn_in=200,
    )

    for temperature in (2.0, 0.5):
        chain = sample(
            GAUSSIAN,
            generator=torch.Generator().manual_seed(3),
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


@pytest.mark.smoke
def test_a_non_positive_temperature_is_refused() -> None:
    with pytest.raises(ValueError, match="temperature must be positive"):
        sample(
            GAUSSIAN,
            generator=torch.Generator().manual_seed(1),
            n_samples=10,
            step_size=0.1,
            temperature=0.0,
        )


@pytest.mark.oracle
def test_a_constant_schedule_at_one_is_the_sampler_draw_for_draw() -> None:
    # Annealing on a constant schedule at temperature 1 reproduces the
    # untempered chain at generators seeded alike, *bitwise*. Both go through
    # one transition, so this is one implementation being one rather than two
    # agreeing.
    chain = sample(
        GAUSSIAN,
        generator=torch.Generator().manual_seed(5),
        n_samples=200,
        step_size=0.2,
        n_steps=10,
    )
    annealed = anneal(
        GAUSSIAN,
        ConstantTempSchedule(1.0, 200),
        generator=torch.Generator().manual_seed(5),
        step_size=0.2,
        n_steps=10,
    )

    assert torch.equal(annealed.final, chain.theta[-1])
    assert annealed.force_evaluations == 200 * leapfrog.force_evaluations(10)


@pytest.mark.smoke
def test_annealing_reports_the_best_point_visited_not_the_last() -> None:
    # The final proposals run cold but not at zero, so the chain can leave
    # the best point it found; what is returned is the best, and its value is
    # the objective there and no larger than the start's or the end's.
    start = torch.tensor([4.0, 3.0], dtype=torch.float64)
    result = anneal(
        GAUSSIAN,
        ExponentialTempSchedule(4.0, 0.01, 300),
        generator=torch.Generator().manual_seed(2),
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


@pytest.mark.analytic
def test_each_tempering_replica_samples_the_gaussian_at_its_own_temperature() -> None:
    # Replica r targets exp(-U / T_r); the ladder is tight enough that swaps
    # happen, since four independent chains would pass without exchanging.
    ladder = (1.0, 2.0, 4.0)
    run = parallel_tempering(
        GAUSSIAN,
        ladder,
        generator=torch.Generator().manual_seed(11),
        n_rounds=2000,
        step_size=0.25,
        n_steps=12,
    )
    draws = run.positions[400:]

    for replica, temperature in enumerate(ladder):
        mean = draws[:, replica].mean(dim=0)
        centred = draws[:, replica] - mean
        covariance = centred.T @ centred / (centred.shape[0] - 1)
        np.testing.assert_allclose(
            mean.numpy(), GAUSSIAN.mean.numpy(), atol=0.12 * math.sqrt(temperature)
        )
        np.testing.assert_allclose(
            covariance.numpy(),
            temperature * GAUSSIAN.covariance.numpy(),
            atol=0.2 * temperature,
        )
    assert bool((run.swap_acceptance > 0.2).all()), run.swap_acceptance
    assert bool((run.swap_acceptance < 1.0).all()), run.swap_acceptance
    assert run.value == pytest.approx(float(GAUSSIAN(run.theta)), rel=EXACT)


@pytest.mark.smoke
def test_tempering_costs_what_its_accounting_says_and_is_reproducible() -> None:
    # One value at the start; then per replica per round one Hamiltonian at
    # the current point, the trajectory's gradients, one at the proposal and
    # the value where the replica landed. Counted, for the reason
    # `test_force_evaluations_counts_what_a_trajectory_actually_costs` gives.
    counted = Counted(GAUSSIAN)
    ladder = (1.0, 2.0, 4.0, 8.0)

    run = parallel_tempering(
        counted,
        ladder,
        generator=torch.Generator().manual_seed(5),
        n_rounds=25,
        step_size=0.2,
        n_steps=6,
    )

    assert run.force_evaluations == 25 * 4 * leapfrog.force_evaluations(6)
    assert counted.calls == 1 + 25 * 4 * (leapfrog.force_evaluations(6) + 3)
    again = parallel_tempering(
        GAUSSIAN,
        ladder,
        generator=torch.Generator().manual_seed(5),
        n_rounds=25,
        step_size=0.2,
        n_steps=6,
    )
    assert torch.equal(run.positions, again.positions)
    assert torch.equal(run.theta, again.theta)
    assert run.positions.shape == (25, 4, 2)


@pytest.mark.analytic
def test_the_walker_trace_counts_the_round_trips_a_two_rung_ladder_makes() -> None:
    # The trace on the continuous ladder (#861): two rungs make it exact, read
    # directly and by `tempered.round_trips`.
    rounds = 200
    run = parallel_tempering(
        GAUSSIAN,
        (1.0, 4.0),
        generator=torch.Generator().manual_seed(12),
        n_rounds=rounds,
        step_size=0.3,
        n_steps=5,
    )
    trace = run.walkers

    assert trace.shape == (rounds, 2)
    np.testing.assert_array_equal(trace.sum(axis=1), np.ones(rounds, dtype=np.int64))
    changed = int((np.diff(trace[:, 0]) != 0).sum()) + int(trace[0, 0] != 0)
    assert float(run.swap_acceptance[0]) == changed / rounds
    returned = ((trace[1:] == 0) & (trace[:-1] == 1)).sum(axis=0)
    np.testing.assert_array_equal(round_trips(trace), returned)
    assert returned.sum() > 0, trace
    # Rung 0 is where the ascent is labelled and the last rung is where the
    # descent is: on a ladder of two there is no rung between them.
    np.testing.assert_array_equal(up_fraction(trace), np.array([1.0, 0.0]))
    assert round_trip_time(trace) == rounds * 2 / float(returned.sum())


@pytest.mark.smoke
def test_a_ladder_that_cannot_exchange_is_refused() -> None:
    with pytest.raises(ValueError, match="at least two temperatures"):
        parallel_tempering(
            GAUSSIAN,
            (1.0,),
            generator=torch.Generator().manual_seed(1),
            n_rounds=5,
            step_size=0.2,
        )
    with pytest.raises(ValueError, match="positive and increasing"):
        parallel_tempering(
            GAUSSIAN,
            (2.0, 1.0),
            generator=torch.Generator().manual_seed(1),
            n_rounds=5,
            step_size=0.2,
        )
    with pytest.raises(ValueError, match="positive and increasing"):
        parallel_tempering(
            GAUSSIAN,
            (0.0, 1.0),
            generator=torch.Generator().manual_seed(1),
            n_rounds=5,
            step_size=0.2,
        )
    with pytest.raises(ValueError, match="n_rounds must be at least 1"):
        parallel_tempering(
            GAUSSIAN,
            (1.0, 2.0),
            generator=torch.Generator().manual_seed(1),
            n_rounds=0,
            step_size=0.2,
        )
    with pytest.raises(ValueError, match="n_steps must be at least 1"):
        parallel_tempering(
            GAUSSIAN,
            (1.0, 2.0),
            generator=torch.Generator().manual_seed(1),
            n_rounds=5,
            step_size=0.2,
            n_steps=0,
        )


#: The chain: draws kept, leapfrog step, steps per proposal, proposals
#: discarded first. The tolerance every comparison below is declared against
#: is four standard errors at this draw count, formed from the chain's own
#: sample standard deviation.
MIXTURE_DRAWS = 800
MIXTURE_STEP = 0.9
MIXTURE_TRAJECTORY = 10
MIXTURE_BURN_IN = 100
MONTE_CARLO_SIGMAS = 4.0


@pytest.mark.oracle
@pytest.mark.release
def test_the_chain_recovers_the_enumerated_assignment_posterior_of_a_mixture() -> None:
    # The rung below (#734): assignment enumeration over 4,096 assignments. At
    # 50 draws the objective is the negated enumerated evidence plus the prior,
    # to 4.4e-16 (1e-12). Over 800 draws the weight is 2.0e-03 off the
    # integrated posterior mean (2.0e-02 declared, four standard errors) and
    # the E-step responsibilities 2.1e-03 at worst (2.0e-02); acceptance 0.83.
    # Twelve observations and one free coordinate: the declared fixture's
    # 5 ** 500 assignments have no reference.
    target, observations, components = weight_posterior()
    reference = enumerated_quadrature(observations, components)

    chain = sample(
        target,
        generator=torch.Generator().manual_seed(734),
        n_samples=MIXTURE_DRAWS,
        step_size=MIXTURE_STEP,
        n_steps=MIXTURE_TRAJECTORY,
        burn_in=MIXTURE_BURN_IN,
    )
    assert chain.acceptance_rate > 0.6, chain.acceptance_rate

    worst = 0.0
    for theta in chain.theta[:: MIXTURE_DRAWS // 50]:
        enumerated = enumerate_mixture_assignments(
            torch.exp(log_simplex(theta)).numpy(), components, observations
        )
        expected = -enumerated.log_evidence + 0.5 * float(theta[0]) ** 2 / (
            MIXTURE_PRIOR_SCALE**2
        )
        worst = max(worst, abs(float(target(theta)) / expected - 1.0))
    assert worst < 1e-12, worst

    assert_recovers_assignment_posterior(
        chain.theta,
        observations,
        components,
        reference,
        sigmas=MONTE_CARLO_SIGMAS,
        size=MIXTURE_DRAWS,
    )


class _Bounded(Objective):
    """A standard Gaussian on ``|theta| < 1`` whose family refuses outside it."""

    def initial(self) -> torch.Tensor:
        return torch.zeros(2, dtype=torch.float64)

    def constrain(self, theta: torch.Tensor) -> Mapping[str, torch.Tensor]:
        return {"theta": theta}

    def theta_from(self, named: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return named["theta"]

    def __call__(self, theta: torch.Tensor) -> torch.Tensor:
        if bool((theta.abs() >= 1.0).any()):
            msg = "every coordinate must lie inside the unit box"
            raise ParameterDomainError(msg)
        return 0.5 * (theta * theta).sum()


@pytest.mark.smoke
def test_a_divergent_trajectory_is_a_rejected_proposal() -> None:
    # Issue #912: a trajectory that leaves the objective's domain, where its
    # family refuses the parameters, is rejected as a proposal of infinite
    # energy is, and the chain runs on inside the domain; before, the refusal
    # stopped the chain.
    chain = hmc.sample(
        _Bounded(), torch.Generator().manual_seed(912), 200, step_size=0.4, n_steps=8
    )
    assert 0.0 < chain.acceptance_rate < 1.0
    assert bool((chain.theta.abs() < 1.0).all())
    assert any(math.isinf(float(error)) for error in chain.energy_error)
