"""Adaptation, pinned where it is exact before it is pinned where it is statistical.

Three statements are arithmetic and need no chain: the dual-averaging
iteration is Hoffman & Gelman's, line for line (``eq:dual-averaging``); a
diagonal mass matrix is a change of coordinates, so the scaled integrator
reproduces a hand-written mass-matrix leapfrog; and the effective sample
size estimator recovers a known integrated autocorrelation time. Then the
statistics: the acceptance lands at its target, the adapted chain and the
fixed one agree on the posterior within Monte Carlo error, and what a draw
costs in gradients is reported for both (issue #333).

The measurement that shaped the module is recorded on
:class:`~snakes_and_ladders.opt.hmc.Adaptation`: on a locally quadratic target
the acceptance is a cliff in the step size, and dual averaging at a
single-proposal statistic oscillates across it. The step jitter and the
dual-averaging gain are set from that measurement, and the tests below pin
what they achieve rather than the constants themselves.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import pytest
import torch
from snakes_and_ladders.likelihood.objective import BranchLengthObjective
from snakes_and_ladders.opt.fit import fit, standard_errors_at
from snakes_and_ladders.opt.hmc import (
    DUAL_AVERAGING_GAMMA,
    DUAL_AVERAGING_KAPPA,
    DUAL_AVERAGING_T0,
    Adaptation,
    HmcChain,
    WithGaussianPrior,
    _DualAveraging,
    _gradient,
    _Scaled,
    effective_sample_size,
    leapfrog,
    sample,
)
from snakes_and_ladders.opt.objective import Objective
from snakes_and_ladders.sim.simulate import simulate_alignment

from tests._fixtures import FOUR_TAXA, load_fixture
from tests._objective_checks import AnalyticGaussian
from tests._scale import at_scale

EXACT = 1e-13

GAUSSIAN = AnalyticGaussian([1.0, -2.0], [[2.0, 0.6], [0.6, 0.5]])
TARGET = 0.65
ADAPTATION = Adaptation(warmup=300, target_acceptance=TARGET, step_jitter=0.4)


def _four_taxon_posterior() -> WithGaussianPrior:
    """Log branch lengths of the four-taxon fixture at 500 sites, under a N(0, 2^2) prior."""
    params = load_fixture(FOUR_TAXA)
    dataset = simulate_alignment(
        tau=params.tau,
        k=params.k,
        pi=params.pi,
        rng=np.random.default_rng(params.seed),
        n_sites=500,
    )
    return WithGaussianPrior(
        BranchLengthObjective(params.tau, params.k, params.pi, dict(dataset.alignment)),
        scale=2.0,
    )


def _ess_per_gradient(chain: HmcChain) -> np.ndarray:
    return (effective_sample_size(chain.theta) / chain.force_evaluations).numpy()


# --- exact ------------------------------------------------------------------


@pytest.mark.oracle
def test_the_dual_averaging_iteration_is_hoffman_and_gelmans() -> None:
    # Algorithm 5 of Hoffman & Gelman (2014), written out with their symbols
    # and stepped by hand for four acceptance statistics, against the class.
    # Every quantity is compared, not only the step, so a wrong constant in
    # the averaging that happened to leave the iterate right would be caught.
    step0, target = 0.1, 0.65
    mu = math.log(10.0 * step0)
    h_bar, log_step_bar = 0.0, 0.0
    averaging = _DualAveraging(step0, target)
    for m, alpha in enumerate((0.2, 0.9, 1.0, 0.5), start=1):
        h_bar = (1.0 - 1.0 / (m + DUAL_AVERAGING_T0)) * h_bar + (target - alpha) / (
            m + DUAL_AVERAGING_T0
        )
        log_step = mu - math.sqrt(m) / DUAL_AVERAGING_GAMMA * h_bar
        weight = m**-DUAL_AVERAGING_KAPPA
        log_step_bar = weight * log_step + (1.0 - weight) * log_step_bar

        step = averaging.update(alpha)

        assert step == pytest.approx(math.exp(log_step), rel=EXACT)
        assert averaging.h_bar == pytest.approx(h_bar, rel=EXACT)
        assert averaging.averaged == pytest.approx(math.exp(log_step_bar), rel=EXACT)


@pytest.mark.mathematical
def test_dual_averaging_moves_the_step_against_the_acceptance() -> None:
    # The sign of the update, which a transposed `target - alpha` would
    # flip while every magnitude stayed plausible: proposals accepted more
    # often than the target grow the step, less often shrink it, and a
    # statistic exactly at the target leaves the iterate at `mu`.
    grows = _DualAveraging(0.1, 0.65)
    shrinks = _DualAveraging(0.1, 0.65)
    holds = _DualAveraging(0.1, 0.65)
    for _ in range(20):
        up = grows.update(1.0)
        down = shrinks.update(0.0)
        flat = holds.update(0.65)

    assert up > 1.0
    assert down < 0.1
    assert flat == pytest.approx(1.0, rel=EXACT)  # mu = log(10 * 0.1)


def _mass_matrix_leapfrog(
    objective: Objective,
    theta: torch.Tensor,
    momentum: torch.Tensor,
    inverse_mass: torch.Tensor,
    step_size: float,
    n_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Kick-drift-kick with ``K = p' M^-1 p / 2``, as Neal (2011) §5.4.1 writes it."""
    position, velocity = theta.clone(), momentum.clone()
    velocity = velocity - 0.5 * step_size * _gradient(objective, position)
    for step in range(n_steps):
        position = position + step_size * inverse_mass * velocity
        if step < n_steps - 1:
            velocity = velocity - step_size * _gradient(objective, position)
    velocity = velocity - 0.5 * step_size * _gradient(objective, position)
    return position, velocity


@pytest.mark.mathematical
def test_a_diagonal_mass_matrix_is_a_change_of_coordinates() -> None:
    # `_Scaled` is the whole implementation of the mass matrix, so this is
    # the statement that it *is* one: the unit-mass leapfrog on the scaled
    # objective, mapped back, equals the mass-matrix leapfrog on the
    # original to round-off, and the Hamiltonian is the same number in
    # both coordinate systems.
    inverse_mass = torch.tensor([2.5, 0.4], dtype=torch.float64)
    scale = inverse_mass.sqrt()
    theta = torch.tensor([0.4, 0.9], dtype=torch.float64)
    momentum = torch.tensor([-0.3, 1.1], dtype=torch.float64)

    reference = _mass_matrix_leapfrog(GAUSSIAN, theta, momentum, inverse_mass, 0.1, 25)
    phi, q = leapfrog(
        _Scaled(GAUSSIAN, scale), theta / scale, momentum * scale, 0.1, 25
    )

    assert float((phi * scale - reference[0]).abs().max()) < 1e-12
    assert float((q / scale - reference[1]).abs().max()) < 1e-12
    kinetic_original = 0.5 * float((momentum * momentum * inverse_mass).sum())
    kinetic_scaled = 0.5 * float(((momentum * scale) ** 2).sum())
    assert kinetic_scaled == pytest.approx(kinetic_original, rel=EXACT)


@pytest.mark.structural
def test_the_scaled_objective_inverts_its_own_map() -> None:
    scale = torch.tensor([2.5, 0.4], dtype=torch.float64)
    scaled = _Scaled(GAUSSIAN, scale)
    point = torch.tensor([0.3, -1.1], dtype=torch.float64)

    assert torch.equal(scaled.theta_from(scaled.constrain(point)), point)
    assert torch.equal(scaled.initial() * scale, GAUSSIAN.initial())
    assert float(scaled(point)) == float(GAUSSIAN(point * scale))


@pytest.mark.oracle
@pytest.mark.parametrize("phi", [0.0, 0.5, 0.9])
def test_the_effective_sample_size_recovers_an_ar1_autocorrelation_time(
    phi: float,
) -> None:
    # An AR(1) with coefficient phi has integrated autocorrelation time
    # (1 + phi) / (1 - phi) in closed form, so the estimator has an exact
    # answer to be held to. Realized ratios of estimate to truth over ten
    # seeds at 20,000 draws: 0.94 to 1.02 at phi = 0, 0.89 to 1.04 at 0.5,
    # 0.83 to 1.12 at 0.9 -- the truncation is a Monte Carlo estimate and
    # the bound below is its spread, not a claim of exactness.
    n = 20_000
    tau = (1.0 + phi) / (1.0 - phi)
    generator = torch.Generator().manual_seed(7)
    noise = torch.randn(n, 2, generator=generator, dtype=torch.float64)
    draws = torch.empty(n, 2, dtype=torch.float64)
    draws[0] = noise[0]
    for index in range(1, n):
        draws[index] = phi * draws[index - 1] + math.sqrt(1.0 - phi**2) * noise[index]

    ratio = (effective_sample_size(draws) / (n / tau)).numpy()

    np.testing.assert_allclose(ratio, np.ones(2), rtol=0.2)


@pytest.mark.edge_case
def test_the_effective_sample_size_of_a_constant_chain_is_its_length() -> None:
    # No autocorrelation to estimate, and a division by a zero variance to
    # avoid; reported as the length rather than as NaN.
    assert torch.equal(
        effective_sample_size(torch.ones(50, 1, dtype=torch.float64)),
        torch.tensor([50.0], dtype=torch.float64),
    )
    with pytest.raises(ValueError, match="at least 4 draws"):
        effective_sample_size(torch.zeros(3, 1, dtype=torch.float64))


# --- refusals ----------------------------------------------------------------


@pytest.mark.edge_case
def test_an_adaptation_out_of_range_is_refused() -> None:
    with pytest.raises(ValueError, match="at least 8 proposals"):
        Adaptation(warmup=7, target_acceptance=0.65, step_jitter=0.0)
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        Adaptation(warmup=100, target_acceptance=1.0, step_jitter=0.0)
    with pytest.raises(ValueError, match="step_jitter must lie"):
        Adaptation(warmup=100, target_acceptance=0.65, step_jitter=1.0)


class _Wall:
    """Zero at the start and enormous everywhere else, so no proposal is accepted."""

    def __init__(self, start: torch.Tensor) -> None:
        self.start = start

    def initial(self) -> torch.Tensor:
        return self.start.clone()

    def constrain(self, theta: torch.Tensor) -> Mapping[str, torch.Tensor]:
        return {"x": theta}

    def theta_from(self, named: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return named["x"]

    def __call__(self, theta: torch.Tensor) -> torch.Tensor:
        return 1e300 * (theta - self.start).abs().sum()


@pytest.mark.edge_case
def test_a_warm_up_whose_chain_did_not_move_is_refused() -> None:
    # A coordinate with zero warm-up variance would get an infinite mass
    # and a chain that never moves there while every diagnostic reads
    # healthy; the refusal names the coordinates.
    wall = _Wall(torch.tensor([0.5, -0.5], dtype=torch.float64))

    with pytest.raises(ValueError, match=r"zero on coordinate\(s\) \[0, 1\]"):
        sample(
            wall,
            seed=1,
            n_samples=10,
            step_size=0.1,
            n_steps=3,
            adaptation=Adaptation(warmup=40, target_acceptance=0.65, step_jitter=0.0),
        )


# --- the fixed-parameter path ------------------------------------------------


@pytest.mark.structural
def test_a_chain_without_adaptation_reports_no_warm_up_and_counts_its_gradients() -> (
    None
):
    # The fixed path is the oracle for the adapted one, so it must be what
    # it was: no report, and the cost is the trajectory count times what one
    # trajectory costs.
    chain = sample(GAUSSIAN, seed=3, n_samples=40, step_size=0.2, n_steps=6, burn_in=10)

    assert chain.adapted is None
    assert chain.force_evaluations == 50 * leapfrog.force_evaluations(6)


# --- the statistics ----------------------------------------------------------


def _pooled_acceptance(objective: Objective, seeds: range, n_samples: int) -> float:
    chains = [
        sample(
            objective,
            seed=seed,
            n_samples=n_samples,
            step_size=0.05,
            n_steps=5,
            adaptation=ADAPTATION,
        )
        for seed in seeds
    ]
    for chain in chains:
        assert chain.adapted is not None
        assert bool(torch.isfinite(chain.adapted.mass_diagonal).all())
        assert bool((chain.adapted.mass_diagonal > 0.0).all())
    return float(np.mean([chain.acceptance_rate for chain in chains]))


@pytest.mark.structural
def test_the_adapted_acceptance_lands_at_its_target_on_the_gaussian() -> None:
    # The contract: a 300-proposal warm-up from a step of 0.05, and the
    # drawn chain accepts at the target. Pooled over 20 seeds, since one
    # seed's rate at 400 draws has a binomial sd of 0.024 before the
    # step's own spread across seeds (0.15 on a mean of 1.09) is counted.
    # Realized 0.650 against 0.65; the per-seed rates ran 0.578 to 0.758
    # with a sd of 0.050.
    pooled = _pooled_acceptance(GAUSSIAN, range(20), 400)

    assert abs(pooled - TARGET) < 0.05, pooled


@pytest.mark.structural
@at_scale("n_seeds", ci=3, stress=20)
def test_the_adapted_acceptance_lands_at_its_target_on_the_four_taxon_posterior(
    n_seeds: int,
) -> None:
    # The same contract on a real posterior: five log branch lengths whose
    # warm-up masses span 5 to 125, so the unit-mass chain the fixed sampler
    # runs has to take the stiffest coordinate's step on every one of them.
    # Realized 0.678 over 3 seeds at 300 draws (0.633 to 0.750) and 0.673
    # over 20 (0.573 to 0.750, sd 0.049); the adapted step ran 0.98 +/- 0.16
    # across the 20 seeds.
    pooled = _pooled_acceptance(_four_taxon_posterior(), range(n_seeds), 300)

    assert abs(pooled - TARGET) < 0.05, pooled


def _agreement(
    objective: Objective,
    fixed_step: float,
    fixed_steps: int,
    n_samples: int,
) -> tuple[HmcChain, HmcChain]:
    """An adapted chain and a fixed-parameter chain of the same length, and their agreement.

    Means must agree within three standard errors, each standard error from
    the chain's own effective sample size; so must the spreads, whose
    standard error is ``sd / sqrt(2 ESS)``. The fixed chain is the oracle:
    it is the sampler every committed result used, and the adapted one may
    be more efficient but not different.
    """
    adapted = sample(
        objective,
        seed=21,
        n_samples=n_samples,
        step_size=0.05,
        n_steps=5,
        adaptation=ADAPTATION,
    )
    fixed = sample(
        objective,
        seed=22,
        n_samples=n_samples,
        step_size=fixed_step,
        n_steps=fixed_steps,
        burn_in=300,
    )
    for chain in (adapted, fixed):
        assert chain.acceptance_rate > 0.5
    ess_adapted = effective_sample_size(adapted.theta)
    ess_fixed = effective_sample_size(fixed.theta)
    sd_adapted, sd_fixed = adapted.theta.std(0), fixed.theta.std(0)

    mean_gap = (adapted.theta.mean(0) - fixed.theta.mean(0)).abs()
    mean_error = (sd_adapted**2 / ess_adapted + sd_fixed**2 / ess_fixed).sqrt()
    assert bool((mean_gap < 3.0 * mean_error).all()), mean_gap / mean_error

    sd_gap = (sd_adapted - sd_fixed).abs()
    sd_error = (
        sd_adapted**2 / (2.0 * ess_adapted) + sd_fixed**2 / (2.0 * ess_fixed)
    ).sqrt()
    assert bool((sd_gap < 3.0 * sd_error).all()), sd_gap / sd_error
    return adapted, fixed


@pytest.mark.oracle
def test_the_adapted_chain_agrees_with_the_fixed_chain_and_the_exact_gaussian() -> None:
    # Two oracles: the fixed-parameter chain, which is what every committed
    # result used, and the closed-form mean and covariance behind both. The
    # delta-method interval of #268 is exact here (a Gaussian's Hessian is
    # its precision), so both chains' spreads are held to it as well.
    #
    # Realized at 2000 draws: means within 1.83 and 0.03 standard errors
    # of each other, spreads within 0.54 and 0.28; the adapted chain's
    # spread is 1.009 and 1.008 of the exact one, the fixed chain's 0.996
    # and 1.018. The warm-up's mass diagonal is 0.60 and 1.81 against the
    # marginal precisions 0.5 and 2.0. Effective samples per gradient:
    # adapted 0.099 and 0.105 against fixed 0.098 and 0.015 -- the fixed
    # chain at a step of 0.25 over 12 leapfrog steps accepts at 0.989 and
    # spends 13 gradients per proposal to move the stiff coordinate by
    # less than its width, 459 effective draws of 2000.
    adapted, fixed = _agreement(
        GAUSSIAN, fixed_step=0.25, fixed_steps=12, n_samples=2000
    )

    laplace = fit(GAUSSIAN, include_intervals=True).standard_errors
    assert laplace is not None
    exact = GAUSSIAN.covariance.diagonal().sqrt()
    np.testing.assert_allclose(laplace["x"].numpy(), exact.numpy(), rtol=1e-8)
    for chain in (adapted, fixed):
        np.testing.assert_allclose(
            chain.theta.mean(0).numpy(), GAUSSIAN.mean.numpy(), atol=0.12
        )
        np.testing.assert_allclose(chain.theta.std(0).numpy(), exact.numpy(), rtol=0.06)

    per_gradient_adapted = _ess_per_gradient(adapted)
    per_gradient_fixed = _ess_per_gradient(fixed)
    assert per_gradient_adapted.min() > per_gradient_fixed.min(), (
        per_gradient_adapted,
        per_gradient_fixed,
    )


@pytest.mark.oracle
def test_the_adapted_chain_agrees_with_the_fixed_chain_on_the_four_taxon_posterior() -> (
    None
):
    # The fixed chain runs at unit mass with the step the stiffest branch
    # allows; the adapted one at the warm-up's metric. Same posterior, same
    # Monte Carlo error bound as the Gaussian case, and the #268 interval
    # -- the delta-method standard error of the branch lengths at the
    # posterior mode -- reported beside both chains' spreads, since on a
    # tree it is an approximation and the chain is what checks it.
    #
    # Realized at 600 draws: means within 1.77 standard errors on every
    # branch, spreads within 2.84 -- the fixed chain's stiffest branch has
    # 102 effective draws, so its spread's standard error is the wide one.
    # Sampled spread over the Laplace interval: adapted 1.01 to 1.13, fixed
    # 0.99 to 1.08. Effective samples per gradient on the slowest branch:
    # adapted 0.038 against fixed 0.019, and the fixed chain's fastest
    # branch is antithetic at 0.58 (3121 effective draws of 600) while its
    # slowest has 102. The warm-up's masses span 5 to 124; the unit-mass
    # chain at a step of 0.1 accepts at 0.873 and moves every branch at the
    # stiffest one's step.
    posterior = _four_taxon_posterior()
    adapted, fixed = _agreement(posterior, fixed_step=0.1, fixed_steps=5, n_samples=600)

    mode = fit(posterior).theta
    laplace = standard_errors_at(posterior, posterior.constrain(mode))
    for chain in (adapted, fixed):
        lengths = torch.stack(
            [
                torch.cat(
                    [
                        tensor.reshape(-1)
                        for tensor in posterior.constrain(draw).values()
                    ]
                )
                for draw in chain.theta
            ]
        )
        interval = torch.cat([tensor.reshape(-1) for tensor in laplace.values()])
        ratio = (lengths.std(0) / interval).numpy()
        np.testing.assert_allclose(ratio, np.ones_like(ratio), rtol=0.25)

    # A chain is as slow as its slowest coordinate, so that is the number
    # compared; the fixed chain's fastest coordinate is antithetic (its
    # size exceeds the draw count) and says nothing about the rest.
    per_gradient_adapted = _ess_per_gradient(adapted)
    per_gradient_fixed = _ess_per_gradient(fixed)
    assert per_gradient_adapted.min() > per_gradient_fixed.min(), (
        per_gradient_adapted,
        per_gradient_fixed,
    )
