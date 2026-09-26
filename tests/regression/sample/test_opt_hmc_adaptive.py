"""Adaptation, pinned where it is exact before it is pinned where it is statistical.

Arithmetic first: dual averaging is Hoffman & Gelman's line for line
(``eq:dual-averaging``); a diagonal mass matrix is a change of coordinates
(a hand-written mass-matrix leapfrog); the ESS estimator recovers a known
autocorrelation time. Then: acceptance lands at target, adapted and fixed
chains agree within Monte Carlo error, and cost per draw in gradients is
reported (issue #333). On a locally quadratic target acceptance is a cliff in
the step (:class:`~sal.sample.hmc.Adaptation`); the jitter and
gain come from that measurement, and what they achieve is pinned.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import pytest
import torch
from sal.likelihood.objective import BranchLengthObjective
from sal.opt.fit import fit, standard_errors_at
from sal.opt.objective import Objective
from sal.sample.chain import _DualAveraging, _Scaled
from sal.sample.hmc import (
    DUAL_AVERAGING_GAMMA,
    DUAL_AVERAGING_KAPPA,
    DUAL_AVERAGING_T0,
    Adaptation,
    HmcChain,
    WithGaussianPrior,
    effective_sample_size,
    gradient_at,
    leapfrog,
    sample,
)
from sal.sim.simulator import simulate_tree

from tests._fixtures import FOUR_TAXA, load_fixture
from tests._posteriors import GAUSSIAN
from tests._rows import every_value
from tests._scale import at_scale

EXACT = 1e-13

TARGET = 0.65
ADAPTATION = Adaptation(warmup=300, target_acceptance=TARGET, step_jitter=0.4)


def _four_taxon_posterior() -> WithGaussianPrior:
    """Log branch lengths of the four-taxon fixture at 500 sites, under a N(0, 2^2) prior."""
    params = load_fixture(FOUR_TAXA)
    dataset = simulate_tree(params, np.random.default_rng(params.seed), n_sites=500)
    return WithGaussianPrior(
        BranchLengthObjective(
            params.tau, params.n_states, params.pi, dict(dataset.alignment)
        ),
        scale=2.0,
    )


def _ess_per_gradient(chain: HmcChain) -> np.ndarray:
    return (effective_sample_size(chain.draws) / chain.force_evaluations).numpy()


# --- exact ------------------------------------------------------------------


@pytest.mark.oracle
def test_the_dual_averaging_iteration_is_hoffman_and_gelmans() -> None:
    # Algorithm 5 of Hoffman & Gelman (2014), written out with their symbols
    # and stepped by hand for four acceptance statistics, against the class.
    # Every quantity is compared, not only the step, so a wrong constant that
    # left the iterate right is still caught.
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


@pytest.mark.analytic
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
    velocity = velocity - 0.5 * step_size * gradient_at(objective, position)
    for step in range(n_steps):
        position = position + step_size * inverse_mass * velocity
        if step < n_steps - 1:
            velocity = velocity - step_size * gradient_at(objective, position)
    velocity = velocity - 0.5 * step_size * gradient_at(objective, position)
    return position, velocity


@pytest.mark.analytic
def test_a_diagonal_mass_matrix_is_a_change_of_coordinates() -> None:
    # `_Scaled` is the whole implementation of the mass matrix: the unit-mass
    # leapfrog on the scaled objective, mapped back, equals the mass-matrix
    # leapfrog on the original to round-off, and the Hamiltonian is the same
    # number in both coordinate systems.
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


@pytest.mark.smoke
def test_the_scaled_objective_inverts_its_own_map() -> None:
    scale = torch.tensor([2.5, 0.4], dtype=torch.float64)
    scaled = _Scaled(GAUSSIAN, scale)
    point = torch.tensor([0.3, -1.1], dtype=torch.float64)

    assert torch.equal(scaled.theta_from(scaled.constrain(point)), point)
    assert torch.equal(scaled.initial() * scale, GAUSSIAN.initial())
    assert float(scaled(point)) == float(GAUSSIAN(point * scale))


@pytest.mark.oracle
def test_the_effective_sample_size_recovers_an_ar1_autocorrelation_time() -> None:
    # AR(1): time (1 + phi) / (1 - phi) exactly. Estimate/truth over ten seeds
    # at 20,000 draws: 0.94-1.02 (phi 0), 0.89-1.04 (0.5), 0.83-1.12 (0.9);
    # the bound is the truncation's spread.
    def check(phi: float) -> None:
        n = 20_000
        tau = (1.0 + phi) / (1.0 - phi)
        generator = torch.Generator().manual_seed(7)
        noise = torch.randn(n, 2, generator=generator, dtype=torch.float64)
        draws = torch.empty(n, 2, dtype=torch.float64)
        draws[0] = noise[0]
        for index in range(1, n):
            draws[index] = (
                phi * draws[index - 1] + math.sqrt(1.0 - phi**2) * noise[index]
            )

        ratio = (effective_sample_size(draws) / (n / tau)).numpy()

        np.testing.assert_allclose(ratio, np.ones(2), rtol=0.2)

    every_value([0.0, 0.5, 0.9], check)


@pytest.mark.smoke
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


@pytest.mark.smoke
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


@pytest.mark.smoke
def test_a_warm_up_whose_chain_did_not_move_is_refused() -> None:
    # A coordinate with zero warm-up variance would get an infinite mass
    # and a chain that never moves there while every diagnostic reads
    # healthy; the refusal names the coordinates.
    wall = _Wall(torch.tensor([0.5, -0.5], dtype=torch.float64))

    with pytest.raises(ValueError, match=r"zero on coordinate\(s\) \[0, 1\]"):
        sample(
            wall,
            generator=torch.Generator().manual_seed(1),
            n_samples=10,
            step_size=0.1,
            n_steps=3,
            adaptation=Adaptation(warmup=40, target_acceptance=0.65, step_jitter=0.0),
        )


# --- the fixed-parameter path ------------------------------------------------


@pytest.mark.smoke
def test_a_chain_without_adaptation_reports_no_warm_up_and_counts_its_gradients() -> (
    None
):
    # The fixed path is the oracle for the adapted one, so it must be what it
    # was: no report, and the cost the trajectory count times one trajectory.
    chain = sample(
        GAUSSIAN,
        torch.Generator().manual_seed(3),
        n_samples=40,
        step_size=0.2,
        n_steps=6,
        burn_in=10,
    )

    assert chain.adapted is None
    assert chain.force_evaluations == 50 * leapfrog.force_evaluations(6)


# --- the statistics ----------------------------------------------------------


def _pooled_acceptance(objective: Objective, seeds: range, n_samples: int) -> float:
    chains = [
        sample(
            objective,
            generator=torch.Generator().manual_seed(seed),
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


@pytest.mark.smoke
def test_the_adapted_acceptance_lands_at_its_target_on_the_gaussian() -> None:
    # 300-proposal warm-up from 0.05; pooled over 20 seeds (one seed's binomial
    # sd 0.024, step spread 0.15 on 1.09). Realized 0.650 against 0.65; per
    # seed 0.578 to 0.758, sd 0.050.
    pooled = _pooled_acceptance(GAUSSIAN, range(20), 400)

    assert abs(pooled - TARGET) < 0.05, pooled


@pytest.mark.smoke
@at_scale("n_seeds", ci=3, stress=20)
@pytest.mark.release  # 12.0 s in the tier, over the 10 s cap (#1088)
def test_the_adapted_acceptance_lands_at_its_target_on_the_four_taxon_posterior(
    n_seeds: int,
) -> None:
    # Five log branch lengths, warm-up masses 5 to 125. Realized 0.678 over 3
    # seeds (0.633-0.750), 0.673 over 20 (0.573-0.750, sd 0.049); step
    # 0.98 +/- 0.16.
    pooled = _pooled_acceptance(_four_taxon_posterior(), range(n_seeds), 300)

    assert abs(pooled - TARGET) < 0.05, pooled


def _agreement(
    objective: Objective,
    fixed_step: float,
    fixed_steps: int,
    n_samples: int,
) -> tuple[HmcChain, HmcChain]:
    """An adapted chain and a fixed-parameter chain of the same length, and their agreement.

    Means and spreads (``sd / sqrt(2 ESS)``) within three errors; fixed is the oracle.
    """
    adapted = sample(
        objective,
        generator=torch.Generator().manual_seed(21),
        n_samples=n_samples,
        step_size=0.05,
        n_steps=5,
        adaptation=ADAPTATION,
    )
    fixed = sample(
        objective,
        generator=torch.Generator().manual_seed(22),
        n_samples=n_samples,
        step_size=fixed_step,
        n_steps=fixed_steps,
        burn_in=300,
    )
    for chain in (adapted, fixed):
        assert chain.acceptance_rate > 0.5
    ess_adapted = effective_sample_size(adapted.draws)
    ess_fixed = effective_sample_size(fixed.draws)
    sd_adapted, sd_fixed = adapted.draws.std(0), fixed.draws.std(0)

    mean_gap = (adapted.draws.mean(0) - fixed.draws.mean(0)).abs()
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
    # Oracles: the fixed chain and the closed form (#268's interval is exact
    # here). At 2000 draws: means 1.83 and 0.03 errors apart, spreads 0.54 and
    # 0.28; spreads/exact adapted 1.009, 1.008, fixed 0.996, 1.018; mass 0.60,
    # 1.81 against 0.5, 2.0. ESS per gradient: adapted 0.099, 0.105; fixed
    # 0.098, 0.015 (step 0.25 x 12, acceptance 0.989, 459 of 2000).
    adapted, fixed = _agreement(
        GAUSSIAN, fixed_step=0.25, fixed_steps=12, n_samples=2000
    )

    laplace = fit(GAUSSIAN, include_intervals=True).standard_errors
    assert laplace is not None
    exact = GAUSSIAN.covariance.diagonal().sqrt()
    np.testing.assert_allclose(laplace["x"].numpy(), exact.numpy(), rtol=1e-8)
    for chain in (adapted, fixed):
        np.testing.assert_allclose(
            chain.draws.mean(0).numpy(), GAUSSIAN.mean.numpy(), atol=0.12
        )
        np.testing.assert_allclose(chain.draws.std(0).numpy(), exact.numpy(), rtol=0.06)

    per_gradient_adapted = _ess_per_gradient(adapted)
    per_gradient_fixed = _ess_per_gradient(fixed)
    assert per_gradient_adapted.min() > per_gradient_fixed.min(), (
        per_gradient_adapted,
        per_gradient_fixed,
    )


@pytest.mark.oracle
@pytest.mark.release  # 11.6 s in the tier, over the 10 s cap (#1088)
def test_the_adapted_chain_agrees_with_the_fixed_chain_on_the_four_taxon_posterior() -> (
    None
):
    # Unit mass at the stiffest branch's step against the warm-up metric; the
    # #268 interval reported beside both. At 600 draws: means within 1.77
    # errors, spreads 2.84 (fixed stiffest ESS 102). Spread/Laplace: adapted
    # 1.01-1.13, fixed 0.99-1.08. ESS per gradient, slowest branch: 0.038
    # against 0.019; fixed fastest antithetic (3121 of 600). Masses 5 to 124;
    # unit mass at 0.1 accepts 0.873.
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
                for draw in chain.draws
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


@pytest.mark.oracle
@pytest.mark.analytic
def test_the_adapted_chains_marginals_are_the_exact_gaussians_within_three_errors() -> (
    None
):
    """The warm-up's own chain against the closed-form marginals, at three
    Monte Carlo standard errors.

    Errors from the chain: ``sd / sqrt(ESS)`` for a mean, the squared
    deviations' own ESS for a variance (320 and 406 against 2,659 and 2,171).
    Warm-up: 300 proposals, target 0.80, jitter 0.4, from 0.05; step 0.9952,
    mass 0.862 and 2.688 (against 0.5, 2.0: order, not value), acceptance
    0.765, 1,800 of 9,000 gradients. At 1,200 draws: means 2.17 and 0.06,
    variances 0.36 and 0.57 errors (3.0); means (1.0596, -1.9990), variances
    (2.0621, 0.5211). MKL's code path moves the last bits (#895):
    ``MKL_CBWR=COMPATIBLE`` (a runner, bitwise) step 1.1013, acceptance 0.770,
    0.04, 1.16, 0.33, 0.73; ``AVX2`` 1.0562, 0.769, 0.78, 0.62, 0.94, 0.70.
    """
    below_the_cliff = sample(
        GAUSSIAN,
        generator=torch.Generator().manual_seed(729),
        n_samples=1200,
        step_size=0.05,
        n_steps=5,
        adaptation=Adaptation(warmup=300, target_acceptance=0.80, step_jitter=0.4),
    )
    adapted = below_the_cliff.adapted
    assert adapted is not None

    squares = (below_the_cliff.draws - GAUSSIAN.mean) ** 2
    exact_variance = GAUSSIAN.covariance.diagonal()
    mean_errors = (
        (below_the_cliff.draws.mean(0) - GAUSSIAN.mean).abs()
        * effective_sample_size(below_the_cliff.draws).sqrt()
        / exact_variance.sqrt()
    )
    variance_errors = (
        (squares.mean(0) - exact_variance).abs()
        * effective_sample_size(squares).sqrt()
        / squares.std(0)
    )
    print(
        f"\nmarginals in standard errors: means {mean_errors.numpy().round(2)}, "
        f"variances {variance_errors.numpy().round(2)}"
    )
    print(
        f"adapted step {adapted.step_size:.4f}, mass "
        f"{adapted.mass_diagonal.numpy().round(3)}, warm-up acceptance "
        f"{adapted.warmup_acceptance:.3f}"
    )

    assert bool((mean_errors < 3.0).all()), mean_errors
    assert bool((variance_errors < 3.0).all()), variance_errors
    assert abs(adapted.warmup_acceptance - 0.80) < 0.05
    assert adapted.force_evaluations == 300 * leapfrog.force_evaluations(5)
    # The stiffer coordinate gets the larger mass: the warm-up recovers the
    # order of 1 / (2.0, 0.5), which is what the metric is for.
    assert float(adapted.mass_diagonal[1]) > float(adapted.mass_diagonal[0])


#: The seeds the energy-error claim is pooled over: the marginal test's 729
#: and the four after it, fixed before any of them was run.
ENERGY_SEEDS = range(729, 734)


def _largest_energy_errors(target: float) -> tuple[list[float], float]:
    """Each seed's largest energy error at ``target``, and the pooled acceptance."""
    chains = [
        sample(
            GAUSSIAN,
            generator=torch.Generator().manual_seed(seed),
            n_samples=1200,
            step_size=0.05,
            n_steps=5,
            adaptation=Adaptation(
                warmup=300, target_acceptance=target, step_jitter=0.4
            ),
        )
        for seed in ENERGY_SEEDS
    ]
    largest = [float(chain.energy_error.max()) for chain in chains]
    return largest, float(np.mean([chain.acceptance_rate for chain in chains]))


@pytest.mark.analytic
@pytest.mark.release  # 10.1 s in the tier, over the 10 s cap (#1088)
def test_the_energy_error_and_not_the_acceptance_rate_says_the_step_is_safe() -> None:
    """Two warm-ups that each land where they were asked, one on the cliff.

    The energy error, not acceptance, says the step is safe (`opt/CLAUDE.md`).
    Pooled over five seeds (#895): seed 729 alone read 27.78 against
    1.190e+05 here, 426.85 on a runner (``MKL_CBWR=COMPATIBLE``, bitwise).
    Medians of the largest energy error, 0.80 against 0.65: 20.88 against
    9.367e+04 (ratio 4,486); ``COMPATIBLE`` 37.98 against 1.123e+05 (2,957);
    ``AVX2`` 23.55 against 7.513e+04 (3,190). Pooled rates within 0.05 of
    target: 0.841/0.621, 0.825/0.609, 0.822/0.627; single chains at 0.65 ran
    0.555 to 0.665.
    """
    safe, safe_rate = _largest_energy_errors(0.80)
    diverging, diverging_rate = _largest_energy_errors(TARGET)
    safe_median = float(np.median(safe))
    diverging_median = float(np.median(diverging))
    print(
        f"\nlargest energy error per seed: {np.round(safe, 2)} below the cliff, "
        f"{[f'{value:.3e}' for value in diverging]} on it; medians {safe_median:.2f} and "
        f"{diverging_median:.3e}; pooled acceptance {safe_rate:.3f} and "
        f"{diverging_rate:.3f}"
    )

    assert safe_median < 100.0, safe
    assert diverging_median > 1000.0 * safe_median, (safe, diverging)
    # Both rates land where they were asked to: the acceptance rate does not
    # tell the chain on the cliff from the one below it.
    assert abs(safe_rate - 0.80) < 0.05, safe_rate
    assert abs(diverging_rate - TARGET) < 0.05, diverging_rate
