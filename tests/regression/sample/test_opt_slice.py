"""Slice sampling, pinned on the property that defines it before the distribution.

The procedure's defining statement is exact and needs no chain: every point
it returns lies in the slice its level was cut at, whatever the target and
whatever the width. That comes first, because a distributional failure says
the sampler is wrong while this says which half.

Then the distribution, against the same two references HMC and MALA are read
against --- an analytic Gaussian and the mixture posterior whose evidence is a
sum over all 4,096 assignments (issue #749) --- and then the shrinkage,
ablated. The shrinkage is a cost device rather than a correctness one: the
interval the stepping-out leaves already contains the current point, so
rejecting from it without narrowing samples the same distribution. What it
buys is an evaluation count that stays bounded, and that is what is measured.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pytest
import torch
from snakes_and_ladders.opt.constrain import log_simplex
from snakes_and_ladders.sample.hmc import effective_sample_size
from snakes_and_ladders.opt.mixture import responsibilities
from snakes_and_ladders.sample.slice import (
    MAX_SHRINKAGES,
    SliceDirection,
    slice_sample,
    slice_update,
)

from tests._objective_checks import AnalyticGaussian
from tests._posteriors import (
    enumerated_quadrature,
    monte_carlo_sigmas,
    weight_posterior,
)

GAUSSIAN = AnalyticGaussian([1.0, -2.0], [[2.0, 0.6], [0.6, 0.5]])

#: The width every chain here runs at, and the one number the sampler takes.
#: 2.0 is about one standard deviation of the wider marginal; the diagnostics
#: it produces are 2.37 expansions and 1.45 shrinkages a sweep, so neither end
#: of the stepping-out is doing all the work.
WIDTH = 2.0
MAX_STEPS_OUT = 10

#: Standard errors a marginal estimate is allowed from its closed form, as in
#: `test_opt_langevin.py`: three on the Gaussian, four on the mixture.
GAUSSIAN_SIGMAS = 3.0
MIXTURE_SIGMAS = 4.0


@pytest.mark.analytic
@pytest.mark.critical
def test_every_slice_update_returns_a_point_in_its_slice() -> None:
    # Neal's (2003) invariant, and the only exact thing a slice sampler
    # asserts: `-objective(theta) >= log_slice` for the point returned. It
    # holds per update and not per draw, which is why `slice_update` is public
    # --- a sweep cuts one level per coordinate and a chain can report none of
    # them.
    generator = torch.Generator().manual_seed(2003)
    position = GAUSSIAN.initial()
    for index in range(200):
        line = torch.zeros(2, dtype=torch.float64)
        line[index % 2] = 1.0
        update = slice_update(
            GAUSSIAN,
            position,
            line,
            generator,
            width=WIDTH,
            max_steps_out=MAX_STEPS_OUT,
        )

        assert -float(GAUSSIAN(update.theta)) >= update.log_slice, index
        # The level is below the current point's density by construction, so
        # the current point is always in its own slice and the interval is
        # never empty.
        assert update.log_slice <= -float(GAUSSIAN(position)), index
        assert update.evaluations >= 2, index
        position = update.theta


@pytest.mark.oracle
@pytest.mark.parametrize(
    "direction", [SliceDirection.COORDINATE, SliceDirection.RANDOM]
)
def test_the_slice_chain_recovers_an_analytic_gaussian(
    direction: SliceDirection,
) -> None:
    # Mean and covariance in closed form, over two seeds and both schemes ---
    # the coordinate sweep and hit-and-run, which differ in how they meet the
    # target's correlation and must agree with it either way.
    #
    # Realized, worst over the two coordinates and in the chain's own standard
    # errors: coordinate 0.53 and 0.64 on the mean, 0.55 and 0.56 on the
    # variance; hit-and-run 0.29 and 0.36 on the mean, 1.66 and 0.33 on the
    # variance. Against the 3.0 declared.
    for seed in (11, 12):
        chain = slice_sample(
            GAUSSIAN,
            generator=torch.Generator().manual_seed(seed),
            n_samples=4000,
            width=WIDTH,
            max_steps_out=MAX_STEPS_OUT,
            burn_in=400,
            direction=direction,
        )

        mean_sigmas, variance_sigmas = monte_carlo_sigmas(
            chain.theta,
            GAUSSIAN.mean.numpy(),
            np.diag(GAUSSIAN.covariance.numpy()),
        )

        assert float(mean_sigmas.max()) < GAUSSIAN_SIGMAS, (seed, mean_sigmas)
        assert float(variance_sigmas.max()) < GAUSSIAN_SIGMAS, (seed, variance_sigmas)
        # The cost unit, reported rather than inferred: what the objective saw.
        assert chain.objective_evaluations == pytest.approx(
            chain.evaluations_per_draw * 4400, rel=1e-9
        )


@pytest.mark.oracle
@pytest.mark.release
def test_the_slice_chain_recovers_the_enumerated_assignment_posterior() -> None:
    # The rung below (issue #734): assignment enumeration, as for HMC and
    # MALA. Realized over the two seeds: the posterior mean weight 0.18 and
    # 1.62 standard errors away and the worst marginal 0.19 and 1.62, against
    # the 4.0 declared, at 5.80 and 5.88 objective evaluations a draw.
    target, observations, components = weight_posterior()
    quadrature_weight, quadrature_marginal = enumerated_quadrature(
        observations, components
    )

    for seed in (756, 757):
        chain = slice_sample(
            target,
            generator=torch.Generator().manual_seed(seed),
            n_samples=1500,
            width=1.0,
            max_steps_out=MAX_STEPS_OUT,
            burn_in=200,
        )

        log_weights = log_simplex(chain.theta)
        drawn_weight = torch.exp(log_weights)[:, 0].numpy()
        drawn_marginal = np.stack(
            [
                responsibilities(
                    torch.as_tensor(observations, dtype=torch.float64), row, components
                ).numpy()[:, 0]
                for row in log_weights
            ]
        )
        size = float(effective_sample_size(chain.theta)[0])
        weight_tolerance = MIXTURE_SIGMAS * float(drawn_weight.std()) / np.sqrt(size)
        marginal_tolerance = (
            MIXTURE_SIGMAS * float(drawn_marginal.std(axis=0).max()) / np.sqrt(size)
        )

        assert abs(drawn_weight.mean() - quadrature_weight) < weight_tolerance, (
            seed,
            drawn_weight.mean(),
            quadrature_weight,
        )
        assert (
            np.abs(drawn_marginal.mean(axis=0) - quadrature_marginal).max()
            < marginal_tolerance
        ), (seed, np.abs(drawn_marginal.mean(axis=0) - quadrature_marginal).max())


@pytest.mark.analytic
def test_shrinkage_is_what_bounds_the_evaluations_a_draw_costs() -> None:
    # The ablation, and the half of it that is refereeable. Bias is not: the
    # interval the stepping-out leaves contains the current point, so
    # rejecting from it without narrowing is a valid update and samples the
    # same distribution. Cost is, and it has no useful bound --- the level is
    # cut at `f(x) + log u`, so a `u` near 1 leaves a slice of arbitrarily
    # small measure inside an interval of fixed width, and the rejection form
    # spends candidates in proportion to the ratio.
    #
    # Measured on this target: the rejection form reaches the 100-candidate
    # refusal at every width tried --- 2.0, 10.0 and 40.0 --- while the
    # shrinkage holds a sweep to 11.3, 12.6 and 16.0 evaluations over that
    # same 20x range, which is the sense in which the width is the only knob
    # and a wrong one costs evaluations rather than correctness.
    costs = []
    for width in (2.0, 10.0, 40.0):
        chain = slice_sample(
            GAUSSIAN,
            generator=torch.Generator().manual_seed(7),
            n_samples=400,
            width=width,
            max_steps_out=MAX_STEPS_OUT,
            burn_in=50,
        )
        costs.append(chain.evaluations_per_draw)
        assert chain.shrinkages_per_draw < 2.0 * MAX_SHRINKAGES, width

        with pytest.raises(ValueError, match="shrinkage is off"):
            slice_sample(
                GAUSSIAN,
                generator=torch.Generator().manual_seed(7),
                n_samples=400,
                width=width,
                max_steps_out=MAX_STEPS_OUT,
                burn_in=50,
                shrink=False,
            )

    # A twentyfold width costs under twice the evaluations, which is the
    # claim that makes the sampler tuning-free rather than under-tuned.
    assert costs[-1] < 2.0 * costs[0], costs


@pytest.mark.smoke
@pytest.mark.critical
def test_a_slice_chain_refuses_a_width_a_step_count_and_a_density_it_cannot_use() -> (
    None
):
    # Refused rather than clamped, per `opt/CLAUDE.md`. The third is the one
    # that matters scientifically: a chain started where the density is not
    # finite has no level to cut a slice at, and silently returning the
    # starting point would look like a chain that did not move.
    with pytest.raises(ValueError, match="width must be positive"):
        slice_sample(
            GAUSSIAN,
            generator=torch.Generator().manual_seed(1),
            n_samples=4,
            width=0.0,
            max_steps_out=MAX_STEPS_OUT,
        )
    with pytest.raises(ValueError, match="max_steps_out must be at least 1"):
        slice_sample(
            GAUSSIAN,
            generator=torch.Generator().manual_seed(1),
            n_samples=4,
            width=WIDTH,
            max_steps_out=0,
        )
    with pytest.raises(ValueError, match="no slice level"):
        slice_sample(
            _Infinite(),
            generator=torch.Generator().manual_seed(1),
            n_samples=4,
            width=WIDTH,
            max_steps_out=MAX_STEPS_OUT,
        )


class _Infinite:
    """An objective that is not finite anywhere: a density with no slice."""

    def initial(self) -> torch.Tensor:
        return torch.zeros(1, dtype=torch.float64)

    def constrain(self, theta: torch.Tensor) -> Mapping[str, torch.Tensor]:
        return {"x": theta}

    def theta_from(self, named: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return named["x"]

    def __call__(self, theta: torch.Tensor) -> torch.Tensor:
        return torch.full((), float("inf"), dtype=torch.float64) + theta.sum() * 0.0
