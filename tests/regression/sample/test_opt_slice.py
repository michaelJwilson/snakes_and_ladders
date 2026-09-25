"""Slice sampling, pinned on the property that defines it before the distribution.

Every returned point lies in the slice its level was cut at, whatever the
target and width. Then the distribution, against HMC and MALA's references
(an analytic Gaussian, the 4,096-assignment mixture; issue #749). Then the
shrinkage, ablated: it bounds the evaluation count, not correctness, since
the stepped-out interval already contains the current point.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pytest
import torch
from sal.backend import Backend
from sal.sample.chain import Adaptation
from sal.sample.hmc import effective_sample_size
from sal.sample.slice import (
    MAX_SHRINKAGES,
    SliceChain,
    SliceDirection,
    slice_sample,
    slice_update,
)

from tests._posteriors import (
    GAUSSIAN,
    assert_recovers_assignment_posterior,
    assert_within_sigmas,
    enumerated_quadrature,
    weight_posterior,
)

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
    # Neal (2003): `-objective(theta) >= log_slice`, per update, so
    # `slice_update` is public; a sweep reports no level.
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
    # Closed form, two seeds, both schemes; worst in chain errors: coordinate
    # mean 0.53, 0.64, variance 0.55, 0.56; hit-and-run mean 0.29, 0.36,
    # variance 1.66, 0.33 (3.0).
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

        assert_within_sigmas(
            chain.draws,
            GAUSSIAN.mean.numpy(),
            np.diag(GAUSSIAN.covariance.numpy()),
            GAUSSIAN_SIGMAS,
            seed,
        )
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
    reference = enumerated_quadrature(observations, components)

    for seed in (756, 757):
        chain = slice_sample(
            target,
            generator=torch.Generator().manual_seed(seed),
            n_samples=1500,
            width=1.0,
            max_steps_out=MAX_STEPS_OUT,
            burn_in=200,
        )

        assert_recovers_assignment_posterior(
            chain.draws,
            observations,
            components,
            reference,
            sigmas=MIXTURE_SIGMAS,
            size=float(effective_sample_size(chain.draws)[0]),
            context=seed,
        )


@pytest.mark.analytic
def test_shrinkage_is_what_bounds_the_evaluations_a_draw_costs() -> None:
    # Bias is not refereeable (rejecting without narrowing is still valid);
    # cost is: the rejection form hits the 100-candidate refusal at widths 2.0,
    # 10.0 and 40.0, shrinkage holds 11.3, 12.6 and 16.0 evaluations per sweep.
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


@pytest.mark.analytic
def test_the_tempered_slice_chain_samples_the_tempered_gaussian() -> None:
    # `exp(-U / T)` on a Gaussian is the Gaussian with covariance `T`-fold,
    # a closed form; issue #1059 gave `slice_sample` the `temperature` the
    # other continuous samplers take.
    temperature = 2.0
    chain = slice_sample(
        GAUSSIAN,
        generator=torch.Generator().manual_seed(1059),
        n_samples=4000,
        width=WIDTH,
        max_steps_out=MAX_STEPS_OUT,
        burn_in=400,
        temperature=temperature,
    )
    assert_within_sigmas(
        chain.draws,
        GAUSSIAN.mean.numpy(),
        temperature * np.diag(GAUSSIAN.covariance.numpy()),
        GAUSSIAN_SIGMAS,
    )


@pytest.mark.analytic
def test_an_unstored_slice_chain_spends_what_the_stored_one_spends() -> None:
    # `store_chain=False` drops the draws and nothing else: one generator
    # state gives the same counters bitwise (issue #1059).
    def run(store_chain: bool) -> SliceChain:
        return slice_sample(
            GAUSSIAN,
            generator=torch.Generator().manual_seed(7),
            n_samples=50,
            width=WIDTH,
            max_steps_out=MAX_STEPS_OUT,
            store_chain=store_chain,
        )

    stored, free = run(True), run(False)
    assert free.draws.shape == (0, 2)
    assert stored.draws.shape == (50, 2)
    assert free.objective_evaluations == stored.objective_evaluations
    assert free.shrinkages_per_draw == stored.shrinkages_per_draw


@pytest.mark.smoke
def test_a_slice_chain_refuses_an_adaptation_a_backend_and_a_temperature() -> None:
    options = {
        "generator": torch.Generator().manual_seed(1),
        "n_samples": 4,
        "width": WIDTH,
        "max_steps_out": MAX_STEPS_OUT,
    }
    with pytest.raises(ValueError, match="no adaptation"):
        slice_sample(
            GAUSSIAN,
            adaptation=Adaptation(8, 0.5, 0.0),
            **options,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="slice_sample runs on python"):
        slice_sample(GAUSSIAN, backend=Backend.RUST, **options)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="temperature must be positive"):
        slice_sample(GAUSSIAN, temperature=0.0, **options)  # type: ignore[arg-type]


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
