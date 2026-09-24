"""A covariate reaching the family through a fit, not through a test (issue #652).

#631 refereed the per-observation covariate at the family; these are the
end-to-end claims at the two seams #652 threads, Baum-Welch over an HMM and the
spatial model's params. Told a varying exposure, a negative-binomial fit
recovers the planted rate; not told it, the same fit recovers the average rate
and reads the spread as overdispersion. The gap is measured. `PoissonEmission`
refuses a covariate by design, so it is not a site here.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from snakes_and_ladders.emissions import NegativeBinomialEmission
from snakes_and_ladders.likelihood.device import CROSS_DEVICE_RTOL_FLOAT64
from snakes_and_ladders.likelihood.spatio_sequential import class_log_density
from snakes_and_ladders.opt.hmm import baum_welch_family
from snakes_and_ladders.sim.spatio_sequential import (
    canonical_spatio_sequential,
    gated_log_density,
)

_DISPERSION = np.array([6.0, 6.0])


def _two_state_chain(
    rate: np.ndarray, spread: float, length: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One sticky two-state chain of counts drawn under a varying exposure.

    Returns the counts, the exposure that produced them, and the hidden path.
    """
    rng = np.random.default_rng(seed)
    stay = 0.94
    states = np.empty(length, dtype=np.int64)
    states[0] = 0
    for t in range(1, length):
        states[t] = states[t - 1] if rng.random() < stay else 1 - states[t - 1]
    exposure = rng.uniform(1.0 / spread, spread, size=length)
    mean = exposure * rate[states]
    dispersion = _DISPERSION[states]
    # NegativeBinomial(r, mu) as a gamma-Poisson mixture, which is the draw the
    # family's own mean-variance relation is stated for.
    counts = rng.poisson(rng.gamma(dispersion, mean / dispersion))
    return counts[None], exposure[None], states


def _fit(
    counts: np.ndarray, exposure: np.ndarray | None, start: np.ndarray
) -> np.ndarray:
    """Baum-Welch from a fixed start, returning the fitted rates sorted."""
    result = baum_welch_family(
        counts,
        torch.log(torch.full((2,), 0.5, dtype=torch.float64)),
        torch.log(torch.tensor([[0.9, 0.1], [0.1, 0.9]], dtype=torch.float64)),
        NegativeBinomialEmission(
            torch.as_tensor(_DISPERSION, dtype=torch.float64),
            torch.as_tensor(start, dtype=torch.float64),
        ),
        max_iterations=200,
        covariate=exposure,
    )
    family = result.emissions
    assert isinstance(family, NegativeBinomialEmission)
    return np.sort(family.mean.detach().numpy())


#: The planted rates, and the chain the two tests below share. They are one
#: claim in two parts --- the same data and the same fit, told the exposure and
#: not told it --- and are two tests only because one fit each keeps both
#: inside the per-test duration cap (`DEV.md`).
_RATE = np.array([2.0, 9.0])
_SPREAD = 4.0
_LENGTH = 1200
_SEED = 11
_START = np.array([1.0, 6.0])


@pytest.mark.end2end
def test_baum_welch_told_the_exposure_recovers_the_planted_rates() -> None:
    """The claim the threading is for: the fit reaches the rate that generated it.

    1.9688 and 8.8088 against a planted 2.0 and 9.0; exposure ``U(0.25, 4)``.
    """
    counts, exposure, _ = _two_state_chain(_RATE, _SPREAD, _LENGTH, _SEED)

    fitted = _fit(counts, exposure, _START)

    np.testing.assert_allclose(fitted, _RATE, rtol=0.1)


@pytest.mark.end2end
def test_the_same_fit_without_the_exposure_misses_them() -> None:
    """And the claim that it is worth something: withheld, the same fit misses.

    It converges to 4.1505 and 18.5904, ~2.07x high against ``E[U(0.25, 4)]`` = 2.125.
    """
    counts, _, _ = _two_state_chain(_RATE, _SPREAD, _LENGTH, _SEED)

    untold = _fit(counts, None, _START)

    assert np.abs(untold / _RATE - 1.0).max() > 0.25


@pytest.mark.smoke
def test_a_constant_exposure_of_one_scores_bitwise() -> None:
    """#631's referee at the seam this threads to: ones change no score at all.

    A rate times exactly 1.0 is that rate, so scoring is bitwise; the fit is below.
    """
    counts, _, _ = _two_state_chain(_RATE, spread=1.0, length=400, seed=5)
    family = NegativeBinomialEmission(
        torch.as_tensor(_DISPERSION, dtype=torch.float64),
        torch.as_tensor(_RATE, dtype=torch.float64),
    )
    observations = torch.as_tensor(counts, dtype=torch.float64)

    assert torch.equal(
        family.log_density(observations),
        family.log_density(
            observations, covariate=torch.ones_like(observations)[..., None]
        ),
    )


@pytest.mark.smoke
def test_a_constant_exposure_of_one_fits_inside_the_declared_tolerance() -> None:
    """The same claim through 200 EM iterations, which is not bitwise.

    One thread 1.5e-13 (two 5.1e-14, four 9.9e-14); bound 1e-12 = 1e-11 / 10.
    """
    counts, _, _ = _two_state_chain(_RATE, spread=1.0, length=400, seed=5)

    ones = np.ones_like(counts, dtype=float)
    told_nothing = _fit(counts, None, _START)
    told_ones = _fit(counts, ones, _START)

    moved = float(np.abs(told_nothing / told_ones - 1.0).max())
    assert moved < 0.1 * CROSS_DEVICE_RTOL_FLOAT64, (
        f"a covariate of ones moved the fit {moved:.2e}, against a declared "
        f"{CROSS_DEVICE_RTOL_FLOAT64:.0e}; it was 1.5e-13 at the single thread "
        "this suite pins when written, and a drift toward the floor is the "
        "tolerance starting to hide something"
    )


@pytest.mark.smoke
def test_the_spatial_seams_run_and_the_covariate_field_is_validated() -> None:
    """What this actually checks: both seams run uncovaried, and the field validates.

    Not that a seam reads `params.covariate`: the instance is categorical (#658 item 5).
    """
    from dataclasses import replace

    from snakes_and_ladders.sim.spatio_sequential import simulate_spatio_sequential

    params = canonical_spatio_sequential()
    data = simulate_spatio_sequential(params, np.random.default_rng(3))

    # Both seams run clean with no covariate, which is every caller today.
    table = gated_log_density(params, data.observations)
    density = class_log_density(params, data.observations, data.labels)
    assert np.isfinite(table).all()
    assert np.isfinite(density).all()

    # And the field is validated against the observations' own axes.
    with pytest.raises(ValueError, match="covariate has shape"):
        replace(params, covariate=np.ones((params.n_positions, 1)))
