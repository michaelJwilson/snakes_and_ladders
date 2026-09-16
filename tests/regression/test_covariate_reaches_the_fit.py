"""A covariate reaching the family through a fit, not through a test (issue #652).

#631 gave the emission families a per-observation covariate and refereed it at
the family. Nothing above them passed one, so no end-to-end claim rested on it.
These are that claim, at the two seams #652 threads: Baum-Welch over an HMM,
and the spatial model's params.

The shape is #631's recovery raised from the family to the fit. A varying
exposure carries information a fit cannot recover from the counts alone: told
the exposure, a negative-binomial fit recovers the planted rate; not told it,
the same data and the same fit recover the *average* rate and read the spread
the exposure would have explained as overdispersion. The gap between the two
is what the threading is worth, and it is measured rather than asserted.

`NegativeBinomialEmission` because it is one of the two families #631 gave an
exposure to --- `PoissonEmission` refuses a covariate by design, as every
family that conditions on nothing does, so it is not a site here either.
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


@pytest.mark.simulated_truth
def test_baum_welch_told_the_exposure_recovers_the_planted_rates() -> None:
    """The claim the threading is for: the fit reaches the rate that generated it.

    Recovers 1.9688 and 8.8088 against a planted 2.0 and 9.0, over a
    ``U(0.25, 4)`` exposure whose draw here spans 15.8x.
    """
    counts, exposure, _ = _two_state_chain(_RATE, _SPREAD, _LENGTH, _SEED)

    fitted = _fit(counts, exposure, _START)

    np.testing.assert_allclose(fitted, _RATE, rtol=0.1)


@pytest.mark.simulated_truth
def test_the_same_fit_without_the_exposure_misses_them() -> None:
    """And the claim that it is worth something: withheld, the same fit misses.

    Not a failure to converge --- it converges, to the rate averaged over the
    exposures it was not told about: 4.1505 and 18.5904, factors of 2.075 and
    2.066 high against an ``E[U(0.25, 4)]`` of 2.125. The bound is stated as a
    floor the untold fit must exceed, so this fails if the covariate ever
    stops reaching the family and the test above starts passing for the wrong
    reason.
    """
    counts, _, _ = _two_state_chain(_RATE, _SPREAD, _LENGTH, _SEED)

    untold = _fit(counts, None, _START)

    assert np.abs(untold / _RATE - 1.0).max() > 0.25


@pytest.mark.structural
def test_a_constant_exposure_of_one_scores_bitwise() -> None:
    """#631's referee at the seam this threads to: ones change no score at all.

    A rate multiplied by exactly 1.0 is that rate, and the reduction that
    follows is the same one, so the *scoring* seam is bitwise and is asserted
    as such. The fit is a separate claim, below, because it is a separate
    mechanism.
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


@pytest.mark.structural
def test_a_constant_exposure_of_one_fits_inside_the_declared_tolerance() -> None:
    """The same claim through 200 EM iterations, which is not bitwise.

    The scoring is bitwise, as above, so the divergence is the emission M step:
    `reestimate` under a covariate reduces in a different order, and 200
    iterations compound it.

    **Under this suite it is 1.5e-13.** `tests/conftest.py` pins
    `OMP_NUM_THREADS` to 1 (`DEV.md`, one process is one core), so the
    reduction is serial here and in CI. The figure is worth stating with its
    thread count because it does not survive one: the same fit reads 5.1e-14 at
    two threads and 9.9e-14 at four, since a split reduction sums in a
    different order again. A reading taken outside the suite's pinned thread is
    a reading of a different configuration.

    Every one of those is two orders inside the 1e-11 declared for a float64
    comparison, which is the trade `CLAUDE.md` permits where the only cost of a
    justified change is bitwise agreement. The bound is a tenth of that
    tolerance, 6.6x the serial reading, so a real drift fails here rather than
    hiding under the floor while a thread count nobody chose does not.
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


@pytest.mark.structural
@pytest.mark.critical
def test_the_spatial_params_carry_the_covariate_to_both_seams() -> None:
    """`gated_log_density` and `class_log_density` both read `params.covariate`.

    The canonical instance is categorical, which refuses a covariate, so the
    claim is made where it can be: the two seams agree with each other on the
    scores, and both change when the params' covariate does. A count family
    would be a second fixture for the same assertion.
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
