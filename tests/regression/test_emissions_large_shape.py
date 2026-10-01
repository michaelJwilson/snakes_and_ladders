"""The count densities at large shape, against `mpmath` at 50 digits (issue #1136).

Above :data:`sal.emissions.rising.LARGE_SHAPE` the negative binomial and the
beta-binomial difference Stirling's series instead of ``lgamma``; at and
below it their arithmetic is unchanged. Each is judged against the exact
density from shape 10 to 1e16, the beta-binomial's pmf against one, the
negative binomial at ``r = inf`` against the Poisson, and the gradient
against a central difference of the exact density.
"""

from __future__ import annotations

import math

import mpmath  # type: ignore[import-untyped]
import numpy as np
import pytest
import torch
from sal.emissions import (
    BetaBinomialEmission,
    CountPairEmission,
    NegativeBinomialEmission,
)
from sal.emissions.rising import LARGE_SHAPE, log_rising_scaled

mpmath.mp.dps = 50

#: Shapes judged: either side of the threshold, then up to the binomial and
#: Poisson limits.
SHAPES = [10.0, 99.0, LARGE_SHAPE, 1e3, 1e6, 1e12, 1e16]

#: Absolute agreement with the exact density, in nats. Measured worst:
#: 7.3e-14 (beta-binomial), 3.7e-14 (negative binomial).
_TOLERANCE = 1e-12

TRIALS = 40
COUNTS = [0.0, 1.0, 3.0, 10.0, 40.0, 200.0]


def _exact_beta_binomial(z: int, a: float, b: float) -> float:
    a_, b_ = mpmath.mpf(a), mpmath.mpf(b)
    return float(
        mpmath.log(mpmath.binomial(TRIALS, z))
        + mpmath.loggamma(z + a_)
        + mpmath.loggamma(TRIALS - z + b_)
        - mpmath.loggamma(TRIALS + a_ + b_)
        + mpmath.loggamma(a_ + b_)
        - mpmath.loggamma(a_)
        - mpmath.loggamma(b_)
    )


def _exact_negative_binomial(y: float, r: float, mu: float) -> float:
    if math.isinf(r):
        return float(y * mpmath.log(mu) - mu - mpmath.loggamma(y + 1))
    r_ = mpmath.mpf(r)
    return float(
        mpmath.loggamma(y + r_)
        - mpmath.loggamma(r_)
        - mpmath.loggamma(y + 1)
        + r_ * mpmath.log(r_ / (r_ + mu))
        + y * mpmath.log(mu / (r_ + mu))
    )


@pytest.mark.oracle
@pytest.mark.parametrize("concentration", SHAPES, ids=str)
def test_the_beta_binomial_is_exact_and_sums_to_one(concentration: float) -> None:
    a, b = 0.3 * concentration, 0.7 * concentration
    family = BetaBinomialEmission([TRIALS], [a], [b])

    got = family.log_density(torch.arange(TRIALS + 1, dtype=torch.float64))[:, 0]
    want = torch.tensor(
        [_exact_beta_binomial(z, a, b) for z in range(TRIALS + 1)], dtype=torch.float64
    )

    assert float((got - want).abs().max()) <= _TOLERANCE
    assert abs(float(torch.logsumexp(got, 0))) <= _TOLERANCE


@pytest.mark.oracle
@pytest.mark.parametrize("dispersion", [*SHAPES, math.inf], ids=str)
def test_the_negative_binomial_is_exact_to_the_poisson_limit(dispersion: float) -> None:
    family = NegativeBinomialEmission([dispersion], [12.0])

    got = family.log_density(torch.tensor(COUNTS, dtype=torch.float64))[:, 0]
    want = [_exact_negative_binomial(y, dispersion, 12.0) for y in COUNTS]

    np.testing.assert_allclose(got.numpy(), want, rtol=0, atol=_TOLERANCE)


@pytest.mark.oracle
def test_zero_overdispersion_is_the_poisson() -> None:
    # alpha = 1 / r = 0 is where a line search over alpha evaluates.
    family = NegativeBinomialEmission.from_overdispersion([0.0, 0.25], [12.0, 12.0])
    y = torch.tensor(COUNTS, dtype=torch.float64)

    poisson = y * math.log(12.0) - 12.0 - torch.lgamma(y + 1.0)
    scored = family.log_density(y)

    np.testing.assert_allclose(scored[:, 0].numpy(), poisson.numpy(), rtol=1e-14)
    assert torch.equal(
        scored[:, 1], NegativeBinomialEmission([4.0], [12.0]).log_density(y)[:, 0]
    )
    assert bool(torch.isfinite(family.bregman_divergence(y)).all())
    draws = family.sample(np.zeros(2_000, dtype=np.int64), np.random.default_rng(0))
    assert abs(draws.mean() - 12.0) < 4.0 * math.sqrt(12.0 / 2_000)
    with pytest.raises(ValueError, match="non-negative"):
        NegativeBinomialEmission.from_overdispersion([-0.1], [12.0])


@pytest.mark.oracle
@pytest.mark.parametrize("dispersion", [1e3, 1e8, 1e12], ids=str)
def test_the_gradient_in_the_dispersion_is_the_exact_derivative(
    dispersion: float,
) -> None:
    # d/dr log NB is the digamma rise plus log(r / (r + mu)) + (mu - y) / (r +
    # mu): every term cancels at large r, which the series form does not.
    r = torch.tensor([dispersion], dtype=torch.float64, requires_grad=True)
    family = NegativeBinomialEmission(r, torch.tensor([12.0], dtype=torch.float64))
    family.log_density(torch.tensor([3.0], dtype=torch.float64)).sum().backward()  # type: ignore[no-untyped-call]
    assert r.grad is not None

    exact = mpmath.diff(
        lambda x: mpmath.loggamma(3 + x)
        - mpmath.loggamma(x)
        + x * mpmath.log(x / (x + 12))
        + 3 * mpmath.log(12 / (x + 12)),
        mpmath.mpf(dispersion),
    )

    assert float(r.grad[0]) == pytest.approx(float(exact), rel=1e-9, abs=1e-30)


@pytest.mark.oracle
def test_below_the_threshold_nothing_moves() -> None:
    # The plain arithmetic is kept bit for bit below the threshold, so every
    # factored table pinned bitwise to `log_density` stays pinned.
    y = torch.arange(60, dtype=torch.float64)
    nb = NegativeBinomialEmission([4.0, 99.0], [20.0, 80.0])
    total = nb.dispersion + nb.mean
    plain = (
        torch.lgamma(y[:, None] + nb.dispersion)
        - torch.lgamma(nb.dispersion)
        - torch.lgamma(y[:, None] + 1.0)
        + nb.dispersion * torch.log(nb.dispersion / total)
        + y[:, None] * torch.log(nb.mean / total)
    )

    assert torch.equal(nb.log_density(y), plain)
    assert float(log_rising_scaled(torch.tensor(math.inf), torch.tensor(5.0))) == 0.0


@pytest.mark.oracle
@pytest.mark.parametrize("concentration", [1e3, 1e12], ids=str)
def test_every_trial_layout_takes_the_large_shape_path_alike(
    concentration: float,
) -> None:
    # Trials per state (the family), per observation and state (the
    # independent pair, which raises them to the successes), and per
    # observation (a covariate) reach the series by three routes.
    a, b = 0.3 * concentration, 0.7 * concentration
    successes = torch.arange(TRIALS + 1, dtype=torch.float64)
    want = torch.tensor(
        [_exact_beta_binomial(z, a, b) for z in range(TRIALS + 1)], dtype=torch.float64
    )
    family = BetaBinomialEmission([TRIALS, TRIALS], [a, 2.0], [b, 3.0])
    pair = CountPairEmission(
        [5.0, 5.0], [10.0, 10.0], [a, 2.0], [b, 3.0], [TRIALS, TRIALS], joint=False
    )
    totals = torch.full_like(successes, 7.0)
    by_pair = pair.log_density(torch.stack([totals, successes], dim=-1))[:, 0]
    by_pair = by_pair - pair.total.log_density(totals)[:, 0]

    for got in (
        family.log_density(successes)[:, 0],
        family.log_density(successes, torch.full((TRIALS + 1, 1), float(TRIALS)))[:, 0],
        by_pair,
    ):
        assert float((got - want).abs().max()) <= _TOLERANCE
