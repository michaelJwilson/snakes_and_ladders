"""The shared categorical sampler, and the guard two copies of it had lost.

`sample_rows` replaced three near-identical private helpers. Two of them
omitted the clamp the third had, so the point of the consolidation is not
that there is now one copy but that the surviving copy is the correct one.
That is what these tests pin.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.testing import assert_allclose
from sal.numerics import logsumexp, sample_rows
from scipy.special import logsumexp as scipy_logsumexp


@pytest.mark.critical
@pytest.mark.smoke
def test_a_degenerate_row_always_yields_its_certain_category() -> None:
    # The one case with an answer that owes nothing to the sampling scheme.
    distributions = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    rows = np.array([0, 1, 0, 1, 0])
    drawn = sample_rows(np.random.default_rng(0), distributions, rows)
    assert drawn.tolist() == [1, 0, 1, 0, 1]


@pytest.mark.critical
@pytest.mark.analytic
def test_the_empirical_frequencies_match_the_distribution() -> None:
    # Against the probabilities themselves, at a sample size where the
    # Monte Carlo error is an order of magnitude below the tolerance:
    # sqrt(p(1-p)/n) <= 5e-4 at n = 1e6, checked to 5e-3.
    distributions = np.array([[0.1, 0.6, 0.3]])
    draws = 1_000_000
    drawn = sample_rows(
        np.random.default_rng(1), distributions, np.zeros(draws, dtype=int)
    )
    frequencies = np.bincount(drawn, minlength=3) / draws
    assert_allclose(frequencies, distributions[0], atol=5e-3)


@pytest.mark.critical
@pytest.mark.smoke
def test_a_row_summing_below_one_cannot_yield_an_index_past_the_end() -> None:
    # A normalized row can sum to 1 - 4e-16, and `rng.random` in [0, 1) lands
    # above it eventually; without the clamp the draw reports category `n`.
    # Constructed, not sampled: ~4e-16 per draw.
    distributions = np.array([[0.3, 0.3, 0.4 - 5e-16]])
    assert distributions.sum() < 1.0

    class _AtTheTop:
        """A generator returning a draw inside that sliver."""

        def random(self, size: tuple[int, ...] | int) -> np.ndarray:
            return np.full(size, 0.9999999999999998)

    drawn = sample_rows(_AtTheTop(), distributions, np.zeros(4, dtype=int))  # type: ignore[arg-type]
    assert drawn.tolist() == [2, 2, 2, 2]
    assert int(drawn.max()) < distributions.shape[1]


@pytest.mark.critical
@pytest.mark.infra
def test_every_row_is_selectable() -> None:
    distributions = np.eye(4)
    rows = np.arange(4)
    drawn = sample_rows(np.random.default_rng(2), distributions, rows)
    assert drawn.tolist() == [0, 1, 2, 3]


@pytest.mark.critical
@pytest.mark.infra
def test_one_draw_is_consumed_per_entry() -> None:
    # The stream cost must not depend on the outcome, or two callers seeded
    # alike would diverge on data rather than on their seeds.
    distributions = np.array([[0.5, 0.5]])
    first = np.random.default_rng(3)
    sample_rows(first, distributions, np.zeros(7, dtype=int))
    remaining = first.random(3)

    second = np.random.default_rng(3)
    second.random(7)
    assert_allclose(remaining, second.random(3))


@pytest.mark.critical
@pytest.mark.smoke
def test_a_one_dimensional_distribution_is_rejected() -> None:
    with pytest.raises(ValueError, match="expected distributions of shape"):
        sample_rows(np.random.default_rng(0), np.array([0.5, 0.5]), np.zeros(2, int))


@pytest.mark.analytic
def test_reducing_several_axes_at_once_is_one_reduction() -> None:
    # A region belief marginalized onto a child sums out several axes (#689).
    # Referee: one reduction over two axes equals two over one.
    rng = np.random.default_rng(20260916)
    values = rng.normal(scale=3.0, size=(4, 5, 6))

    together = logsumexp(values, axis=(0, 2))
    apart = logsumexp(logsumexp(values, axis=2), axis=0)

    assert together.shape == (5,)
    np.testing.assert_allclose(together, apart, rtol=0, atol=1e-13)


@pytest.mark.analytic
def test_the_shift_survives_an_exponent_the_linear_domain_would_lose() -> None:
    # What the shift is for, over several axes as over one: 800 in an exponent
    # overflows a float64 and the shifted form returns it exactly.
    values = np.full((2, 2), 800.0)

    assert float(logsumexp(values, axis=(0, 1))) == pytest.approx(
        800.0 + math.log(4.0), rel=1e-15
    )


@pytest.mark.oracle
def test_logsumexp_is_scipys_at_exponents_the_linear_domain_cannot_hold() -> None:
    """`scipy.special.logsumexp` is the referee, and it is the same algorithm.

    Bitwise at +800, -750 and ``800 + log 4``; two axes: 1.78e-16 against 1e-15.
    """
    rng = np.random.default_rng(20260919)
    vector = rng.normal(scale=5.0, size=128)
    overflowing = rng.normal(loc=800.0, scale=20.0, size=(4, 7))
    underflowing = rng.normal(loc=-750.0, scale=15.0, size=(4, 7))
    several = rng.normal(scale=3.0, size=(4, 5, 6))

    assert float(logsumexp(vector, axis=0)) == float(scipy_logsumexp(vector, axis=0))
    for values, axis in ((overflowing, 1), (underflowing, 0)):
        assert np.array_equal(
            logsumexp(values, axis=axis), scipy_logsumexp(values, axis=axis)
        )
    assert_allclose(
        logsumexp(several, axis=(0, 2)),
        scipy_logsumexp(several, axis=(0, 2)),
        rtol=1e-15,
        atol=0.0,
    )
    constant = np.full((2, 2), 800.0)
    assert float(logsumexp(constant, axis=(0, 1))) == 800.0 + math.log(4.0)
