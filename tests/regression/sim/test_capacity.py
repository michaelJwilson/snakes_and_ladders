"""Capacity against an independent route, a published pin, and a converse.

Issue #594. The erasure and symmetric channels have closed forms, the symmetric
checked against mutual information summed over the ``2 x 2`` joint. The
Gaussian integral is held to Monte Carlo at the sample's tolerance and to the
published rate-1/2 ``sigma* = 0.9787``. The inverse is a converse: the (3,6)
erasure threshold sits below it, the suite's one comparison of the two.
"""

from __future__ import annotations

import math
from itertools import pairwise, product

import numpy as np
import pytest
from sal.likelihood.ldpc import erasure_threshold
from sal.likelihood.turbo import noise_scale
from sal.sim.capacity import (
    DEFAULT_NODES,
    binary_entropy,
    capacity,
    decibels_at_capacity,
    eb_n0_decibels,
    erasure_capacity,
    gaussian_capacity,
    noise_at_capacity,
    symmetric_capacity,
)
from sal.sim.ldpc import (
    BinaryErasureChannel,
    BinaryInputGaussianChannel,
    BinarySymmetricChannel,
    Channel,
)
from scipy.optimize import brentq

from tests._rows import every_row, every_value

# The rate-1/2 binary-input Gaussian limit (Richardson and Urbanke 2008,
# §4.10), quoted to four places, which is what fixes the tolerance below.
PUBLISHED_SIGMA_STAR = 0.9787
RATES = (0.1, 1.0 / 3.0, 0.5, 4.0 / 7.0, 0.9)


@pytest.mark.analytic
def test_the_binary_entropy_peaks_at_one_half_and_vanishes_at_certainty() -> None:
    # Three properties that fix H up to nothing: the endpoints by the
    # `0 log 0 = 0` limit, the symmetry, and the maximum.
    assert binary_entropy(0.0) == 0.0
    assert binary_entropy(1.0) == 0.0
    assert binary_entropy(0.5) == 1.0
    for probability in (0.01, 0.1, 0.25, 0.4):
        assert binary_entropy(probability) == pytest.approx(
            binary_entropy(1.0 - probability), rel=1e-15
        )
        assert binary_entropy(probability) < binary_entropy(0.5)


@pytest.mark.analytic
def test_the_erasure_capacity_is_the_delivered_fraction() -> None:
    # Exact, not approximate: an unerased symbol arrives intact, so the
    # capacity is the probability of delivery and nothing is integrated.
    for erasure in (0.0, 0.1, 0.4294, 0.5, 1.0):
        assert erasure_capacity(erasure) == 1.0 - erasure
    assert capacity(BinaryErasureChannel(0.4)) == 0.6


@pytest.mark.oracle
@pytest.mark.analytic
def test_the_symmetric_capacity_matches_the_mutual_information_it_stands_for() -> None:
    # The independent route: `1 - H(p)` is the closed form, and this sums
    # `sum_xy p(x, y) log2 p(y | x) / p(y)` over the 2x2 joint at a uniform
    # input. A sign or a base error in one does not survive the other.
    for flip in (0.01, 0.05, 0.11, 0.25, 0.45):
        transition = np.array([[1.0 - flip, flip], [flip, 1.0 - flip]])
        joint = 0.5 * transition
        received = joint.sum(axis=0)
        mutual = float(
            sum(
                joint[sent, seen] * math.log2(transition[sent, seen] / received[seen])
                for sent in (0, 1)
                for seen in (0, 1)
            )
        )
        assert symmetric_capacity(flip) == pytest.approx(mutual, rel=1e-14)
        assert capacity(BinarySymmetricChannel(flip)) == pytest.approx(
            mutual, rel=1e-14
        )


@pytest.mark.oracle
@pytest.mark.analytic
def test_the_gaussian_capacity_recovers_the_published_rate_half_limit() -> None:
    # The pin: at the published sigma* the capacity is one half. The tolerance
    # is the quotation, not the quadrature -- 0.9787 is four places, and the
    # measured value is 0.49999606, which is 3.9e-6 from a half.
    assert gaussian_capacity(PUBLISHED_SIGMA_STAR) == pytest.approx(0.5, abs=1e-5)


@pytest.mark.oracle
@pytest.mark.analytic
def test_the_quadrature_agrees_with_monte_carlo_within_its_sampling_error() -> None:
    # Quadrature against sampling, which shares no node, weight or recursion
    # with it. The tolerance is four standard errors of the sample mean, so it
    # is set by the draw count and not chosen to pass.
    sigma = PUBLISHED_SIGMA_STAR
    rng = np.random.default_rng(594)
    received = 1.0 + sigma * rng.standard_normal(2_000_000)
    terms = np.logaddexp(0.0, -2.0 * received / sigma**2) / np.log(2.0)
    sampled = 1.0 - float(terms.mean())
    standard_error = float(terms.std(ddof=1)) / math.sqrt(terms.size)
    assert abs(gaussian_capacity(sigma) - sampled) < 4.0 * standard_error


@pytest.mark.analytic
def test_the_gaussian_capacity_falls_with_the_noise_and_stays_in_the_unit_interval() -> (
    None
):
    # A capacity outside [0, 1] bits on a binary input would be a bug the pin
    # at one point cannot see, and monotonicity is what licenses the bisection
    # in `noise_at_capacity`.
    scales = [0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 4.0, 8.0]
    values = [gaussian_capacity(sigma) for sigma in scales]
    assert all(0.0 < value <= 1.0 for value in values)
    assert all(later < earlier for earlier, later in pairwise(values))


@pytest.mark.analytic
def test_the_node_count_is_not_what_the_number_rests_on() -> None:
    # Halving and doubling the default both move the answer under 1e-9, so the
    # quadrature has converged rather than landed near the pin by luck.
    sigma = PUBLISHED_SIGMA_STAR
    reference = gaussian_capacity(sigma, nodes=DEFAULT_NODES)
    for nodes in (50, 200):
        assert abs(gaussian_capacity(sigma, nodes=nodes) - reference) < 1e-9


@pytest.mark.analytic
@pytest.mark.parametrize(
    "family",
    [BinaryErasureChannel, BinarySymmetricChannel, BinaryInputGaussianChannel],
)
def test_the_shannon_limit_round_trips_through_capacity(family: type) -> None:
    # What `noise_at_capacity` claims, checked by putting the answer back in:
    # the capacity at the returned noise level is the rate asked for.
    def check(rate: float) -> None:
        noise = noise_at_capacity(rate, family)
        assert capacity(family(noise)) == pytest.approx(rate, abs=1e-7)

    every_value(RATES, check)


@pytest.mark.oracle
@pytest.mark.analytic
def test_the_erasure_limit_is_one_minus_the_rate() -> None:
    # Closed form, so the bisection is held to it rather than to itself.
    def check(rate: float) -> None:
        limit = noise_at_capacity(rate, BinaryErasureChannel)
        assert limit == pytest.approx(1.0 - rate, abs=1e-9)

    every_value(RATES, check)


@pytest.mark.oracle
@pytest.mark.analytic
def test_no_ensemble_threshold_reaches_the_limit_it_is_bounded_by() -> None:
    # (3,6) threshold 0.4294 against capacity 0.5: 14.1% unused, the ensemble's
    # gap at any length.
    threshold = erasure_threshold(3, 6, iterations=2000, precision=1e-4)
    limit = noise_at_capacity(0.5, BinaryErasureChannel)
    assert threshold < limit
    assert threshold == pytest.approx(0.4294, abs=1e-3)
    assert (limit - threshold) / limit == pytest.approx(0.141, abs=5e-3)


@pytest.mark.smoke
def test_a_channel_with_no_declared_capacity_is_refused() -> None:
    # The `Channel` protocol exposes log-likelihood ratios alone, so a fourth
    # channel has no capacity to derive. Returning one anyway would attribute
    # a number to a channel this module has never seen.
    class Lossless(Channel):
        def log_likelihood_ratios(
            self, codeword: np.ndarray, rng: np.random.Generator
        ) -> np.ndarray:
            del rng
            return np.zeros(np.asarray(codeword).size)

    with pytest.raises(NotImplementedError, match="no capacity is declared"):
        capacity(Lossless())
    with pytest.raises(NotImplementedError, match="no capacity is declared"):
        noise_at_capacity(0.5, Lossless)  # type: ignore[arg-type]


@pytest.mark.smoke
def test_a_rate_outside_the_open_unit_interval_is_refused() -> None:
    # At 0 and 1 the limit is an interval end rather than a crossing, and the
    # bisection would return that end as though it had solved for it.
    def check(rate: float) -> None:
        with pytest.raises(ValueError, match="strictly in"):
            noise_at_capacity(rate, BinaryErasureChannel)

    every_value([0.0, 1.0, -0.1, 1.5], check)


@pytest.mark.smoke
def test_the_arguments_each_function_cannot_answer_for_are_refused() -> None:
    with pytest.raises(ValueError, match="probability lies in"):
        binary_entropy(1.2)
    with pytest.raises(ValueError, match="erasure probability lies in"):
        erasure_capacity(-0.1)
    with pytest.raises(ValueError, match="noise scale is positive"):
        gaussian_capacity(0.0)
    with pytest.raises(ValueError, match="at least two nodes"):
        gaussian_capacity(1.0, nodes=1)


@pytest.mark.analytic
def test_the_two_directions_of_the_decibel_map_are_inverses() -> None:
    # `noise_scale` sets a waterfall's points and `eb_n0_decibels` reads them
    # back. They are one formula written twice, in opposite directions, in two
    # modules -- so they are pinned to each other here rather than restated.
    def check(rate: float, decibels: float) -> None:
        read = eb_n0_decibels(noise_scale(decibels, rate), rate)
        assert read == pytest.approx(decibels, abs=1e-12)

    every_row(product(RATES, [-1.0, 0.0, 1.5, 4.0, 9.0]), check)


@pytest.mark.oracle
@pytest.mark.analytic
def test_the_rate_half_limit_is_the_published_decibel_figure() -> None:
    # +0.187 dB, the binary-input limit at rate 1/2 every coding text quotes
    # (Richardson and Urbanke 2008, §4.10). Reached here from the capacity
    # integral rather than from the quoted sigma*, so it is a second route to
    # the same number.
    assert decibels_at_capacity(0.5) == pytest.approx(0.187, abs=1e-3)


@pytest.mark.analytic
def test_the_limit_in_decibels_rises_with_the_rate() -> None:
    # A code that sends more information per symbol needs a better channel.
    # The turbo fixture's transmitted rate is the low end, at -0.508 dB.
    rates = [0.1, 0.3299, 0.5, 4.0 / 7.0, 0.8]
    limits = [decibels_at_capacity(rate) for rate in rates]
    assert all(later > earlier for earlier, later in pairwise(limits))
    assert limits[1] == pytest.approx(-0.508, abs=1e-3)


@pytest.mark.smoke
def test_the_decibel_map_refuses_what_it_cannot_answer_for() -> None:
    with pytest.raises(ValueError, match="noise scale is positive"):
        eb_n0_decibels(0.0, 0.5)
    with pytest.raises(ValueError, match="rate must be in"):
        eb_n0_decibels(1.0, 0.0)


@pytest.mark.oracle
def test_each_family_s_rate_half_limit_is_the_figure_its_text_quotes() -> None:
    # Published rate-1/2 limits (Richardson and Urbanke 2008, §4.1, §4.10;
    # erasure `1 - eps`), 1e-3 declared: eps* 0.4999999995 (4.66e-10, the
    # bisection's precision), p* 0.11002786 (2.79e-5), sigma* 0.97869412 (5.88e-6).
    erasure = noise_at_capacity(0.5, BinaryErasureChannel)
    crossover = noise_at_capacity(0.5, BinarySymmetricChannel)
    sigma = noise_at_capacity(0.5, BinaryInputGaussianChannel)

    assert erasure == pytest.approx(0.5, abs=1e-9)
    assert crossover == pytest.approx(0.11, abs=1e-3)
    assert sigma == pytest.approx(PUBLISHED_SIGMA_STAR, abs=1e-3)

    # And the forward direction at the quoted parameters: the BSC at p = 0.11
    # carries one half a bit, 8.40e-5 above it at the quotation's four places.
    assert symmetric_capacity(0.11) == pytest.approx(0.5, abs=1e-3)
    assert erasure_capacity(0.5) == 0.5


@pytest.mark.oracle
def test_the_symmetric_limit_is_the_entropy_equation_solved_by_hand() -> None:
    # `scipy.optimize.brentq` on `H(p) = 1 - R` shares no step with bisection:
    # worst 4.21e-10 over five rates, against 1e-9.
    worst = 0.0
    for rate in RATES:
        expected = float(
            brentq(
                lambda p, target=rate: binary_entropy(p) - (1.0 - target), 1e-12, 0.5
            )
        )
        worst = max(
            worst, abs(noise_at_capacity(rate, BinarySymmetricChannel) - expected)
        )

    assert worst < 1e-9
