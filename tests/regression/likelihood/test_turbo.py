"""Turbo decoding: what it is held to, and where it departs from the exact answer.

The joint graph has cycles, so the iteration is the Bethe approximation and
equality against the exact bitwise MAP would be a false assertion. Four
things are asserted instead, and each names its referee.

* *Enumeration over all ``2 ** K`` messages* referees the direction and the
  size of the gap: the iteration's bit error rate falls from its first
  iteration and stays within a stated factor of the exact MAP's.
* *The all-zero symmetry* referees the shortcut every measurement past the
  encoder's reach rests on, per realization on a real codeword.
* *The uncoded closed form* ``Q(sqrt(2 E_b / N_0))`` referees the waterfall:
  a coded curve above it at the operating point is a code buying nothing.
* *The binomial interval the frame count supports* referees the
  monotonicity claim, stated in the weak form the measurement supports, with
  the departure reported in the docstring of the test that measures it.

The tiers are set from the measured wall clock on the reference host: the
CI-tier tests here cost 4.4 s together, the stress waterfall 12.1 s and the
release waterfall 182 s.
"""

from __future__ import annotations

import numpy as np
import pytest
from snakes_and_ladders.likelihood.convolutional import bcjr
from snakes_and_ladders.likelihood.ldpc import MapEstimate
from snakes_and_ladders.likelihood.turbo import (
    ErrorRates,
    decode_turbo,
    decode_turbo_per_iteration,
    exact_turbo_posterior,
    measure_error_rates,
    noise_scale,
    split_streams,
    uncoded_bit_error_rate,
)
from snakes_and_ladders.sim.convolutional import TurboCode, turbo_encode
from snakes_and_ladders.sim.fixtures import Fixture, fixture

#: The exact bitwise MAP's bit error rate, times this, bounds the iteration's
#: at the enumerable size. Measured at 1.26, 1.66, 1.74 and 3.20 over the
#: four declared points of the CI fixture: a K = 12 interleaver is far too
#: short for the iteration to approach the MAP. A bound on the *gap*, so the
#: claim is that the iteration decodes the same code, not that it is optimal.
MAP_GAP_FACTOR = 4.0

#: How many one-sigma binomial intervals an increase in the ensemble bit
#: error rate across iterations may span before it is a departure rather
#: than sampling noise. Two: the largest increase measured over the CI and
#: stress ensembles is 1.0 intervals.
NOISE_INTERVALS = 2.0


def _transmit(
    code: TurboCode, message: np.ndarray, eb_n0_db: float, rng: np.random.Generator
) -> np.ndarray:
    """Encode, send through the Gaussian channel at that point, return the ratios."""
    sigma = noise_scale(eb_n0_db, code.rate)
    word = turbo_encode(code, message)
    received = (1.0 - 2.0 * word) + sigma * rng.standard_normal(code.block_length)
    return np.asarray(2.0 * received / sigma**2)


def _increase_within_noise(rates: ErrorRates) -> float:
    """The largest rise across iterations, in one-sigma binomial intervals.

    Zero when the ensemble rate never rises. The interval is
    :attr:`ErrorRates.bit_error_interval`, which understates the true spread
    because a turbo failure is bursty, so the comparison is conservative in
    the direction that would fail.
    """
    rise = np.diff(rates.bit_error_rate)
    interval = rates.bit_error_interval[:-1]
    scaled = np.where(
        interval > 0.0, rise / np.where(interval > 0.0, interval, 1.0), 0.0
    )
    return float(max(scaled.max(initial=0.0), 0.0))


# --- against enumeration, at the enumerable size ---------------------------------


@pytest.mark.oracle
def test_the_iteration_reduces_the_bit_error_rate_towards_the_exact_map() -> None:
    """Iteration 8 beats iteration 1 and stays within `MAP_GAP_FACTOR` of the MAP.

    Without the first the iteration could be doing nothing; without the
    second it could be converging to the wrong code's answer. Neither is
    equality, because the joint graph has cycles: the ordering the
    approximation must respect is what is asserted.

    Per point the claim is *no worse*, because the fixture's 1,200 message
    bits per point cannot separate 40 errors from 40: at 2 dB the first and
    last iterations tie exactly. The strict improvement is asserted over the
    four points together, where the sample is 4,800 bits and 306 errors
    against 267.
    """
    params = fixture("turbo", "ci").params
    code = params.code()
    first_total = last_total = 0

    for eb_n0_db in params.eb_n0_db:
        rng = np.random.default_rng(params.seed)
        exact_errors = 0
        iteration_errors = np.zeros(params.iterations, dtype=np.int64)
        for _ in range(params.frames):
            message = rng.integers(0, 2, params.message_length).astype(np.uint8)
            llr = _transmit(code, message, eb_n0_db, rng)
            exact_errors += int(
                (exact_turbo_posterior(code, llr).bits != message).sum()
            )
            decisions = decode_turbo_per_iteration(
                code, llr, iterations=params.iterations
            )
            iteration_errors += (decisions != message[None, :]).sum(axis=1)

        assert iteration_errors[-1] <= iteration_errors[0], eb_n0_db
        assert exact_errors <= iteration_errors[-1], eb_n0_db
        assert iteration_errors[-1] <= MAP_GAP_FACTOR * exact_errors, eb_n0_db
        first_total += int(iteration_errors[0])
        last_total += int(iteration_errors[-1])

    assert last_total < first_total


@pytest.mark.oracle
def test_the_joint_posterior_is_the_enumerated_one_where_the_two_chains_agree() -> None:
    """At high signal the iteration's posterior ratios are the exact ones.

    The regime in which the approximation is exact, which
    ``likelihood/CLAUDE.md`` says an approximate evaluator must state: when
    every message bit is decided far from zero, the cycles carry no ambiguity
    around them and the Bethe beliefs are the true marginals. The sign
    agreement is asserted and the ratios are not, since Bethe over-counts the
    evidence around a cycle.
    """
    params = fixture("turbo", "ci").params
    code = params.code()
    rng = np.random.default_rng(101)
    agreed = 0

    for _ in range(50):
        message = rng.integers(0, 2, params.message_length).astype(np.uint8)
        llr = _transmit(code, message, 8.0, rng)
        exact = exact_turbo_posterior(code, llr)
        decoding = decode_turbo(code, llr, iterations=params.iterations)

        np.testing.assert_array_equal(decoding.bits, exact.bits)
        np.testing.assert_array_equal(exact.bits, message)
        agreed += 1

    assert agreed == 50


# --- the structure of one decoding ----------------------------------------------


@pytest.mark.structural
def test_the_per_iteration_run_is_the_capped_run_at_every_cap() -> None:
    # One run reports every iteration count, which is what makes the release
    # waterfall cost the deepest iteration rather than the sum. The two paths
    # would drift silently, so they are pinned to each other.
    params = fixture("turbo", "ci").params
    code = params.code()
    rng = np.random.default_rng(103)
    message = rng.integers(0, 2, params.message_length).astype(np.uint8)
    llr = _transmit(code, message, 1.0, rng)

    decisions = decode_turbo_per_iteration(code, llr, iterations=params.iterations)

    for cap in range(1, params.iterations + 1):
        result = decode_turbo(code, llr, iterations=cap)
        np.testing.assert_array_equal(result.bits, decisions[cap - 1])
        assert result.iterations == cap
        assert result.estimate is MapEstimate.BITWISE


@pytest.mark.mathematical
def test_the_second_decoder_reads_the_first_decoders_systematic_stream() -> None:
    """No message bit crosses the channel twice: the streams partition the word.

    The interleaver reorders what decoder two reads; it adds no evidence. A
    decoder given a second independent copy of the systematic bits would
    decode better than the code allows, and no error-rate test would
    attribute that failure.
    """
    params = fixture("turbo", "ci").params
    code = params.code()
    rng = np.random.default_rng(107)
    message = rng.integers(0, 2, params.message_length).astype(np.uint8)
    llr = _transmit(code, message, 2.0, rng)

    streams = split_streams(code, llr)

    assert (
        streams.systematic.size
        + streams.parity_first.size
        + streams.parity_second.size
        + streams.tail_second.size
        == code.block_length
    )
    assert streams.systematic.size == code.steps
    # A single BCJR over the first chain alone decodes worse than the pair:
    # the second parity stream is what the iteration adds.
    single = bcjr(code.trellis, streams.systematic, streams.parity_first)
    single_errors = int(
        ((single.posterior_llr[: code.message_length] < 0.0) != message).sum()
    )
    turbo_errors = int((decode_turbo(code, llr, iterations=8).bits != message).sum())

    assert turbo_errors <= single_errors


@pytest.mark.edge_case
def test_a_word_of_the_wrong_length_or_a_zero_iteration_cap_is_refused() -> None:
    params = fixture("turbo", "ci").params
    code = params.code()

    with pytest.raises(ValueError, match="shape"):
        decode_turbo(code, np.zeros(code.block_length - 1))
    with pytest.raises(ValueError, match="at least 1"):
        decode_turbo(code, np.zeros(code.block_length), iterations=0)
    with pytest.raises(ValueError, match="rate must be in"):
        noise_scale(1.0, 0.0)


# --- the all-zero shortcut every measurement past the encoder rests on ------------


@pytest.mark.mathematical
def test_the_error_pattern_under_a_codeword_is_the_pattern_under_zero() -> None:
    """Per realization, on a real codeword, for the recursive encoder.

    Issue #340 states the argument for the parity-check decoder; a turbo
    code's encoder is recursive and its decoder a different message map, so
    the property is re-checked here rather than inherited. It licenses
    :func:`~snakes_and_ladders.likelihood.turbo.measure_error_rates` sending
    the zero message at every length.
    """
    params = fixture("turbo", "ci").params
    code = params.code()
    rng = np.random.default_rng(109)
    sigma = noise_scale(1.0, code.rate)

    for _ in range(20):
        message = rng.integers(0, 2, params.message_length).astype(np.uint8)
        word = turbo_encode(code, message)
        # One noise realization, read in each codeword's own frame. The
        # channel is output symmetric, so sending `c` instead of `0` negates
        # the ratio exactly at the bits where `c_i = 1`; the equivalent
        # realization is the sign map below, not the same additive noise.
        under_zero_llr = 2.0 * (1.0 + sigma * rng.standard_normal(code.block_length))
        under_zero_llr /= sigma**2
        under_word_llr = (1.0 - 2.0 * word) * under_zero_llr

        under_word = decode_turbo(code, under_word_llr, iterations=4)
        under_zero = decode_turbo(code, under_zero_llr, iterations=4)

        np.testing.assert_array_equal(under_word.bits ^ message, under_zero.bits)


# --- the waterfall ---------------------------------------------------------------


def _waterfall(instance: Fixture) -> list[ErrorRates]:
    """Every declared point of one instance, under that instance's own seed."""
    params = instance.params
    code = params.code()
    return [
        measure_error_rates(
            code,
            eb_n0_db,
            params.frames,
            np.random.default_rng(params.seed),
            iterations=params.iterations,
        )
        for eb_n0_db in params.eb_n0_db
    ]


@pytest.mark.stress
@pytest.mark.oracle
def test_the_waterfall_falls_below_the_uncoded_closed_form() -> None:
    """The coded curve is under `Q(sqrt(2 E_b / N_0))` at every declared point.

    It also falls with `E_b / N_0` and improves from the first iteration to
    the last.

    *The departure, reported rather than asserted:* the ensemble bit error
    rate is **not** monotone in the iteration count. Over the six declared
    points it rises between consecutive iterations at four of them --- the
    largest rise 2.3e-3 at 0 dB, between iterations 7 and 8 --- and every
    rise is inside one binomial interval of the 12,800 message bits the
    point rests on. So the strong claim is not made: asserted is that the
    last iteration beats the first and that no rise exceeds
    `NOISE_INTERVALS` intervals.
    """
    # 12.1 s measured, so `stress` and not the CI tier. The CI-tier sibling
    # is the enumeration test above, at a size the budget holds.
    instance = fixture("turbo", "stress")
    params = instance.params
    rates = _waterfall(instance)
    uncoded = uncoded_bit_error_rate(np.array(params.eb_n0_db))
    final = np.array([rate.bit_error_rate[-1] for rate in rates])

    assert instance.oracle == "closed-form"
    assert np.all(final < uncoded), (final, uncoded)
    assert np.all(np.diff(final) <= 0.0), final
    for rate in rates:
        assert rate.bits == params.frames * params.message_length
        assert rate.bit_error_rate[-1] < rate.bit_error_rate[0]
        assert _increase_within_noise(rate) <= NOISE_INTERVALS


@pytest.mark.release
@pytest.mark.oracle
def test_the_release_waterfall_turns_where_the_ensemble_says() -> None:
    """At `K = 1024` the curve falls two orders of magnitude across the span.

    It stays under the uncoded closed form throughout. The claim a short block
    cannot carry: the interleaver gain grows with the block length, so the
    turn is sharp at 1,024 and gradual at 256. The stress tier's monotonicity
    departure applies here and is checked the same way.
    """
    # 182 s measured over 1,200 decodings, so `release`: past both the 5- and
    # the 10-minute budgets `DEV.md` sets.
    instance = fixture("turbo", "release")
    params = instance.params
    rates = _waterfall(instance)
    uncoded = uncoded_bit_error_rate(np.array(params.eb_n0_db))
    final = np.array([rate.bit_error_rate[-1] for rate in rates])

    assert instance.oracle == "none"
    assert np.all(final < uncoded)
    assert np.all(np.diff(final) <= 0.0), final
    # The turn: two orders of magnitude across the declared span, which a
    # K = 256 block does not reach.
    assert final[0] > 100.0 * max(
        final[-1], 1.0 / (params.frames * params.message_length)
    )
    for rate in rates:
        assert rate.bit_error_rate[-1] < rate.bit_error_rate[0]
        assert _increase_within_noise(rate) <= NOISE_INTERVALS
