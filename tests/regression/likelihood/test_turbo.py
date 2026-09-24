"""Turbo decoding: what it is held to, and where it departs from the exact answer.

The joint graph has cycles (Bethe), so exactness is not asserted. Referees:
enumeration over ``2 ** K`` messages (the gap to the exact MAP falls and stays
bounded); the all-zero symmetry, per realization; the uncoded closed form
``Q(sqrt(2 E_b / N_0))``; and the binomial interval of the frame count for
monotonicity, stated weakly. CI tests cost 4.4 s; the waterfalls fell from
12.1 s and 182 s to 0.46 s and 6.0 s with #754's Rust trellis (stress runs
both backends, 11.8 s).
"""

from __future__ import annotations

import numpy as np
import pytest
from snakes_and_ladders.backend import Backend
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
from snakes_and_ladders.sim.convolutional import TurboCode, TurboParams, turbo_encode
from snakes_and_ladders.sim.fixtures import Fixture, fixture
from snakes_and_ladders.sim.ldpc import BinaryInputGaussianChannel

#: Bounds the iteration's bit error rate by this times the exact MAP's; measured
#: 1.26, 1.66, 1.74, 3.20 over the four CI points (K = 12): a bound on the gap.
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

    Bursty failures make the interval understate the spread: conservative.
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

    Per point "no worse" (2 dB ties); strictly over all four: 306 against 267 of 4,800.
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

    The exact regime (``likelihood/CLAUDE.md``): signs asserted; Bethe over-counts ratios.
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


#: Declared 1e-12 nats, realized 1.8e-14 over 100 draws: the second pass
#: reorders the sums; decisions are held to equality.
BCJR_TOLERANCE = 1e-12

#: Draws per declared point of the CI fixture.
CONSTITUENT_DRAWS = 25


@pytest.mark.critical
@pytest.mark.oracle
@pytest.mark.parametrize("backend", [Backend.PYTHON, Backend.RUST])
def test_the_turbo_posterior_is_bcjr_on_the_first_constituent_alone(
    backend: Backend,
) -> None:
    """With the second constituent carrying no evidence, the pair is one BCJR.

    The rung below (#734). Zeroing the second parity and tail ratios leaves
    `L_sys + L_ext,1 = bcjr(trellis, systematic, parity_first).posterior_llr`,
    true only if eq:extrinsic is subtracted once; both backends. Over four
    points x 25 draws: 1.8e-14, all 1,200 decisions equal, one and eight
    iterations 2.1e-14 apart. Erasing parity alone fails by 18.9 nats; the
    intact code by 51.9, what the second parity stream adds.
    """
    params = fixture("turbo", "ci").params
    code = params.code()
    where = code.slices()

    def departure(received: np.ndarray, reference: np.ndarray) -> float:
        """The largest gap, in nats, between the turbo posterior and BCJR's."""
        decoding = decode_turbo(
            code, received, iterations=params.iterations, backend=backend
        )
        return float(np.abs(decoding.posterior_llr - reference).max())

    worst = stationary = parity_only = intact = 0.0
    mismatched = 0

    for eb_n0_db in params.eb_n0_db:
        rng = np.random.default_rng(params.seed)
        for _ in range(CONSTITUENT_DRAWS):
            message = rng.integers(0, 2, params.message_length).astype(np.uint8)
            llr = _transmit(code, message, eb_n0_db, rng)
            streams = split_streams(code, llr)
            alone = bcjr(
                code.trellis,
                streams.systematic,
                streams.parity_first,
                backend=backend,
            )
            reference = alone.posterior_llr[: code.message_length]

            erased = llr.copy()
            erased[where["parity_second"]] = 0.0
            erased[where["tail_second"]] = 0.0
            decoding = decode_turbo(
                code, erased, iterations=params.iterations, backend=backend
            )
            first = decode_turbo(code, erased, iterations=1, backend=backend)

            worst = max(worst, departure(erased, reference))
            stationary = max(
                stationary,
                float(np.abs(first.posterior_llr - decoding.posterior_llr).max()),
            )
            mismatched += int(
                (decoding.bits != (reference < 0.0).astype(np.uint8)).sum()
            )

            kept = llr.copy()
            kept[where["parity_second"]] = 0.0
            parity_only = max(parity_only, departure(kept, reference))
            intact = max(intact, departure(llr, reference))

    assert worst < BCJR_TOLERANCE
    assert stationary < BCJR_TOLERANCE
    assert mismatched == 0
    assert parity_only == pytest.approx(18.93, abs=0.01)
    assert intact == pytest.approx(51.88, abs=0.01)


@pytest.mark.smoke
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


@pytest.mark.analytic
def test_the_second_decoder_reads_the_first_decoders_systematic_stream() -> None:
    """No message bit crosses the channel twice: the streams partition the word.

    A second systematic copy would decode better than the code allows.
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


@pytest.mark.smoke
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


@pytest.mark.analytic
def test_the_error_pattern_under_a_codeword_is_the_pattern_under_zero() -> None:
    """Per realization, on a real codeword, for the recursive encoder.

    Re-checked, not inherited from #340; licenses `measure_error_rates`'s zero message.
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


def _waterfall(
    instance: Fixture[TurboParams], backend: Backend = Backend.RUST
) -> list[ErrorRates]:
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
            backend=backend,
        )
        for eb_n0_db in params.eb_n0_db
    ]


@pytest.mark.stress
@pytest.mark.oracle
@pytest.mark.parametrize("backend", [Backend.PYTHON, Backend.RUST])
def test_the_waterfall_falls_below_the_uncoded_closed_form(backend: Backend) -> None:
    """The coded curve is under `Q(sqrt(2 E_b / N_0))` at every declared point.

    It falls with `E_b / N_0` and improves first to last iteration. Not
    monotone in iterations: rises at four of six points, largest 2.3e-3 at
    0 dB (7 to 8), each inside one interval of 12,800 bits, so asserted: last
    beats first, no rise over `NOISE_INTERVALS`.
    """
    # 12.1 s measured, so `stress` and not the CI tier. The CI-tier sibling
    # is the enumeration test above, at a size the budget holds.
    instance = fixture("turbo", "stress")
    params = instance.params
    rates = _waterfall(instance, backend)
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

    Under the uncoded form; the turn sharpens with block length (1,024 against 256).
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


# --- end to end: the planted message against the rate STATUS.md records ---------

#: The bit error rate `STATUS.md` records for the `ci` instance at 8
#: iterations, at 0, 1, 2 and 3 dB, in that order: 100 frames and 1,200
#: message bits per point.
RECORDED_BIT_ERROR_RATE = (0.115, 0.061, 0.033, 0.013)

#: Frames of 100, per point, that carried at least one wrong bit.
RECORDED_FRAME_ERRORS = (46, 23, 11, 5)

#: What a figure recorded to three decimals is recomputed to (`DEV.md`): half
#: of its last digit. A rate over seeded draws reproduces exactly, so the
#: recomputation is held to the precision the record has and no looser.
RECORDED_PRECISION = 5e-4


@pytest.mark.critical
@pytest.mark.end2end
def test_the_iteration_recovers_the_planted_message_at_the_recorded_rate() -> None:
    """Realized 0.115000, 0.060833, 0.033333 and 0.013333 against the
    0.115, 0.061, 0.033 and 0.013 `STATUS.md` records, inside the 5e-4 a
    three-decimal record is recomputed to; 46, 23, 11 and 5 of 100 frames
    carried a wrong bit.

    End to end with no oracle: seeded message, `turbo_encode`, `sim.ldpc`'s
    Gaussian channel at each declared `E_b / N_0`, `decode_turbo` at 8
    iterations, judged bit for bit. The rate is the claim: an ordering test
    survives both sides of `STATUS.md`'s MAP comparison drifting together.
    """
    params = fixture("turbo", "ci").params
    code = params.code()
    realized = []
    frames_lost = []

    for eb_n0_db in params.eb_n0_db:
        rng = np.random.default_rng(params.seed)
        channel = BinaryInputGaussianChannel(noise_scale(eb_n0_db, code.rate))
        wrong_bits = wrong_frames = 0
        for _ in range(params.frames):
            message = rng.integers(0, 2, params.message_length).astype(np.uint8)
            llr = channel.log_likelihood_ratios(turbo_encode(code, message), rng)
            decoded = decode_turbo(code, llr, iterations=params.iterations).bits
            wrong_bits += int((decoded != message).sum())
            wrong_frames += int(not np.array_equal(decoded, message))
        realized.append(wrong_bits / (params.frames * params.message_length))
        frames_lost.append(wrong_frames)

    assert tuple(frames_lost) == RECORDED_FRAME_ERRORS
    for rate, recorded in zip(realized, RECORDED_BIT_ERROR_RATE, strict=True):
        assert abs(rate - recorded) < RECORDED_PRECISION, (realized, rate)
    assert all(np.diff(realized) < 0.0)
