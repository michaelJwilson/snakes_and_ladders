"""Turbo decoding: two BCJR passes exchanging extrinsic log-likelihood ratios.

The turbo code's factor graph is two trellis chains sharing the message
variables through a permutation, so it has cycles and no exact algorithm is
available at the lengths it is used at. What Berrou, Glavieux and
Thitimajshima (1993) proposed is loopy sum-product on a *serial* schedule:
decoder one runs a full BCJR pass over its own chain, its extrinsic output
(``eq:extrinsic``) is interleaved and handed to decoder two as an a priori
ratio, decoder two runs, and its extrinsic output is deinterleaved and
handed back (``alg:turbo``). Two half-iterations are one iteration.

**Extrinsic, and why it must be.** A decoder handed the other's *posterior*
would be handed back its own previous output inside it, and the two would
agree on a shared error after a few iterations regardless of the channel.
The extrinsic ratio removes both the systematic channel term and the a
priori term the decoder was given, so what crosses is only what that chain's
own parity stream said. Nothing in :func:`decode_turbo` is a free choice
here: the subtraction is what makes the exchange the sum-product message.

**This is an approximation, and the suite reports the departure.** Exactly
one thing may be asserted against the exact bitwise MAP that
:func:`exact_turbo_posterior` enumerates: the iteration's bit error rate
approaches it, with a margin the fixture pins. Asserting equality would
assert something false --- the graph has cycles --- and asserting only that
the loop ran is the coverage theatre root ``CLAUDE.md`` forbids
(``likelihood/CLAUDE.md``, *An approximate evaluator states which regime
carries its correctness*).

**Refusing versus failing.** Issue #340's :class:`Decoding` is returned
unchanged. ``decoded`` is the turbo analogue of its syndrome check: the two
constituent decoders' hard decisions on the message agree, which is the
stopping rule Hagenauer's hard-decision-aid criterion uses and which the
error-rate measurements read as *this frame stopped early* rather than as
*this frame is right*. ``residual`` is the largest change in an extrinsic
ratio at the last iteration.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from snakes_and_ladders.enumeration import refuse_oversized
from snakes_and_ladders.likelihood.convolutional import bcjr, enumerate_messages
from snakes_and_ladders.likelihood.ldpc import Decoding, MapEstimate
from snakes_and_ladders.numerics import logsumexp
from snakes_and_ladders.sim.convolutional import (
    TurboCode,
    inverse_permutation,
    turbo_generator_matrix,
)
from snakes_and_ladders.sim.ldpc import LLR_CAP

DEFAULT_ITERATIONS = 8


@dataclass(frozen=True)
class TurboStreams:
    """One received word split into the streams each constituent decoder reads.

    Parameters
    ----------
    systematic : np.ndarray
        Shape ``(K + m,)``: the first encoder's input stream, message bits
        then its tail.
    parity_first, parity_second : np.ndarray
        Shape ``(K + m,)`` each.
    tail_second : np.ndarray
        Shape ``(m,)``: the second encoder's tail inputs, which the
        interleaver does not reach and which are therefore transmitted.
    """

    systematic: np.ndarray
    parity_first: np.ndarray
    parity_second: np.ndarray
    tail_second: np.ndarray


def split_streams(code: TurboCode, llr: np.ndarray) -> TurboStreams:
    """Cut a received word's ratios into the four streams of ``eq:turbo-word``.

    Parameters
    ----------
    code : TurboCode
    llr : np.ndarray
        Shape ``(3 K + 4 m,)``, in the convention of ``eq:ldpc-llr``.

    Returns
    -------
    TurboStreams

    Raises
    ------
    ValueError
        If ``llr`` is not the code's block length.
    """
    ratios = np.asarray(llr, dtype=float)
    if ratios.shape != (code.block_length,):
        msg = f"llr has shape {ratios.shape}, the code has {code.block_length} bits"
        raise ValueError(msg)
    where = code.slices()
    return TurboStreams(
        systematic=ratios[where["systematic"]],
        parity_first=ratios[where["parity_first"]],
        parity_second=ratios[where["parity_second"]],
        tail_second=ratios[where["tail_second"]],
    )


def _second_systematic(code: TurboCode, streams: TurboStreams) -> np.ndarray:
    """The second decoder's systematic ratios: the first's, interleaved.

    Its ``K`` message positions carry the same bits the first encoder sent,
    in the interleaver's order, so the channel evidence is the first
    stream's permuted --- the interleaver moves no bit through the channel
    twice. Its ``m`` tail positions carry bits nothing else sent, so they
    are the transmitted tail stream.
    """
    return np.concatenate(
        [
            streams.systematic[: code.message_length][code.interleaver],
            streams.tail_second,
        ]
    )


def _iterate(
    code: TurboCode, llr: np.ndarray, iterations: int
) -> Iterator[tuple[np.ndarray, bool, float]]:
    """One extrinsic exchange per step: the message posterior, agreement, residual.

    The single loop both public decoders read, so a waterfall at four
    iteration counts costs the deepest one rather than their sum -- 8
    passes against 36 at the release tier's cap -- and there is one
    implementation of the schedule rather than two to keep in step.

    Yields
    ------
    tuple[np.ndarray, bool, float]
        After each iteration: the ``K`` message posterior ratios, whether the
        two constituent decoders' hard decisions agree, and the largest
        change in an extrinsic ratio since the previous iteration.
    """
    if iterations < 1:
        msg = f"iterations must be at least 1, got {iterations}"
        raise ValueError(msg)
    streams = split_streams(code, llr)
    trellis, order = code.trellis, code.interleaver
    inverse = inverse_permutation(order)
    k, steps = code.message_length, code.steps
    second_systematic = _second_systematic(code, streams)

    apriori_first = np.zeros(steps)
    extrinsic_second = np.zeros(steps)
    for _ in range(iterations):
        first = bcjr(trellis, streams.systematic, streams.parity_first, apriori_first)
        # Only the K message positions cross the interleaver: the tails
        # belong to their own encoder, and a decoder given another's tail
        # extrinsic would be given evidence about a different bit.
        apriori_second = np.zeros(steps)
        apriori_second[:k] = np.clip(first.extrinsic_llr[:k], -LLR_CAP, LLR_CAP)[order]
        second = bcjr(trellis, second_systematic, streams.parity_second, apriori_second)
        previous = extrinsic_second
        extrinsic_second = np.zeros(steps)
        extrinsic_second[:k] = np.clip(second.extrinsic_llr[:k], -LLR_CAP, LLR_CAP)[
            inverse
        ]
        apriori_first = extrinsic_second
        # The joint posterior of a message bit: the channel's systematic
        # ratio plus what each chain's own parity said about it, which is
        # the second decoder's posterior deinterleaved.
        posterior = second.posterior_llr[:k][inverse]
        yield (
            posterior,
            bool(np.array_equal(first.posterior_llr[:k] < 0.0, posterior < 0.0)),
            float(np.abs(extrinsic_second - previous).max()),
        )


def decode_turbo(
    code: TurboCode,
    llr: np.ndarray,
    *,
    iterations: int = DEFAULT_ITERATIONS,
    early_stop: bool = False,
) -> Decoding:
    """Run ``iterations`` extrinsic exchanges and return the message decision.

    Parameters
    ----------
    code : TurboCode
    llr : np.ndarray
        Shape ``(3 K + 4 m,)``.
    iterations : int
        Full iterations, each two BCJR passes. At least one.
    early_stop : bool
        Stop at the first iteration whose two constituent decoders agree on
        every message bit. Off by default, because an error rate measured
        per iteration wants every iteration run; on, it is the cost saving a
        deployment makes and the ``iterations`` field reports where it
        stopped.

    Returns
    -------
    Decoding
        Issue #340's type. ``bits`` is the ``K``-bit message decision, not
        the transmitted word: the tail inputs are not message bits and a bit
        error rate over them would count the code's overhead as errors.
        ``estimate`` is :attr:`~snakes_and_ladders.likelihood.ldpc.MapEstimate.BITWISE`,
        since both passes are sum-product.

    Raises
    ------
    ValueError
        If ``iterations`` is below one, or ``llr`` is the wrong length.
    """
    posterior = np.zeros(code.message_length)
    agree, residual, ran = False, np.inf, 0
    for ran, (posterior, agree, residual) in enumerate(
        _iterate(code, llr, iterations), start=1
    ):
        if early_stop and agree:
            break
    return Decoding(
        bits=(posterior < 0.0).astype(np.uint8),
        posterior_llr=posterior,
        iterations=ran,
        decoded=agree,
        residual=residual,
        estimate=MapEstimate.BITWISE,
    )


def decode_turbo_per_iteration(
    code: TurboCode, llr: np.ndarray, *, iterations: int = DEFAULT_ITERATIONS
) -> np.ndarray:
    """The message decision after each of ``1 .. iterations`` iterations.

    The rows are what :func:`decode_turbo` returns at each cap -- both read
    :func:`_iterate`, and a test asserts the two agree at every cap rather
    than trusting that they must.

    Returns
    -------
    np.ndarray
        ``uint8`` of shape ``(iterations, K)``.
    """
    return np.array(
        [
            (posterior < 0.0).astype(np.uint8)
            for posterior, _, _ in _iterate(code, llr, iterations)
        ],
        dtype=np.uint8,
    )


# --- the exact oracle -------------------------------------------------------------


@dataclass(frozen=True)
class ExactTurboDecoding:
    """What enumerating all ``2 ** K`` messages says about a received word.

    Parameters
    ----------
    posterior_llr : np.ndarray
        Shape ``(K,)``: the exact bitwise MAP ratio per message bit.
    bits : np.ndarray
        Its hard decision: the bitwise MAP message, which minimizes the bit
        error rate and is what an iterative decoder is approaching.
    ml_message : np.ndarray
        The blockwise MAP: the single most likely message, a different
        answer.
    log_evidence : float
        ``log sum_u exp(-c(u) . L)``, up to the constant every word shares.
    """

    posterior_llr: np.ndarray
    bits: np.ndarray
    ml_message: np.ndarray
    log_evidence: float


def exact_turbo_posterior(code: TurboCode, llr: np.ndarray) -> ExactTurboDecoding:
    """Bit posteriors of the turbo code by enumeration over its messages.

    Encodes every ``K``-bit message and softmaxes ``-c . L``; it shares no
    recursion with :func:`decode_turbo` and reaches
    only ``K`` small, which is the ceiling the textbook's *Supported sizes*
    paragraph states.

    Parameters
    ----------
    code : TurboCode
    llr : np.ndarray
        Shape ``(3 K + 4 m,)``.

    Returns
    -------
    ExactTurboDecoding

    Raises
    ------
    ValueError
        Past :data:`snakes_and_ladders.enumeration.MAX_ENUMERABLE_CONFIGURATIONS`
        messages, or if ``llr`` is the wrong length.
    """
    ratios = np.asarray(llr, dtype=float)
    if ratios.shape != (code.block_length,):
        msg = f"llr has shape {ratios.shape}, the code has {code.block_length} bits"
        raise ValueError(msg)
    refuse_oversized(
        2**code.message_length, what=f"2 ** {code.message_length} turbo messages"
    )
    messages = enumerate_messages(code.message_length)
    # The codewords through the generator matrix rather than `2 ** K` calls
    # to the encoder: the two are the same set because encoding is
    # GF(2)-linear, which a test asserts against `turbo_encode` rather than
    # assuming, and the matrix product is what makes the oracle affordable
    # over an ensemble of frames rather than over one.
    generator = turbo_generator_matrix(code).astype(np.int64)
    words = (messages.astype(np.int64) @ generator) & 1
    scores = -(words.astype(float) @ ratios)
    log_evidence = float(logsumexp(scores[None, :], axis=1)[0])
    ones = np.where(messages == 1, scores[:, None], -np.inf)
    zeros = np.where(messages == 0, scores[:, None], -np.inf)
    posterior = np.logaddexp.reduce(zeros, axis=0) - np.logaddexp.reduce(ones, axis=0)
    return ExactTurboDecoding(
        posterior_llr=np.asarray(posterior),
        bits=(np.asarray(posterior) < 0.0).astype(np.uint8),
        ml_message=messages[int(np.argmax(scores))],
        log_evidence=log_evidence,
    )


# --- error rates over a declared ensemble -----------------------------------------


def uncoded_bit_error_rate(eb_n0_db: np.ndarray) -> np.ndarray:
    """``Q(sqrt(2 E_b / N_0))``: uncoded antipodal signalling, the analytic pin.

    The closed form every coded curve is read against: a code that does not
    beat it at the operating point is not buying anything. Written through
    ``erfc`` because ``Q(x) = erfc(x / sqrt 2) / 2`` and ``erfc`` keeps its
    significant digits where ``1 - Phi(x)`` cancels them; through the
    standard library's rather than ``scipy.special``'s, because a handful of
    points is a comprehension and the package imports scipy nowhere else.

    Parameters
    ----------
    eb_n0_db : np.ndarray
        ``E_b / N_0`` in dB.

    Returns
    -------
    np.ndarray
        The bit error rate at each point.
    """
    points = np.atleast_1d(np.asarray(eb_n0_db, dtype=float))
    return np.array(
        [0.5 * math.erfc(math.sqrt(10.0 ** (point / 10.0))) for point in points]
    )


def noise_scale(eb_n0_db: float, rate: float) -> float:
    """The Gaussian channel's ``sigma`` at an ``E_b / N_0`` and a code rate.

    Antipodal signalling puts one unit of energy in each transmitted symbol,
    so ``E_s = 1`` and ``E_b = 1 / R``; with ``N_0 = 2 sigma ** 2`` this is
    ``sigma ** 2 = 1 / (2 R E_b / N_0)``. The rate is the *transmitted* one,
    ``K / (3 K + 4 m)``, tail bits included: they are energy spent, and
    charging them to the code is what makes a short block's curve honest.

    Parameters
    ----------
    eb_n0_db : float
        In dB.
    rate : float
        ``K / n``, in ``(0, 1]``.

    Returns
    -------
    float
    """
    if not 0.0 < rate <= 1.0:
        msg = f"rate must be in (0, 1], got {rate}"
        raise ValueError(msg)
    return float(1.0 / np.sqrt(2.0 * rate * 10.0 ** (eb_n0_db / 10.0)))


@dataclass(frozen=True)
class ErrorRates:
    """A waterfall point: what was measured, and how many frames it rests on.

    A bit error rate without its frame count is a number nothing bounds, so
    the count travels with it and :attr:`bit_error_interval` is read from
    both (``sec:turbo:validation``).

    Parameters
    ----------
    eb_n0_db : float
        The point.
    bit_errors : np.ndarray
        Shape ``(iterations,)``: message-bit errors summed over frames, per
        iteration count.
    frame_errors : np.ndarray
        Shape ``(iterations,)``: frames with at least one message-bit error.
    frames : int
        Frames simulated.
    bits : int
        Message bits simulated, ``frames * K``.
    """

    eb_n0_db: float
    bit_errors: np.ndarray
    frame_errors: np.ndarray
    frames: int
    bits: int

    @property
    def bit_error_rate(self) -> np.ndarray:
        """Errors over bits, per iteration count."""
        return np.asarray(self.bit_errors / self.bits)

    @property
    def frame_error_rate(self) -> np.ndarray:
        """Frames in error over frames, per iteration count."""
        return np.asarray(self.frame_errors / self.frames)

    @property
    def bit_error_interval(self) -> np.ndarray:
        """The one-sigma binomial interval half-width on :attr:`bit_error_rate`.

        ``sqrt(p (1 - p) / N)`` with ``N`` the message bits simulated. Bit
        errors inside one frame are correlated --- a turbo failure is bursty
        --- so this understates the spread of the *estimate* and is stated
        as what it is: the error bar the sample size supports if the bits
        were independent, and a floor on the true one.

        Returns
        -------
        np.ndarray
            Shape ``(iterations,)``.
        """
        p = self.bit_error_rate
        return np.asarray(np.sqrt(p * (1.0 - p) / self.bits))


def measure_error_rates(
    code: TurboCode,
    eb_n0_db: float,
    frames: int,
    rng: np.random.Generator,
    *,
    iterations: int = DEFAULT_ITERATIONS,
) -> ErrorRates:
    """Simulate ``frames`` transmissions at one ``E_b / N_0`` and count errors.

    The all-zero message is sent. The turbo code is linear and the Gaussian
    channel output symmetric, so the error pattern's distribution does not
    depend on the message sent (``sec:ldpc``, the argument issue #340 states
    and this suite re-checks for the recursive encoder); sending zero costs
    nothing and removes the encoder from the inner loop.

    Parameters
    ----------
    code : TurboCode
    eb_n0_db : float
        The point, in dB.
    frames : int
        Frames to simulate. It is the whole of the resolution of the
        result, so it is a parameter and never a default.
    rng : np.random.Generator
        ``sim/CLAUDE.md``: a generator, never a seed.
    iterations : int
        The deepest iteration count; every shallower one is read from the
        same run.

    Returns
    -------
    ErrorRates
    """
    sigma = noise_scale(eb_n0_db, code.rate)
    zero = np.zeros(code.block_length, dtype=np.uint8)
    bit_errors = np.zeros(iterations, dtype=np.int64)
    frame_errors = np.zeros(iterations, dtype=np.int64)
    for _ in range(frames):
        received = (1.0 - 2.0 * zero) + sigma * rng.standard_normal(code.block_length)
        decisions = decode_turbo_per_iteration(
            code, 2.0 * received / sigma**2, iterations=iterations
        )
        errors = decisions.sum(axis=1, dtype=np.int64)
        bit_errors += errors
        frame_errors += errors > 0
    return ErrorRates(
        eb_n0_db=eb_n0_db,
        bit_errors=bit_errors,
        frame_errors=frame_errors,
        frames=frames,
        bits=frames * code.message_length,
    )
