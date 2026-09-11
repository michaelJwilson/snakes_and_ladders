"""Decoding a parity-check code: sum-product, min-sum, and the enumeration oracle.

The general :func:`snakes_and_ladders.likelihood.message_passing.sum_product`
already decodes a code through
:func:`snakes_and_ladders.sim.factor_graph.from_parity_check`, one table per
parity factor and one dictionary entry per message. At 60,000 edges that is
the wrong shape, so this module is the same algorithm specialised to binary
variables and parity factors, where a message is one log-likelihood ratio and
the factor update collapses to the ``tanh`` rule, ``eq:tanh-rule`` (Gallager
1962; Richardson and Urbanke 2008). It is held to the general implementation
on small codes: same fixed point, to a stated tolerance, reached in a
different order.

**Which Markov chain.** The schedule is flooding: every check reads the
variable messages of the previous half-iteration and every variable reads the
check messages just written, so an iteration is one Jacobi sweep over the
checks followed by one over the variables. Two message buffers hold it, one
per direction, each ``n_edges`` long in variable order and read in check order
through the code's one permutation; both are allocated once and written in
place. A layered (Gauss--Seidel) schedule is a different iteration with a
different transient, and is not what the general oracle runs either.

**Two decodings, and they are different answers.** Sum-product's hard decision
is the *bitwise* MAP -- each bit at the mode of its own marginal -- and can be
a word outside the code. Min-sum, ``eq:min-sum``, is max-product in the log
domain: its decision is the *blockwise* MAP, exact on a cycle-free code with a
unique maximum. :class:`Decoding.estimate` names which one a result is.

**Refusing versus failing.** A decoder that reaches its iteration cap without
a codeword has not failed to converge; it has failed to decode, and that is
the event a block error rate counts. The result says so through ``decoded``
rather than raising, and its bits are the beliefs at the last iteration.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from snakes_and_ladders.enumeration import refuse_oversized
from snakes_and_ladders.numerics import logsumexp
from snakes_and_ladders.sim.ldpc import LLR_CAP, ParityCheck, generator_matrix

DEFAULT_MAX_ITERATIONS = 100


class DecodingAlgorithm(StrEnum):
    """The check update, and therefore which MAP the decision is."""

    SUM_PRODUCT = "sum-product"
    """The ``tanh`` rule: exact marginals on a tree, Bethe elsewhere."""

    MIN_SUM = "min-sum"
    """Sign times minimum magnitude: max-product, so the blockwise MAP."""


class MapEstimate(StrEnum):
    """Which decoding a hard decision is."""

    BITWISE = "bitwise"
    BLOCKWISE = "blockwise"


@dataclass(frozen=True)
class Decoding:
    """What the decoder returns.

    Parameters
    ----------
    bits : np.ndarray
        The hard decision, ``uint8`` of shape ``(n_bits,)``: one where the
        posterior ratio is negative, zero where it is positive or zero.
    posterior_llr : np.ndarray
        Per bit, the channel ratio plus every incoming check message: the
        log-odds of the marginal (sum-product) or max-marginal (min-sum).
        Exactly zero is an *erasure*, not a decision for zero, which is why
        the field below is not the syndrome alone: the zero word through an
        erasure channel would otherwise satisfy every check at the first
        iteration.
    iterations : int
        Iterations run, including the one that produced ``bits``.
    decoded : bool
        Whether every bit is decided and ``H bits = 0``: the stop condition.
    residual : float
        The largest change in a check-to-variable message at the last
        iteration: zero at a fixed point, and what ``tolerance`` is held to.
    estimate : MapEstimate
        Which decoding ``bits`` is.
    """

    bits: np.ndarray
    posterior_llr: np.ndarray
    iterations: int
    decoded: bool
    residual: float
    estimate: MapEstimate


def _segments(code: ParityCheck) -> np.ndarray:
    """The check each edge belongs to, in check order."""
    return np.repeat(np.arange(code.n_checks), code.row_weights)


def _tanh_rule(
    incoming: np.ndarray, starts: np.ndarray, segment: np.ndarray
) -> np.ndarray:
    """``eq:tanh-rule``: ``2 atanh prod_{j != i} tanh(m_j / 2)`` per edge.

    The leave-one-out product is the segment product divided by the edge's
    own factor, with zeros counted rather than divided by: a zero (an
    erasure) makes every other edge's product zero and leaves its own as
    the product of the rest.
    """
    factors = np.tanh(0.5 * incoming)
    zero = factors == 0.0
    safe = np.where(zero, 1.0, factors)
    product = np.multiply.reduceat(safe, starts)[segment]
    zeros = np.add.reduceat(zero.astype(np.int64), starts)[segment]
    leave_one_out = np.where(
        zeros == 0, product / safe, np.where((zeros == 1) & zero, product, 0.0)
    )
    return np.asarray(2.0 * np.arctanh(leave_one_out))


def _min_sum(
    incoming: np.ndarray, starts: np.ndarray, segment: np.ndarray
) -> np.ndarray:
    """``eq:min-sum``: the product of the other signs times their smallest magnitude.

    The leave-one-out minimum is the segment's minimum unless the edge holds
    it alone, in which case it is the second smallest.
    """
    magnitude = np.abs(incoming)
    sign = np.where(incoming < 0.0, -1.0, 1.0)
    others_sign = np.multiply.reduceat(sign, starts)[segment] * sign
    smallest = np.minimum.reduceat(magnitude, starts)[segment]
    holds_it = magnitude == smallest
    ties = np.add.reduceat(holds_it.astype(np.int64), starts)[segment]
    second = np.minimum.reduceat(np.where(holds_it, np.inf, magnitude), starts)[segment]
    return np.asarray(others_sign * np.where(holds_it & (ties == 1), second, smallest))


_UPDATES = {
    DecodingAlgorithm.SUM_PRODUCT: _tanh_rule,
    DecodingAlgorithm.MIN_SUM: _min_sum,
}
_ESTIMATES = {
    DecodingAlgorithm.SUM_PRODUCT: MapEstimate.BITWISE,
    DecodingAlgorithm.MIN_SUM: MapEstimate.BLOCKWISE,
}


def decode(
    code: ParityCheck,
    llr: np.ndarray,
    *,
    algorithm: DecodingAlgorithm = DecodingAlgorithm.SUM_PRODUCT,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    early_stop: bool = True,
    tolerance: float | None = None,
) -> Decoding:
    """Iterative decoding from channel log-likelihood ratios.

    Messages in both directions are clipped to ``+-LLR_CAP`` before the check
    update, so ``tanh`` never saturates; the posterior differs from the
    unclipped one by less than ``exp(-LLR_CAP)``, inside every tolerance the
    suite states.

    Parameters
    ----------
    code : ParityCheck
    llr : np.ndarray
        Shape ``(n_bits,)``, in the convention of ``eq:ldpc-llr``.
    algorithm : DecodingAlgorithm
        The check update.
    max_iterations : int
        The cap; at least one.
    early_stop : bool
        Stop at the first iteration whose hard decision is a codeword with
        no bit left undecided.
    tolerance : float | None
        Also stop once no check-to-variable message moved by more than
        this; ``0.0`` stops at an exact fixed point, which the erasure
        channel reaches. ``None``, the default, never stops on the residual.
        With ``early_stop`` off a tolerance is how a fixed point is compared
        to the general implementation's.

    Raises
    ------
    ValueError
        If ``llr`` is the wrong length or the cap is below one.
    """
    ratios = np.asarray(llr, dtype=float)
    if ratios.shape != (code.n_bits,):
        msg = f"llr has shape {ratios.shape}, the code has {code.n_bits} bits"
        raise ValueError(msg)
    if max_iterations < 1:
        msg = f"max_iterations must be at least 1, got {max_iterations}"
        raise ValueError(msg)
    update = _UPDATES[algorithm]
    order = code.check_order
    variable_starts = code.variable_offsets[:-1]
    check_starts = code.check_offsets[:-1]
    segment = _segments(code)
    # The two buffers, in variable order; each iteration writes both in place.
    to_check = np.zeros(code.n_edges)
    to_variable = np.zeros(code.n_edges)
    posterior = ratios.copy()
    bits = np.zeros(code.n_bits, dtype=np.uint8)
    decoded = False
    residual = np.inf
    iteration = 0
    while iteration < max_iterations:
        iteration += 1
        np.subtract(posterior[code.edge_variable], to_variable, out=to_check)
        np.clip(to_check, -LLR_CAP, LLR_CAP, out=to_check)
        proposal = np.clip(
            update(to_check[order], check_starts, segment), -LLR_CAP, LLR_CAP
        )
        residual = float(np.abs(proposal - to_variable[order]).max())
        to_variable[order] = proposal
        np.add(ratios, np.add.reduceat(to_variable, variable_starts), out=posterior)
        bits = (posterior < 0.0).astype(np.uint8)
        decoded = bool(np.all(posterior != 0.0)) and not np.any(code.syndrome(bits))
        if (early_stop and decoded) or (
            tolerance is not None and residual <= tolerance
        ):
            break
    return Decoding(
        bits, posterior, iteration, decoded, residual, _ESTIMATES[algorithm]
    )


# --- the exact oracle -----------------------------------------------------------


@dataclass(frozen=True)
class ExactDecoding:
    """The answers enumeration gives, for the decoder to be held to.

    Parameters
    ----------
    posterior_llr : np.ndarray
        Per bit, ``log P(x_i = 0 | y) - log P(x_i = 1 | y)``.
    ml_codeword : np.ndarray
        The codeword of highest likelihood; the first in enumeration order
        on a tie, which a fixture rules out by pinning the margin.
    log_evidence : float
        ``log sum_c exp(-c . L)``, the normalizer, up to the constant
        ``sum_i log p(y_i | 0)`` that every word shares.
    n_codewords : int
        ``2 ** k``.
    """

    posterior_llr: np.ndarray
    ml_codeword: np.ndarray
    log_evidence: float
    n_codewords: int


def enumerate_codewords(code: ParityCheck) -> np.ndarray:
    """Every codeword, as a ``(2 ** k, n_bits)`` ``uint8`` array.

    The words are the span of
    :func:`snakes_and_ladders.sim.ldpc.generator_matrix`, each checked against
    ``H c = 0`` here, so the set is the code by definition rather than by
    trust in the elimination. A (3,6) code at ``n = 24`` has ``k >= 14``:
    ``16,384`` words.

    Raises
    ------
    ValueError
        Past :data:`snakes_and_ladders.enumeration.MAX_ENUMERABLE_CONFIGURATIONS`
        words, or past the length the elimination allows.
    """
    generator = generator_matrix(code)
    k = generator.shape[0]
    refuse_oversized(2**k, what=f"2 ** {k} codewords of a [{code.n_bits}, {k}] code")
    coefficients = (np.arange(2**k)[:, None] >> np.arange(k)) & 1
    words = np.asarray((coefficients @ generator.astype(np.int64)) & 1, dtype=np.uint8)
    if np.any((code.dense().astype(np.int64) @ words.T) & 1):
        msg = "the generator matrix spans a word outside the code"
        raise RuntimeError(msg)
    return words


def exact_decoding(code: ParityCheck, llr: np.ndarray) -> ExactDecoding:
    """Bit posteriors and the maximum-likelihood codeword by enumeration.

    Under ``eq:ldpc-llr`` a word's log-likelihood is ``-c . L`` plus a
    constant, so the posterior over the code is a softmax of ``-c . L``
    and the bit posteriors are its marginals.
    """
    ratios = np.asarray(llr, dtype=float)
    words = enumerate_codewords(code)
    log_weight = -(words.astype(float) @ ratios)
    log_evidence = float(logsumexp(log_weight[np.newaxis, :], axis=1)[0])
    # `logaddexp` rather than `numerics.logsumexp` for the marginals: a bit
    # that is one in no codeword has an all-`-inf` column, which the shifted
    # form turns into `nan` and `logaddexp` correctly leaves at `-inf`.
    ones = np.where(words == 1, log_weight[:, np.newaxis], -np.inf)
    zeros = np.where(words == 0, log_weight[:, np.newaxis], -np.inf)
    posterior = np.logaddexp.reduce(zeros, axis=0) - np.logaddexp.reduce(ones, axis=0)
    return ExactDecoding(
        np.asarray(posterior),
        words[int(np.argmax(log_weight))],
        log_evidence,
        int(words.shape[0]),
    )


# --- density evolution on the erasure channel -----------------------------------


def erasure_density_evolution(
    epsilon: float, column_weight: int, row_weight: int, iterations: int
) -> np.ndarray:
    """``eq:density-evolution``: the erasure probability of a variable-to-check
    message on a cycle-free ``(column_weight, row_weight)`` code, per iteration.

    ``x_0 = epsilon`` and ``x_l = epsilon (1 - (1 - x_{l-1})^(k-1))^(j-1)``:
    a check message is erased unless every other incoming message is known,
    and a variable message is erased when the channel and every other check
    are. It is the infinite-length limit of the flooding decoder on the
    erasure channel, and :func:`erasure_threshold` reads the largest
    ``epsilon`` it goes to zero from (Richardson and Urbanke 2008, §3.12).
    """
    trajectory = np.empty(iterations + 1)
    trajectory[0] = epsilon
    for step in range(1, iterations + 1):
        known = (1.0 - trajectory[step - 1]) ** (row_weight - 1)
        trajectory[step] = epsilon * (1.0 - known) ** (column_weight - 1)
    return trajectory


def erasure_threshold(
    column_weight: int,
    row_weight: int,
    *,
    iterations: int = 20_000,
    precision: float = 1e-6,
) -> float:
    """The BP threshold ``epsilon*`` of the ensemble on the erasure channel.

    Bisection on ``epsilon`` between zero and one, ``epsilon`` counted as
    below the threshold when ``eq:density-evolution`` falls under
    ``1e-10`` within ``iterations``. Convergence slows as the threshold is
    approached from below, so the cap bounds how close the bisection can
    resolve; the (3,6) value ``0.4294`` is the published pin.
    """
    low, high = 0.0, 1.0
    while high - low > precision:
        middle = 0.5 * (low + high)
        final = erasure_density_evolution(middle, column_weight, row_weight, iterations)
        if final[-1] < 1e-10:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)
