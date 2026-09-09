"""Decoding one convolutional code: BCJR, Viterbi, and the enumeration oracle.

A terminated shift register's trellis is a hidden Markov chain whose state is
the register contents, so its two classical decoders are the two this
repository already has on a chain: BCJR is forward--backward
(``eq:forward``, ``eq:posterior``) and Viterbi is max-product. What the chain
modules cannot be handed directly is the *edge*: the observation belongs to
the transition and not to the state, since the bits transmitted at a step are
a function of the edge taken. This module is therefore forward--backward with
the emission on the edge, in the log domain, plus the *extrinsic* output a
turbo iteration exchanges (``eq:bcjr``, ``eq:extrinsic``).

**One convention throughout.** Every log-likelihood ratio is
``eq:ldpc-llr``'s, imported from issue #340 rather than restated: ``L =
log p(y | bit = 0) - log p(y | bit = 1)``, positive when the channel favours
zero. A bit's log-likelihood contribution is then ``-bit * L`` up to a
constant shared by every path, which is what makes the branch metric a sum
of three such terms and the posterior ratio decompose as ``eq:extrinsic``
into channel, a priori and extrinsic parts.

**Which oracle referees which claim.** At ``K`` small enough to enumerate,
:func:`exact_bitwise_posterior` sums over all ``2 ** K`` messages and is the
referee for BCJR; the maximum of the same sum is the referee for Viterbi.
Independently of size, :func:`snakes_and_ladders.sim.factor_graph.from_trellis`
presents the same chain to the general
:func:`snakes_and_ladders.likelihood.message_passing.sum_product`, whose tree
schedule is exact on it and shares no recursion with the code here. Both are
used: the first says the answer is right, the second says the specialization
is the same computation as the general one, which is the pairing issue #340
established for the parity-check decoder.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from snakes_and_ladders.enumeration import refuse_oversized
from snakes_and_ladders.numerics import logsumexp
from snakes_and_ladders.sim.convolutional import (
    IMPOSSIBLE_EDGE,
    Trellis,
    encode_stream,
    terminate,
)


@dataclass(frozen=True)
class TrellisDecoding:
    """What one BCJR pass returns.

    Parameters
    ----------
    posterior_llr : np.ndarray
        Per trellis step, ``log P(u_t = 0 | y) - log P(u_t = 1 | y)`` under
        the channel and a priori ratios given.
    extrinsic_llr : np.ndarray
        ``posterior_llr`` less the systematic channel ratio and the a priori
        ratio: what this decoder learned about ``u_t`` from every step but
        ``t`` (``eq:extrinsic``). This is what a turbo iteration passes on,
        and passing the posterior instead would feed a decoder its own
        previous output.
    log_evidence : float
        ``log p(y)`` up to the constant ``sum_i log p(y_i | 0)`` every path
        shares: the forward recursion's normalizer.
    """

    posterior_llr: np.ndarray
    extrinsic_llr: np.ndarray
    log_evidence: float


def _branch_metrics(
    trellis: Trellis,
    systematic_llr: np.ndarray,
    parity_llr: np.ndarray,
    apriori_llr: np.ndarray,
) -> np.ndarray:
    """``(T, n_states, 2)``: the log weight of each edge at each step.

    An edge with input ``u`` and parity ``p`` at step ``t`` carries
    ``-(u (L_s + L_a) + p L_p)``, the log-likelihood of the two bits it
    transmits plus the a priori log-odds of its input, each in
    ``eq:ldpc-llr``'s convention and each up to the per-bit constant every
    edge shares.
    """
    total = systematic_llr + apriori_llr
    inputs = np.array([0.0, 1.0])
    metrics = -inputs[None, None, :] * total[:, None, None]
    return np.asarray(metrics - trellis.parity[None, :, :] * parity_llr[:, None, None])


def bcjr(
    trellis: Trellis,
    systematic_llr: np.ndarray,
    parity_llr: np.ndarray,
    apriori_llr: np.ndarray | None = None,
    *,
    terminated: bool = True,
) -> TrellisDecoding:
    """Forward--backward over the trellis, with the observation on the edge.

    ``alpha`` and ``beta`` are the log-domain forward and backward state
    metrics of ``eq:bcjr``; the posterior ratio at step ``t`` is the
    log-sum-exp over edges with input zero less that over edges with input
    one. Exact log-MAP, not the max-log approximation: the sums are formed
    with :func:`snakes_and_ladders.numerics.logsumexp`, so nothing here is an
    approximation to be reported under
    ``likelihood/CLAUDE.md``'s rule and equality against enumeration may be
    asserted.

    Parameters
    ----------
    trellis : Trellis
    systematic_llr, parity_llr : np.ndarray
        Shape ``(T,)`` each: the channel's ratio for the systematic and the
        parity bit transmitted at each step.
    apriori_llr : np.ndarray | None
        Shape ``(T,)``: the ratio a previous decoder supplies for the input
        bit. ``None`` is the uninformative ``0``, and is the whole
        difference between a stand-alone convolutional decoding and a turbo
        half-iteration.
    terminated : bool
        Whether the register was driven back to the zero state, so the
        backward recursion starts concentrated there rather than uniform.
        ``sim.convolutional.terminate`` is what makes this true, and a
        decoder told the wrong thing computes a posterior for a different
        model.

    Returns
    -------
    TrellisDecoding

    Raises
    ------
    ValueError
        If the three arrays disagree in length or are not one-dimensional.
    """
    systematic = np.asarray(systematic_llr, dtype=float)
    parity = np.asarray(parity_llr, dtype=float)
    apriori = (
        np.zeros_like(systematic)
        if apriori_llr is None
        else np.asarray(apriori_llr, dtype=float)
    )
    if systematic.ndim != 1 or not (
        systematic.shape == parity.shape == apriori.shape
    ):
        msg = (
            f"systematic {systematic.shape}, parity {parity.shape} and a priori "
            f"{apriori.shape} must be the same one-dimensional shape"
        )
        raise ValueError(msg)
    length = systematic.size
    n_states = trellis.n_states
    gamma = _branch_metrics(trellis, systematic, parity, apriori)

    alpha = np.full((length + 1, n_states), IMPOSSIBLE_EDGE)
    alpha[0, 0] = 0.0
    for t in range(length):
        # One scatter per input bit: edge (s, u) carries alpha[t, s] + gamma
        # into next_state[s, u]. Two `logaddexp.at` calls rather than a
        # Python loop over states, which is the cost that matters at K = 1024.
        arriving = np.full((2, n_states), IMPOSSIBLE_EDGE)
        for u in (0, 1):
            np.logaddexp.at(
                arriving[u], trellis.next_state[:, u], alpha[t] + gamma[t, :, u]
            )
        alpha[t + 1] = np.logaddexp(arriving[0], arriving[1])

    beta = np.full((length + 1, n_states), IMPOSSIBLE_EDGE)
    if terminated:
        beta[length, 0] = 0.0
    else:
        beta[length] = 0.0
    for t in range(length - 1, -1, -1):
        # A gather: from state s the two edges lead to next_state[s, u].
        beta[t] = np.logaddexp(
            beta[t + 1][trellis.next_state[:, 0]] + gamma[t, :, 0],
            beta[t + 1][trellis.next_state[:, 1]] + gamma[t, :, 1],
        )

    edge = (
        alpha[:length, :, None]
        + gamma
        + beta[1:][np.arange(length)[:, None, None], trellis.next_state[None, :, :]]
    )
    per_input = logsumexp(np.ascontiguousarray(edge.transpose(0, 2, 1)), axis=2)
    posterior = per_input[:, 0] - per_input[:, 1]
    # `alpha + beta` at the final step and not `alpha` alone: on a terminated
    # trellis only paths ending at the zero state are in the model, and
    # summing every end state would be the evidence of a different code.
    log_evidence = float(logsumexp((alpha[length] + beta[length])[None, :], axis=1)[0])
    return TrellisDecoding(
        posterior_llr=np.asarray(posterior),
        extrinsic_llr=np.asarray(posterior - systematic - apriori),
        log_evidence=log_evidence,
    )


def viterbi(
    trellis: Trellis,
    systematic_llr: np.ndarray,
    parity_llr: np.ndarray,
    apriori_llr: np.ndarray | None = None,
    *,
    terminated: bool = True,
) -> np.ndarray:
    """The single most likely input sequence: max-product on the same trellis.

    The blockwise MAP, against BCJR's bitwise one --- two decodings of one
    model and different answers, as ``likelihood/CLAUDE.md`` states of the
    chain. The arguments are :func:`bcjr`'s.

    Returns
    -------
    np.ndarray
        ``uint8`` of shape ``(T,)``: the input bits of the best path.
    """
    systematic = np.asarray(systematic_llr, dtype=float)
    parity = np.asarray(parity_llr, dtype=float)
    apriori = (
        np.zeros_like(systematic)
        if apriori_llr is None
        else np.asarray(apriori_llr, dtype=float)
    )
    length, n_states = systematic.size, trellis.n_states
    gamma = _branch_metrics(trellis, systematic, parity, apriori)
    metric = np.full((length + 1, n_states), IMPOSSIBLE_EDGE)
    metric[0, 0] = 0.0
    back_state = np.zeros((length, n_states), dtype=np.int64)
    back_input = np.zeros((length, n_states), dtype=np.uint8)
    for t in range(length):
        best = np.full(n_states, IMPOSSIBLE_EDGE)
        for u in (0, 1):
            scores = metric[t] + gamma[t, :, u]
            target = trellis.next_state[:, u]
            # `maximum.at` keeps the largest of the (usually two) edges into
            # each state; the argmax is recovered by the equality below,
            # which costs one pass rather than a Python loop over states.
            np.maximum.at(best, target, scores)
        for u in (0, 1):
            scores = metric[t] + gamma[t, :, u]
            target = trellis.next_state[:, u]
            chosen = scores >= best[target]
            back_state[t, target[chosen]] = np.arange(n_states)[chosen]
            back_input[t, target[chosen]] = u
        metric[t + 1] = best
    end = 0 if terminated else int(np.argmax(metric[length]))
    path = np.empty(length, dtype=np.uint8)
    state = end
    for t in range(length - 1, -1, -1):
        path[t] = back_input[t, state]
        state = int(back_state[t, state])
    return path


# --- the enumeration oracle ------------------------------------------------------


@dataclass(frozen=True)
class ExactTrellisDecoding:
    """The answers enumeration over all ``2 ** K`` messages gives.

    Parameters
    ----------
    posterior_llr : np.ndarray
        Shape ``(K,)``: the exact bitwise MAP ratio per message bit.
    ml_message : np.ndarray
        The maximum-likelihood message; the first in enumeration order on a
        tie, which a fixture rules out by pinning the margin.
    log_evidence : float
        ``log sum_u exp(score(u))``, in the same additive constant as
        :attr:`TrellisDecoding.log_evidence`.
    margin : float
        The gap in log-likelihood between the best message and the runner-up:
        zero exactly when the maximum is not unique.
    """

    posterior_llr: np.ndarray
    ml_message: np.ndarray
    log_evidence: float
    margin: float


def enumerate_messages(message_length: int) -> np.ndarray:
    """Every ``K``-bit message, as a ``(2 ** K, K)`` ``uint8`` array.

    Raises
    ------
    ValueError
        Past :data:`snakes_and_ladders.enumeration.MAX_ENUMERABLE_CONFIGURATIONS`.
    """
    refuse_oversized(2**message_length, what=f"2 ** {message_length} messages")
    return np.asarray(
        (np.arange(2**message_length)[:, None] >> np.arange(message_length - 1, -1, -1))
        & 1,
        dtype=np.uint8,
    )


def exact_bitwise_posterior(
    trellis: Trellis,
    systematic_llr: np.ndarray,
    parity_llr: np.ndarray,
    message_length: int,
) -> ExactTrellisDecoding:
    """Bit posteriors of a terminated convolutional code, by enumeration.

    Sums ``exp(-c . L)`` over the ``2 ** K`` messages, with ``c`` the
    terminated input sequence and its parity; it shares no recursion with
    :func:`bcjr` and reaches only ``K`` small, which is the ceiling the
    textbook's *Supported sizes* paragraph states.

    Parameters
    ----------
    trellis : Trellis
    systematic_llr, parity_llr : np.ndarray
        Shape ``(K + m,)``: the channel's ratios over the terminated stream.
    message_length : int
        ``K``; the trellis runs ``K + m`` steps.

    Returns
    -------
    ExactTrellisDecoding
    """
    systematic = np.asarray(systematic_llr, dtype=float)
    parity = np.asarray(parity_llr, dtype=float)
    messages = enumerate_messages(message_length)
    scores = np.empty(messages.shape[0])
    for index, message in enumerate(messages):
        inputs = terminate(trellis, message)
        bits, _ = encode_stream(trellis, inputs)
        scores[index] = -(inputs @ systematic) - (bits @ parity)
    log_evidence = float(logsumexp(scores[None, :], axis=1)[0])
    ones = np.where(messages == 1, scores[:, None], -np.inf)
    zeros = np.where(messages == 0, scores[:, None], -np.inf)
    posterior = np.logaddexp.reduce(zeros, axis=0) - np.logaddexp.reduce(ones, axis=0)
    order = np.argsort(scores)[::-1]
    margin = float(scores[order[0]] - scores[order[1]]) if scores.size > 1 else np.inf
    return ExactTrellisDecoding(
        posterior_llr=np.asarray(posterior),
        ml_message=messages[int(order[0])],
        log_evidence=log_evidence,
        margin=margin,
    )
