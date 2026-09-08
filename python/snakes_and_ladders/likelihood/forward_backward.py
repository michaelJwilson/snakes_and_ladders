"""Forward--backward as an evaluator, not as Baum--Welch's internals (issue #173).

One chain, one pass each way in the log domain: the evidence, the posterior
at every position, and the pairwise posterior across every transition --
``eq:forward`` and ``eq:posterior`` of ``docs/tex/textbook.tex``, derived in
``app:forward-backward`` as pruning on a chain, and the E step every
chain-shaped model here shares. Baum--Welch in :mod:`snakes_and_ladders.opt.hmm` keeps its own
recursion for the gradient it needs; this one exists so the coupled model,
and any caller that wants a posterior rather than a fit, does not reach into
an optimizer's internals to get one. Pinned against the path enumeration,
which shares no recursion with it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from snakes_and_ladders.numerics import logsumexp


@dataclass(frozen=True)
class ForwardBackward:
    """What the two passes return.

    Parameters
    ----------
    log_evidence : float
        ``log p(y_1..T)``.
    posterior : np.ndarray
        ``p(z_t = i | y)``, shape ``(T, K)``, each row summing to one.
    pairwise : np.ndarray
        ``p(z_{t-1} = i, z_t = j | y)``, shape ``(T - 1, K, K)``, each slice
        summing to one; empty when ``T = 1``.
    """

    log_evidence: float
    posterior: np.ndarray
    pairwise: np.ndarray


def forward_backward(
    log_density: np.ndarray, log_initial: np.ndarray, log_transition: np.ndarray
) -> ForwardBackward:
    """Run both passes on one chain.

    Parameters
    ----------
    log_density : np.ndarray
        Per-position emission scores, shape ``(T, K)``: a log-probability for
        a discrete family, a log-density otherwise.
    log_initial : np.ndarray
        Shape ``(K,)``.
    log_transition : np.ndarray
        Shape ``(K, K)``, rows the source state.

    Raises
    ------
    ValueError
        If the shapes disagree or the chain is empty.
    """
    log_density = np.asarray(log_density, dtype=float)
    log_initial = np.asarray(log_initial, dtype=float)
    log_transition = np.asarray(log_transition, dtype=float)
    if log_density.ndim != 2 or log_density.shape[0] < 1:
        msg = f"log_density must be (T, K) with T >= 1, got {log_density.shape}"
        raise ValueError(msg)
    length, n_states = log_density.shape
    if log_initial.shape != (n_states,) or log_transition.shape != (n_states, n_states):
        msg = (
            f"log_initial {log_initial.shape} and log_transition {log_transition.shape} "
            f"do not match {n_states} states"
        )
        raise ValueError(msg)

    alpha = np.empty((length, n_states))
    alpha[0] = log_initial + log_density[0]
    for t in range(1, length):
        alpha[t] = (
            logsumexp(alpha[t - 1][:, None] + log_transition, axis=0) + log_density[t]
        )
    beta = np.zeros((length, n_states))
    for t in range(length - 2, -1, -1):
        beta[t] = logsumexp(
            log_transition + (log_density[t + 1] + beta[t + 1])[None, :], axis=1
        )
    log_evidence = float(logsumexp(alpha[-1][None, :], axis=1)[0])
    posterior = np.exp(alpha + beta - log_evidence)
    pairwise = np.empty((max(length - 1, 0), n_states, n_states))
    for t in range(1, length):
        pairwise[t - 1] = np.exp(
            alpha[t - 1][:, None]
            + log_transition
            + (log_density[t] + beta[t])[None, :]
            - log_evidence
        )
    return ForwardBackward(log_evidence, posterior, pairwise)


def sample_path(
    log_density: np.ndarray,
    log_initial: np.ndarray,
    log_transition: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """One draw of the hidden path from its posterior: forward filter, backward sample.

    The block Gibbs move a chain-shaped model needs: exact, because the
    posterior over paths factorizes backward given the forward messages.
    """
    log_density = np.asarray(log_density, dtype=float)
    log_initial = np.asarray(log_initial, dtype=float)
    log_transition = np.asarray(log_transition, dtype=float)
    length, n_states = log_density.shape
    alpha = np.empty((length, n_states))
    alpha[0] = log_initial + log_density[0]
    for t in range(1, length):
        alpha[t] = (
            logsumexp(alpha[t - 1][:, None] + log_transition, axis=0) + log_density[t]
        )
    path = np.empty(length, dtype=np.int64)
    weights = np.exp(alpha[-1] - logsumexp(alpha[-1][None, :], axis=1)[0])
    path[-1] = rng.choice(n_states, p=weights / weights.sum())
    for t in range(length - 2, -1, -1):
        scores = alpha[t] + log_transition[:, path[t + 1]]
        weights = np.exp(scores - logsumexp(scores[None, :], axis=1)[0])
        path[t] = rng.choice(n_states, p=weights / weights.sum())
    return path
