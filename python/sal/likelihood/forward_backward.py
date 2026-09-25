"""Forward--backward as an evaluator, not as Baum--Welch's internals (issue #173).

One chain, one pass each way in the log domain: the evidence, the posterior
at every position, and the pairwise posterior across every transition --
``eq:forward`` and ``eq:posterior`` of ``docs/tex/textbook.tex``, derived in
``app:forward-backward`` as pruning on a chain, and the E step every
chain-shaped model here shares. Baum--Welch in :mod:`sal.opt.hmm` keeps its own
recursion for the gradient it needs; this one exists so the coupled model,
and any caller that wants a posterior rather than a fit, does not reach into
an optimizer's internals to get one. Pinned against the path enumeration,
which shares no recursion with it.

**No `backend` here, and `likelihood.rust.ragged.posteriors` says why** (issue #860).
The compiled ragged kernel returns log marginals and the transition counts
*summed over the segments*; this returns probabilities and the per-step
pairwise posterior. The sum is not the steps, so no conversion recovers this
return from that one, and a door onto it would be a second implementation
rather than a route to the same answer.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np

from sal.numerics import constant_chain_kernel, logsumexp


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


@dataclass(frozen=True)
class StepKernels:
    """A chain's kernels, and the one matrix to read instead where there is one.

    Parameters
    ----------
    kernels : np.ndarray
        ``(T - 1, K, K)``, one matrix per transition. For a constant kernel
        this is :func:`numpy.broadcast_to` over the caller's matrix: stride
        zero, not a copy.
    constant : np.ndarray | None
        The ``(K, K)`` matrix itself where the kernel does not vary, so the
        constant path indexes nothing; ``None`` where it does vary.
    """

    kernels: np.ndarray
    constant: np.ndarray | None

    def __iter__(self) -> Iterator[Any]:
        """``(kernels, constant)``: the order callers unpack.

        ``Any`` because only ``constant`` is optional, and the union would
        mistype ``kernels`` at every unpacking.
        """
        yield from (self.kernels, self.constant)


def step_kernels(log_transition: np.ndarray, length: int, n_states: int) -> StepKernels:
    """One transition kernel per step, and the matrix to use instead if there is one.

    A chain may carry a kernel that is a function of position --- a spacing
    between sites, a rate that varies along the sequence --- so the recursions
    here take either shape (issue #653):

    ``(K, K)``
        One matrix for the whole chain, what every caller passed before this
        was written, and the form to pass when the kernel does not vary.
    ``(T - 1, K, K)``
        One matrix per transition, indexed by the step it governs, so
        ``log_transition[t - 1]`` carries ``z_{t-1} -> z_t``.

    Returns both the ``(T - 1, K, K)`` view and, for the constant case, the
    ``(K, K)`` matrix itself, so a caller writes
    ``(kernels[t - 1] if constant is None else constant) if constant is None else constant`` and the constant path
    indexes nothing. That is not a micro-optimization looking for a home: at
    ``T = 100,000``, ``K = 8`` the index costs 390 ns a step, 2.6% of the
    forward pass, and 26 call sites pass a matrix. Hoisting the choice returns
    the constant path to the time it took before this shape was admitted
    (1.319 s against 1.332 s, inside the spread of three runs).

    The constant view is :func:`numpy.broadcast_to`, stride zero and not a
    copy, so the ``T``-fold memory the varying form costs --- 0.50 KiB against
    48.83 MiB at that size --- is paid only by a caller who asks for it.

    Which of the two shapes arrived is read by
    :func:`~sal.numerics.constant_chain_kernel`, so this and the
    torch recursion in :mod:`sal.opt.hmm` refuse the same shapes
    in the same words (issue #857).

    Raises
    ------
    ValueError
        If ``log_transition`` is neither shape.
    """
    if not constant_chain_kernel(log_transition.shape, length, n_states):
        return StepKernels(log_transition, None)
    steps = max(length - 1, 0)
    return StepKernels(
        np.broadcast_to(log_transition, (steps, n_states, n_states)), log_transition
    )


def _forward(
    log_density: np.ndarray, log_initial: np.ndarray, log_transition: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """The forward pass, checked, with the kernels the caller reads after it.

    One recursion for both the evaluator and the block sampler (issue #857).
    The two carried a copy each, and the copy checked no shape: a
    ``log_initial`` of the wrong length broadcast into the sampler and
    returned a path.

    Parameters
    ----------
    log_density, log_initial, log_transition : np.ndarray
        As :func:`forward_backward`, already float arrays.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray | None]
        ``alpha``, shape ``(T, K)``; the per-step kernels; and the one matrix
        to read instead where the chain carries one --- :func:`step_kernels`'
        pair, hoisted out of the recursion for its reason.

    Raises
    ------
    ValueError
        If the shapes disagree or the chain is empty.
    """
    if log_density.ndim != 2 or log_density.shape[0] < 1:
        msg = f"log_density must be (T, K) with T >= 1, got {log_density.shape}"
        raise ValueError(msg)
    length, n_states = log_density.shape
    if log_initial.shape != (n_states,):
        msg = f"log_initial {log_initial.shape} does not match {n_states} states"
        raise ValueError(msg)
    # Read by name and bound to locals before the loop: the hoist is what
    # `step_kernels` returns a `constant` for, and an attribute lookup per
    # step would spend what it saved.
    steps = step_kernels(log_transition, length, n_states)
    kernels, constant = steps.kernels, steps.constant
    alpha = np.empty((length, n_states))
    alpha[0] = log_initial + log_density[0]
    for t in range(1, length):
        alpha[t] = (
            logsumexp(
                alpha[t - 1][:, None]
                + (kernels[t - 1] if constant is None else constant),
                axis=0,
            )
            + log_density[t]
        )
    return alpha, kernels, constant


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
        Rows the source state. Either ``(K, K)``, one kernel for the whole
        chain, or ``(T - 1, K, K)``, one per step --- see :func:`step_kernels`.

    Raises
    ------
    ValueError
        If the shapes disagree or the chain is empty.
    """
    log_density = np.asarray(log_density, dtype=float)
    log_initial = np.asarray(log_initial, dtype=float)
    log_transition = np.asarray(log_transition, dtype=float)
    alpha, kernels, constant = _forward(log_density, log_initial, log_transition)
    length, n_states = alpha.shape

    beta = np.zeros((length, n_states))
    for t in range(length - 2, -1, -1):
        beta[t] = logsumexp(
            (kernels[t] if constant is None else constant)
            + (log_density[t + 1] + beta[t + 1])[None, :],
            axis=1,
        )
    log_evidence = float(logsumexp(alpha[-1], axis=0))
    posterior = np.exp(alpha + beta - log_evidence)
    pairwise = np.empty((max(length - 1, 0), n_states, n_states))
    for t in range(1, length):
        pairwise[t - 1] = np.exp(
            alpha[t - 1][:, None]
            + (kernels[t - 1] if constant is None else constant)
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
    ``log_transition`` takes either shape :func:`step_kernels` accepts. The
    filter is :func:`forward_backward`'s own (:func:`_forward`), so the two
    agree by construction and the draws are refused on the shapes the
    evaluator refuses.

    Raises
    ------
    ValueError
        If the shapes disagree or the chain is empty.
    """
    log_density = np.asarray(log_density, dtype=float)
    log_initial = np.asarray(log_initial, dtype=float)
    log_transition = np.asarray(log_transition, dtype=float)
    alpha, kernels, constant = _forward(log_density, log_initial, log_transition)
    length, n_states = alpha.shape
    path = np.empty(length, dtype=np.int64)
    weights = np.exp(alpha[-1] - logsumexp(alpha[-1], axis=0))
    path[-1] = rng.choice(n_states, p=weights / weights.sum())
    for t in range(length - 2, -1, -1):
        step = kernels[t] if constant is None else constant
        scores = alpha[t] + step[:, path[t + 1]]
        weights = np.exp(scores - logsumexp(scores, axis=0))
        path[t] = rng.choice(n_states, p=weights / weights.sum())
    return path
