"""The ragged forward-backward: the gateway, and the oracle the Rust twin is pinned to.

Issue #666. The Python path pads every segment to the longest and masks, which
is what lets it take one batched step per position of the longest. The Rust
twin, :mod:`sal.likelihood.ragged.rust`, walks the segments in place and pads nothing, so it does the work the problem
has rather than the work the longest segment implies.

**Which is faster is a question about the lengths, not about the language**, and
the benchmark answers it rather than this docstring: even lengths waste no
padding and the batched path has nothing to beat, while one long segment among
short ones is where the padding is nearly all of the block.

Per root `CLAUDE.md`, the pure-Python route, :func:`posteriors_oracle`, stays
as the oracle and the regression test pins the twin against it.

**The selector is here rather than on `forward_backward` (issue #860).** The
ragged kernel returns *log* gamma, the log transition counts **summed over
the segments** and one log evidence per segment; `forward_backward` returns
probabilities on one dense chain, including the per-step pairwise posterior
`(T - 1, K, K)`. The sum is not the steps, so no conversion recovers a
`ForwardBackward` from what the kernel returns, and a `backend` on the
evaluator would have to be a second implementation rather than a door. The
choice the two do share is this function's --- kernel or oracle, over the
same inputs and the same return --- and that is where `Backend` names it.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum
from typing import cast

import numpy as np

from sal.backend import Backend, twin
from sal.likelihood.forward_backward import forward_backward
from sal.ragged import Ragged


class SwitchKind(StrEnum):
    """How a per-position switch probability ``s_t`` enters the step into ``t``.

    ``STAY_OR_MOVE`` is ``(1 - s_t) I + s_t A`` (issue #1082). The other two
    are a slow chain ``A`` over ``K`` states coupled to a fast binary layer,
    ``2 K`` states with ``(i, a)`` at ``2 i + a`` as ``np.kron`` lays them
    out (issue #1133):

    - ``KRONECKER``: ``A ⊗ S_t``, ``S_t = [[1 - s_t, s_t], [s_t, 1 - s_t]]``.
      No ``A'`` makes it ``(1 - s) I + s A'`` unless ``A = I``, since both of
      ``(1 - s)(A ⊗ I) + s (A ⊗ J)`` carry ``A``.
    - ``KRONECKER_DIAGONAL``: ``S_t`` on ``A``'s diagonal blocks alone, the
      layer switching only where the slow chain stays; an off-diagonal move
      lands on either layer with probability one half.
    """

    STAY_OR_MOVE = "stay_or_move"
    KRONECKER = "kronecker"
    KRONECKER_DIAGONAL = "kronecker_diagonal"


@dataclass(frozen=True)
class Posteriors:
    """What one ragged forward--backward pass returns, in the log domain.

    Parameters
    ----------
    log_posterior : np.ndarray
        ``(total, n_states)``, the log marginal at every position.
    log_counts : np.ndarray
        ``(n_states, n_states)``, the log transition counts summed over
        segments. The pair spanning a boundary is in none of them.
    log_evidence : np.ndarray
        One log evidence per segment.
    """

    log_posterior: np.ndarray
    log_counts: np.ndarray
    log_evidence: np.ndarray

    def __iter__(self) -> Iterator[np.ndarray]:
        """``(log_posterior, log_counts, log_evidence)``: the order callers unpack."""
        yield from (self.log_posterior, self.log_counts, self.log_evidence)


def posteriors(
    log_density: Ragged,
    log_initial: np.ndarray,
    log_transition: np.ndarray,
    *,
    switch: np.ndarray | None = None,
    switch_kind: SwitchKind = SwitchKind.STAY_OR_MOVE,
    backend: Backend = Backend.RUST,
) -> Posteriors:
    """Marginals, transition counts and per-segment evidence, by default in Rust.

    Parameters
    ----------
    log_density : Ragged
        Per-position scores, ``(total, n_states)`` with the segment lengths.
    log_initial : np.ndarray
        ``(n_states,)``, the distribution each segment restarts at.
    log_transition : np.ndarray
        ``(n_states, n_states)`` in log space.
    switch : np.ndarray | None
        One probability ``s`` in ``[0, 1]`` per position, or ``None`` for
        ``log_transition`` at every step (issue #1082). The step into
        position ``t`` then takes ``(1 - s[t]) I + s[t] A``, ``A`` the
        transition: the chain stays with probability ``1 - s[t]`` and
        otherwise moves by ``A``. The kernel builds each step's transition
        from the one ``A`` rather than a ``(T, K, K)`` stack, and a
        segment's first entry is never read.
    switch_kind : SwitchKind
        How ``switch`` enters. Under either Kronecker kind ``log_density``
        and ``log_initial`` are over the ``2 K`` states, ``log_transition``
        is the slow chain's ``(K, K)``, and ``switch`` is required; the
        kernel takes each step in its factors, ``2 K^2 + 4 K`` terms against
        the ``4 K^2`` of the explicit matrix, and the counts are over the
        ``2 K`` states.
    backend : Backend
        Which implementation runs it. ``RUST`` is the compiled kernel,
        :func:`sal.likelihood.ragged.rust.posteriors`, and is the default;
        ``PYTHON`` is :func:`posteriors_oracle`, the sibling it is pinned
        against, reached through the enum rather than by naming the function
        (issue #860). Nothing else.

    Returns
    -------
    Posteriors
        The three arrays the kernel writes: ``gamma`` as ``(total,
        n_states)``, the log transition counts summed over segments as
        ``(n_states, n_states)``, and one log evidence per segment; the pair
        spanning a boundary is in none of the counts. On ``RUST`` they are the
        extension's own buffers, wrapped by the twin and not copied, so a pin reads
        what the kernel wrote.

    Raises
    ------
    ValueError
        If ``backend`` is neither ``RUST`` nor ``PYTHON``.
    """
    if (rust := twin("ragged posteriors", backend, __name__)) is not None:
        return cast(
            "Posteriors",
            rust.posteriors(
                log_density, log_initial, log_transition, switch, switch_kind
            ),
        )
    return posteriors_oracle(
        log_density, log_initial, log_transition, switch, switch_kind
    )


def step_transitions(
    log_transition: np.ndarray, steps: np.ndarray, switch_kind: SwitchKind
) -> np.ndarray:
    """Each step's transition, materialized: ``(len(steps), n, n)`` in log space.

    The storage the kernel avoids, built here for the oracle. ``steps`` are
    the switch probabilities of the steps taken.
    """
    moved = np.exp(np.asarray(log_transition, dtype=float))
    s = np.asarray(steps, dtype=float)[:, None, None]
    if switch_kind is SwitchKind.STAY_OR_MOVE:
        matrices = (1.0 - s) * np.eye(moved.shape[0]) + s * moved
    else:
        layer = (1.0 - s) * np.eye(2) + s * (1.0 - np.eye(2))
        if switch_kind is SwitchKind.KRONECKER:
            matrices = np.stack([np.kron(moved, one) for one in layer])
        else:
            kept = np.diag(np.diag(moved))
            spread = np.kron(moved - kept, np.full((2, 2), 0.5))
            matrices = np.stack([spread + np.kron(kept, one) for one in layer])
    with np.errstate(divide="ignore"):
        return np.asarray(np.log(matrices))


def posteriors_oracle(
    log_density: Ragged,
    log_initial: np.ndarray,
    log_transition: np.ndarray,
    switch: np.ndarray | None = None,
    switch_kind: SwitchKind = SwitchKind.STAY_OR_MOVE,
) -> Posteriors:
    """The same, one segment at a time through `forward_backward`.

    The oracle the compiled path is pinned against: it reuses the per-chain
    recursion this repository already refereed rather than writing a second
    batched one, so what it adds is only the segmentation. With ``switch``
    it materializes each segment's ``(T - 1, n, n)`` stack of the step's
    transition (:func:`step_transitions`), the storage the kernel avoids, and
    hands it to `forward_backward`'s per-step form.
    """
    switch_kind = SwitchKind(switch_kind)
    if switch_kind is not SwitchKind.STAY_OR_MOVE and switch is None:
        msg = f"a {switch_kind} switch needs a switch probability per position"
        raise ValueError(msg)
    gamma = np.empty_like(log_density.values)
    n_states = log_density.values.shape[1]
    counts = np.full((n_states, n_states), -np.inf)
    evidence = np.empty(log_density.n_segments)
    at = 0
    if switch is not None:
        switch = np.asarray(switch, dtype=float).reshape(-1)
    for index, segment in enumerate(log_density.segments()):
        kernel = np.asarray(log_transition, dtype=float)
        if switch is not None:
            kernel = step_transitions(
                log_transition, switch[at + 1 : at + len(segment)], switch_kind
            )
        run = forward_backward(segment, log_initial, kernel)
        # `forward_backward` returns probabilities; this returns logs, which
        # is what the accumulator below needs and what the compiled kernel
        # carries. Converting here keeps the comparison in one space.
        with np.errstate(divide="ignore"):
            gamma[at : at + len(segment)] = np.log(run.posterior)
            counts = np.logaddexp(counts, np.log(run.pairwise.sum(axis=0)))
        evidence[index] = run.log_evidence
        at += len(segment)
    return Posteriors(gamma, counts, evidence)
