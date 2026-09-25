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
from typing import cast

import numpy as np

from sal.backend import Backend, twin
from sal.likelihood.forward_backward import forward_backward
from sal.ragged import Ragged


@dataclass(frozen=True)
class Posteriors:
    """What one ragged forward--backward pass returns, in the log domain.

    Parameters
    ----------
    gamma : np.ndarray
        ``(total, n_states)``, the log marginal at every position.
    counts : np.ndarray
        ``(n_states, n_states)``, the log transition counts summed over
        segments. The pair spanning a boundary is in none of them.
    evidence : np.ndarray
        One log evidence per segment.
    """

    gamma: np.ndarray
    counts: np.ndarray
    evidence: np.ndarray

    def __iter__(self) -> Iterator[np.ndarray]:
        """``(gamma, counts, evidence)``: the order callers unpack."""
        yield from (self.gamma, self.counts, self.evidence)


def posteriors(
    log_density: Ragged,
    log_initial: np.ndarray,
    log_transition: np.ndarray,
    *,
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
            "Posteriors", rust.posteriors(log_density, log_initial, log_transition)
        )
    return posteriors_oracle(log_density, log_initial, log_transition)


def posteriors_oracle(
    log_density: Ragged,
    log_initial: np.ndarray,
    log_transition: np.ndarray,
) -> Posteriors:
    """The same, one segment at a time through `forward_backward`.

    The oracle the compiled path is pinned against: it reuses the per-chain
    recursion this repository already refereed rather than writing a second
    batched one, so what it adds is only the segmentation.
    """
    gamma = np.empty_like(log_density.values)
    n_states = log_density.values.shape[1]
    counts = np.full((n_states, n_states), -np.inf)
    evidence = np.empty(log_density.n_segments)
    at = 0
    for index, segment in enumerate(log_density.segments()):
        run = forward_backward(segment, log_initial, log_transition)
        # `forward_backward` returns probabilities; this returns logs, which
        # is what the accumulator below needs and what the compiled kernel
        # carries. Converting here keeps the comparison in one space.
        with np.errstate(divide="ignore"):
            gamma[at : at + len(segment)] = np.log(run.posterior)
            counts = np.logaddexp(counts, np.log(run.pairwise.sum(axis=0)))
        evidence[index] = run.log_evidence
        at += len(segment)
    return Posteriors(gamma, counts, evidence)
