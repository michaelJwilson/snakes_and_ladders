"""The ragged forward-backward in Rust, pinned to the NumPy oracle.

Issue #666. The Python path pads every segment to the longest and masks, which
is what lets it take one batched step per position of the longest. This one
walks the segments in place and pads nothing, so it does the work the problem
has rather than the work the longest segment implies.

**Which is faster is a question about the lengths, not about the language**, and
the benchmark answers it rather than this docstring: even lengths waste no
padding and the batched path has nothing to beat, while one long segment among
short ones is where the padding is nearly all of the block.

Per root `CLAUDE.md`, the pure-Python route stays as the oracle and the
regression test pins this against it.
"""

from __future__ import annotations

import numpy as np

from snakes_and_ladders.likelihood.forward_backward import forward_backward
from snakes_and_ladders.oxi_snakes_and_ladders import ragged_posteriors
from snakes_and_ladders.ragged import Ragged


def posteriors(
    log_density: Ragged,
    log_initial: np.ndarray,
    log_transition: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Marginals, transition counts and per-segment evidence, in Rust.

    Parameters
    ----------
    log_density : Ragged
        Per-position scores, ``(total, n_states)`` with the segment lengths.
    log_initial : np.ndarray
        ``(n_states,)``, the distribution each segment restarts at.
    log_transition : np.ndarray
        ``(n_states, n_states)`` in log space.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        ``gamma`` as ``(total, n_states)``, the log transition counts summed
        over segments as ``(n_states, n_states)``, and one log evidence per
        segment. The pair spanning a boundary is in none of the counts.
    """
    values = np.ascontiguousarray(log_density.values, dtype=np.float64)
    n_states = values.shape[1]
    gamma = np.empty_like(values)
    counts = np.empty((n_states, n_states), dtype=np.float64)
    evidence = np.empty(log_density.n_segments, dtype=np.float64)
    ragged_posteriors(
        values,
        np.asarray(log_density.lengths, dtype=np.int64),
        np.ascontiguousarray(log_initial, dtype=np.float64),
        np.ascontiguousarray(log_transition, dtype=np.float64),
        gamma,
        counts,
        evidence,
    )
    return gamma, counts, evidence


def posteriors_oracle(
    log_density: Ragged,
    log_initial: np.ndarray,
    log_transition: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        gamma[at : at + len(segment)] = run.posterior
        evidence[index] = run.log_evidence
        counts = np.logaddexp(counts, run.pairwise.sum(axis=0))
        at += len(segment)
    return gamma, counts, evidence
