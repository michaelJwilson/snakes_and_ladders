"""The ragged forward-backward and Viterbi in Rust, pinned to the oracles in :mod:`sal.likelihood.ragged`.

``sal.oxisal.ragged_posteriors`` is pinned to
:func:`~sal.likelihood.ragged.posteriors_oracle` and ``sal.oxisal.ragged_viterbi``
(issue #1138) to :func:`~sal.likelihood.ragged.viterbi_oracle`.

The Rust twin of :mod:`sal.likelihood.ragged` (issues #666, #1059): it walks
the segments in place and pads nothing. The gateway,
:func:`sal.likelihood.ragged.posteriors`, reaches it with ``backend=RUST``,
its default; why the selector sits there rather than on
``forward_backward`` is stated in that module.
"""

from __future__ import annotations

import numpy as np

from sal.likelihood.ragged import Paths, Posteriors, SwitchKind
from sal.oxisal import ragged_posteriors, ragged_viterbi
from sal.ragged import Ragged


def posteriors(
    log_density: Ragged,
    log_initial: np.ndarray,
    log_transition: np.ndarray,
    switch: np.ndarray | None = None,
    switch_kind: SwitchKind = SwitchKind.STAY_OR_MOVE,
) -> Posteriors:
    """Marginals, transition counts and per-segment evidence, in Rust.

    Parameters
    ----------
    log_density : Ragged
        Per-position scores, ``(total, n_states)`` with the segment lengths.
    log_initial : np.ndarray
        ``(n_states,)``, the distribution each segment restarts at.
    log_transition : np.ndarray
        ``(n_states, n_states)`` in log space.
    switch : np.ndarray | None
        One switch probability per position, or ``None``; see
        :func:`sal.likelihood.ragged.posteriors`.
    switch_kind : SwitchKind
        How ``switch`` enters; see :class:`sal.likelihood.ragged.SwitchKind`.

    Returns
    -------
    Posteriors
        The extension's own buffers, wrapped and not copied, so a pin reads
        what the kernel wrote.
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
        None
        if switch is None
        else np.ascontiguousarray(switch, dtype=np.float64).reshape(-1),
        str(SwitchKind(switch_kind)),
    )
    return Posteriors(gamma, counts, evidence)


def viterbi(
    log_density: Ragged,
    log_initial: np.ndarray,
    log_transition: np.ndarray,
    switch: np.ndarray | None = None,
    switch_kind: SwitchKind = SwitchKind.STAY_OR_MOVE,
) -> Paths:
    """The most probable path of every segment, in Rust; segments in parallel.

    Parameters
    ----------
    log_density, log_initial, log_transition, switch, switch_kind
        As :func:`posteriors`; see :func:`sal.likelihood.ragged.viterbi`.

    Returns
    -------
    Paths
        The extension's own buffers, wrapped and not copied.
    """
    values = np.ascontiguousarray(log_density.values, dtype=np.float64)
    path = np.empty(values.shape[0], dtype=np.int64)
    log_joint = np.empty(log_density.n_segments, dtype=np.float64)
    ragged_viterbi(
        values,
        np.asarray(log_density.lengths, dtype=np.int64),
        np.ascontiguousarray(log_initial, dtype=np.float64),
        np.ascontiguousarray(log_transition, dtype=np.float64),
        path,
        log_joint,
        None
        if switch is None
        else np.ascontiguousarray(switch, dtype=np.float64).reshape(-1),
        str(SwitchKind(switch_kind)),
    )
    return Paths(path, log_joint)
