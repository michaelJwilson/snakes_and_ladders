"""A per-step circulant rate, superseded by the matrix forms (issue #658).

`SpatioSequentialParams.self_transition` briefly took a `(S - 1,)` vector of
self-transition rates and built one circulant per step from it. The matrix
forms that replaced it --- `(K, K)` for a chain and `(S - 1, K, K)` per step
--- express a kernel this cannot: a kernel assembled from parts, a base over
one latent times a kernel over another, is not a circulant at any rate, and an
application that reassembles its kernel per position has exactly that shape.

Conserved under `sandbox/CLAUDE.md`'s **superseded in capability** clause. It
is not faster and there is nothing to measure; what it referees is the
circulant case, where both routes describe the same chain and must agree bit
for bit. `tests/regression/sandbox/test_circulant_schedule.py` pins that.

The rate form remains the shorter statement where the kernel *is* circulant:
one number per step against `K * K`, and the row-stochastic property holds by
construction rather than by validation.
"""

from __future__ import annotations

import numpy as np

from snakes_and_ladders.sim.spatio_sequential import circulant_transition


def circulant_schedule(n_states: int, rates: np.ndarray) -> np.ndarray:
    """One circulant per step, from one self-transition rate per step.

    Parameters
    ----------
    n_states : int
        ``K``, the states each circulant is over.
    rates : np.ndarray
        Shape ``(S - 1,)``, the self-transition rate at each transition. Each
        is validated by :func:`circulant_transition`, which is what makes this
        a conserved route rather than a second construction: the arithmetic is
        the live one's, called once per step.

    Returns
    -------
    np.ndarray
        Shape ``(S - 1, K, K)``, the form the chain recursions take since #656
        and `SpatioSequentialParams` takes since #658.

    Raises
    ------
    ValueError
        If ``rates`` is not one-dimensional, or any rate is one
        :func:`circulant_transition` refuses.
    """
    rates = np.asarray(rates, dtype=float)
    if rates.ndim != 1:
        msg = f"rates must be one rate per transition, got shape {rates.shape}"
        raise ValueError(msg)
    return np.stack([circulant_transition(n_states, float(rate)) for rate in rates])
