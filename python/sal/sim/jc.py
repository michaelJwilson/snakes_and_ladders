"""Closed-form k-state Jukes-Cantor model.

Implements ``eq:jc`` of ``docs/tex/textbook.tex``: the transition-probability
matrix P(t) and its
generating rate matrix Q for k equally exchangeable states with a uniform
stationary distribution, under the branch-length normalization
``-sum_i pi_i q_ii = 1`` (``eq:normalization``). This is the oracle
``simulate.py`` is validated against, per ``sim/CLAUDE.md`` -- it is derived
independently of any pruning/likelihood code and must stay that way.
"""

from __future__ import annotations

import numpy as np


def jc_transition_probabilities(t: float, n_states: int = 4) -> np.ndarray:
    """Closed-form k-state Jukes-Cantor transition probabilities P(t).

    Parameters
    ----------
    t : float
        Branch length, in expected substitutions per site.
    n_states : int
        Number of states (4 for JC69 nucleotides).

    Returns
    -------
    np.ndarray
        Array of shape (k, k); entry [i, j] is Pr(state j at the branch's
        end | state i at its start). Rows sum to 1.

    Notes
    -----
    ``P_ii(t) = 1/k + (k-1)/k * exp(-k*t/(k-1))``,
    ``P_ij(t) = 1/k - 1/k * exp(-k*t/(k-1))`` for ``i != j``.
    """
    if n_states < 2:
        msg = f"k must be >= 2, got {n_states}"
        raise ValueError(msg)
    if t < 0:
        msg = f"t must be non-negative, got {t}"
        raise ValueError(msg)

    decay = np.exp(-n_states * t / (n_states - 1))
    off_diagonal = (1.0 - decay) / n_states
    diagonal = 1.0 / n_states + (n_states - 1) / n_states * decay

    p = np.full((n_states, n_states), off_diagonal, dtype=np.float64)
    np.fill_diagonal(p, diagonal)
    return p


def jc_rate_matrix(n_states: int = 4) -> np.ndarray:
    """The k-state Jukes-Cantor rate matrix Q underlying ``jc_transition_probabilities``.

    Parameters
    ----------
    n_states : int
        Number of states.

    Returns
    -------
    np.ndarray
        Array of shape (k, k) with off-diagonal entries ``1/(k-1)`` and
        diagonal entries ``-1``, satisfying the branch-length normalization
        ``-sum_i pi_i q_ii = 1`` under the uniform stationary distribution
        ``pi = 1/k``.
    """
    if n_states < 2:
        msg = f"k must be >= 2, got {n_states}"
        raise ValueError(msg)

    q = np.full((n_states, n_states), 1.0 / (n_states - 1), dtype=np.float64)
    np.fill_diagonal(q, -1.0)
    return q
