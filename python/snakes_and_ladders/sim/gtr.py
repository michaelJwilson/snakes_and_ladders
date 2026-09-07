"""The general time-reversible substitution model, and its reduction to JC.

Jukes-Cantor has no free rate parameters: ``jc_rate_matrix(k)`` takes only
``k``, and its stationary distribution is uniform by construction. So there
is nothing in ``Q`` or ``pi`` to fit under it, and issue #104's "fit branch
lengths, rates etc." needs a model that has some. GTR is the general
reversible one (Felsenstein, *Inferring Phylogenies*, ch. 13):

    Q[i, j] = s[i, j] * pi[j]   for i != j,   Q[i, i] = -sum_{j != i} Q[i, j]

with ``s`` symmetric -- the *exchangeabilities* -- and ``pi`` the stationary
distribution. Reversibility is by construction: ``pi[i] Q[i, j] == pi[j]
Q[j, i]`` follows from ``s`` being symmetric.

Two normalizations are not optional, and both are gauges rather than
conventions -- without them the model has directions along which the
likelihood is exactly flat, and no parameter in it has a confidence
interval.

* **Rate.** ``Q`` and ``c * Q`` describe the same process at branch lengths
  ``t`` and ``t / c``, so ``Q`` is scaled to one expected substitution per
  unit time: ``-sum_i pi[i] Q[i, i] == 1``. This is what makes a branch
  length mean "expected substitutions per site".
* **Exchangeability scale.** The same freedom appears again in ``s``, since
  scaling every entry is undone by the rate normalization. The conventional
  fix is to hold one entry at 1; :func:`exchangeabilities_from_free` does
  that with the last.

Setting every exchangeability equal and ``pi`` uniform must reproduce
``jc_rate_matrix(k)`` exactly, which is the oracle this module is pinned
against rather than a restatement of its own formula.
"""

from __future__ import annotations

import numpy as np


def n_exchangeabilities(k: int) -> int:
    """Number of free entries in a ``k``-state symmetric exchangeability matrix.

    Parameters
    ----------
    k : int
        Number of states, >= 2.

    Returns
    -------
    int
        ``k * (k - 1) / 2`` -- the upper triangle.
    """
    if k < 2:
        msg = f"k must be >= 2, got {k}"
        raise ValueError(msg)
    return k * (k - 1) // 2


def exchangeability_matrix(values: np.ndarray, k: int) -> np.ndarray:
    """Fill a symmetric ``(k, k)`` matrix from its upper triangle.

    Parameters
    ----------
    values : np.ndarray
        The ``k * (k - 1) / 2`` upper-triangular entries, row-major.
    k : int
        Number of states.

    Returns
    -------
    np.ndarray
        Symmetric ``(k, k)`` array with a zero diagonal.

    Raises
    ------
    ValueError
        If ``values`` has the wrong length or any entry is not positive.
    """
    expected = n_exchangeabilities(k)
    if values.shape != (expected,):
        msg = f"expected {expected} exchangeabilities for k={k}, got {values.shape}"
        raise ValueError(msg)
    if not bool(np.all(values > 0.0)):
        msg = "exchangeabilities must be strictly positive"
        raise ValueError(msg)

    matrix = np.zeros((k, k), dtype=np.float64)
    rows, columns = np.triu_indices(k, k=1)
    matrix[rows, columns] = values
    matrix[columns, rows] = values
    return matrix


def gtr_rate_matrix(values: np.ndarray, pi: np.ndarray) -> np.ndarray:
    """The normalized GTR rate matrix for these exchangeabilities and ``pi``.

    Parameters
    ----------
    values : np.ndarray
        Upper-triangular exchangeabilities, ``k * (k - 1) / 2`` entries.
    pi : np.ndarray
        Stationary distribution, shape ``(k,)``, positive and summing to 1.

    Returns
    -------
    np.ndarray
        Rate matrix of shape ``(k, k)``: rows sum to zero, ``pi`` is
        stationary, detailed balance holds, and the rate is normalized to one
        expected substitution per unit time.

    Raises
    ------
    ValueError
        If ``pi`` is not a positive distribution, or ``values`` is malformed.
    """
    k = pi.shape[0]
    if not np.isclose(pi.sum(), 1.0):
        msg = f"pi sums to {pi.sum()}, expected 1.0"
        raise ValueError(msg)
    if not bool(np.all(pi > 0.0)):
        msg = "pi must be strictly positive for a reversible model"
        raise ValueError(msg)

    symmetric = exchangeability_matrix(values, k)
    rate = symmetric * pi[np.newaxis, :]
    np.fill_diagonal(rate, 0.0)
    np.fill_diagonal(rate, -rate.sum(axis=1))

    # The rate gauge: without it Q and c*Q are the same model at rescaled
    # branch lengths, and neither the rates nor the lengths are identifiable.
    scale = -float((pi * np.diag(rate)).sum())
    normalized: np.ndarray = rate / scale
    return normalized


def exchangeabilities_from_free(free: np.ndarray, k: int) -> np.ndarray:
    """Complete a free parameter vector to a full exchangeability vector.

    The last entry is held at 1. Scaling every exchangeability by a constant
    is undone by :func:`gtr_rate_matrix`'s rate normalization, so without
    pinning one the model has an exactly flat direction.

    Parameters
    ----------
    free : np.ndarray
        The first ``k * (k - 1) / 2 - 1`` exchangeabilities.
    k : int
        Number of states.

    Returns
    -------
    np.ndarray
        All ``k * (k - 1) / 2`` entries, the last equal to 1.
    """
    expected = n_exchangeabilities(k) - 1
    if free.shape != (expected,):
        msg = f"expected {expected} free exchangeabilities for k={k}, got {free.shape}"
        raise ValueError(msg)
    return np.concatenate([free, np.ones(1)])


def reversible_transition_probabilities(
    rate_matrix: np.ndarray, pi: np.ndarray, t: float
) -> np.ndarray:
    """``P(t) = exp(Q t)`` for a reversible ``Q``, by symmetric eigendecomposition.

    Reversibility is what makes this exact rather than merely convenient.
    With ``D = diag(sqrt(pi))``, detailed balance makes ``D Q D^-1``
    symmetric, so ``numpy.linalg.eigh`` applies -- real eigenvalues, an
    orthogonal basis, and no general-purpose matrix exponential. This is
    equation (eigen) of the technical document, and the reason a single
    decomposition serves every branch length.

    Parameters
    ----------
    rate_matrix : np.ndarray
        Reversible rate matrix, shape ``(k, k)``.
    pi : np.ndarray
        Its stationary distribution, shape ``(k,)``, strictly positive.
    t : float
        Branch length, non-negative.

    Returns
    -------
    np.ndarray
        Transition probabilities, shape ``(k, k)``, rows summing to 1.

    Raises
    ------
    ValueError
        If ``t`` is negative.
    """
    if t < 0:
        msg = f"t must be non-negative, got {t}"
        raise ValueError(msg)

    root = np.sqrt(pi)
    symmetric = rate_matrix * root[:, np.newaxis] / root[np.newaxis, :]
    # Force exact symmetry: detailed balance makes the two halves equal in
    # exact arithmetic, and eigh reads only one triangle anyway, but an
    # asymmetry of 1e-17 here is a silent claim that the model is reversible
    # when it is not.
    eigenvalues, vectors = np.linalg.eigh((symmetric + symmetric.T) / 2.0)
    scaled = (vectors * np.exp(eigenvalues * t)) @ vectors.T
    probabilities: np.ndarray = scaled / root[:, np.newaxis] * root[np.newaxis, :]
    return probabilities
