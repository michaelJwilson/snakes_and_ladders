"""Two estimators of a labelling, and which loss each one minimizes.

Issue #696. A posterior over labellings supports more than one point estimate,
and they are not interchangeable:

* the **maximum a posteriori** labelling maximizes ``p(x)`` over whole
  configurations, so it minimizes the probability of getting the *entire* field
  wrong;
* the **maximum posterior marginal** labelling takes each site's own argmax, so
  it minimizes the *expected number of sites* that are wrong.

**Which one is wanted is decided by the reported metric, not by taste.**
:func:`sal.search.spatio_sequential.label_accuracy` scores
per-site agreement with a planted labelling, and the estimator that minimizes
per-site error is the marginal one. Reporting a MAP labelling against a
per-site score is answering a question nobody asked, and on a frustrated or
near-critical field the two labellings differ.

The identity that makes this a theorem rather than a preference: for any
labelling ``y``, the expected number of wrong sites is
``sum_i (1 - p_i(y_i))``, which is minimized term by term by taking each
``y_i`` at its own marginal's argmax. So the marginal decoder is optimal for
that loss by construction, and :mod:`tests.regression.search.test_decoding`
checks it against exhaustive enumeration rather than against this argument.

**A marginal decoder can return a labelling of probability zero**, and that is
not a defect but the loss speaking: minimizing per-site error does not require
the answer to be jointly consistent. Where joint consistency is the
requirement --- a decoded codeword, a ground state --- the MAP is the estimator
and :mod:`sal.search.tightening` is what bounds it.
"""

from __future__ import annotations

import numpy as np


def marginal_decode(marginals: np.ndarray) -> np.ndarray:
    """The maximum-posterior-marginal labelling: each site at its own argmax.

    Parameters
    ----------
    marginals : np.ndarray
        Per-site marginals, shape ``(n_nodes, n_states)``, each row a
        distribution. Rows are **not** renormalized here: a caller whose rows
        do not sum to one has not converged, and normalizing would hide it.

    Returns
    -------
    np.ndarray
        One state per site.

    Raises
    ------
    ValueError
        If ``marginals`` is not two-dimensional, carries a negative entry, or
        has a row that does not sum to one within ``1e-6``.
    """
    array = np.asarray(marginals, dtype=np.float64)
    if array.ndim != 2:
        msg = f"marginals must be (n_nodes, n_states), got {array.shape}"
        raise ValueError(msg)
    if array.min() < 0.0:
        msg = "marginals carry a negative entry, so they are not a distribution"
        raise ValueError(msg)
    mass = array.sum(axis=1)
    if not np.allclose(mass, 1.0, atol=1e-6):
        worst = int(np.argmax(np.abs(mass - 1.0)))
        msg = (
            f"row {worst} of the marginals sums to {mass[worst]}, not one; a "
            "caller that has not normalized has not converged"
        )
        raise ValueError(msg)
    return np.asarray(array.argmax(axis=1), dtype=np.int64)


def expected_site_errors(marginals: np.ndarray, labelling: np.ndarray) -> float:
    """``sum_i (1 - p_i(y_i))``: how many sites a labelling gets wrong, in mean.

    The loss the marginal decoder minimizes, written out so a comparison
    between estimators is a number rather than an argument. It reads the
    single-site marginals only, which is exactly why it is minimized site by
    site.

    Parameters
    ----------
    marginals : np.ndarray
        Per-site marginals, ``(n_nodes, n_states)``.
    labelling : np.ndarray
        One state per site.

    Returns
    -------
    float
        The expected number of wrong sites, between ``0`` and ``n_nodes``.
    """
    array = np.asarray(marginals, dtype=np.float64)
    states = np.asarray(labelling, dtype=np.int64)
    if states.shape != (array.shape[0],):
        msg = (
            f"labelling is {states.shape}, expected {(array.shape[0],)} to "
            "match the marginals"
        )
        raise ValueError(msg)
    chosen = array[np.arange(array.shape[0]), states]
    return float((1.0 - chosen).sum())
