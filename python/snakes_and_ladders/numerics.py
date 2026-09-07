"""Numerical helpers with no model knowledge, importable from anywhere.

Everything here is arithmetic. It names no model, no alignment and no tree, so
`snakes_and_ladders.opt` and `snakes_and_ladders.learn` may import it without acquiring an application
reference, and `snakes_and_ladders.sim` may import it without inverting the layering.

The module exists because the alternative failed. Vectorized categorical
sampling was written three times -- in `snakes_and_ladders.sim.simulate`, `snakes_and_ladders.opt.potts`
and `snakes_and_ladders.opt.hmm` -- and the three copies drifted: two of them omitted the
guard the third had, which is the difference between a valid state index and
one past the end of the alphabet. A helper with one home cannot drift from
itself.

`logsumexp` arrived here for the same reason and by the same route (issue
#230): four private copies in two spellings, across `snakes_and_ladders.sim.potts`,
`snakes_and_ladders.opt.potts`, `snakes_and_ladders.likelihood.potts` and
`snakes_and_ladders.likelihood.belief_propagation`. They had not drifted in *behaviour* --
the two spellings compute the same thing -- which is what makes consolidating
them safe, and what would have made a later divergence hard to notice.
"""

from __future__ import annotations

import numpy as np


def sample_rows(
    rng: np.random.Generator, distributions: np.ndarray, rows: np.ndarray
) -> np.ndarray:
    """Draw one categorical index per entry of ``rows``, from the row it selects.

    Inverse-CDF sampling, vectorized over the whole batch: one uniform draw
    per entry, placed against the cumulative probabilities of the row that
    entry names.

    **The last cumulative column is clamped to 1.** A probability row that
    sums to ``1 - 4e-16`` after rounding leaves a sliver of the unit interval
    above its own total, and a draw landing there is past every column --
    which the obvious formulations report as an index one past the end of the
    alphabet. The clamp sends that draw to the last category, which is where
    it belongs. This is not hypothetical arithmetic: the drift is ordinary
    ``float64`` behaviour for a normalized row, and the draw is reachable
    because ``rng.random`` returns values in ``[0, 1)``.

    Parameters
    ----------
    rng : np.random.Generator
        Seeded generator. One value is drawn per entry of ``rows``, in order,
        so a caller's stream consumption does not depend on the outcome.
    distributions : np.ndarray
        Row-stochastic, shape ``(n_rows, n_categories)``. Rows need only sum
        to 1 within rounding.
    rows : np.ndarray
        Which row each draw comes from, shape ``(n_draws,)``, entries in
        ``[0, n_rows)``.

    Returns
    -------
    np.ndarray
        Sampled category per entry, shape ``(n_draws,)``, entries in
        ``[0, n_categories)``.

    Raises
    ------
    ValueError
        If ``distributions`` is not 2-D. A 1-D distribution has no row to
        select, and passing one is a mistake worth naming rather than
        broadcasting past.
    """
    if distributions.ndim != 2:
        msg = (
            f"expected distributions of shape (n_rows, n_categories), got "
            f"{distributions.shape}"
        )
        raise ValueError(msg)

    cumulative = np.cumsum(distributions, axis=1)
    cumulative[:, -1] = 1.0
    draws = rng.random(size=(int(rows.shape[0]),))
    selected: np.ndarray = np.argmax(draws[:, np.newaxis] < cumulative[rows], axis=1)
    return selected


def logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
    """``log(sum(exp(values)))`` along ``axis``, shifted by the maximum.

    The shift is the whole point: a Potts coupling of ``J = 2`` on a 4x4
    lattice puts ``exp(32)`` inside a product of sixteen messages, and the
    linear-domain recursion loses it. Subtracting the row maximum before
    exponentiating bounds every term at 1, and adding it back afterwards is
    exact.

    Parameters
    ----------
    values : np.ndarray
        Log-domain values.
    axis : int
        Axis to reduce. It is removed from the result, as ``np.max`` without
        ``keepdims`` would remove it.

    Returns
    -------
    np.ndarray
        The reduction, with ``axis`` removed.

    Examples
    --------
    >>> import numpy as np
    >>> float(logsumexp(np.array([[0.0, 0.0]]), axis=1)[0])
    0.6931471805599453
    """
    peak = values.max(axis=axis, keepdims=True)
    shifted = np.log(np.exp(values - peak).sum(axis=axis, keepdims=True))
    result: np.ndarray = (peak + shifted).squeeze(axis)
    return result
