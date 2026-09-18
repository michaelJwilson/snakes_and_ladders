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

from snakes_and_ladders import oxi_snakes_and_ladders
from snakes_and_ladders.backend import Backend


def sample_rows(
    rng: np.random.Generator,
    distributions: np.ndarray,
    rows: np.ndarray,
    *,
    backend: Backend = Backend.RUST,
) -> np.ndarray:
    """Draw one categorical index per entry of ``rows``, from the row it selects.

    Inverse-CDF sampling, vectorized over the whole batch: one uniform draw
    per entry, placed against the cumulative probabilities of the row that
    entry names. The uniforms are drawn here whichever backend runs the
    lookup, in the same order and the same count, so the two differ in
    arithmetic alone and agree **bitwise** for the same generator state ---
    which is why the compiled path is the default (issue #187) and why
    `sim`'s reproducibility contract, that a seeded generator determines the
    sample, holds on both. Until issue #717 the compiled path was a twin
    module, `numerics_rust`, and a caller chose by import.

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
    backend : Backend
        :data:`~snakes_and_ladders.backend.Backend.RUST` performs the lookup
        in the extension over borrowed arrays (issue #202: passing lists
        cost more marshalling than the NumPy oracle's whole run);
        :data:`~snakes_and_ladders.backend.Backend.PYTHON` is the NumPy
        oracle. :data:`~snakes_and_ladders.backend.Backend.NUMBA` is
        refused: no such kernel exists here.

    Returns
    -------
    np.ndarray
        Sampled category per entry, shape ``(n_draws,)``, entries in
        ``[0, n_categories)``, ``int64``.

    Raises
    ------
    ValueError
        If ``distributions`` is not 2-D. A 1-D distribution has no row to
        select, and passing one is a mistake worth naming rather than
        broadcasting past. The extension refuses a row index outside the
        distribution count.
    """
    if distributions.ndim != 2:
        msg = (
            f"expected distributions of shape (n_rows, n_categories), got "
            f"{distributions.shape}"
        )
        raise ValueError(msg)

    draws = rng.random(size=(int(rows.shape[0]),))
    if backend is Backend.RUST:
        # `ascontiguousarray` is what makes the borrow safe: Rust's
        # `as_slice` accepts only a C-contiguous array, and this is free when
        # the input already is one.
        sampled = np.empty(int(rows.shape[0]), dtype=np.int64)
        oxi_snakes_and_ladders.sample_rows(
            np.ascontiguousarray(distributions, dtype=np.float64).reshape(-1),
            int(distributions.shape[1]),
            np.ascontiguousarray(rows, dtype=np.int64),
            draws,
            sampled,
        )
        return sampled
    if backend is not Backend.PYTHON:
        msg = f"sample_rows runs on {Backend.PYTHON} or {Backend.RUST}, not {backend}"
        raise ValueError(msg)
    cumulative = np.cumsum(distributions, axis=1)
    cumulative[:, -1] = 1.0
    selected: np.ndarray = np.argmax(draws[:, np.newaxis] < cumulative[rows], axis=1)
    return selected


def logsumexp(values: np.ndarray, axis: int | tuple[int, ...]) -> np.ndarray:
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
    axis : int | tuple[int, ...]
        Axis or axes to reduce. Several at once is one reduction and not a
        loop of them: marginalizing a region belief onto a child region sums
        out every variable the child does not carry, and doing that one axis
        at a time re-shifts by a new maximum each pass for the same answer at
        more cost (issue #689). Removed from the result, as ``np.max`` without
        ``keepdims`` would remove it. A one-dimensional vector of scores ---
        a log normalizer over an enumeration, which is most of the callers
        --- reduces with ``axis=0`` and returns a zero-dimensional array that
        ``float`` takes. Fourteen sites across seven modules wrote
        ``values[None, :]`` and indexed ``[0]`` back off instead, which is
        the same arithmetic on the same values and was removed by issue #586.

    Returns
    -------
    np.ndarray
        The reduction, with ``axis`` removed.

    Examples
    --------
    >>> import numpy as np
    >>> float(logsumexp(np.array([0.0, 0.0]), axis=0))
    0.6931471805599453
    >>> float(logsumexp(np.zeros((2, 3)), axis=(0, 1)))
    1.791759469228055
    """
    peak = values.max(axis=axis, keepdims=True)
    shifted = np.log(np.exp(values - peak).sum(axis=axis, keepdims=True))
    result: np.ndarray = (peak + shifted).squeeze(axis)
    return result
