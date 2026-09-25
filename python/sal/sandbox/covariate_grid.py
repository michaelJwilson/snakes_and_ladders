"""A continuous covariate coded on a power-of-two grid: declined, conserved (issue #1064).

The coupled Rust E step (:mod:`sal.likelihood.spatio_sequential_rust`)
indexes a table by row. A continuous covariate gives a table by count and
distinct covariate a row per observation, so this rounds the covariate to the
nearest multiple of ``1 / scale``, tabulates every ``(count, grid value)``
pair through the families' own ``log_density``, and bounds the change per
observation by ``G / (2 scale)``, ``G`` the largest ``|d log p / dc|`` taken
through each family's autograd. :func:`grid_scale` picks the coarsest power of
two within a stated tolerance; a power of two makes ``code / scale`` exact in
``float64``, so a covariate already on the grid is reproduced bit for bit.

:func:`log_grid_scale` codes ``ln c`` instead, ``code = round(ln c * scale)``
tabulated at ``exp(code / scale)``: a level per fixed ratio, which is relative
precision, with ``d log p / d ln c = c d log p / dc`` in the bound. A zero
covariate, the channel unobserved, has no logarithm and takes a level of its
own. On the ci exposure, which spans a ratio of 72, it codes 4.5x fewer
levels than the linear grid at tolerances 1.0, 0.25 and 0.05 (129, 516 and
2,059 against 581, 2,324 and 9,289), each within its tolerance.

**Why it is not on the live path.** No family reaches it. The negative
binomial's exposure factors exactly (:mod:`sal.emissions.nb`): the table is by
count and the kernel adds the exposure's two terms per observation. The
beta-binomial's trial count is an integer with three exact layouts,
``CovariateRows``, and a grid over an integer is refused: a trial count
rounded below its own successes leaves the support, where the density is
``-inf`` and no slope bounds the change. It lives here on
``sandbox/CLAUDE.md``'s rule for a declined route that was finished.

**What it referees.** The continuous exposure, the one continuous covariate
the model carries: on the grid each score is within its stated bound of the
family's ``log_density``, and each class's log evidence within the sum of its
members' bounds of the NumPy oracle, on either grid; on a power-of-two grid,
and on the log grid's own levels, the table is the family's ``log_density``
bitwise.
``tests/regression/sandbox/test_covariate_grid.py`` pins both.

**What would bring it back.** A continuous covariate on a family whose density
does not factor into a table by count and a per-observation term.

The byte ceiling is the live module's
:data:`~sal.likelihood.spatio_sequential_rust.COVARIATE_TABLE_CEILING`, one
ceiling for every covariate table. The tabulation and the ``uint32`` refusal
are carried rather than imported: the live module's are private (issue #1010).
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from sal.emissions import EmissionFamily
from sal.likelihood.spatio_sequential_rust import COVARIATE_TABLE_CEILING
from sal.sim.spatio_sequential import SpatioSequentialParams

#: The largest row a ``uint32`` index carries. A table with more rows would
#: wrap on the cast and read the wrong row, so it is refused.
_ROW_LIMIT = 2**32 - 1


def _slope(
    sides: list[EmissionFamily], values: np.ndarray, covariate: np.ndarray
) -> np.ndarray:
    """``max |d log p / dc|`` per observation, over every class and state.

    Through each family's own ``log_density`` and autograd: the observations
    are independent, so the gradient of one state's column summed over the
    observations is that state's derivative at each observation.
    """
    import torch

    counts = torch.from_numpy(np.asarray(values, dtype=np.float64).reshape(-1))
    peak = np.zeros(counts.shape[0])
    for side in sides:
        at = torch.tensor(
            np.asarray(covariate, dtype=np.float64).reshape(-1, 1), requires_grad=True
        )
        scores = side.log_density(counts, covariate=at)
        for k in range(scores.shape[-1]):
            (gradient,) = torch.autograd.grad(scores[:, k].sum(), at, retain_graph=True)
            peak = np.maximum(peak, np.abs(gradient.detach().numpy().reshape(-1)))
    return peak


def grid_scale(
    sides: list[EmissionFamily],
    values: np.ndarray,
    covariate: np.ndarray,
    covariate_tolerance: float,
) -> float:
    """The grid a covariate is coded on: the coarsest power of two within ``covariate_tolerance``.

    Rounding ``c`` to the nearest multiple of ``1 / scale`` moves it by at most
    ``1 / (2 scale)``, so ``|Delta log p| <= G / (2 scale)`` per observation,
    ``G = max |d log p / dc|`` over every class, state and observation, the
    slope taken through each family's autograd at every covariate and at its
    grid value. The scale returned is the smallest power of two meeting
    ``covariate_tolerance`` with that bound; since the grid values move with
    the scale, it is doubled from the observations' own slope until they
    agree. The bound is exact where ``|d log p / dc|`` is largest at a sampled
    point, and a slope peaking between the samples is the case it does not
    cover: a per-observation bound from the two ends of one interval was
    exceeded by 1.3e-6 of itself at the ci instance, at a count whose slope
    turns inside the interval.

    **An integer covariate is refused**: a table over its values is exact,
    and no coarser grid is admissible, because a trial count rounded below its
    own successes leaves the beta-binomial's support, where the density is
    ``-inf`` and no slope bounds the change.

    Parameters
    ----------
    sides : list[EmissionFamily]
        One channel's family per class.
    values : np.ndarray
        That channel's counts.
    covariate : np.ndarray
        That channel's covariate, the counts' shape.
    covariate_tolerance : float
        The largest ``|Delta log p|`` admitted per observation.

    Returns
    -------
    float

    Raises
    ------
    ValueError
        If ``covariate_tolerance`` is not positive and finite, or the
        covariate is an integer.
    """
    _refuse_tolerance(covariate, covariate_tolerance)
    return _coarsest_scale(
        float(_slope(sides, values, covariate).max()),
        covariate_tolerance,
        lambda scale: float(
            _slope(sides, values, _on_grid(_grid, covariate, scale)).max()
        ),
    )


def _refuse_tolerance(covariate: np.ndarray, covariate_tolerance: float) -> None:
    """Refuse a tolerance that is not positive and finite, or an integer covariate."""
    if not (math.isfinite(covariate_tolerance) and covariate_tolerance > 0.0):
        msg = f"covariate_tolerance must be positive and finite, got {covariate_tolerance}"
        raise ValueError(msg)
    if bool((covariate == np.rint(covariate)).all()):
        msg = (
            f"covariate_tolerance={covariate_tolerance}: the covariate is an "
            f"integer, and a table over its values is exact at scale 1; a "
            f"tolerance is for a continuous covariate"
        )
        raise ValueError(msg)


def _coarsest_scale(
    peak: float, covariate_tolerance: float, peak_on_grid: Callable[[float], float]
) -> float:
    """The smallest power of two with ``peak / (2 scale) <= covariate_tolerance``.

    ``peak`` is the slope's largest magnitude at the observations, and
    ``peak_on_grid(scale)`` at their grid values; the scale is doubled from
    the observations' own until the two agree.
    """
    if peak == 0.0:
        return 1.0
    scale = float(2.0 ** math.ceil(math.log2(peak / (2.0 * covariate_tolerance))))
    while True:
        peak = max(peak, peak_on_grid(scale))
        if peak / (2.0 * scale) <= covariate_tolerance:
            return scale
        scale *= 2.0


def _on_grid(
    grid: Callable[[np.ndarray, float], tuple[np.ndarray, np.ndarray]],
    covariate: np.ndarray,
    scale: float,
) -> np.ndarray:
    """Each observation's grid value, flat."""
    codes, levels = grid(covariate, scale)
    return np.asarray(levels[codes.reshape(-1)])


def _grid(covariate: np.ndarray, scale: float) -> tuple[np.ndarray, np.ndarray]:
    """The covariate's codes from zero, and the grid value of each code.

    The codes span the least to the greatest, contiguously, rather than the
    distinct codes alone: the range needs no sort, and the distinct codes need
    one per call, which is what issue #658 measured as the cost of a table.
    """
    codes = np.rint(np.asarray(covariate, dtype=np.float64) * scale).astype(np.int64)
    low = int(codes.min())
    levels = (low + np.arange(int(codes.max()) - low + 1, dtype=np.float64)) / scale
    return codes - low, levels


def _outer_table(
    sides: list[EmissionFamily],
    params: SpatioSequentialParams,
    extent: int,
    levels: np.ndarray,
) -> np.ndarray:
    """Every ``(count, level)`` pair's log-density, row ``count * len(levels) + code``.

    The outer product of the counts ``0 .. extent - 1`` and ``levels``, built
    vectorized and never sorted.
    """
    import torch

    n_levels = levels.size
    counts = torch.from_numpy(np.repeat(np.arange(extent, dtype=np.float64), n_levels))
    covariate = torch.from_numpy(
        np.tile(np.asarray(levels, dtype=np.float64), extent)[:, None]
    )
    table = np.empty((extent * n_levels, params.n_classes, params.n_states))
    for m, side in enumerate(sides):
        table[:, m, :] = side.log_density(counts, covariate=covariate).numpy()
    return table


def _refuse_table(
    n_rows: int, n_levels: int, params: SpatioSequentialParams, what: str
) -> None:
    """Refuse a grid's table past a ``uint32`` row index or the byte ceiling."""
    if n_rows - 1 > _ROW_LIMIT:
        msg = (
            f"{what}: {n_rows} table rows, past the {_ROW_LIMIT + 1} a uint32 "
            f"row index carries"
        )
        raise ValueError(msg)
    size = n_rows * params.n_classes * params.n_states * 8
    if size > COVARIATE_TABLE_CEILING:
        msg = (
            f"{what} spans {n_levels} codes: its table is {size} bytes, past "
            f"the {COVARIATE_TABLE_CEILING} of COVARIATE_TABLE_CEILING; a larger "
            f"covariate_tolerance gives a coarser grid"
        )
        raise ValueError(msg)


def _grid_rows(
    sides: list[EmissionFamily],
    params: SpatioSequentialParams,
    values: np.ndarray,
    covariate: np.ndarray,
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """One channel's table on the grid at ``scale``, and each observation's row.

    The row is ``count * len(levels) + code``.

    Raises
    ------
    ValueError
        If ``scale`` is not positive and finite, or the table's rows overflow
        a ``uint32`` index or its bytes pass
        :data:`~sal.likelihood.spatio_sequential_rust.COVARIATE_TABLE_CEILING`.
        The message names ``scale``.
    """
    return _tabulated(_grid, "covariate grid", sides, params, values, covariate, scale)


def _tabulated(
    grid: Callable[[np.ndarray, float], tuple[np.ndarray, np.ndarray]],
    name: str,
    sides: list[EmissionFamily],
    params: SpatioSequentialParams,
    values: np.ndarray,
    covariate: np.ndarray,
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """The table on ``grid`` at ``scale`` and each observation's row, refused first."""
    if not (math.isfinite(scale) and scale > 0.0):
        msg = f"a {name}'s scale must be positive and finite, got scale={scale}"
        raise ValueError(msg)
    codes, levels = grid(covariate, scale)
    n_rows = (int(values.max()) + 1) * levels.size
    _refuse_table(n_rows, levels.size, params, f"the {name} at scale={scale}")
    table = _outer_table(sides, params, int(values.max()) + 1, levels)
    rows = values.astype(np.int64) * levels.size + codes.reshape(values.shape)
    return table, rows


def _grid_bound(
    sides: list[EmissionFamily],
    values: np.ndarray,
    covariate: np.ndarray,
    scale: float,
) -> np.ndarray:
    """Per observation, the bound on ``|Delta log p|`` the grid at ``scale`` admits.

    ``G * |c - c_hat|``, with ``G`` the largest ``|d log p / dc|`` over every
    class, state and observation, at the covariate and at its grid value
    (:func:`grid_scale`). It is at most ``G / (2 scale)``, and zero where the
    covariate is on the grid.
    """
    codes, levels = _grid(covariate, scale)
    flat = np.asarray(covariate, dtype=np.float64).reshape(-1)
    at_grid = levels[codes.reshape(-1)]
    moved = np.abs(flat - at_grid)
    if not bool(moved.any()):
        return np.zeros(np.shape(values))
    peak = max(
        float(_slope(sides, values, flat).max()),
        float(_slope(sides, values, at_grid).max()),
    )
    return np.asarray(peak * moved).reshape(np.shape(values))


def log_grid_scale(
    sides: list[EmissionFamily],
    values: np.ndarray,
    covariate: np.ndarray,
    covariate_tolerance: float,
) -> float:
    """The log grid's scale: the coarsest power of two within ``covariate_tolerance``.

    :func:`grid_scale` in ``ln c``: rounding ``ln c`` to the nearest multiple
    of ``1 / scale`` moves it by at most ``1 / (2 scale)``, so
    ``|Delta log p| <= G / (2 scale)``, ``G = max |d log p / d ln c|`` over
    every class, state and observation, at every covariate and at its grid
    value. ``d log p / d ln c = c d log p / dc`` by the chain rule, the
    derivative in ``c`` taken through each family's autograd. The precision is
    relative: a level per fixed ratio of the covariate, not per fixed step.
    An unobserved channel, ``c = 0``, has no logarithm; it scores ``log 1``
    at every state and is carried on a level of its own, where its slope is
    zero.

    Parameters
    ----------
    sides : list[EmissionFamily]
        One channel's family per class.
    values : np.ndarray
        That channel's counts.
    covariate : np.ndarray
        That channel's covariate, the counts' shape, non-negative.
    covariate_tolerance : float
        The largest ``|Delta log p|`` admitted per observation.

    Returns
    -------
    float

    Raises
    ------
    ValueError
        As :func:`grid_scale` refuses, or if a covariate is negative or not
        finite.
    """
    _refuse_tolerance(covariate, covariate_tolerance)
    _refuse_log(covariate)
    return _coarsest_scale(
        float(_log_slope(sides, values, covariate).max()),
        covariate_tolerance,
        lambda scale: float(
            _log_slope(sides, values, _on_grid(_log_grid, covariate, scale)).max()
        ),
    )


def _refuse_log(covariate: np.ndarray) -> None:
    """Refuse a covariate the log grid cannot code: negative or not finite."""
    flat = np.asarray(covariate, dtype=np.float64)
    if bool(((flat < 0.0) | ~np.isfinite(flat)).any()):
        msg = (
            "the log grid codes ln c: every covariate must be finite and "
            "non-negative, zero marking the channel unobserved"
        )
        raise ValueError(msg)


def _log_slope(
    sides: list[EmissionFamily], values: np.ndarray, covariate: np.ndarray
) -> np.ndarray:
    """``max |d log p / d ln c|`` per observation: ``c`` times :func:`_slope`."""
    flat = np.asarray(covariate, dtype=np.float64).reshape(-1)
    return np.asarray(flat * _slope(sides, values, flat))


def _log_grid(covariate: np.ndarray, scale: float) -> tuple[np.ndarray, np.ndarray]:
    """The covariate's codes on ``ln c`` from zero, and the grid value of each code.

    ``code = round(ln c * scale)``, tabulated at ``exp(code / scale)``, the
    codes offset to start at zero and spanning the least to the greatest.
    Where a covariate is zero, the channel unobserved, level zero is ``0.0``
    and every observed code moves up by one.
    """
    flat = np.asarray(covariate, dtype=np.float64)
    _refuse_log(flat)
    observed = flat > 0.0
    if not bool(observed.any()):
        return np.zeros(flat.shape, dtype=np.int64), np.zeros(1)
    logs = np.rint(np.log(flat[observed]) * scale).astype(np.int64)
    low = int(logs.min())
    levels = np.exp(
        (low + np.arange(int(logs.max()) - low + 1, dtype=np.float64)) / scale
    )
    if bool(observed.all()):
        return np.asarray(logs - low).reshape(flat.shape), levels
    codes = np.zeros(flat.shape, dtype=np.int64)
    codes[observed] = logs - low + 1
    return codes, np.concatenate([np.zeros(1), levels])


def _log_grid_rows(
    sides: list[EmissionFamily],
    params: SpatioSequentialParams,
    values: np.ndarray,
    covariate: np.ndarray,
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """One channel's table on the log grid at ``scale``, and each observation's row.

    The row is ``count * len(levels) + code``.

    Raises
    ------
    ValueError
        As :func:`_grid_rows` refuses, or if a covariate is negative or not
        finite.
    """
    return _tabulated(
        _log_grid, "log covariate grid", sides, params, values, covariate, scale
    )


def _log_grid_bound(
    sides: list[EmissionFamily],
    values: np.ndarray,
    covariate: np.ndarray,
    scale: float,
) -> np.ndarray:
    """Per observation, the bound on ``|Delta log p|`` the log grid at ``scale`` admits.

    ``G * |ln c - ln c_hat|``, with ``G`` the largest ``|d log p / d ln c|``
    over every class, state and observation, at the covariate and at its grid
    value (:func:`log_grid_scale`). It is at most ``G / (2 scale)``, and zero
    where the covariate is on the grid or unobserved.
    """
    flat = np.asarray(covariate, dtype=np.float64).reshape(-1)
    at_grid = _on_grid(_log_grid, covariate, scale)
    observed = flat > 0.0
    moved = np.zeros(flat.shape)
    moved[observed] = np.abs(np.log(flat[observed]) - np.log(at_grid[observed]))
    if not bool(moved.any()):
        return np.zeros(np.shape(values))
    peak = max(
        float(_log_slope(sides, values, flat).max()),
        float(_log_slope(sides, values, at_grid).max()),
    )
    return np.asarray(peak * moved).reshape(np.shape(values))
