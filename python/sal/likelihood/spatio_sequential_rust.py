"""The coupled E step and its field in Rust, pinned to the NumPy oracle (issue #399).

The same two quantities :mod:`sal.likelihood.spatio_sequential`
computes --- ``class_posteriors`` and ``external_field`` --- over the
two-channel count emission of :mod:`sal.sim.count_pairs`,
through ``src/coupled.rs``. The NumPy path stays and is the oracle
(``likelihood/CLAUDE.md``); this is pinned against it at the ci instance and
on a 64-vertex slice of the declared 5,041-vertex one.

**Why there is a Rust path at all.** `cProfile` on the NumPy path at the 5K
instance, bin factor 10: one ``class_posteriors`` is 20.1 s, of which
``torch.lgamma`` is 10.4 s of self time (51.5%) and the two emission
``log_density`` bodies 18.9 s cumulative (94.3%); one ``external_field`` is
191.3 s, of which ``torch.lgamma`` is 102.8 s (53.7%) and the emission
densities 188.1 s (98.3%). Everything else --- forward--backward, the
``einsum``, the boundary --- is under 4%. The measured speedups are in
``STATUS.md``.

**A table row indexes a table instead of calling `lgamma`.** Both channels'
observations are integers in a range of a few thousand, so
``log p(count | class, state)`` is a function of an integer and is tabulated
once per call by the families themselves. The kernel then does two loads and
an add where the oracle does three ``lgamma`` calls, and the tables are the
oracle's own arithmetic rather than a second implementation of it. The tables
are row-major, ``[row, M, K]``, for the reason ``src/coupled.rs`` states.

**Under a covariate the row is not the count** (issue #658). The density is
then a function of the count *and* the exposure or trial count it is scored
against, so there is no table indexed by the count alone. The trial count's
channel tabulates every ``(count, covariate code)`` pair and each observation
carries the row it falls in; the kernel never knew what the index meant, only
that the table was built from it.

**The exposure is factored instead** (issue #1064). A continuous exposure takes
one value per observation, and a table by count and distinct exposure then has
``extent * S * V`` rows: 2.3 GB per channel at the ci instance of
``spatio_sequential_counts_covariate``. The negative binomial splits as
``A_k(y) + r_k log(r_k / t) + y log(mu_k c / t)`` with ``t = r_k + mu_k c``
(:meth:`~sal.emissions.NegativeBinomialEmission.count_log_factor`), so
the total's table is ``A`` by count and the kernel forms the two exposure terms
per observation, in the family's order. Each score is then the family's to
2.3 ulp relative at that instance, the difference between two ``log``
implementations.

**A trial count may be tabulated on a grid** (issue #1064). With
``covariate_tolerance`` set, a covariate is coded ``round(c * scale)`` for the
coarsest power-of-two ``scale`` whose bound
``max |d log p / dc| / (2 scale)`` is within the tolerance, and the table spans
the codes from the least to the greatest. An integer covariate is coded at
``scale = 1`` whatever the tolerance, which reproduces it exactly; see
:func:`grid_scale`.

**One crossing per call, contiguous.** The counts cross as the ``(S, V)``
arrays they are already held in, cast to ``uint16`` --- which is the same
statement the table makes, that the counts are small --- and the results are
written into arrays allocated here.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from sal import oxisal
from sal.emissions import EmissionFamily, NegativeBinomialEmission
from sal.likelihood.spatio_sequential import (
    ClassPosteriors,
    log_prior,
)
from sal.sim.count_pairs import (
    SUCCESSES,
    TOTAL,
    IndependentCountPair,
)
from sal.sim.spatio_sequential import SpatioSequentialParams


def _families(params: SpatioSequentialParams) -> list[IndependentCountPair]:
    """The two-channel families of every class, at their own type.

    Raises
    ------
    TypeError
        If a class does not carry one. The kernel tabulates two channels by
        their integer counts, and a family that is not a pair of count
        families has nothing to tabulate.
    """
    families = []
    for m, family in enumerate(params.emissions):
        if not isinstance(family, IndependentCountPair):
            msg = (
                f"class {m} carries a {type(family).__name__}; the Rust E step "
                f"is over the two-channel count emission"
            )
            raise TypeError(msg)
        families.append(family)
    return families


#: Bytes one channel's grid table may take (issue #1064): the refusal a
#: fine ``scale`` meets before it allocates. 1 GiB is 6.2x the covariate of the
#: stress instance of ``spatio_sequential_counts_covariate``, the largest array
#: the E step already holds there.
GRID_TABLE_CEILING = 2**30

#: The largest row a ``uint32`` index carries. A table with more rows would
#: wrap on the cast and read the wrong row, so it is refused.
_ROW_LIMIT = 2**32 - 1


@dataclass(frozen=True)
class ExposureTerm:
    """The negative binomial's exposure, factored out of the total's table (issue #1064).

    With it, the total's table is ``A_k(y)`` by count and its row is the
    count; the kernel adds ``r log(r / t) + y log(mu c / t)``, ``t = r + mu c``,
    per observation.

    Parameters
    ----------
    exposure : np.ndarray
        ``(S, n_nodes)`` contiguous ``float64``; zero marks the total unobserved.
    dispersion : np.ndarray
        ``(M, K)`` contiguous ``float64``, each class's ``r``.
    mean : np.ndarray
        ``(M, K)`` contiguous ``float64``, each class's ``mu`` at unit exposure.
    """

    exposure: np.ndarray
    dispersion: np.ndarray
    mean: np.ndarray

    def arguments(self) -> dict[str, np.ndarray]:
        """The kernel's three keyword arguments, flat."""
        return {
            "exposure": self.exposure.reshape(-1),
            "dispersion": self.dispersion.reshape(-1),
            "mean": self.mean.reshape(-1),
        }


def _side(family: IndependentCountPair, channel: int) -> EmissionFamily:
    """One channel's family of a pair."""
    return family.total if channel == TOTAL else family.successes


def _refuse_rows(n_rows: int, what: str) -> None:
    """Refuse a table whose rows a ``uint32`` index cannot carry."""
    if n_rows - 1 > _ROW_LIMIT:
        msg = (
            f"{what}: {n_rows} table rows, past the {_ROW_LIMIT + 1} a uint32 "
            f"row index carries"
        )
        raise ValueError(msg)


def _tabulate(
    sides: list[EmissionFamily],
    params: SpatioSequentialParams,
    values: np.ndarray,
    levels: np.ndarray,
    codes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Every ``(count, level)`` pair's log-density, and each observation's row.

    The outer product of the counts ``0 .. max`` and ``levels``, built
    vectorized and never sorted; the row is ``count * len(levels) + code``.
    """
    # `log_density` is the families' tensor API, which the objectives
    # differentiate through; torch is imported at this call rather than with
    # the module (issue #1011).
    import torch

    extent = int(values.max()) + 1
    n_levels = levels.size
    counts = torch.from_numpy(np.repeat(np.arange(extent, dtype=np.float64), n_levels))
    covariate = torch.from_numpy(
        np.tile(np.asarray(levels, dtype=np.float64), extent)[:, None]
    )
    table = np.empty((extent * n_levels, params.n_classes, params.n_states))
    for m, side in enumerate(sides):
        table[:, m, :] = side.log_density(counts, covariate=covariate).numpy()
    rows = values.astype(np.int64) * n_levels + codes.reshape(values.shape)
    return np.ascontiguousarray(table), rows


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
    turns inside the interval. A power of two makes ``code / scale`` exact in
    ``float64``, so a covariate already on the grid is reproduced bit for bit.

    **An integer covariate is coded at ``scale = 1``**, whatever the tolerance.
    At that scale the code is the value and the grid is exact, so no finer one
    is needed; and no coarser one is admissible, because a trial count rounded
    below its own successes leaves the beta-binomial's support, where the
    density is ``-inf`` and no slope bounds the change.

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
        If ``covariate_tolerance`` is not positive and finite.
    """
    if not (math.isfinite(covariate_tolerance) and covariate_tolerance > 0.0):
        msg = f"covariate_tolerance must be positive and finite, got {covariate_tolerance}"
        raise ValueError(msg)
    if bool((covariate == np.rint(covariate)).all()):
        return 1.0
    peak = float(_slope(sides, values, covariate).max())
    if peak == 0.0:
        return 1.0
    scale = float(2.0 ** math.ceil(math.log2(peak / (2.0 * covariate_tolerance))))
    while True:
        codes, levels = _grid(covariate, scale)
        at_grid = levels[codes.reshape(-1)]
        peak = max(peak, float(_slope(sides, values, at_grid).max()))
        if peak / (2.0 * scale) <= covariate_tolerance:
            return scale
        scale *= 2.0


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


def _grid_rows(
    sides: list[EmissionFamily],
    params: SpatioSequentialParams,
    values: np.ndarray,
    covariate: np.ndarray,
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """One channel's table on the grid at ``scale``, and each observation's row.

    Raises
    ------
    ValueError
        If ``scale`` is not positive and finite, or the table's rows overflow
        a ``uint32`` index or its bytes pass :data:`GRID_TABLE_CEILING`. The
        message names ``scale``.
    """
    if not (math.isfinite(scale) and scale > 0.0):
        msg = f"a covariate grid's scale must be positive and finite, got scale={scale}"
        raise ValueError(msg)
    codes, levels = _grid(covariate, scale)
    n_rows = (int(values.max()) + 1) * levels.size
    what = f"the covariate grid at scale={scale}"
    _refuse_rows(n_rows, what)
    size = n_rows * params.n_classes * params.n_states * 8
    if size > GRID_TABLE_CEILING:
        msg = (
            f"{what} spans {levels.size} codes: its table is {size} bytes, past "
            f"the {GRID_TABLE_CEILING} of GRID_TABLE_CEILING; a larger "
            f"covariate_tolerance gives a coarser grid"
        )
        raise ValueError(msg)
    return _tabulate(sides, params, values, levels, codes)


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


def _channel_rows(
    families: list[IndependentCountPair],
    params: SpatioSequentialParams,
    values: np.ndarray,
    covariate: np.ndarray | None,
    channel: int,
    covariate_tolerance: float | None = None,
) -> tuple[np.ndarray, np.ndarray, ExposureTerm | None]:
    """One channel's table, the row each observation falls in, and its exposure term.

    Without a covariate the row **is** the count, the table is count-major and
    the extent is the largest count plus one --- exactly what this module
    tabulated before issue #658, and the same numbers.

    With one on a negative binomial, the table is
    :meth:`~sal.emissions.NegativeBinomialEmission.count_log_factor`
    by count, the row is the count, and the exposure travels as an
    :class:`ExposureTerm` the kernel completes the density with (issue #1064).

    With one on any other family, the density is a function of the count
    *and* the covariate. Every ``(count, covariate)`` combination is tabulated,
    and the row is ``count * n_levels + code``. Where ``covariate_tolerance``
    is ``None`` the codes are the distinct covariate values, factorized once
    per call (issue #658), bit for bit what this module computed before
    issue #1064. Where it is set, the codes are the covariate's on the grid
    :func:`grid_scale` chooses, spanning the least to the greatest.

    The rows are the families' own arithmetic either way, which is the property
    that keeps this a table rather than a second implementation of the oracle.
    """
    # `log_density` is the families' tensor API, which the objectives
    # differentiate through: the table's arguments are built as arrays and
    # cross into it once per family, and torch is imported at this call rather
    # than with the module (issue #1011).
    import torch

    sides = [_side(family, channel) for family in families]
    if covariate is None:
        extent = int(values.max()) + 1
        counts = torch.from_numpy(np.arange(extent, dtype=np.float64))
        table = np.empty((extent, params.n_classes, params.n_states))
        for m, side in enumerate(sides):
            table[:, m, :] = side.log_density(counts).numpy()
        return np.ascontiguousarray(table), values, None
    if all(isinstance(side, NegativeBinomialEmission) for side in sides):
        negative_binomials = [
            side for side in sides if isinstance(side, NegativeBinomialEmission)
        ]
        exposure = np.ascontiguousarray(covariate, dtype=np.float64)
        if bool(((exposure < 0.0) | ~np.isfinite(exposure)).any()):
            msg = "every exposure must be finite and non-negative; zero marks the count unobserved"
            raise ValueError(msg)
        extent = int(values.max()) + 1
        counts = torch.from_numpy(np.arange(extent, dtype=np.float64))
        table = np.empty((extent, params.n_classes, params.n_states))
        for m, side in enumerate(negative_binomials):
            table[:, m, :] = side.count_log_factor(counts).numpy()
        term = ExposureTerm(
            exposure,
            np.ascontiguousarray(
                np.stack([side.dispersion.numpy() for side in negative_binomials])
            ),
            np.ascontiguousarray(
                np.stack([side.mean.numpy() for side in negative_binomials])
            ),
        )
        return np.ascontiguousarray(table), values, term
    if covariate_tolerance is not None:
        scale = grid_scale(sides, values, covariate, covariate_tolerance)
        table, rows = _grid_rows(sides, params, values, covariate, scale)
        return table, rows, None
    # The covariate alone is factorized, and the row is arithmetic on the two
    # codes: `count * n_distinct + code`. Factorizing the *pairs* instead --- one
    # `np.unique` over an `(S * V, 2)` array --- is a lexsort per call, and it
    # cost 135.6 ms of a 141.4 ms E step at the ci instance, taking the backend
    # to 0.6x the oracle it exists to beat. This is one sort of a single column
    # and two integer operations, and the table it addresses is the outer
    # product rather than the distinct pairs: larger, built vectorized, and
    # never sorted.
    distinct, codes = np.unique(covariate.reshape(-1), return_inverse=True)
    _refuse_rows(
        (int(values.max()) + 1) * distinct.size, "the distinct covariate values"
    )
    table, rows = _tabulate(sides, params, values, distinct, codes)
    return table, rows, None


@dataclass(frozen=True)
class EmissionTables:
    """One tabulated log-density per channel.

    Parameters
    ----------
    total : np.ndarray
        The first channel's table, ``(row, M, K)`` contiguous ``float64``.
    successes : np.ndarray
        The second channel's, the same shape.
    """

    total: np.ndarray
    successes: np.ndarray

    def __iter__(self) -> Iterator[np.ndarray]:
        """``(total, successes)``: the order callers unpack."""
        yield from (self.total, self.successes)


@dataclass(frozen=True)
class EmissionRows:
    """The tables and the row each observation falls in.

    A row index is only interpretable against the table it was built with,
    so the two travel together (issue #658).

    Parameters
    ----------
    total_rows : np.ndarray
        The first channel's rows, ``(S, n_nodes)`` contiguous ``uint32``.
    success_rows : np.ndarray
        The second channel's, the same shape.
    total_table : np.ndarray
        The first channel's table, as :class:`EmissionTables` carries it.
    success_table : np.ndarray
        The second channel's.
    exposure : ExposureTerm | None
        The first channel's factored exposure, where it carries one; its
        table is then ``A`` by count (issue #1064).
    """

    total_rows: np.ndarray
    success_rows: np.ndarray
    total_table: np.ndarray
    success_table: np.ndarray
    exposure: ExposureTerm | None = None

    def kernel_arguments(self) -> dict[str, np.ndarray]:
        """The exposure term as the kernels' keyword arguments; empty without one."""
        return {} if self.exposure is None else self.exposure.arguments()

    def __iter__(self) -> Iterator[np.ndarray | ExposureTerm | None]:
        """The rows, the tables, then the exposure term: the order callers unpack."""
        yield from (
            self.total_rows,
            self.success_rows,
            self.total_table,
            self.success_table,
            self.exposure,
        )


def emission_tables(
    params: SpatioSequentialParams,
    observations: np.ndarray,
    *,
    covariate_tolerance: float | None = None,
) -> EmissionTables:
    """Both channels' log-density for every class, state and table row.

    Parameters
    ----------
    params : SpatioSequentialParams
        Its emissions the two-channel families, and the covariate the rows are
        tabulated against where it carries one.
    observations : np.ndarray
        Shape ``(S, n_nodes, 2)``.
    covariate_tolerance : float | None
        As :func:`emission_rows` takes it.

    Returns
    -------
    EmissionTables
        The first channel's table and the second's. :func:`emission_rows`
        returns these with the row indices and the exposure term, which is
        how the two entry points take them.
    """
    rows = emission_rows(params, observations, covariate_tolerance=covariate_tolerance)
    return EmissionTables(rows.total_table, rows.success_table)


def emission_rows(
    params: SpatioSequentialParams,
    observations: np.ndarray,
    *,
    covariate_tolerance: float | None = None,
) -> EmissionRows:
    """The rows and the tables together, since neither is meaningful alone.

    A row index is only interpretable against the table it was built with, so
    the two are returned from one call rather than derived twice from the same
    inputs and trusted to agree (issue #658).

    Parameters
    ----------
    params : SpatioSequentialParams
        The two-channel families, and the covariate where it carries one.
    observations : np.ndarray
        Shape ``(S, n_nodes, 2)``.
    covariate_tolerance : float | None
        The largest ``|Delta log p|`` per observation a covariate grid may
        admit, for a channel whose covariate is tabulated rather than factored
        (the trial count's). ``None``, the default, tabulates the distinct
        covariate values exactly, as before issue #1064. An integer covariate
        is coded exactly at any tolerance (:func:`grid_scale`), so for the
        trial count the grid changes the table's layout and not its values.

    Returns
    -------
    EmissionRows
        The total's rows, the successes' rows, the total's table, the
        successes' table, and the total's exposure term where it has one.

    Raises
    ------
    ValueError
        If ``covariate_tolerance`` is not positive and finite, or a table's
        rows overflow a ``uint32`` index.
    """
    families = _families(params)
    covariate = params.covariate
    out = []
    for channel in (TOTAL, SUCCESSES):
        table, rows, term = _channel_rows(
            families,
            params,
            observations[..., channel],
            None if covariate is None else covariate[..., channel],
            channel,
            covariate_tolerance,
        )
        out.append((np.ascontiguousarray(rows, dtype=np.uint32), table, term))
    return EmissionRows(out[0][0], out[1][0], out[0][1], out[1][1], out[0][2])


def class_posteriors(
    params: SpatioSequentialParams,
    observations: np.ndarray,
    labels: np.ndarray,
    *,
    covariate_tolerance: float | None = None,
) -> ClassPosteriors:
    """Forward--backward on every class's chain over its members' summed scores.

    The signature and the return of
    :func:`sal.likelihood.spatio_sequential.class_posteriors`,
    which is the oracle this is pinned to.

    Parameters
    ----------
    params : SpatioSequentialParams
        The model, its emissions the two-channel count families.
    observations : np.ndarray
        Shape ``(S, n_nodes, 2)``, integer counts.
    labels : np.ndarray
        One class per vertex, shape ``(n_nodes,)``.
    covariate_tolerance : float | None
        As :func:`emission_rows` takes it.

    Returns
    -------
    ClassPosteriors
        The posterior ``(M, S, K)``, the pairwise ``(M, S - 1, K, K)`` and the
        per-class log evidence ``(M,)``.
    """
    rows = emission_rows(params, observations, covariate_tolerance=covariate_tolerance)
    n_positions, n_nodes = observations.shape[:2]
    posterior = np.empty((params.n_classes, n_positions, params.n_states))
    pairwise = np.empty(
        (
            params.n_classes,
            max(n_positions - 1, 0),
            params.n_states,
            params.n_states,
        )
    )
    log_evidence = np.empty(params.n_classes)
    log_transition = np.ascontiguousarray(
        np.broadcast_to(
            np.log(params.transition),
            (params.n_classes, params.n_states, params.n_states),
        )
    )
    oxisal.class_posteriors(
        rows.total_rows.reshape(-1),
        rows.success_rows.reshape(-1),
        np.ascontiguousarray(labels, dtype=np.int64),
        rows.total_table.reshape(-1),
        rows.success_table.reshape(-1),
        np.ascontiguousarray(np.log(params.initial)).reshape(-1),
        log_transition.reshape(-1),
        n_positions,
        n_nodes,
        params.n_classes,
        params.n_states,
        posterior.reshape(-1),
        pairwise.reshape(-1),
        log_evidence,
        **rows.kernel_arguments(),
    )
    return ClassPosteriors(posterior, pairwise, log_evidence)


def external_field(
    params: SpatioSequentialParams,
    observations: np.ndarray,
    labels: np.ndarray,
    posterior: np.ndarray | None = None,
    *,
    covariate_tolerance: float | None = None,
) -> np.ndarray:
    """``H[n, m]``, minus the posterior-expected emission score, shape ``(n_nodes, M)``.

    The signature and the return of
    :func:`sal.likelihood.spatio_sequential.external_field`.
    ``posterior`` defaults to this module's own E step at ``labels``;
    ``covariate_tolerance`` is as :func:`emission_rows` takes it.

    Returns
    -------
    np.ndarray
        Shape ``(n_nodes, M)``.
    """
    if posterior is None:
        posterior = class_posteriors(
            params, observations, labels, covariate_tolerance=covariate_tolerance
        ).posterior
    rows = emission_rows(params, observations, covariate_tolerance=covariate_tolerance)
    n_positions, n_nodes = observations.shape[:2]
    field = np.empty((n_nodes, params.n_classes))
    # (M, S, K) to (S, M, K): the kernel wants one position's weights
    # contiguous beside the tables' rows, which are count-major.
    weights = np.ascontiguousarray(np.moveaxis(posterior, 0, 1))
    oxisal.external_field(
        rows.total_rows.reshape(-1),
        rows.success_rows.reshape(-1),
        rows.total_table.reshape(-1),
        rows.success_table.reshape(-1),
        weights.reshape(-1),
        n_positions,
        n_nodes,
        params.n_classes,
        params.n_states,
        field.reshape(-1),
        **rows.kernel_arguments(),
    )
    return field


def labelled_log_likelihood(
    params: SpatioSequentialParams,
    observations: np.ndarray,
    labels: np.ndarray,
    *,
    covariate_tolerance: float | None = None,
) -> float:
    """``log p(x, l | theta)`` with the chains marginalized, up to ``log Z_Potts``.

    The signature and the return of
    :func:`sal.likelihood.spatio_sequential.labelled_log_likelihood`,
    over this module's E step. The Potts term is the oracle's own: it is a sum
    over edges and costs nothing beside the emission densities.
    ``covariate_tolerance`` is as :func:`emission_rows` takes it.

    Returns
    -------
    float
    """
    own = float(log_prior(params, np.asarray(labels, dtype=np.int64)[None, :])[0])
    return own + float(
        class_posteriors(
            params, observations, labels, covariate_tolerance=covariate_tolerance
        ).log_evidence.sum()
    )


def covariate_grid_error(
    params: SpatioSequentialParams,
    observations: np.ndarray,
    covariate_tolerance: float,
) -> np.ndarray:
    """Per observation, the bound on ``|Delta log p|`` the covariate grid admits, shape ``(S, n_nodes)``.

    Summed over the channels that are tabulated on a grid, each
    ``G * |c - c_hat|`` with ``G`` the largest ``|d log p / dc|`` through the
    family's autograd, over every class, state and observation and at both
    ``c`` and ``c_hat``, and ``c_hat`` the grid value at :func:`grid_scale`'s
    scale. It is at most ``G / (2 scale)``, which :func:`grid_scale` holds
    within ``covariate_tolerance``. A class's log evidence is a
    log-sum over paths of a sum over its members' scores, so it moves by at
    most the sum of this array over the class's members. The factored
    exposure contributes zero, and so does an integer covariate, which the
    grid reproduces exactly.

    Parameters
    ----------
    params : SpatioSequentialParams
        The two-channel families and their covariate.
    observations : np.ndarray
        Shape ``(S, n_nodes, 2)``.
    covariate_tolerance : float
        As :func:`emission_rows` takes it.

    Returns
    -------
    np.ndarray
        Zero everywhere where the params carry no covariate.
    """
    families = _families(params)
    bound = np.zeros(observations.shape[:2])
    if params.covariate is None:
        return bound
    for channel in (TOTAL, SUCCESSES):
        sides = [_side(family, channel) for family in families]
        if all(isinstance(side, NegativeBinomialEmission) for side in sides):
            continue
        values = observations[..., channel]
        covariate = params.covariate[..., channel]
        scale = grid_scale(sides, values, covariate, covariate_tolerance)
        bound += _grid_bound(sides, values, covariate, scale)
    return bound
