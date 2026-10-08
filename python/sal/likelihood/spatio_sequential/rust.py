"""The coupled E step and its field in Rust, pinned to the NumPy oracle (issue #399).

The same two quantities :mod:`sal.likelihood.spatio_sequential`
computes --- ``class_posteriors`` and ``external_field`` --- over the
two-channel count emission of :mod:`sal.sim.count_pairs` and over every
one-channel family (issues #1298, #1308), through ``src/coupled.rs``. The
NumPy path stays and is the oracle (``likelihood/CLAUDE.md``); this is pinned
against it at the ci instance and on a 64-vertex slice of the declared
5,041-vertex one.

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

**Under a covariate the row is not the count** (issues #658, #1064). The
density is then a function of the count *and* the exposure or trial count.

- The negative binomial's exposure is continuous, and a table by count and
  distinct exposure has a row per observation: 2.3 GB per channel at the ci
  instance of ``spatio_sequential_counts_covariate``. It is factored: the
  table is :func:`~sal.emissions.nb.exposure_table`, ``T_k(y) = S(r_k, y) -
  lgamma(y + 1)``, and the kernel completes it in :mod:`sal.emissions.nb`'s
  order, ``(T + y log(lambda / (1 + q))) - r log1p(q)``, ``lambda = mu_k c``,
  ``q = lambda / r_k``: one ``log`` and one ``log1p`` per score, no term of
  size ``r log r`` cancelled (issue #1335).
- The beta-binomial's trial count is an integer, and its rows take one of
  three exact layouts, :data:`CovariateRows`, bitwise to one another.
  ``range`` and ``distinct`` tabulate every ``(successes, trial count)`` pair
  --- over every trial count from the least to the greatest, or over those
  that occur, found by ``oxisal.factorize`` in one pass in first-appearance
  order --- and hand each observation its row. ``factored`` keeps the table
  by successes, ``U_k(z) = lgamma(z + a_k)``, and the kernel adds
  ``V_k(n - z) - W_k(n)``, the Beta function's three terms and the
  state-free ``lgamma(j + 1)`` at ``n``, ``z`` and ``n - z``, all tabulated
  by :mod:`sal.emissions.bb` and summed in the family's order, so each score
  is its ``log_density`` bit for bit. ``range`` is the default: in a fit at
  the stress instance, with the rows built once, it and ``distinct`` are
  tied fastest and ``factored`` is 4% slower over the E steps and fields,
  its field 14% slower per call for the six additions per score the
  family's order costs; ``factored`` is the fastest where rows are built per
  call, and its tables the smallest. The numbers are in
  ``changelog.d/1064.added.md``.

**A one-channel model reads one table, keyed by what determines its density**
(issues #1298, #1308). The kernel adds nothing to a row it reads, so no
family's arithmetic is in Rust, and the families differ only in the row:

- The categorical emission, the default model of
  :func:`~sal.search.spatio_sequential.fit_spatio_sequential`, is a function
  of the symbol: its table is the log emission matrix, ``[x, M, K]``
  (:func:`symbol_table`).
- A negative binomial or beta-binomial without a covariate is a function of
  the count, and a beta-binomial under a trial count of the
  ``(count, trial count)`` pair, in the ``range`` or ``distinct`` layout
  above (:func:`one_channel_table`).
- The Gaussian, a continuous exposure, the ``factored`` layout, and classes
  of different families are scored by site: the table is each family's
  ``log_density`` at every observation, ``(S * n_nodes, M, K)``, its row
  ``s * n_nodes + v``. So is a keyed table with more rows than there are
  sites, which would be more density evaluations than the sites themselves.

The two-channel count emission and a one-channel family do not share a row,
and classes mixing the two are refused. A table by site is built once per set
of parameters where a fit hands its rows in (:class:`ObservationRows`), so
the E step, the labelled joints and the field after one M step share it.

A covariate grid, for a continuous covariate on a family that does not
factor, serves neither family and is conserved in
:mod:`sal.sandbox.covariate_grid`.

**The rows depend on the observations and the covariate alone.** No parameter
enters them, so a fit builds them once with :func:`observation_rows` and hands
the result to every E step, which then builds only the tables.

**One crossing per call, contiguous.** The counts cross as the ``(S, V)``
arrays they are already held in, cast to ``uint32``, and the results are
written into arrays allocated here.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Literal, cast

import numpy as np

from sal import oxisal
from sal.emissions import (
    BetaBinomialEmission,
    CategoricalEmission,
    EmissionFamily,
    NegativeBinomialEmission,
    validated_trials,
)
from sal.emissions.bb import (
    beta_binomial_log_pmf,
    density_table,
    log_factorial,
    trial_tables,
)
from sal.emissions.nb import exposure_table
from sal.likelihood.spatio_sequential import (
    COVARIATE_ROWS,
    COVARIATE_TABLE_CEILING,
    ChannelRows,
    ClassPosteriors,
    CovariateRows,
    ObservationRows,
    covariate_block,
)
from sal.sim.count_pairs import (
    SUCCESSES,
    TOTAL,
    IndependentCountPair,
)
from sal.sim.spatio_sequential import SpatioSequentialParams


def _categorical(params: SpatioSequentialParams) -> list[CategoricalEmission] | None:
    """Every class's categorical family, or ``None`` where a class carries another (issue #1298).

    A mixture of the categorical and another one-channel family is scored by
    site (:func:`one_channel_table`, issue #1308).

    Raises
    ------
    ValueError
        If the classes' alphabets differ: one table serves every class, and a
        symbol past one class's alphabet has no density under it.
    """
    families = [
        family for family in params.emissions if isinstance(family, CategoricalEmission)
    ]
    if len(families) < params.n_classes:
        return None
    alphabets = {family.n_symbols for family in families}
    if len(alphabets) > 1:
        msg = (
            f"the classes' alphabets are {sorted(alphabets)} symbols; the Rust "
            f"E step tabulates one alphabet for every class"
        )
        raise ValueError(msg)
    return families


def _families(params: SpatioSequentialParams) -> list[IndependentCountPair]:
    """The two-channel families of every class, at their own type.

    Raises
    ------
    TypeError
        If a class does not carry one: its table is over one channel
        (:func:`one_channel_table`), and the two cannot share a row.
    """
    families = []
    for m, family in enumerate(params.emissions):
        if not isinstance(family, IndependentCountPair):
            msg = (
                f"class {m} carries a {type(family).__name__}; a two-channel "
                f"table is over the two-channel count emission in every class"
            )
            raise TypeError(msg)
        families.append(family)
    return families


def _refuse_mixed_channels(params: SpatioSequentialParams) -> bool:
    """``True`` where every class carries the two-channel count emission.

    Raises
    ------
    TypeError
        If some classes carry it and others do not. It scores ``(S, n_nodes,
        2)`` observations and a one-channel family ``(S, n_nodes)``, so no
        table of either aligns with the other by site (issue #1308).
    """
    pairs = [isinstance(family, IndependentCountPair) for family in params.emissions]
    if all(pairs):
        return True
    if any(pairs):
        m = pairs.index(not pairs[0])
        msg = (
            f"class {m} carries a {type(params.emissions[m]).__name__} and class "
            f"0 a {type(params.emissions[0]).__name__}; the two-channel count "
            f"emission's rows do not align by site with a one-channel family's"
        )
        raise TypeError(msg)
    return False


#: The largest row a ``uint32`` index carries. A table with more rows would
#: wrap on the cast and read the wrong row, so it is refused.
_ROW_LIMIT = 2**32 - 1


@dataclass(frozen=True)
class ExposureTerm:
    """The negative binomial's exposure, factored out of the total's table (issue #1064).

    With it, the total's table is :func:`~sal.emissions.nb.exposure_table` by
    count and its row is the count; the kernel adds ``y log c - (y + r) log t``,
    ``t = r + mu c``, per observation.

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


@dataclass(frozen=True)
class TrialTerm:
    """The beta-binomial's trial count, factored out of the successes' table (issue #1064).

    With it, the successes' table is ``U[z, m, k] = lgamma(z + a_mk)`` and its
    row is the successes; the kernel adds the other eight terms of the density
    per observation, in the family's order (:mod:`sal.emissions.bb`).

    Parameters
    ----------
    trials : np.ndarray
        ``(S, n_nodes)`` contiguous ``uint32``; zero marks the successes
        unobserved.
    failure : np.ndarray
        ``V[j, m, k] = S(b_mk, j)``, ``(extent, M, K)`` contiguous.
    trial : np.ndarray
        ``W[n, m, k] = S(a_mk + b_mk, n)``, ``(extent, M, K)`` contiguous.
    log_factorial : np.ndarray
        ``lgamma(j + 1)``, ``(extent,)``.
    log_rate : np.ndarray
        ``log(a / (a + b))`` and ``log(b / (a + b))``, ``(2, M, K)`` contiguous.
    """

    trials: np.ndarray
    failure: np.ndarray
    trial: np.ndarray
    log_factorial: np.ndarray
    log_rate: np.ndarray

    def arguments(self) -> dict[str, np.ndarray]:
        """The kernel's five keyword arguments, flat."""
        return {
            "trials": self.trials.reshape(-1),
            "failure_table": self.failure.reshape(-1),
            "trial_table": self.trial.reshape(-1),
            "log_factorial": self.log_factorial,
            "log_rate": self.log_rate.reshape(-1),
        }

    @property
    def nbytes(self) -> int:
        """The bytes of the four tables, the trial counts excluded."""
        return (
            self.failure.nbytes
            + self.trial.nbytes
            + self.log_factorial.nbytes
            + self.log_rate.nbytes
        )


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


def _refuse_bytes(n_rows: int, params: SpatioSequentialParams, what: str) -> None:
    """Refuse a covariate table past :data:`COVARIATE_TABLE_CEILING` before it is built."""
    size = n_rows * params.n_classes * params.n_states * 8
    if size > COVARIATE_TABLE_CEILING:
        msg = (
            f"{what}: its table is {size} bytes, past the {COVARIATE_TABLE_CEILING} "
            f"of COVARIATE_TABLE_CEILING"
        )
        raise ValueError(msg)


def _outer_table(
    sides: Sequence[EmissionFamily],
    params: SpatioSequentialParams,
    extent: int,
    levels: np.ndarray,
) -> np.ndarray:
    """Every ``(count, level)`` pair's log-density, row ``count * len(levels) + code``.

    The outer product of the counts ``0 .. extent - 1`` and ``levels``, built
    vectorized and never sorted.
    """
    # `log_density` is the families' tensor API, which the objectives
    # differentiate through; torch is imported at this call rather than with
    # the module (issue #1011).
    import torch

    n_levels = levels.size
    counts = torch.from_numpy(np.repeat(np.arange(extent, dtype=np.float64), n_levels))
    covariate = torch.from_numpy(
        np.tile(np.asarray(levels, dtype=np.float64), extent)[:, None]
    )
    table = np.empty((extent * n_levels, params.n_classes, params.n_states))
    for m, side in enumerate(sides):
        if isinstance(side, BetaBinomialEmission):
            # The factored layout's rising factorials, so every layout scores
            # the same bits (issue #1332); a zero trial count is unobserved.
            trials = covariate.numpy()
            pmf = beta_binomial_log_pmf(
                counts.numpy()[:, None],
                trials,
                side.alpha.detach().numpy(),
                side.beta.detach().numpy(),
            )
            table[:, m, :] = np.where(trials == 0.0, 0.0, pmf)
            continue
        table[:, m, :] = side.log_density(counts, covariate=covariate).numpy()
    return table


def _range_codes(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Each integer's offset from the least, and the value of each offset.

    The codes span the least to the greatest, contiguously, rather than the
    distinct values alone: the range needs no sort, and the distinct values
    need one per call, which is what issue #658 measured as the cost of a
    table.
    """
    codes = np.rint(np.asarray(values, dtype=np.float64)).astype(np.int64)
    low = int(codes.min())
    levels = low + np.arange(int(codes.max()) - low + 1, dtype=np.float64)
    return codes - low, levels


def _uint32(values: np.ndarray, what: str) -> np.ndarray:
    """``values`` as contiguous ``uint32`` rows, refused past what the type carries."""
    if values.size:
        _refuse_rows(int(values.max()) + 1, what)
    return np.ascontiguousarray(values, dtype=np.uint32)


def observation_rows(
    observations: np.ndarray,
    covariate: np.ndarray | None,
    *,
    covariate_rows: CovariateRows = "range",
) -> ObservationRows:
    """Every observation's table row in every channel, for ``covariate_rows``.

    ``(S, n_nodes)`` observations are one channel (issues #1298, #1308): its
    row is the value where every value is a non-negative integer --- a symbol
    or a count --- and the observation's own index ``s * n_nodes + v``
    otherwise. Under an integer covariate and ``range`` or ``distinct``, the
    row is the ``(count, trial count)`` pair's, as for the second channel
    below; a family the pair does not key ignores it and is scored by site
    (:func:`one_channel_table`).
    Without a covariate the row **is** the count, in both channels, and the
    layout does not enter. With one, the total's row is the count and its
    exposure is carried for the kernel; the successes' row is the count and
    the trial count carried under ``factored``, and ``count * n_codes + code``
    under ``range`` and ``distinct``, the code being the trial count's offset
    from the least under ``range`` and its first-appearance index
    (``oxisal.factorize``) under ``distinct``.

    Parameters
    ----------
    observations : np.ndarray
        Shape ``(S, n_nodes, 2)``, integer counts, or ``(S, n_nodes)``, one
        channel.
    covariate : np.ndarray | None
        Shape ``(S, n_nodes, 2)``: the exposure, then the trial count; or
        ``(S, n_nodes)`` beside one channel.
    covariate_rows : CovariateRows
        The trial count's layout; ``range``, the default, is tied fastest
        with ``distinct`` in a fit at stress (``changelog.d/1064.added.md``).

    Returns
    -------
    ObservationRows

    Raises
    ------
    ValueError
        If ``covariate_rows`` is not a :data:`CovariateRows`, an exposure of
        two channels is negative or not finite, their trial count is not a
        non-negative integer, or a row overflows a ``uint32`` index.
    """
    if covariate_rows not in COVARIATE_ROWS:
        msg = f"covariate_rows must be one of {COVARIATE_ROWS}, got {covariate_rows!r}"
        raise ValueError(msg)
    if observations.ndim == 2:
        return _one_channel_rows(observations, covariate, covariate_rows)
    totals, successes = observations[..., TOTAL], observations[..., SUCCESSES]
    total = ChannelRows(_uint32(totals, "the totals"), int(totals.max()) + 1)
    plain = ChannelRows(_uint32(successes, "the successes"), int(successes.max()) + 1)
    if covariate is None:
        return ObservationRows(covariate_rows, observations, None, total, plain)
    exposure = np.ascontiguousarray(covariate[..., TOTAL], dtype=np.float64)
    if bool(((exposure < 0.0) | ~np.isfinite(exposure)).any()):
        msg = "every exposure must be finite and non-negative; zero marks the count unobserved"
        raise ValueError(msg)
    total = ChannelRows(total.rows, total.extent, covariate=exposure)
    return ObservationRows(
        covariate_rows,
        observations,
        covariate,
        total,
        _trial_rows(successes, covariate[..., SUCCESSES], covariate_rows),
    )


def _counts(values: np.ndarray) -> bool:
    """Whether every value is a non-negative integer a ``uint32`` row carries."""
    if values.size == 0:
        return False
    if not np.issubdtype(values.dtype, np.integer):
        if not bool(np.isfinite(values).all()):
            return False
        if not bool((values == np.floor(values)).all()):
            return False
    return float(values.min()) >= 0 and float(values.max()) < _ROW_LIMIT


def _one_channel_rows(
    observations: np.ndarray, covariate: np.ndarray | None, layout: CovariateRows
) -> ObservationRows:
    """One channel's rows: by value, by ``(count, trial count)`` pair, or by site (issue #1308).

    The categorical model's symbols are their own rows (issue #1298), and so
    are counts without a covariate. Values that are not non-negative
    integers have no table by value, and each is its own site.
    """
    if not _counts(observations):
        sites = np.arange(observations.size, dtype=np.int64).reshape(observations.shape)
        _refuse_rows(observations.size, "the sites")
        rows = ChannelRows(
            np.ascontiguousarray(sites, dtype=np.uint32),
            observations.size,
            by_site=True,
        )
        return ObservationRows(layout, observations, covariate, rows, None)
    plain = ChannelRows(
        _uint32(observations, "the counts"), int(observations.max()) + 1
    )
    if covariate is None or layout == "factored" or not _counts(covariate):
        return ObservationRows(layout, observations, covariate, plain, None)
    try:
        keyed = _trial_rows(observations, covariate, layout)
    except ValueError:
        # A pair past a `uint32` row: no family is keyed by it, and each is
        # scored by site.
        keyed = plain
    return ObservationRows(layout, observations, covariate, keyed, None)


def _trial_rows(
    successes: np.ndarray, trials: np.ndarray, layout: CovariateRows
) -> ChannelRows:
    """The successes' rows under a trial count, in ``layout``."""
    import torch

    # The family's own check, so the refusal is the one its `log_density`
    # would make.
    validated_trials(
        torch.from_numpy(np.asarray(trials, dtype=np.float64)),
        torch.ones(1, dtype=torch.float64),
    )
    extent = int(successes.max()) + 1
    if layout == "factored":
        return ChannelRows(
            _uint32(successes, "the successes"),
            extent,
            covariate=_uint32(trials, "the trial counts"),
        )
    if layout == "range":
        codes, levels = _range_codes(trials)
    else:
        flat, levels = oxisal.factorize(
            np.ascontiguousarray(trials, dtype=np.float64).reshape(-1)
        )
        codes = flat.astype(np.int64).reshape(trials.shape)
    what = f"the trial counts' {layout} table"
    _refuse_rows(extent * levels.size, what)
    rows = successes.astype(np.int64) * levels.size + codes
    return ChannelRows(np.ascontiguousarray(rows, dtype=np.uint32), extent, levels)


def _count_table(
    sides: Sequence[EmissionFamily], params: SpatioSequentialParams, extent: int
) -> np.ndarray:
    """Each class's log-density at every count ``0 .. extent - 1``, ``(extent, M, K)``."""
    import torch

    counts = torch.from_numpy(np.arange(extent, dtype=np.float64))
    table = np.empty((extent, params.n_classes, params.n_states))
    for m, side in enumerate(sides):
        if isinstance(side, BetaBinomialEmission):
            # The NumPy pmf every beta-binomial route reads (issue #1332).
            table[:, m, :] = density_table(side, extent)
            continue
        table[:, m, :] = side.log_density(counts).numpy()
    return table


def _total_table(
    families: list[IndependentCountPair],
    params: SpatioSequentialParams,
    rows: ChannelRows,
) -> tuple[np.ndarray, ExposureTerm | None]:
    """The first channel's table, and its exposure term where it carries one."""
    sides = [family.total for family in families]
    if rows.covariate is None:
        return _count_table(sides, params, rows.extent), None
    table = np.empty((rows.extent, params.n_classes, params.n_states))
    for m, side in enumerate(sides):
        table[:, m, :] = exposure_table(side, rows.extent)
    term = ExposureTerm(
        rows.covariate,
        np.ascontiguousarray(np.stack([side.dispersion.numpy() for side in sides])),
        np.ascontiguousarray(np.stack([side.mean.numpy() for side in sides])),
    )
    return table, term


def _success_table(
    families: list[IndependentCountPair],
    params: SpatioSequentialParams,
    rows: ChannelRows,
) -> tuple[np.ndarray, TrialTerm | None]:
    """The second channel's table, and its trial term where it carries one."""
    sides: list[BetaBinomialEmission] = [family.successes for family in families]
    if rows.levels is not None:
        n_rows = rows.extent * rows.levels.size
        _refuse_bytes(n_rows, params, f"the trial counts' {rows.levels.size} codes")
        return _outer_table(sides, params, rows.extent, rows.levels), None
    if rows.covariate is None:
        return _count_table(sides, params, rows.extent), None
    trials_extent = int(rows.covariate.max()) + 1
    _refuse_bytes(trials_extent, params, f"a trial count of {trials_extent - 1}")
    shape = (params.n_classes, params.n_states)
    success = np.empty((rows.extent, *shape))
    failure = np.empty((trials_extent, *shape))
    trial = np.empty((trials_extent, *shape))
    log_rate = np.empty((2, *shape))
    for m, side in enumerate(sides):
        tables = trial_tables(side, rows.extent, trials_extent)
        success[:, m, :] = tables.success
        failure[:, m, :] = tables.failure
        trial[:, m, :] = tables.trial
        log_rate[:, m, :] = tables.log_rate
    term = TrialTerm(
        rows.covariate, failure, trial, log_factorial(trials_extent), log_rate
    )
    return success, term


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
        table is then ``B`` by count (issue #1064).
    trials : TrialTerm | None
        The second channel's factored trial count, where it carries one; its
        table is then ``U`` by successes (issue #1064).
    """

    total_rows: np.ndarray
    success_rows: np.ndarray
    total_table: np.ndarray
    success_table: np.ndarray
    exposure: ExposureTerm | None = None
    trials: TrialTerm | None = None

    def kernel_arguments(self) -> dict[str, np.ndarray]:
        """The factored terms as the kernels' keyword arguments; empty without one."""
        arguments = {} if self.exposure is None else self.exposure.arguments()
        if self.trials is not None:
            arguments.update(self.trials.arguments())
        return arguments

    @property
    def table_bytes(self) -> int:
        """The bytes of every table built from the parameters, rows excluded."""
        return (
            self.total_table.nbytes
            + self.success_table.nbytes
            + (0 if self.trials is None else self.trials.nbytes)
        )

    def __iter__(self) -> Iterator[np.ndarray | ExposureTerm | TrialTerm | None]:
        """The rows, the tables, then the two terms: the order callers unpack."""
        yield from (
            self.total_rows,
            self.success_rows,
            self.total_table,
            self.success_table,
            self.exposure,
            self.trials,
        )


def _rows_for(
    params: SpatioSequentialParams,
    observations: np.ndarray,
    covariate_rows: CovariateRows | ObservationRows,
) -> ObservationRows:
    """``covariate_rows`` if it is already rows for these inputs, else rows built in that layout.

    Raises
    ------
    ValueError
        If ``covariate_rows`` is rows built from other observations or another
        covariate. They are compared by identity, which is what a fit holds.
    """
    if not isinstance(covariate_rows, ObservationRows):
        return observation_rows(
            observations, params.covariate, covariate_rows=covariate_rows
        )
    if (
        covariate_rows.observations is not observations
        or covariate_rows.covariate is not params.covariate
    ):
        msg = (
            "these rows were built from other observations or another covariate; "
            "build them with observation_rows for these"
        )
        raise ValueError(msg)
    return covariate_rows


def emission_tables(
    params: SpatioSequentialParams,
    observations: np.ndarray,
    *,
    covariate_rows: CovariateRows | ObservationRows = "range",
) -> EmissionTables:
    """Both channels' log-density for every class, state and table row.

    Parameters
    ----------
    params : SpatioSequentialParams
        Its emissions the two-channel families, and the covariate the rows are
        tabulated against where it carries one.
    observations : np.ndarray
        Shape ``(S, n_nodes, 2)``.
    covariate_rows : CovariateRows | ObservationRows
        As :func:`emission_rows` takes it.

    Returns
    -------
    EmissionTables
        The first channel's table and the second's. :func:`emission_rows`
        returns these with the row indices and the factored terms, which is
        how the two entry points take them.
    """
    rows = emission_rows(params, observations, covariate_rows=covariate_rows)
    return EmissionTables(rows.total_table, rows.success_table)


def emission_rows(
    params: SpatioSequentialParams,
    observations: np.ndarray,
    *,
    covariate_rows: CovariateRows | ObservationRows = "range",
) -> EmissionRows:
    """The rows and the tables together, since neither is meaningful alone.

    A row index is only interpretable against the table it was built with, so
    the two are returned from one call rather than derived twice from the same
    inputs and trusted to agree (issue #658). The tables are the families' own
    arithmetic whatever the layout, which is the property that keeps this a
    table rather than a second implementation of the oracle.

    Parameters
    ----------
    params : SpatioSequentialParams
        The two-channel families, and the covariate where it carries one.
    observations : np.ndarray
        Shape ``(S, n_nodes, 2)``.
    covariate_rows : CovariateRows | ObservationRows
        The trial count's layout, or rows :func:`observation_rows` already
        built for these observations and this covariate, in which case only
        the tables are built.

    Returns
    -------
    EmissionRows

    Raises
    ------
    ValueError
        As :func:`observation_rows` refuses, if handed rows built for other
        inputs, or if a covariate table passes :data:`COVARIATE_TABLE_CEILING`.
    """
    families = _families(params)
    rows = _rows_for(params, observations, covariate_rows)
    if rows.successes is None:
        msg = (
            "the two-channel count emission takes (S, n_nodes, 2) observations; "
            f"these are {observations.shape}"
        )
        raise ValueError(msg)
    total_table, exposure = _total_table(families, params, rows.total)
    success_table, trials = _success_table(families, params, rows.successes)
    return EmissionRows(
        rows.total.rows,
        rows.successes.rows,
        np.ascontiguousarray(total_table),
        np.ascontiguousarray(success_table),
        exposure,
        trials,
    )


#: The key :attr:`ObservationRows.tables` holds a symbol table under.
_SYMBOL = "symbol"


@dataclass(frozen=True)
class SymbolTable:
    """The categorical model's one channel: the symbols and their table (issue #1298).

    Its own type rather than :class:`EmissionRows` with the second channel
    absent: the two are different inputs to the kernel, and a reader of
    either should not have to ask which it holds.

    Parameters
    ----------
    symbols : np.ndarray
        ``(S, n_nodes)`` contiguous ``uint32``, each observation's symbol and
        so its table row.
    table : np.ndarray
        ``table[x, m, k] = log B_m[k, x]``, ``(n_symbols, M, K)`` contiguous
        ``float64``.
    """

    symbols: np.ndarray
    table: np.ndarray


def symbol_table(
    params: SpatioSequentialParams,
    observations: np.ndarray,
    *,
    covariate_rows: CovariateRows | ObservationRows = "range",
) -> SymbolTable:
    """The categorical model's symbols and table, the kernel's one-channel input.

    The table is each family's stored log emission matrix, transposed, so
    every score is its ``log_density`` bit for bit.

    Parameters
    ----------
    params : SpatioSequentialParams
        Every class a :class:`~sal.emissions.CategoricalEmission` over one
        alphabet.
    observations : np.ndarray
        Shape ``(S, n_nodes)``, symbols.
    covariate_rows : CovariateRows | ObservationRows
        As :func:`emission_rows` takes it; the layout does not enter one
        channel by symbol. Rows already built keep the last table, as
        :func:`one_channel_table`'s do (issue #1308).

    Returns
    -------
    SymbolTable

    Raises
    ------
    TypeError
        If a class carries no categorical family.
    ValueError
        If the alphabets differ, the observations carry two channels or a
        covariate, or a symbol lies outside the alphabet, as the family's
        ``validate`` refuses it.
    """
    families = _categorical(params)
    if families is None:
        msg = "a symbol table is over the categorical emission in every class"
        raise TypeError(msg)
    rows = _rows_for(params, observations, covariate_rows)
    held = rows.tables.get(_SYMBOL)
    if held is not None and held[0] is params:
        return cast(SymbolTable, held[1])
    if rows.successes is not None:
        msg = (
            "the categorical model takes (S, n_nodes) symbols; these observations "
            "carry two channels"
        )
        raise ValueError(msg)
    if params.covariate is not None:
        msg = "the categorical model's symbols take no covariate"
        raise ValueError(msg)
    families[0].validate(observations)
    table = np.empty((families[0].n_symbols, params.n_classes, params.n_states))
    for m, family in enumerate(families):
        table[:, m, :] = family.log_matrix.t().numpy()
    built = SymbolTable(rows.total.rows, table)
    if isinstance(covariate_rows, ObservationRows):
        rows.tables[_SYMBOL] = (params, built)
    return built


#: What a one-channel table's row is keyed on (issue #1308): the count, the
#: ``(count, trial count)`` pair of :data:`CovariateRows`'s ``range`` or
#: ``distinct`` layout, or the site ``s * n_nodes + v``.
RowKey = Literal["count", "pair", "site"]


@dataclass(frozen=True)
class OneChannelTable:
    """A one-channel model's rows and table, the kernel's one-channel input (issue #1308).

    Parameters
    ----------
    rows : np.ndarray
        ``(S, n_nodes)`` contiguous ``uint32``, each observation's table row.
    table : np.ndarray
        ``table[row, m, k]``, the log-density under state ``k`` of class
        ``m``, ``(n_rows, M, K)`` contiguous ``float64``.
    key : RowKey
        What a row is keyed on.
    """

    rows: np.ndarray
    table: np.ndarray
    key: RowKey


def _row_key(
    family: EmissionFamily, params: SpatioSequentialParams, rows: ChannelRows
) -> RowKey:
    """The row one class's family is tabulated by.

    A negative binomial or beta-binomial without a covariate is a function of
    the count; a beta-binomial under a trial count, of the pair where the
    layout keys it. Everything else --- the Gaussian, a continuous exposure,
    the ``factored`` layout, any other family --- is scored by site.
    """
    if rows.by_site:
        return "site"
    if isinstance(family, NegativeBinomialEmission | BetaBinomialEmission):
        if params.covariate is None:
            return "count"
        if isinstance(family, BetaBinomialEmission) and rows.levels is not None:
            return "pair"
    return "site"


def _site_table(
    params: SpatioSequentialParams,
    observations: np.ndarray,
    labels: np.ndarray | None,
) -> np.ndarray:
    """Each site's log-density under every class and state, ``(S * n_nodes, M, K)``.

    Each family's own ``log_density`` at the site's own covariate, the
    oracle's arithmetic. With ``labels``, a class is scored at its members
    alone, which is all the E step reads, and its other entries are zero.
    """
    import torch

    n_positions, n_nodes = observations.shape[:2]
    _refuse_bytes(n_positions * n_nodes, params, f"{n_positions * n_nodes} sites")
    table = np.zeros((n_positions, n_nodes, params.n_classes, params.n_states))
    every = np.arange(n_nodes)
    for m, family in enumerate(params.emissions):
        members = every if labels is None else np.flatnonzero(labels == m)
        if members.size == 0:
            continue
        scores = family.log_density(
            torch.as_tensor(observations[:, members], dtype=family.observation_dtype),
            covariate=covariate_block(params, members),
        )  # (S, members, K)
        table[:, members, m, :] = scores.detach().numpy()
    return table.reshape(n_positions * n_nodes, params.n_classes, params.n_states)


def _keyed_table(
    params: SpatioSequentialParams, observations: np.ndarray, rows: ChannelRows
) -> OneChannelTable | None:
    """The table by count or by pair, where every class is keyed alike and it is the smaller.

    ``None`` where a class is scored by site, or where the keyed table has
    more rows than there are sites: it is then more density evaluations than
    scoring each site, and the cost is the data's, not its size's. At the
    stress instance of ``spatio_sequential`` with trial counts in ``[10,
    40)``, the pair table is 1,230 rows against 400 sites.
    """
    keys = {_row_key(family, params, rows) for family in params.emissions}
    if len(keys) != 1:
        return None
    key = keys.pop()
    n_sites = observations.shape[0] * observations.shape[1]
    if key == "count" and rows.extent <= n_sites:
        table = _count_table(params.emissions, params, rows.extent)
        return OneChannelTable(rows.rows, table, key)
    if key == "pair":
        assert rows.levels is not None
        n_rows = rows.extent * rows.levels.size
        size = n_rows * params.n_classes * params.n_states * 8
        if n_rows <= n_sites and size <= COVARIATE_TABLE_CEILING:
            table = _outer_table(params.emissions, params, rows.extent, rows.levels)
            return OneChannelTable(rows.rows, table, key)
    return None


def _may_key(
    params: SpatioSequentialParams, covariate_rows: CovariateRows | ObservationRows
) -> bool:
    """Whether a table by count or by pair can serve every class, read from the types alone.

    Every class a negative binomial or a beta-binomial, and under a covariate
    every class a beta-binomial in a layout that keys the pair. Anything else
    is scored by site whatever the rows hold, and they are not built for it.
    """
    if params.covariate is None:
        return all(
            isinstance(family, NegativeBinomialEmission | BetaBinomialEmission)
            for family in params.emissions
        )
    layout = (
        covariate_rows.layout
        if isinstance(covariate_rows, ObservationRows)
        else covariate_rows
    )
    return layout != "factored" and all(
        isinstance(family, BetaBinomialEmission) for family in params.emissions
    )


#: The key :attr:`ObservationRows.tables` holds a one-channel table under.
_ONE_CHANNEL = "one_channel"


def one_channel_table(
    params: SpatioSequentialParams,
    observations: np.ndarray,
    labels: np.ndarray | None = None,
    *,
    covariate_rows: CovariateRows | ObservationRows = "range",
) -> OneChannelTable:
    """A one-channel model's rows and table, every class's family tabulated (issue #1308).

    One kernel path serves every family; what differs is the row. Where
    every class is a negative binomial or a beta-binomial keyed alike --- by
    count, or by ``(count, trial count)`` pair --- and that table has no more
    rows than there are sites, the table is over those, built by the families
    themselves. Otherwise every class is scored by site, so the tables align
    whatever families the classes mix. The categorical model has its own
    table, :func:`symbol_table`.

    Parameters
    ----------
    params : SpatioSequentialParams
        One-channel families, and the covariate where it carries one.
    observations : np.ndarray
        Shape ``(S, n_nodes)``, or ``(S, n_nodes, ...)`` for a family with
        axes of its own, which is scored by site.
    labels : np.ndarray | None
        Where given, a by-site table scores each class at its members alone,
        the E step's reads; ``None`` scores every class at every site, the
        field's. Rows already built ignore it: their table serves every call
        under the same parameters, and is built whole.
    covariate_rows : CovariateRows | ObservationRows
        As :func:`emission_rows` takes it. Rows already built keep the last
        table in :attr:`ObservationRows.tables`, and return it while the
        parameters are the same object.

    Returns
    -------
    OneChannelTable

    Raises
    ------
    ValueError
        If ``covariate_rows`` is not a :data:`CovariateRows`, or a by-site
        table passes :data:`COVARIATE_TABLE_CEILING`.
    """
    built = isinstance(covariate_rows, ObservationRows)
    if built:
        rows = _rows_for(params, observations, covariate_rows)
        held = rows.tables.get(_ONE_CHANNEL)
        if held is not None and held[0] is params:
            return cast(OneChannelTable, held[1])
    elif covariate_rows not in COVARIATE_ROWS:
        msg = f"covariate_rows must be one of {COVARIATE_ROWS}, got {covariate_rows!r}"
        raise ValueError(msg)
    table = None
    if _may_key(params, covariate_rows):
        rows = _rows_for(params, observations, covariate_rows)
        table = _keyed_table(params, observations, rows.total)
    if table is None:
        n_positions, n_nodes = observations.shape[:2]
        sites = np.arange(n_positions * n_nodes, dtype=np.uint32).reshape(
            n_positions, n_nodes
        )
        scored = _site_table(params, observations, None if built else labels)
        table = OneChannelTable(sites, scored, "site")
    if built:
        rows.tables[_ONE_CHANNEL] = (params, table)
    return table


@dataclass(frozen=True)
class _KernelInputs:
    """What both kernels take: the rows, the tables and the factored terms."""

    totals: np.ndarray
    successes: np.ndarray | None
    total_table: np.ndarray
    success_table: np.ndarray | None
    terms: dict[str, np.ndarray]


def _kernel_inputs(
    params: SpatioSequentialParams,
    observations: np.ndarray,
    covariate_rows: CovariateRows | ObservationRows,
    labels: np.ndarray | None = None,
) -> _KernelInputs:
    """The kernels' arrays, flat: one channel by row, or two by count.

    ``successes`` and ``success_table`` are ``None`` for a one-channel model,
    which is how the kernel is told it has one channel. ``labels`` is as
    :func:`one_channel_table` takes it.
    """
    if not _refuse_mixed_channels(params):
        if _categorical(params) is not None:
            symbols = symbol_table(params, observations, covariate_rows=covariate_rows)
            return _KernelInputs(
                symbols.symbols.reshape(-1), None, symbols.table.reshape(-1), None, {}
            )
        one = one_channel_table(
            params, observations, labels, covariate_rows=covariate_rows
        )
        return _KernelInputs(
            one.rows.reshape(-1), None, one.table.reshape(-1), None, {}
        )
    rows = emission_rows(params, observations, covariate_rows=covariate_rows)
    return _KernelInputs(
        rows.total_rows.reshape(-1),
        rows.success_rows.reshape(-1),
        rows.total_table.reshape(-1),
        rows.success_table.reshape(-1),
        rows.kernel_arguments(),
    )


def class_posteriors(
    params: SpatioSequentialParams,
    observations: np.ndarray,
    labels: np.ndarray,
    *,
    covariate_rows: CovariateRows | ObservationRows = "range",
) -> ClassPosteriors:
    """Forward--backward on every class's chain over its members' summed scores.

    The signature and the return of
    :func:`sal.likelihood.spatio_sequential.class_posteriors`,
    which is the oracle this is pinned to.

    Parameters
    ----------
    params : SpatioSequentialParams
        The model: the two-channel count family in every class, or one-channel
        families.
    observations : np.ndarray
        Shape ``(S, n_nodes, 2)``, integer counts, or ``(S, n_nodes)``.
    labels : np.ndarray
        One class per vertex, shape ``(n_nodes,)``.
    covariate_rows : CovariateRows | ObservationRows
        As :func:`emission_rows` takes it.

    Returns
    -------
    ClassPosteriors
        The posterior ``(M, S, K)``, the pairwise ``(M, S - 1, K, K)`` and the
        per-class log evidence ``(M,)``.
    """
    labels = np.ascontiguousarray(labels, dtype=np.int64)
    inputs = _kernel_inputs(params, observations, covariate_rows, labels)
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
        inputs.totals,
        inputs.successes,
        labels,
        inputs.total_table,
        inputs.success_table,
        np.ascontiguousarray(np.log(params.initial)).reshape(-1),
        log_transition.reshape(-1),
        n_positions,
        n_nodes,
        params.n_classes,
        params.n_states,
        posterior.reshape(-1),
        pairwise.reshape(-1),
        log_evidence,
        **inputs.terms,
    )
    return ClassPosteriors(posterior, pairwise, log_evidence)


def external_field(
    params: SpatioSequentialParams,
    observations: np.ndarray,
    labels: np.ndarray,
    posterior: np.ndarray | None = None,
    *,
    covariate_rows: CovariateRows | ObservationRows = "range",
) -> np.ndarray:
    """``H[n, m]``, minus the posterior-expected emission score, shape ``(n_nodes, M)``.

    The signature and the return of
    :func:`sal.likelihood.spatio_sequential.external_field`.
    ``posterior`` defaults to this module's own E step at ``labels``, over
    the same rows; ``covariate_rows`` is as :func:`emission_rows` takes it.

    Returns
    -------
    np.ndarray
        Shape ``(n_nodes, M)``.
    """
    built = _rows_for(params, observations, covariate_rows)
    if posterior is None:
        posterior = class_posteriors(
            params, observations, labels, covariate_rows=built
        ).posterior
    inputs = _kernel_inputs(params, observations, built)
    n_positions, n_nodes = observations.shape[:2]
    field = np.empty((n_nodes, params.n_classes))
    # (M, S, K) to (S, M, K): the kernel wants one position's weights
    # contiguous beside the tables' rows, which are count-major.
    weights = np.ascontiguousarray(np.moveaxis(posterior, 0, 1))
    oxisal.external_field(
        inputs.totals,
        inputs.successes,
        inputs.total_table,
        inputs.success_table,
        weights.reshape(-1),
        n_positions,
        n_nodes,
        params.n_classes,
        params.n_states,
        field.reshape(-1),
        **inputs.terms,
    )
    return field


def labelled_log_likelihood(
    params: SpatioSequentialParams,
    observations: np.ndarray,
    labels: np.ndarray,
    *,
    covariate_rows: CovariateRows | ObservationRows = "range",
) -> float:
    """``log p(x, l | theta)`` with the chains marginalized, up to ``log Z_Potts``.

    The signature and the return of
    :func:`sal.likelihood.spatio_sequential.labelled_log_likelihood`,
    over this module's E step. The Potts term is ``oxisal.coupled_log_prior``,
    the oracle's :func:`~sal.likelihood.spatio_sequential.log_prior` term by
    term and in its order, so bit for bit (issue #1308): at the stress
    instance of ``spatio_sequential``, 180 edges, 1.21 us against the
    vectorised NumPy's 13.1 us and the loop over edges' 543 us.
    ``covariate_rows`` is as :func:`emission_rows` takes it.

    Returns
    -------
    float
    """
    graph = params.graph
    own = oxisal.coupled_log_prior(
        np.ascontiguousarray(labels, dtype=np.int64),
        graph.edge_index.reshape(-1),
        graph.edge_coupling,
        float(params.beta),
    )
    return own + float(
        class_posteriors(
            params, observations, labels, covariate_rows=covariate_rows
        ).log_evidence.sum()
    )
