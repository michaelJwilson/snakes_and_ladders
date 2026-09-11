"""Site-pattern compression: the alignment as unique columns with integer weights (issue #408).

``eq:site-independence`` of ``docs/tex/textbook.tex`` makes the log-likelihood
a sum over sites of a term that reads only that site's column. Two identical
columns therefore contribute the same term twice, and the sum over ``L``
columns equals a weighted sum over the ``P`` distinct ones::

    sum_s log Pr(x_s) = sum_p w_p log Pr(y_p)

with ``w_p`` the number of columns equal to pattern ``p``. The identity is
exact -- the arithmetic is reassociated, not approximated -- and the work
falls from ``L`` to ``P`` columns. ``P`` is bounded by ``k ** n_leaves`` and
by ``L``, so the win grows with the alignment and saturates once every
distinct column has appeared: at the four-taxon fixtures ``P`` is at most 256
whatever ``L`` is.

**The compression lives here and each backend takes weights**: the pattern
table is a property of the alignment, and a backend that built its own would
have to be checked against the others for agreeing on what a pattern is.
``pruning``, ``pruning_torch`` and ``pruning_rust`` all take a ``weights``
argument of the shape :func:`compress` returns;
:attr:`SitePatterns.alignment` is the compressed alignment they read
alongside it.

Weights are exact counts, not a normalisation, so ``weights.sum()`` is the
site count and a fractional weight is never introduced. The Rust binding
cannot take them (``pruning_rust`` says how it works around that), which is
the one place the compression is not carried by the kernel itself.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SitePatterns:
    """An alignment's distinct columns and how often each occurs.

    Parameters
    ----------
    names : tuple[str, ...]
        Leaf names, in the row order of ``columns``. Sorted, so two
        alignments over the same taxa compress to comparable tables.
    columns : np.ndarray
        Shape ``(n_leaves, n_patterns)``, ``int64``: one column per distinct
        pattern, in first-occurrence order of the original alignment.
    weights : np.ndarray
        Shape ``(n_patterns,)``, ``int64``: how many columns of the original
        alignment equal each pattern. Sums to ``n_sites``.
    n_sites : int
        Sites the weights sum to --- the original alignment's length.
    """

    names: tuple[str, ...]
    columns: np.ndarray
    weights: np.ndarray
    n_sites: int

    @property
    def n_patterns(self) -> int:
        """Distinct columns."""
        return int(self.columns.shape[1])

    @property
    def compression_ratio(self) -> float:
        """``n_sites / n_patterns``: the factor by which the column work falls.

        It bounds the win rather than stating it, since a backend pays costs
        that do not scale with the column count.
        """
        return self.n_sites / self.n_patterns

    @property
    def alignment(self) -> dict[str, np.ndarray]:
        """The pattern table in the mapping form a backend's ``alignment`` takes."""
        return {
            name: np.ascontiguousarray(self.columns[row])
            for row, name in enumerate(self.names)
        }


def compress(alignment: Mapping[str, np.ndarray]) -> SitePatterns:
    """Collapse identical alignment columns to distinct patterns with counts.

    Parameters
    ----------
    alignment : Mapping[str, np.ndarray]
        Leaf name to its observed states, each of shape ``(n_sites,)`` --
        what ``snakes_and_ladders.sim.simulate.SimulatedDataset.alignment``
        produces.

    Returns
    -------
    SitePatterns
        The distinct columns, their counts, and the site count they sum to.

    Raises
    ------
    ValueError
        If ``alignment`` is empty or ragged. A ragged alignment has no
        columns to compress, and silently truncating to the shortest row
        would drop data.
    """
    if not alignment:
        msg = "an alignment with no taxa has no columns to compress"
        raise ValueError(msg)
    names = tuple(sorted(alignment))
    rows = [np.asarray(alignment[name], dtype=np.int64) for name in names]
    shapes = {row.shape for row in rows}
    if len(shapes) != 1 or len(next(iter(shapes))) != 1:
        msg = f"the alignment is ragged: rows have shapes {sorted(shapes)}"
        raise ValueError(msg)
    return compress_columns(names, np.ascontiguousarray(np.stack(rows)))


def compress_columns(
    names: tuple[str, ...],
    columns: np.ndarray,
    weights: np.ndarray | None = None,
) -> SitePatterns:
    """Collapse identical columns of a raw block, adding the weights they carry.

    :func:`compress` is this over an alignment mapping with every weight
    one. The weighted form is what a caller holding columns that already
    carry multiplicities needs --- the block-frequency bound's frequent
    blocks, whose columns repeat both inside a block and across blocks
    (:mod:`snakes_and_ladders.likelihood.blocks`).

    Parameters
    ----------
    names : tuple[str, ...]
        Leaf names in the row order of ``columns``.
    columns : np.ndarray
        Shape ``(len(names), n_columns)``, ``int64``.
    weights : np.ndarray | None
        One multiplicity per column, or ``None`` for one each.

    Returns
    -------
    SitePatterns
        The distinct columns and the summed weights, ``n_sites`` being the
        total multiplicity rather than the number of columns passed.

    Raises
    ------
    ValueError
        If ``columns`` is not two-dimensional with one row per name, or
        ``weights`` does not have one entry per column.
    """
    if columns.ndim != 2 or columns.shape[0] != len(names):
        msg = (
            f"columns has shape {columns.shape}, expected "
            f"({len(names)}, n_columns), one row per name"
        )
        raise ValueError(msg)
    multiplicity = (
        np.ones(columns.shape[1], dtype=np.int64)
        if weights is None
        else np.asarray(weights, dtype=np.int64)
    )
    if multiplicity.shape != (columns.shape[1],):
        msg = (
            f"weights has shape {multiplicity.shape}, expected "
            f"({columns.shape[1]},), one per column"
        )
        raise ValueError(msg)

    # np.unique over the columns: the transposed view is made contiguous once
    # so the row-wise uniqueness test walks memory in stride order rather
    # than gathering one taxon at a time (root CLAUDE.md, "Memory layout").
    _, first, inverse = np.unique(
        np.ascontiguousarray(columns.T),
        axis=0,
        return_index=True,
        return_inverse=True,
    )
    totals = np.bincount(
        inverse.reshape(-1), weights=multiplicity, minlength=first.shape[0]
    )
    # First-occurrence order, so a pattern table read beside the original
    # alignment lists patterns in the order the alignment introduces them.
    order = np.argsort(first)
    return SitePatterns(
        names=names,
        columns=np.ascontiguousarray(columns[:, first[order]]),
        weights=np.ascontiguousarray(np.rint(totals[order]).astype(np.int64)),
        n_sites=int(multiplicity.sum()),
    )


def check_weights(weights: np.ndarray | None, n_sites: int) -> np.ndarray | None:
    """Validate a backend's ``weights`` argument against its column count.

    Every backend takes the same argument under the same rules, and this is
    the one statement of them.

    Parameters
    ----------
    weights : np.ndarray | None
        Per-column weights, or ``None`` for unweighted -- every column once,
        which is what an uncompressed alignment means.
    n_sites : int
        Columns the caller passed.

    Returns
    -------
    np.ndarray | None
        ``weights`` as a ``float64`` array of shape ``(n_sites,)``, or
        ``None`` unchanged.

    Raises
    ------
    ValueError
        If ``weights`` does not have one entry per column, or any entry is
        negative. A negative weight is not a count of anything, and it would
        turn a log-likelihood into a quantity with no sign.
    """
    if weights is None:
        return None
    array = np.asarray(weights, dtype=np.float64)
    if array.shape != (n_sites,):
        msg = f"weights has shape {array.shape}, expected ({n_sites},), one per column"
        raise ValueError(msg)
    if bool(np.any(array < 0.0)):
        msg = "weights must be non-negative: a column cannot occur a negative number of times"
        raise ValueError(msg)
    return array


__all__ = ["SitePatterns", "check_weights", "compress", "compress_columns"]
