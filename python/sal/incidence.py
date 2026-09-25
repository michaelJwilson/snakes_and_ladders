"""One compressed-row incidence, for every relation the package stores.

A relation between two index sets -- bits and checks, sites and sites,
variables and factors -- is an edge list, and every consumer of one wants
the same three things: the entries of a row contiguously, the degrees, and
the same relation read the other way round. Three modules derived that
layout separately (``sim.ldpc.ParityCheck`` by ``lexsort`` and
``searchsorted``, ``sim.graph.PottsGraph`` by a stable ``argsort`` on each
call, ``sim.factor_graph.FactorGraph`` not at all, scanning every factor per
query), and the survey issue #586 ran found them by their field names rather
than by anyone recalling it. :class:`SparseIncidence` is that layout,
written once.

**The layout.** ``indices[offsets[r]:offsets[r + 1]]`` are the columns of row
``r``, and ``order`` is the permutation that put the caller's pair list into
this row-major order --- so a per-entry array the caller holds in *its* order
is read in this one by :meth:`SparseIncidence.gather`, with no second copy
and no dictionary. That is the layout rule root ``CLAUDE.md`` states: one
contiguous array walked in stride order, not a list of lists, and what a
compiled kernel takes across the FFI boundary without marshalling per row.

**Stability is part of the contract.** The sort is stable, so entries within
a row keep the order the caller supplied them in. Both consumers depend on
it: ``PottsGraph.compressed_adjacency`` promises each node's neighbours in
the graph's edge order, because a sweep consumes one random draw per site in
that order and a permutation would change which draw a site sees without
changing any distribution a chi-square could catch; and ``ParityCheck``'s
two orientations are related by one permutation only because each is stable.

**Transpose, not a second build.** :meth:`transpose` returns the same
relation with the roles swapped, together with the permutation from this
layout's entry order into that one. A decoder holding one message per edge
reads it from both ends through that permutation rather than keeping a
second copy, which is what ``ParityCheck.check_order`` already was.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np


def _row_major(
    n_rows: int, rows: np.ndarray, columns: np.ndarray, *, ascending: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The counting sort itself: ``(offsets, indices, order, degrees)``.

    Written once and called twice --- by the public builder, which validates
    first, and by :meth:`SparseIncidence.transpose`, whose inputs came from
    an incidence that was already validated and which would otherwise pay
    four range reductions to re-establish what it knows. Duplicating the
    sort to avoid that would be the thing issue #586 is about.
    """
    degrees = np.bincount(rows, minlength=n_rows)
    offsets = np.zeros(n_rows + 1, dtype=np.int64)
    np.cumsum(degrees, out=offsets[1:])
    order = (
        np.lexsort((columns, rows)) if ascending else np.argsort(rows, kind="stable")
    )
    # `columns[order]` is fancy-indexed, so it is already a fresh
    # C-contiguous array; wrapping it in `ascontiguousarray` was a call
    # per build that copied nothing.
    return offsets, columns[order], order, degrees


@dataclass(frozen=True)
class SparseIncidence:
    """A relation between ``n_rows`` and ``n_cols`` indices, row-major.

    Parameters
    ----------
    n_rows, n_cols : int
        The two index sets. ``n_cols`` is carried so :meth:`transpose` needs
        no second argument and an out-of-range column is caught at the build.
    offsets : np.ndarray
        ``int64``, length ``n_rows + 1``: the start of each row in
        ``indices``, and ``offsets[-1] == n_entries``.
    indices : np.ndarray
        ``int64``, length ``n_entries``: the column of each entry, rows
        consecutive and ascending.
    order : np.ndarray
        ``int64``, length ``n_entries``: entry ``e`` of this layout is the
        caller's pair ``order[e]``. The permutation, not its inverse, so
        :meth:`gather` is one fancy index.
    """

    n_rows: int
    n_cols: int
    offsets: np.ndarray
    indices: np.ndarray
    order: np.ndarray

    @classmethod
    def from_pairs(
        cls,
        n_rows: int,
        n_cols: int,
        rows: np.ndarray,
        columns: np.ndarray,
        *,
        every_row: bool = False,
        distinct: bool = False,
        ascending: bool = False,
    ) -> SparseIncidence:
        """Build from the entries ``(rows[e], columns[e])`` in any order.

        Parameters
        ----------
        rows, columns : np.ndarray
            One-dimensional and the same length; the entries of the relation.
        every_row : bool
            Refuse a row with no entry. A segmented reduction over an empty
            segment returns its neighbour's value rather than an identity, so
            a consumer that reduces per row must ask for this rather than
            compute on the case; one that does not (a graph may hold an
            isolated node) leaves it false.
        distinct : bool
            Refuse a repeated ``(row, column)``. False by default, because a
            periodic lattice of extent two along a dimension legitimately
            produces the same pair twice --- the "+1" and "-1" neighbour
            coincide, which is a doubled bond and not a duplicate.
        ascending : bool
            Order the entries within a row by column, rather than keeping the
            caller's order. The two consumers want opposite things and both
            are contracts, so neither is the default silently: a parity check
            reads its rows ascending, and a Potts sweep needs each node's
            neighbours in the graph's edge order, because it consumes one
            draw per site in that order.

        Raises
        ------
        ValueError
            If the two arrays disagree in shape or are not one-dimensional,
            an index lies outside its set, or one of the two options above is
            asked for and violated.
        """
        rows = np.asarray(rows, dtype=np.int64).reshape(-1)
        columns = np.asarray(columns, dtype=np.int64).reshape(-1)
        if rows.shape != columns.shape:
            msg = (
                f"rows and columns must be the same length, got "
                f"{rows.size} and {columns.size}"
            )
            raise ValueError(msg)
        if n_rows < 0 or n_cols < 0:
            msg = f"an incidence has non-negative extents, got {n_rows} x {n_cols}"
            raise ValueError(msg)
        if rows.size and (
            rows.min() < 0
            or rows.max() >= n_rows
            or columns.min() < 0
            or columns.max() >= n_cols
        ):
            msg = f"an entry lies outside the {n_rows} x {n_cols} relation"
            raise ValueError(msg)
        offsets, indices, order, degrees = _row_major(
            n_rows, rows, columns, ascending=ascending
        )
        if every_row and not degrees.all():
            empty = int(np.flatnonzero(degrees == 0)[0])
            msg = f"every row must carry an entry; row {empty} carries none"
            raise ValueError(msg)
        if distinct:
            # Under `ascending` the keys are sorted by construction, so equal
            # entries are adjacent and the test is a comparison. Without it
            # they need not be, and an adjacency test would pass the
            # duplicate it exists to refuse, so the keys are sorted first.
            keys = (
                rows[order] * n_cols + indices
                if ascending
                else np.sort(rows * n_cols + columns)
            )
            if np.any(keys[1:] == keys[:-1]):
                msg = "an entry (row, column) is listed twice"
                raise ValueError(msg)
        incidence = cls(n_rows, n_cols, offsets, indices, order)
        incidence.__dict__["degrees"] = degrees
        return incidence

    @property
    def n_entries(self) -> int:
        """The entries of the relation."""
        return int(self.indices.size)

    @cached_property
    def degrees(self) -> np.ndarray:
        """Entries per row, length ``n_rows``.

        Cached, and seeded by the builder, which counted them to place the
        offsets: five consumers asked for them per build, and ``np.diff``
        over the offsets was the second line of the profile.
        """
        return np.asarray(np.diff(self.offsets))

    @cached_property
    def rows(self) -> np.ndarray:
        """The row owning each entry, length ``n_entries``: ``offsets`` expanded.

        Cached because :meth:`transpose` and every consumer that wants the
        entries as pairs ask for the same expansion, and it is the one part
        of the layout that is not already stored.
        """
        return np.repeat(np.arange(self.n_rows, dtype=np.int64), self.degrees)

    def row(self, index: int) -> np.ndarray:
        """The columns of row ``index``, a view into ``indices``."""
        return np.asarray(self.indices[self.offsets[index] : self.offsets[index + 1]])

    def gather(self, values: np.ndarray) -> np.ndarray:
        """``values``, one per caller pair, read in this layout's entry order."""
        values = np.asarray(values)
        if values.shape[0] != self.n_entries:
            msg = (
                f"values carries {values.shape[0]} entries for "
                f"{self.n_entries} in the relation"
            )
            raise ValueError(msg)
        return np.asarray(values[self.order])

    def transpose(
        self, *, every_row: bool = False
    ) -> tuple[SparseIncidence, np.ndarray]:
        """The relation the other way round, and the permutation that reads it.

        Returns
        -------
        tuple[SparseIncidence, np.ndarray]
            The transposed incidence, whose ``order`` is into the same caller
            pair list as this one's; and ``read``, so that ``values[read]`` is
            a per-entry array *held in this layout's order* read in the
            transpose's --- what lets a decoder keep one message per edge and
            no second copy. The read direction rather than its inverse,
            because that is the one every consumer indexes with, and an
            ``argsort`` to produce it would be paid per build.
        """
        offsets, indices, read, degrees = _row_major(
            self.n_cols, self.indices, self.rows, ascending=False
        )
        if every_row and not degrees.all():
            empty = int(np.flatnonzero(degrees == 0)[0])
            msg = f"every row must carry an entry; row {empty} carries none"
            raise ValueError(msg)
        # `read` is into this layout's entry order, because it sorted this
        # layout's arrays: that is the read permutation. The transpose's own
        # `order` is restated against the caller's original pairs, so both
        # orientations index one pair list.
        other = SparseIncidence(
            self.n_cols, self.n_rows, offsets, indices, self.order[read]
        )
        other.__dict__["degrees"] = degrees
        return other, read

    def dense(self) -> np.ndarray:
        """The relation as an ``(n_rows, n_cols)`` count array, for small cases.

        Counts rather than zeros and ones, so a doubled entry reads as two;
        an oracle that asserts a matrix over GF(2) takes it modulo two.
        """
        matrix = np.zeros((self.n_rows, self.n_cols), dtype=np.int64)
        np.add.at(matrix, (self.rows, self.indices), 1)
        return matrix


__all__ = ["SparseIncidence"]
