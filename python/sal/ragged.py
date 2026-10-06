"""Chains of unequal length, laid end to end with the lengths beside them.

Issue #666. A batch of chains is not one long chain: the recursions restart at
each boundary, at the initial distribution rather than at the transition. The
number of boundaries is therefore part of the problem, not of its size --- one
segment of length 400 and twenty of length 20 are different problems at the
same total, because a state change *across* a boundary is free where the same
change one step earlier costs a transition.

**This is the only batch form.** A rectangular batch is a `Ragged` whose
lengths happen to be equal, and the entry points that take an ``(n, T, ...)``
array convert rather than branch. Two implementations of one recursion is the
drift this avoids; the conserved rectangular route lives in `sandbox` and
referees the equal-length case bit for bit.

**A segment of one position is admitted** (issue #1233). It reads no
transition: its posterior is ``initial * emission`` normalised, its Viterbi
state that product's arg max, and its log evidence the log-sum-exp of the
same, and it adds nothing to the transition counts. Every recursion over a
`Ragged` --- forward, backward, Viterbi, sampling, in NumPy, Rust, torch and
JAX --- is pinned to brute-force enumeration on lengths 1, 2 and 7. Only an
empty segment is refused.

The layout is the compressed-row form this package already uses wherever it
stores a relation: one array, and offsets into it.

**A reduction and a floor live beside the layout** (issue #1141). `Ragged.reduce`
is ``ufunc.reduceat`` on the segment starts, which every caller otherwise
restates; `Ragged.floored` merges adjacent segments greedily until each reaches
a floor on extent and on summed weight, never across a change of group. Both
leave `values` as it is: segments are contiguous, so a merge is a relabelling of
the lengths and nothing is copied. The rule reads lengths, weights, groups and
extents and never `values`, so it is :func:`floor_lengths`, which a caller
holding only lengths calls directly (issue #1233).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from itertools import accumulate

import numpy as np
from numpy.typing import ArrayLike

#: The shortest segment allowed: one position, an initial distribution times an
#: emission and no transition (issue #1233).
MINIMUM_LENGTH = 1


@dataclass(frozen=True)
class Ragged:
    """Segments of a flat array, addressed by their lengths.

    Parameters
    ----------
    values : np.ndarray
        The segments end to end, shape ``(total, ...)``. Whatever trailing axes
        an observation carries are its own and are not read here.
    lengths : tuple[int, ...]
        One length per segment, each at least `MINIMUM_LENGTH` (one), summing
        to ``values.shape[0]``.
    """

    values: np.ndarray
    lengths: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.lengths:
            msg = "a ragged batch needs at least one segment, got none"
            raise ValueError(msg)
        short = [
            (index, length)
            for index, length in enumerate(self.lengths)
            if length < MINIMUM_LENGTH
        ]
        if short:
            index, length = short[0]
            msg = (
                f"segment {index} has length {length}; a segment carries at "
                f"least {MINIMUM_LENGTH} position"
            )
            raise ValueError(msg)
        total = int(sum(self.lengths))
        if self.values.shape[0] != total:
            msg = (
                f"lengths sum to {total} and values has {self.values.shape[0]} "
                "rows; the segments must tile the array exactly"
            )
            raise ValueError(msg)

    @property
    def offsets(self) -> tuple[int, ...]:
        """Where each segment starts, and where the last one ends."""
        return tuple(accumulate(self.lengths, initial=0))

    def _edges(self) -> np.ndarray:
        """`offsets` as an int64 array, without the tuple."""
        edges = np.zeros(self.n_segments + 1, dtype=np.int64)
        np.cumsum(
            np.fromiter(self.lengths, dtype=np.int64, count=self.n_segments),
            out=edges[1:],
        )
        return edges

    @property
    def n_segments(self) -> int:
        """How many chains the batch holds."""
        return len(self.lengths)

    @property
    def longest(self) -> int:
        """The longest segment, which is what a padded form is padded to."""
        return max(self.lengths)

    @property
    def rectangular(self) -> bool:
        """Whether every segment is the same length."""
        return len(set(self.lengths)) == 1

    def segments(self) -> Iterator[np.ndarray]:
        """Each segment as a view, in order; no copy is taken."""
        edges = self.offsets
        for index in range(self.n_segments):
            yield self.values[edges[index] : edges[index + 1]]

    @classmethod
    def from_rectangular(cls, array: np.ndarray) -> Ragged:
        """The ragged form of an ``(n_segments, length, ...)`` batch.

        The conversion every existing caller goes through, so a rectangular
        argument keeps working and there is still only one recursion below it.
        """
        if array.ndim < 2:
            msg = f"a rectangular batch is (n_segments, length, ...), got {array.shape}"
            raise ValueError(msg)
        n_segments, length = array.shape[:2]
        flat = array.reshape((n_segments * length, *array.shape[2:]))
        return cls(values=flat, lengths=(int(length),) * int(n_segments))

    def padded(self, fill: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        """The batch as ``(n_segments, longest, ...)``, and a live-position mask.

        The padded form exists so the recursion can keep batching over segments
        at one step per position of the longest. It is an implementation, never
        a likelihood: every consumer masks it, in the E step **and** in the
        emission M step, or it fits the padding.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            The padded values, and a ``(n_segments, longest)`` boolean mask that
            is true where a position is real.
        """
        shape = (self.n_segments, self.longest, *self.values.shape[1:])
        block = np.full(shape, fill, dtype=self.values.dtype)
        mask = np.zeros((self.n_segments, self.longest), dtype=bool)
        for index, segment in enumerate(self.segments()):
            block[index, : len(segment)] = segment
            mask[index, : len(segment)] = True
        return block, mask

    def reduce(
        self, ufunc: np.ufunc = np.add, values: np.ndarray | None = None
    ) -> np.ndarray:
        """One reduction per segment, along the position axis.

        Parameters
        ----------
        ufunc : np.ufunc
            A binary ufunc, e.g. `np.add`, `np.maximum`, `np.minimum`.
        values : np.ndarray, optional
            The array reduced, shape ``(total, ...)``; `self.values` when
            omitted. Any array laid out like the batch is admitted, so a weight
            per position reduces to a weight per segment without a copy of the
            batch.

        Returns
        -------
        np.ndarray
            Shape ``(n_segments, ...)``. ``ufunc.reduceat`` orders a sum
            differently from ``ufunc.reduce`` on the slice: on 10^5 standard
            normals ``np.add`` differs from it by 1.1e-14 at the widest, and
            ``maximum`` and ``minimum`` agree bitwise.

        Raises
        ------
        ValueError
            If `values` does not have the batch's number of positions.
        """
        array = self.values if values is None else np.asarray(values)
        if array.ndim == 0 or array.shape[0] != self.values.shape[0]:
            msg = (
                f"values has shape {array.shape}; the batch has "
                f"{self.values.shape[0]} positions on the leading axis"
            )
            raise ValueError(msg)
        return np.asarray(ufunc.reduceat(array, self._edges()[:-1], axis=0))

    def floored(
        self,
        min_length: float,
        *,
        weight: ArrayLike | None = None,
        min_weight: float = 0.0,
        groups: ArrayLike | None = None,
        extent: ArrayLike | None = None,
    ) -> tuple[Ragged, np.ndarray]:
        """Adjacent segments merged until each reaches a floor.

        The rule is greedy along each group. A merged segment opens at a
        segment, takes the segments after it, and closes at the first one at
        which its extent reaches `min_length` **and** its summed `weight`
        reaches `min_weight`. A merge never crosses a change in `groups`: at
        the end of a group, an unclosed remainder joins the merged segment
        before it in that group, and stands alone when there is none --- the
        only merged segment of a group may fall short of the floor.

        Parameters
        ----------
        min_length : float
            The floor on a merged segment's extent.
        weight : array_like, optional
            One weight per segment, shape ``(n_segments,)``; zero when omitted.
            A weight per position is reduced to this first, ``self.reduce(values=w)``.
        min_weight : float
            The floor on a merged segment's summed weight.
        groups : array_like, optional
            One label per segment, shape ``(n_segments,)``; a new group starts
            wherever the label changes. One group when omitted.
        extent : array_like, optional
            A ``(start, end)`` coordinate per segment, shape ``(n_segments, 2)``,
            non-decreasing along the batch. A merged segment's extent is the
            ``end`` of its last segment minus the ``start`` of its first, so a
            gap between segments counts toward it --- a span in base pairs, say.
            When omitted the coordinates are the offsets, and the extent is the
            number of positions.

        Returns
        -------
        tuple[Ragged, np.ndarray]
            The merged batch over the same `values`, and the index of its
            merged segment for each old one, shape ``(n_segments,)``: starting
            at zero, non-decreasing, and stepping by at most one.

        Raises
        ------
        ValueError
            If `weight`, `groups` or `extent` does not have one entry per segment.
        """
        lengths, parent = floor_lengths(
            self.lengths,
            min_length,
            weight=weight,
            min_weight=min_weight,
            groups=groups,
            extent=extent,
        )
        return Ragged(values=self.values, lengths=lengths), parent


def floor_lengths(
    lengths: tuple[int, ...],
    min_length: float,
    *,
    weight: ArrayLike | None = None,
    min_weight: float = 0.0,
    groups: ArrayLike | None = None,
    extent: ArrayLike | None = None,
) -> tuple[tuple[int, ...], np.ndarray]:
    """The rule of `Ragged.floored` on lengths alone (issue #1233).

    `Ragged.floored` calls this and wraps the merged lengths around its own
    `values`; a caller holding lengths and no array calls it directly.

    Parameters
    ----------
    lengths : tuple[int, ...]
        One length per segment, as `Ragged.lengths`.
    min_length, weight, min_weight, groups, extent
        As `Ragged.floored`.

    Returns
    -------
    tuple[tuple[int, ...], np.ndarray]
        The merged lengths, summing to ``sum(lengths)``, and the index of the
        merged segment for each old one, shape ``(len(lengths),)``.

    Raises
    ------
    ValueError
        If `weight`, `groups` or `extent` does not have one entry per segment.
    """
    n = len(lengths)
    edges = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.fromiter(lengths, dtype=np.int64, count=n), out=edges[1:])
    if extent is None:
        starts = edges[:-1].astype(np.float64)
        ends = edges[1:].astype(np.float64)
    else:
        span = np.asarray(extent, dtype=np.float64)
        if span.shape != (n, 2):
            msg = f"extent has shape {span.shape}; expected ({n}, 2)"
            raise ValueError(msg)
        starts, ends = span[:, 0], span[:, 1]
    mass = _per_segment(weight, n, "weight").astype(np.float64)
    labels = _per_segment(groups, n, "groups")
    opens = np.ones(n, dtype=bool)
    opens[1:] = labels[1:] != labels[:-1]
    parent = _greedy_floor(
        starts.tolist(),
        ends.tolist(),
        mass.tolist(),
        opens.tolist(),
        float(min_length),
        float(min_weight),
    )
    merged = np.add.reduceat(
        np.diff(edges), np.flatnonzero(np.diff(parent, prepend=-1))
    )
    return tuple(int(length) for length in merged), parent


def _per_segment(array: ArrayLike | None, n: int, name: str) -> np.ndarray:
    """One entry per segment, zeros when omitted, or a refusal."""
    if array is None:
        return np.zeros(n, dtype=np.float64)
    out = np.asarray(array)
    if out.shape != (n,):
        msg = f"{name} has shape {out.shape}; expected ({n},), one per segment"
        raise ValueError(msg)
    return out


def _greedy_floor(
    starts: list[float],
    ends: list[float],
    mass: list[float],
    opens: list[bool],
    min_length: float,
    min_weight: float,
) -> np.ndarray:
    """The parent of each segment under the greedy floor of `Ragged.floored`.

    The rule is sequential --- whether a segment opens a merged segment depends
    on where the previous one closed --- so it is one pass over segments, not
    positions, on Python floats: 10^5 segments take tens of milliseconds.
    """
    n = len(starts)
    parent = np.empty(n, dtype=np.int64)
    label = -1
    first = 0
    total = 0.0
    closed = True
    previous_closed_in_group = False
    for index in range(n):
        if opens[index]:
            if not closed and previous_closed_in_group:
                parent[first:index] = label - 1
                label -= 1
            previous_closed_in_group = False
            closed = True
        if closed:
            label += 1
            first = index
            total = 0.0
        parent[index] = label
        total += mass[index]
        closed = ends[index] - starts[first] >= min_length and total >= min_weight
        if closed:
            previous_closed_in_group = True
    if not closed and previous_closed_in_group:
        parent[first:] = label - 1
    return parent
