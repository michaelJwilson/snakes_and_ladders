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

**A segment shorter than two steps is refused.** One position is all initial
distribution and no transition --- a degenerate case that a padded, masked
implementation gets wrong before it gets anything else wrong --- so it is
refused here, where the shape is declared, rather than left to produce a number
a caller would have to distrust.

The layout is the compressed-row form this package already uses wherever it
stores a relation: one array, and offsets into it.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

#: The shortest segment that carries a transition, and so the shortest allowed.
MINIMUM_LENGTH = 2


@dataclass(frozen=True)
class Ragged:
    """Segments of a flat array, addressed by their lengths.

    Parameters
    ----------
    values : np.ndarray
        The segments end to end, shape ``(total, ...)``. Whatever trailing axes
        an observation carries are its own and are not read here.
    lengths : tuple[int, ...]
        One length per segment, each at least `MINIMUM_LENGTH`, summing to
        ``values.shape[0]``.
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
                f"least {MINIMUM_LENGTH} positions, since one position is an "
                "initial distribution and no transition (issue #666)"
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
        edges = [0]
        for length in self.lengths:
            edges.append(edges[-1] + length)
        return tuple(edges)

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
