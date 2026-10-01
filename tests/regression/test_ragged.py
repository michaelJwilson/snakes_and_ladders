"""Segments of unequal length tile one array, reduce per segment, and merge to a floor.

Issues #666 and #1141.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from sal.ragged import MINIMUM_LENGTH, Ragged


@pytest.mark.critical
@pytest.mark.infra
def test_the_segments_tile_the_array_and_are_views() -> None:
    """Every row belongs to exactly one segment, and none is copied."""
    values = np.arange(9.0)
    batch = Ragged(values, (2, 3, 4))
    assert batch.offsets == (0, 2, 5, 9)
    assert [segment.tolist() for segment in batch.segments()] == [
        [0.0, 1.0],
        [2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
    ]
    first = next(iter(batch.segments()))
    first[0] = 99.0
    assert values[0] == 99.0, "a segment is a view, not a copy"


@pytest.mark.critical
@pytest.mark.smoke
def test_a_one_position_segment_is_refused() -> None:
    """All initial distribution and no transition, so the shape is refused.

    The ruling on #666: a padded, masked recursion goes wrong here first.
    """
    with pytest.raises(ValueError, match="at least 2 positions"):
        Ragged(np.zeros(3), (1, 2))
    assert MINIMUM_LENGTH == 2


@pytest.mark.critical
@pytest.mark.smoke
def test_lengths_that_do_not_tile_are_refused() -> None:
    """A batch whose lengths do not sum to its rows is not a batch."""
    with pytest.raises(ValueError, match="tile the array exactly"):
        Ragged(np.zeros(5), (2, 2))
    with pytest.raises(ValueError, match="at least one segment"):
        Ragged(np.zeros(0), ())


@pytest.mark.critical
@pytest.mark.infra
def test_a_rectangular_batch_is_a_ragged_one_with_equal_lengths() -> None:
    """The conversion every existing caller goes through, with its trailing axes."""
    array = np.arange(24.0).reshape(2, 3, 4)
    batch = Ragged.from_rectangular(array)
    assert batch.lengths == (3, 3)
    assert batch.rectangular
    np.testing.assert_array_equal(batch.values, array.reshape(6, 4))


@pytest.mark.critical
@pytest.mark.analytic
def test_padding_is_recoverable_and_the_mask_says_where() -> None:
    """The padded form loses nothing: the mask selects exactly the real rows.

    The mask keeps padding out of a likelihood batched over segments.
    """
    batch = Ragged(np.arange(9.0), (2, 3, 4))
    block, mask = batch.padded(fill=np.nan)
    assert block.shape == (3, 4)
    assert mask.sum() == 9
    np.testing.assert_array_equal(block[mask], batch.values)
    assert np.isnan(block[~mask]).all(), "every padded cell is the fill"
    for index, segment in enumerate(batch.segments()):
        np.testing.assert_array_equal(block[index, : len(segment)], segment)


def _random_batch(
    rng: np.random.Generator, n_segments: int, trailing: tuple[int, ...] = ()
) -> Ragged:
    """A batch of `n_segments` segments of 2-12 positions, standard normal values."""
    lengths = tuple(int(x) for x in rng.integers(2, 13, n_segments))
    return Ragged(rng.standard_normal((sum(lengths), *trailing)), lengths)


@pytest.mark.critical
@pytest.mark.oracle
@pytest.mark.parametrize("ufunc", [np.maximum, np.minimum], ids=lambda u: u.__name__)
@pytest.mark.parametrize("trailing", [(), (3,), (2, 4)], ids=str)
def test_reduce_equals_a_loop_over_segments_bitwise(
    ufunc: np.ufunc, trailing: tuple[int, ...]
) -> None:
    """``maximum`` and ``minimum`` per segment equal the per-segment ``reduce``, bitwise.

    Issue #1141. An extremum does not depend on the order it is taken in.
    """
    batch = _random_batch(np.random.default_rng(1141), 200, trailing)
    expected = np.stack([ufunc.reduce(s, axis=0) for s in batch.segments()])
    got = batch.reduce(ufunc)
    assert got.shape == (200, *trailing)
    np.testing.assert_array_equal(got, expected)


@pytest.mark.critical
@pytest.mark.oracle
@pytest.mark.parametrize("trailing", [(), (3,), (2, 4)], ids=str)
def test_reduce_sums_equal_exact_sums(trailing: tuple[int, ...]) -> None:
    """``np.add`` per segment: bitwise on integers, within 1e-14 of ``math.fsum`` on floats.

    Issue #1141. Integer-valued float64 sums below 2^53 are exact in any order,
    so there the referee is bitwise; on standard normals ``reduceat`` and the
    correctly rounded ``fsum`` differ by summation order alone.
    """
    rng = np.random.default_rng(1142)
    batch = _random_batch(rng, 200, trailing)
    integral = np.round(1e6 * batch.values)
    exact = np.stack(
        [
            np.sum(segment.astype(np.int64), axis=0)
            for segment in Ragged(integral, batch.lengths).segments()
        ]
    ).astype(np.float64)
    np.testing.assert_array_equal(batch.reduce(values=integral), exact)

    flat = batch.values.reshape(batch.values.shape[0], -1)
    fsum = np.array(
        [
            [math.fsum(column) for column in segment.T]
            for segment in Ragged(flat, batch.lengths).segments()
        ]
    ).reshape((batch.n_segments, *trailing))
    np.testing.assert_allclose(batch.reduce(), fsum, rtol=0.0, atol=1e-14)


@pytest.mark.critical
@pytest.mark.smoke
def test_reduce_refuses_values_of_another_length() -> None:
    """An array not laid out like the batch has no segments to reduce."""
    with pytest.raises(ValueError, match="positions on the leading axis"):
        Ragged(np.zeros(5), (2, 3)).reduce(values=np.zeros(4))


def _floor_reference(
    lengths: list[int],
    starts: list[float],
    ends: list[float],
    weight: list[float],
    groups: list[int],
    min_length: float,
    min_weight: float,
) -> list[int]:
    """The greedy floor written out group by group, as lists of segment indices."""
    blocks: list[list[int]] = []
    index = 0
    while index < len(lengths):
        group_end = index
        while group_end < len(lengths) and groups[group_end] == groups[index]:
            group_end += 1
        merged: list[list[int]] = []
        current: list[int] = []
        for k in range(index, group_end):
            current.append(k)
            span = ends[current[-1]] - starts[current[0]]
            mass = sum(weight[j] for j in current)
            if span >= min_length and mass >= min_weight:
                merged.append(current)
                current = []
        if current:
            if merged:
                merged[-1].extend(current)
            else:
                merged.append(current)
        blocks.extend(merged)
        index = group_end
    parent = [0] * len(lengths)
    for label, block in enumerate(blocks):
        for k in block:
            parent[k] = label
    return parent


@pytest.mark.critical
@pytest.mark.oracle
@pytest.mark.parametrize("seed", range(40))
def test_floored_equals_the_greedy_rule_written_out(seed: int) -> None:
    """`floored` against a straight-line reference, with groups, weights and spans.

    Issue #1141. Half the instances take coordinates with gaps between
    segments, so the extent is a span rather than a count; the floors are
    drawn so that remainders and short groups both occur.
    """
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 60))
    batch = _random_batch(rng, n)
    lengths = list(batch.lengths)
    groups = np.sort(rng.integers(0, int(rng.integers(1, 6)), n))
    weight = rng.exponential(1.0, n)
    min_length = float(rng.integers(2, 30))
    min_weight = float(rng.choice([0.0, 1.0, 3.0]))
    if seed % 2:
        gaps = rng.integers(0, 5, n)
        starts = np.cumsum(np.r_[0, np.asarray(lengths[:-1]) + gaps[:-1]])
        ends = starts + np.asarray(lengths)
        extent: np.ndarray | None = np.column_stack([starts, ends])
    else:
        edges = np.asarray(batch.offsets)
        starts, ends, extent = edges[:-1], edges[1:], None
    merged, parent = batch.floored(
        min_length,
        weight=weight,
        min_weight=min_weight,
        groups=groups,
        extent=extent,
    )
    expected = _floor_reference(
        lengths,
        [float(x) for x in starts],
        [float(x) for x in ends],
        weight.tolist(),
        groups.tolist(),
        min_length,
        min_weight,
    )
    assert parent.tolist() == expected
    assert merged.values is batch.values, "values is shared, not copied"
    assert merged.lengths == tuple(
        int(np.sum(np.asarray(lengths)[parent == label]))
        for label in range(merged.n_segments)
    )


@pytest.mark.critical
@pytest.mark.analytic
@pytest.mark.parametrize("seed", range(20))
def test_every_merged_segment_meets_the_floor_unless_alone_in_its_group(
    seed: int,
) -> None:
    """The floor holds for every merged segment but a group's only one.

    Issue #1141. The parent index starts at zero and steps by zero or one, so
    each merged segment is a contiguous run of old ones, inside one group.
    """
    rng = np.random.default_rng(10_000 + seed)
    n = int(rng.integers(1, 200))
    batch = _random_batch(rng, n)
    groups = np.sort(rng.integers(0, 8, n))
    weight = rng.exponential(1.0, n)
    min_length, min_weight = 15.0, 2.0
    merged, parent = batch.floored(
        min_length, weight=weight, min_weight=min_weight, groups=groups
    )
    assert parent[0] == 0
    assert set(np.diff(parent).tolist()) <= {0, 1}
    assert parent[-1] == merged.n_segments - 1
    for label in range(merged.n_segments):
        members = parent == label
        assert len(set(groups[members].tolist())) == 1, "a merge crossed a group"
        alone = len(set(parent[groups == groups[members][0]].tolist())) == 1
        if not alone:
            assert merged.lengths[label] >= min_length
            assert weight[members].sum() >= min_weight


@pytest.mark.critical
@pytest.mark.analytic
def test_floored_by_hand() -> None:
    """Lengths (2, 3, 2, 4, 2, 5, 2) at a floor of five positions, worked by hand.

    One group: 2+3 closes, 2+4 closes, 2+5 closes, the trailing 2 joins it.
    Split after the third: 2+3 closes and the 2 joins it; 4+2 closes, 5 closes
    alone, and the trailing 2 joins it.
    """
    batch = Ragged(np.arange(20.0), (2, 3, 2, 4, 2, 5, 2))
    merged, parent = batch.floored(5)
    assert merged.lengths == (5, 6, 9)
    assert parent.tolist() == [0, 0, 1, 1, 2, 2, 2]
    merged, parent = batch.floored(5, groups=[0, 0, 0, 1, 1, 1, 1])
    assert merged.lengths == (7, 6, 7)
    assert parent.tolist() == [0, 0, 0, 1, 1, 2, 2]
    merged, parent = batch.floored(100)
    assert merged.lengths == (20,), "a group that never closes stands alone"


@pytest.mark.critical
@pytest.mark.smoke
def test_floored_refuses_arrays_not_one_per_segment() -> None:
    """A weight, a group label or a coordinate pair is one per segment."""
    batch = Ragged(np.zeros(5), (2, 3))
    with pytest.raises(ValueError, match="one per segment"):
        batch.floored(3, weight=np.ones(3))
    with pytest.raises(ValueError, match="one per segment"):
        batch.floored(3, groups=[0])
    with pytest.raises(ValueError, match=r"expected \(2, 2\)"):
        batch.floored(3, extent=np.zeros((2, 3)))
