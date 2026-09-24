"""Segments of unequal length tile one array, and a one-position segment is refused.

Issue #666.
"""

from __future__ import annotations

import numpy as np
import pytest
from snakes_and_ladders.ragged import MINIMUM_LENGTH, Ragged


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
