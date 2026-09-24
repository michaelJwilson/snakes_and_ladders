"""Regression test pinning the example hot function's expected output.

See tests/_example_hotpath.py for what this exercises and why it's scaffolding.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from tests._example_hotpath import pairwise_distance


@pytest.mark.oracle
def test_pairwise_distance_small_fixed_input() -> None:
    # Small, hand-checkable input.
    x = np.array(
        [
            [0.0, 0.0],
            [3.0, 4.0],
        ]
    )
    y = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )

    # Hand-computed: x[0]=(0,0) to y = (0,0), (1,0), (0,1) -> 0, 1, 1;
    # x[1]=(3,4) to the same -> 5, sqrt(20), sqrt(18).
    expected = np.array(
        [
            [0.0, 1.0, 1.0],
            [5.0, np.sqrt(20.0), np.sqrt(18.0)],
        ]
    )

    result = pairwise_distance(x, y)

    assert result.shape == expected.shape
    assert_allclose(result, expected, rtol=1e-10)
