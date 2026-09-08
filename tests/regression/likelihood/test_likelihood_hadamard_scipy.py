"""`likelihood.hadamard`'s transform against `scipy.linalg.hadamard` (issue #376).

`walsh_hadamard` is the Sylvester--Hadamard product ``H v`` computed by an
in-place butterfly in ``O(N log N)``, which is a different computation from
forming ``H`` and multiplying. `scipy.linalg.hadamard` builds the matrix by
the Kronecker recursion, so the product against it is an independent answer
and not a rearrangement of ours.

What this establishes is the transform, exactly, at every order the
conjugation is offered at, and with it the two directions of
``eq:hadamard``, which are the transform composed with a logarithm and an
exponential. What it does not establish is the phylogenetic content --- that
the edge spectrum a real alignment implies is the generating tree's --- which
is the pruning likelihood's job and is refereed there.

``scipy`` is not a declared dependency of this repository, so this skips
unless it is installed; whether to declare it is the open question issue #376
leaves standing.
"""

from __future__ import annotations

import numpy as np
import pytest
from snakes_and_ladders.likelihood.hadamard import (
    MAX_TAXA,
    expected_spectrum,
    hadamard_conjugation,
    walsh_hadamard,
)

linalg = pytest.importorskip("scipy.linalg")

#: The orders the conjugation runs at: ``2^(n-1)`` for 3 to `MAX_TAXA` taxa.
ORDERS = tuple(1 << (n_taxa - 1) for n_taxa in range(3, MAX_TAXA + 1))


@pytest.mark.oracle
@pytest.mark.parametrize("order", ORDERS)
def test_the_fast_transform_is_the_dense_product_with_scipy_s_matrix(
    order: int,
) -> None:
    rng = np.random.default_rng(order)
    vector = rng.normal(size=order)

    ours = walsh_hadamard(vector)
    theirs = linalg.hadamard(order, dtype=np.float64) @ vector

    # Both sum the same +-1 terms; the butterfly sums them in a different
    # order, so the agreement is to the accumulation of that reordering.
    np.testing.assert_allclose(ours, theirs, rtol=0.0, atol=1e-12 * order)


@pytest.mark.oracle
@pytest.mark.parametrize("order", ORDERS)
def test_scipy_s_matrix_is_the_sign_convention_our_indexing_assumes(
    order: int,
) -> None:
    # The docstring's claim: entry (A, B) is (-1) to the size of A and B as
    # bit masks, which is what makes a subset index a split index.
    theirs = linalg.hadamard(order, dtype=np.float64)
    rows = np.arange(order)
    expected = np.array(
        [[(-1.0) ** bin(row & column).count("1") for column in rows] for row in rows]
    )

    np.testing.assert_array_equal(theirs, expected)


@pytest.mark.oracle
def test_both_directions_of_the_conjugation_invert_through_scipy_s_matrix() -> None:
    # `eq:hadamard` is s = H^-1 exp(H q) and q = H^-1 log(H s), with
    # H^-1 = H / N. Built here from scipy's matrix rather than from the
    # butterfly, so the round trip is checked against the dense algebra.
    order = 1 << 4
    rng = np.random.default_rng(376)
    edges = rng.uniform(0.01, 0.2, size=order)
    edges[0] = -edges[1:].sum()
    matrix = linalg.hadamard(order, dtype=np.float64)

    spectrum = matrix @ np.exp(matrix @ edges) / order
    np.testing.assert_allclose(expected_spectrum(edges), spectrum, atol=1e-12)

    recovered = matrix @ np.log(matrix @ spectrum) / order
    np.testing.assert_allclose(hadamard_conjugation(spectrum), recovered, atol=1e-12)
    np.testing.assert_allclose(hadamard_conjugation(spectrum), edges, atol=1e-10)
