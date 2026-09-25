"""`likelihood.hadamard`'s transform against `scipy.linalg.hadamard` (issue #376).

`walsh_hadamard` is an in-place ``O(N log N)`` butterfly; `scipy` builds ``H``
by the Kronecker recursion, an independent answer. Established: the transform
exactly at every offered order, and so both directions of ``eq:hadamard``; the
phylogenetic content is the pruning likelihood's. Skips without ``scipy``,
which is undeclared (issue #376).
"""

from __future__ import annotations

import numpy as np
import pytest
from sal.likelihood.hadamard import (
    MAX_TAXA,
    expected_spectrum,
    hadamard_conjugation,
    walsh_hadamard,
)

linalg = pytest.importorskip("scipy.linalg")

#: The orders the conjugation runs at: ``2^(n-1)`` for 3 to `MAX_TAXA` taxa.
ORDERS = tuple(1 << (n_taxa - 1) for n_taxa in range(3, MAX_TAXA + 1))


@pytest.mark.oracle
def test_the_transform_and_both_directions_of_the_conjugation_are_scipy_s_algebra() -> (
    None
):
    # Issue #982 folded three tests into this one. The sign convention the
    # indexing assumes is asserted against the bit-mask definition in
    # `test_likelihood_hadamard.py` at every order; agreeing with scipy's
    # product here at every order carries it to scipy's matrix too.
    for order in ORDERS:
        vector = np.random.default_rng(order).normal(size=order)

        ours = walsh_hadamard(vector)
        theirs = linalg.hadamard(order, dtype=np.float64) @ vector

        # Both sum the same +-1 terms; the butterfly sums them in a different
        # order, so the agreement is to the accumulation of that reordering.
        np.testing.assert_allclose(ours, theirs, rtol=0.0, atol=1e-12 * order)

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
