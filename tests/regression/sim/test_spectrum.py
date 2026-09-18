"""The spectrum seam, pinned to closed forms and to itself.

Issue #718. Two referees. The periodic ``3x3`` lattice's coupling matrix is a
circulant-of-circulants whose eigenvalues are closed form,
``2J (cos(2 pi a / 3) + cos(2 pi b / 3))``, so both ends of its spectrum are
known without solving anything. And the dense route is the oracle of the
sparse one: ``eigh`` depends on nothing, ``eigsh`` on a start vector and a
tolerance, so the sparse answer is pinned to the dense one at the tolerance
the module declares, on every graph the suite reaches.
"""

from __future__ import annotations

import numpy as np
import pytest
from snakes_and_ladders.sim.graph import (
    BoundaryCondition,
    erdos_renyi_graph,
    lattice_graph,
    triangular_lattice_graph,
)
from snakes_and_ladders.sim.spectrum import (
    SPARSE_TOLERANCE,
    coupling_matrix,
    extreme_eigenpairs,
    laplacian,
)


@pytest.mark.mathematical
@pytest.mark.critical
def test_the_torus_spectrum_is_its_closed_form() -> None:
    graph = lattice_graph((3, 3), BoundaryCondition.PERIODIC, 0.7)
    matrix = coupling_matrix(graph)

    largest, _ = extreme_eigenpairs(matrix, largest=True)
    smallest, _ = extreme_eigenpairs(matrix, largest=False)

    # a = b = 0 gives 4J; a = b = 1 gives 2J(-1/2 - 1/2) = -2J.
    assert largest[0] == pytest.approx(4 * 0.7, abs=1e-12)
    assert smallest[0] == pytest.approx(-2 * 0.7, abs=1e-12)


@pytest.mark.mathematical
def test_a_duplicated_edge_counts_twice() -> None:
    # A periodic lattice two sites wide names each wrap-around pair twice;
    # both couplings act, so the matrix sums them rather than keeping one.
    graph = lattice_graph((2, 3), BoundaryCondition.PERIODIC, 1.0)
    dense = coupling_matrix(graph).toarray()

    counts: dict[tuple[int, int], int] = {}
    for first, second in graph.edges:
        key = (min(first, second), max(first, second))
        counts[key] = counts.get(key, 0) + 1
    assert max(counts.values()) == 2
    for (first, second), count in counts.items():
        assert dense[first, second] == count
        assert dense[second, first] == count


@pytest.mark.mathematical
def test_the_laplacian_kills_the_constant_vector_whatever_the_signs() -> None:
    graph = triangular_lattice_graph((4, 4), BoundaryCondition.OPEN, -1.0)
    matrix = laplacian(graph)

    assert np.allclose(matrix @ np.ones(graph.n_nodes), 0.0, atol=1e-12)
    assert np.allclose(matrix.toarray(), matrix.toarray().T)


@pytest.mark.oracle
@pytest.mark.parametrize("largest", [True, False])
def test_the_sparse_route_reproduces_the_dense_oracle(largest: bool) -> None:
    # Mean degree 4 on 200 nodes: the disordered structure beside the lattices.
    graph = erdos_renyi_graph(200, 4.0 / 199.0, 1.0, np.random.default_rng(718))
    matrix = coupling_matrix(graph)

    dense_values, dense_vectors = extreme_eigenpairs(
        matrix, largest=largest, count=2, dense_below=10**6
    )
    sparse_values, sparse_vectors = extreme_eigenpairs(
        matrix, largest=largest, count=2, dense_below=0
    )

    assert np.allclose(sparse_values, dense_values, atol=100 * SPARSE_TOLERANCE)
    # The sign convention makes the lines comparable as vectors.
    assert np.allclose(np.abs(sparse_vectors), np.abs(dense_vectors), atol=1e-6)
    assert np.all(
        dense_vectors[np.argmax(np.abs(dense_vectors) > 1e-12, axis=0), [0, 1]] > 0
    )


@pytest.mark.structural
def test_the_sparse_route_is_a_function_of_its_input() -> None:
    graph = lattice_graph((12, 12), BoundaryCondition.OPEN, 0.5)
    matrix = coupling_matrix(graph)

    first = extreme_eigenpairs(matrix, largest=True, dense_below=0)
    second = extreme_eigenpairs(matrix, largest=True, dense_below=0)

    assert first[0].tolist() == second[0].tolist()
    assert np.array_equal(first[1], second[1])


@pytest.mark.edge_case
def test_a_count_outside_the_spectrum_is_refused() -> None:
    graph = lattice_graph((2, 2), BoundaryCondition.OPEN, 1.0)
    with pytest.raises(ValueError, match="count must lie"):
        extreme_eigenpairs(coupling_matrix(graph), largest=True, count=4)
