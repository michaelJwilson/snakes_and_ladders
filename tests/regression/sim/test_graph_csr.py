"""`PottsGraph.from_csr` is the tuple constructor on a sparse adjacency (issue #1081).

Referee: the tuple constructor. Each lattice is written out as the symmetric
CSR a caller would hold, and the graph read back carries the same edges and
couplings as a multiset --- the order is row-major rather than the lattice
builder's, and nothing downstream reads the order.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse
from sal.sim.graph import (
    BoundaryCondition,
    PottsGraph,
    lattice_graph,
    triangular_lattice_graph,
)
from sal.sim.potts import energies


def _csr(graph: PottsGraph) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``graph``'s adjacency as symmetric CSR: both directions of each edge."""
    edges, coupling = graph.edge_index, graph.edge_coupling
    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    columns = np.concatenate([edges[:, 1], edges[:, 0]])
    weights = np.concatenate([coupling, coupling])
    order = np.lexsort((columns, rows))
    indptr = np.searchsorted(rows[order], np.arange(graph.n_nodes + 1))
    return indptr, columns[order], weights[order]


def _multiset(graph: PottsGraph) -> list[tuple[tuple[int, int], float]]:
    ordered = np.sort(graph.edge_index, axis=1)
    return sorted(
        zip(map(tuple, ordered.tolist()), graph.edge_coupling.tolist(), strict=True)
    )


GRAPHS = {
    "square open": lattice_graph((5, 4), BoundaryCondition.OPEN, 0.7),
    # Extent 2 along a periodic axis doubles a bond, which the multiset keeps.
    "square periodic, extent 2": lattice_graph((2, 3), BoundaryCondition.PERIODIC, 1.1),
    "triangular periodic": triangular_lattice_graph(
        (4, 4), BoundaryCondition.PERIODIC, -0.4
    ),
}


@pytest.mark.oracle
@pytest.mark.parametrize("name", sorted(GRAPHS))
def test_a_csr_adjacency_is_the_graph_its_edges_build(name: str) -> None:
    graph = GRAPHS[name]

    read = PottsGraph.from_csr(*_csr(graph))

    assert read.n_nodes == graph.n_nodes
    assert _multiset(read) == _multiset(graph)
    assert np.array_equal(read.edge_coupling, np.asarray(read.coupling))
    assert bool(np.all(read.edge_index[:, 0] < read.edge_index[:, 1]))


@pytest.mark.smoke
def test_one_coupling_is_every_edges() -> None:
    graph = lattice_graph((3, 3), BoundaryCondition.OPEN, 0.5)
    indptr, indices, _ = _csr(graph)

    assert _multiset(PottsGraph.from_csr(indptr, indices, 0.5)) == _multiset(graph)


@pytest.mark.smoke
def test_an_asymmetric_or_self_looped_adjacency_is_refused() -> None:
    graph = lattice_graph((3, 3), BoundaryCondition.OPEN, 0.5)
    indptr, indices, weights = _csr(graph)
    skewed = weights.copy()
    skewed[0] = 0.9
    with pytest.raises(ValueError, match="not symmetric"):
        PottsGraph.from_csr(indptr, indices, skewed)
    with pytest.raises(ValueError, match="not symmetric"):
        PottsGraph.from_csr(np.array([0, 1, 1]), np.array([1]), 1.0)
    with pytest.raises(ValueError, match="self loop"):
        PottsGraph.from_csr(np.array([0, 1]), np.array([0]), 1.0)


# --- a directed adjacency (issue #1140) -----------------------------------------
#
# Referee: scipy's symmetrisation, ``(A + A^T) / 2``, read by `from_csr`.


def _directed(
    seed: int, n_nodes: int = 60, n_entries: int = 400
) -> tuple[np.ndarray, ...]:
    """A random directed ``A`` as raw CSR: one-way and two-way pairs, repeats, both signs.

    The columns of a row are left unsorted and repeated, as a caller may
    hold them before scipy canonicalises.
    """
    rng = np.random.default_rng(seed)
    rows = rng.integers(0, n_nodes, n_entries)
    columns = (rows + rng.integers(1, n_nodes, n_entries)) % n_nodes
    # Every fourth entry reciprocated, so both kinds of pair are drawn.
    rows, columns = (
        np.concatenate([rows, columns[::4]]),
        np.concatenate([columns, rows[::4]]),
    )
    weights = rng.normal(0.5, 1.0, rows.size)
    order = np.argsort(rows, kind="stable")
    indptr = np.searchsorted(rows[order], np.arange(n_nodes + 1))
    return indptr, columns[order], weights[order]


def _scipy_symmetrised(
    indptr: np.ndarray, indices: np.ndarray, weights: np.ndarray
) -> PottsGraph:
    """The route a caller took before: ``(A + A^T) / 2`` in scipy, then `from_csr`."""
    n_nodes = indptr.size - 1
    adjacency = scipy.sparse.csr_array(
        (weights, indices, indptr), shape=(n_nodes, n_nodes)
    )
    symmetric = scipy.sparse.csr_array((adjacency + adjacency.T) * 0.5)
    symmetric.sum_duplicates()
    return PottsGraph.from_csr(symmetric.indptr, symmetric.indices, symmetric.data)


@pytest.mark.oracle
@pytest.mark.parametrize("seed", range(5))
def test_a_directed_adjacency_scores_as_its_scipy_symmetrisation(seed: int) -> None:
    indptr, indices, weights = _directed(seed)
    reference = _scipy_symmetrised(indptr, indices, weights)
    n_nodes = indptr.size - 1
    field = np.random.default_rng(seed + 100).normal(0.0, 1.0, (n_nodes, 4))
    states = np.random.default_rng(seed + 200).integers(0, 4, (32, n_nodes))

    read = PottsGraph.from_directed_csr(indptr, indices, weights)

    assert read.edges == reference.edges
    np.testing.assert_allclose(
        read.edge_coupling, reference.edge_coupling, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        energies(read, field, states),
        energies(reference, field, states),
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.oracle
@pytest.mark.parametrize("name", ["square open", "triangular periodic"])
def test_a_symmetric_adjacency_reads_as_from_csr_bitwise(name: str) -> None:
    indptr, indices, weights = _csr(GRAPHS[name])

    read = PottsGraph.from_directed_csr(indptr, indices, weights)

    assert read == PottsGraph.from_csr(indptr, indices, weights)
    assert np.array_equal(read.edge_coupling, np.asarray(read.coupling))


@pytest.mark.oracle
def test_a_one_way_entry_is_half_a_coupling_and_a_pair_their_mean() -> None:
    # 0 -> 1 at 2.0 alone; 1 <-> 2 at 1.0 and 3.0.
    indptr, indices = np.array([0, 1, 2, 3]), np.array([1, 2, 1])

    read = PottsGraph.from_directed_csr(indptr, indices, np.array([2.0, 1.0, 3.0]))

    assert read.edges == ((0, 1), (1, 2))
    assert read.coupling == (1.0, 2.0)


@pytest.mark.smoke
def test_a_self_loop_in_a_directed_adjacency_is_refused() -> None:
    with pytest.raises(ValueError, match="self loop"):
        PottsGraph.from_directed_csr(np.array([0, 1, 1]), np.array([0]), 1.0)
