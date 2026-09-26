"""`PottsGraph.from_csr` is the tuple constructor on a sparse adjacency (issue #1081).

Referee: the tuple constructor. Each lattice is written out as the symmetric
CSR a caller would hold, and the graph read back carries the same edges and
couplings as a multiset --- the order is row-major rather than the lattice
builder's, and nothing downstream reads the order.
"""

from __future__ import annotations

import numpy as np
import pytest
from sal.sim.graph import (
    BoundaryCondition,
    PottsGraph,
    lattice_graph,
    triangular_lattice_graph,
)


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
