"""What `PottsGraph.from_csr` costs, beside a caller building the tuples itself.

Correctness is pinned in `tests/regression/sim/test_graph_csr.py`. The claim
issue #1081 makes is not speed: `edges` is a tuple of pairs, so both routes
pay for the tuples, and `from_csr` adds a symmetry check. Measured at
5,041 sites 3.6 ms against 2.8 ms, and at 10^6 sites 1,018 ms against 884 ms:
the check costs 15%, and the tuples the rest.

`PottsGraph.from_directed_csr` (issue #1140) is timed against the route it
replaces, scipy's ``(A + A^T) / 2`` then `from_csr`, at 10^6 directed entries
over 2 x 10^5 sites: median 598 ms against 931 ms. Both build the same
~10^6 tuples, about 560 ms of either; the 333 ms between them is scipy's
transpose, sum and the symmetry check `from_csr` repeats.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse
from pytest_benchmark.fixture import BenchmarkFixture
from sal.sim.graph import BoundaryCondition, PottsGraph, lattice_graph


def _csr(side: int) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    graph = lattice_graph((side, side), BoundaryCondition.OPEN, 0.7)
    edges, coupling = graph.edge_index, graph.edge_coupling
    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    columns = np.concatenate([edges[:, 1], edges[:, 0]])
    weights = np.concatenate([coupling, coupling])
    order = np.lexsort((columns, rows))
    indptr = np.searchsorted(rows[order], np.arange(graph.n_nodes + 1))
    return graph.n_nodes, indptr, columns[order], weights[order]


def _by_hand(
    n_nodes: int, indptr: np.ndarray, indices: np.ndarray, weights: np.ndarray
) -> PottsGraph:
    """The route a caller took before: the upper entries as tuples, unchecked."""
    rows = np.repeat(np.arange(n_nodes), np.diff(indptr))
    keep = rows < indices
    return PottsGraph(
        n_nodes,
        tuple(zip(rows[keep].tolist(), indices[keep].tolist(), strict=True)),
        tuple(weights[keep].tolist()),
    )


@pytest.mark.parametrize("route", ["from_csr", "by_hand"])
def test_graph_from_csr_benchmark(benchmark: BenchmarkFixture, route: str) -> None:
    n_nodes, indptr, indices, weights = _csr(71)
    if route == "from_csr":
        graph = benchmark(PottsGraph.from_csr, indptr, indices, weights)
    else:
        graph = benchmark(_by_hand, n_nodes, indptr, indices, weights)
    assert len(graph.edges) == 2 * 71 * 70


def _directed(n_nodes: int, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A directed k-out adjacency, ``n_nodes * k`` entries: each site's ``k`` draws."""
    rng = np.random.default_rng(1140)
    rows = np.repeat(np.arange(n_nodes), k)
    columns = (rows + rng.integers(1, n_nodes, rows.size)) % n_nodes
    indptr = np.arange(0, rows.size + 1, k)
    return indptr, columns, rng.uniform(0.1, 1.0, rows.size)


def _scipy_symmetrised(
    indptr: np.ndarray, indices: np.ndarray, weights: np.ndarray
) -> PottsGraph:
    """The route a caller took before issue #1140: ``(A + A^T) / 2``, then `from_csr`."""
    n_nodes = indptr.size - 1
    adjacency = scipy.sparse.csr_array(
        (weights, indices, indptr), shape=(n_nodes, n_nodes)
    )
    symmetric = scipy.sparse.csr_array((adjacency + adjacency.T) * 0.5)
    symmetric.sum_duplicates()
    return PottsGraph.from_csr(symmetric.indptr, symmetric.indices, symmetric.data)


@pytest.mark.release
@pytest.mark.parametrize("route", ["from_directed_csr", "scipy_symmetrised"])
def test_graph_from_directed_csr_benchmark(
    benchmark: BenchmarkFixture, route: str
) -> None:
    """At stress size, 10^6 directed entries over 2 x 10^5 sites."""
    indptr, indices, weights = _directed(200_000, 5)
    build = (
        PottsGraph.from_directed_csr
        if route == "from_directed_csr"
        else _scipy_symmetrised
    )
    graph = benchmark(build, indptr, indices, weights)
    assert graph.edge_coupling.size > 900_000
