"""What `PottsGraph.from_csr` costs, beside a caller building the tuples itself.

Correctness is pinned in `tests/regression/sim/test_graph_csr.py`. The claim
issue #1081 makes is not speed: `edges` is a tuple of pairs, so both routes
pay for the tuples, and `from_csr` adds a symmetry check. Measured at
5,041 sites 3.6 ms against 2.8 ms, and at 10^6 sites 1,018 ms against 884 ms:
the check costs 15%, and the tuples the rest.
"""

from __future__ import annotations

import numpy as np
import pytest
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
