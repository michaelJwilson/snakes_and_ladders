"""The compressed-row adjacency is the neighbour lists, flattened, in the same order.

Every kernel and every Python sweep reads it, and all of them consume their
inputs in neighbour order, so a permutation here would change which draw a
site sees without changing any distribution a chi-square could catch.

The oracle is :func:`_neighbour_lists` below: the list-of-lists loop the five
builders issue #277 retired each held --- `search.potts_mcmc._adjacency`,
`search.potts_mcmc_rust.flatten_adjacency`, and the open-coded copies in
`search.ground_state`, `search.alpha_expansion` and `sim.potts`. Each was
asserted equal to `compressed_adjacency` on these graphs before its call site
moved; the loop stays here, in the suite, as the independent reference the
package no longer contains.
"""

from __future__ import annotations

import numpy as np
import pytest
from snakes_and_ladders.sim.canonical import planted_spin_glass
from snakes_and_ladders.sim.graph import BoundaryCondition, PottsGraph, lattice_graph


def _neighbour_lists(graph: PottsGraph) -> list[list[tuple[int, float]]]:
    """Neighbour lists with the coupling on each incident edge, from the edges."""
    neighbours: list[list[tuple[int, float]]] = [[] for _ in range(graph.n_nodes)]
    for (first, second), coupling in graph.weighted_edges():
        neighbours[first].append((second, coupling))
        neighbours[second].append((first, coupling))
    return neighbours


def _graphs() -> list[PottsGraph]:
    return [
        lattice_graph((2, 2), BoundaryCondition.PERIODIC, 0.5),  # doubled bonds
        lattice_graph((4, 5), BoundaryCondition.OPEN, 0.3),
        lattice_graph((3, 3, 2), BoundaryCondition.PERIODIC, 0.7),
        planted_spin_glass(60, 4.0, 0.2, np.random.default_rng(1)).graph,
    ]


@pytest.mark.smoke
@pytest.mark.parametrize(
    "graph", _graphs(), ids=["2x2-periodic", "4x5", "3x3x2", "glass"]
)
def test_the_compressed_rows_are_the_list_adjacency_in_order(graph: PottsGraph) -> None:
    offsets, neighbours, couplings = graph.compressed_adjacency()
    lists = _neighbour_lists(graph)

    assert offsets.shape == (graph.n_nodes + 1,)
    assert offsets[0] == 0
    assert (
        offsets[-1] == 2 * len(graph.edges) == neighbours.shape[0] == couplings.shape[0]
    )
    for node, row in enumerate(lists):
        start, stop = int(offsets[node]), int(offsets[node + 1])
        assert [int(n) for n in neighbours[start:stop]] == [n for n, _ in row]
        assert [float(c) for c in couplings[start:stop]] == [c for _, c in row]


@pytest.mark.smoke
def test_the_compressed_rows_are_contiguous_native_arrays() -> None:
    # What the kernels require: no marshalling per node, no copy at the FFI
    # boundary (root CLAUDE.md's layout and boundary rules).
    offsets, neighbours, couplings = _graphs()[1].compressed_adjacency()

    for array, dtype in (
        (offsets, np.int64),
        (neighbours, np.int64),
        (couplings, np.float64),
    ):
        assert array.dtype == dtype
        assert array.flags["C_CONTIGUOUS"]
