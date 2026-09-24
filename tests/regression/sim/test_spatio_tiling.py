"""The tiling field and the tiles it is planted on (issue #1050).

`sim.potts.tile_partition` splits a lattice into connected tiles from seeded
centres, and `sim.potts.tiling_field` plants one favoured state per tile. The
tiling is judged by a breadth-first search written here over the edge list,
which shares no code with the compressed-adjacency search the tiling is built
on; the field by its definition, entry by entry, at both declared tiers.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import pytest
from snakes_and_ladders.sim.fixtures import fixture
from snakes_and_ladders.sim.graph import (
    BoundaryCondition,
    PottsGraph,
    lattice_graph,
    triangular_lattice_graph,
)
from snakes_and_ladders.sim.potts import (
    SpatioTilingParams,
    tile_partition,
    tiling_field,
)

#: The two declared instances, loaded once each.
CI: SpatioTilingParams = fixture("spatio_tiling", "ci").params
RELEASE: SpatioTilingParams = fixture("spatio_tiling", "release").params

#: The lattices the ensemble draws tilings on: a chain, a square grid open
#: and periodic, and the triangular lattice the fixtures use.
LATTICES: dict[str, PottsGraph] = {
    "chain": lattice_graph((17,), BoundaryCondition.OPEN, 1.0),
    "square": lattice_graph((9, 7), BoundaryCondition.OPEN, 1.0),
    "torus": lattice_graph((8, 8), BoundaryCondition.PERIODIC, 1.0),
    "triangular": triangular_lattice_graph((9, 8), BoundaryCondition.OPEN, 1.0),
}


def _neighbours(graph: PottsGraph) -> list[list[int]]:
    """Each node's neighbours, from the edge list and not the compressed rows."""
    rows: list[list[int]] = [[] for _ in range(graph.n_nodes)]
    for left, right in graph.edges:
        rows[left].append(right)
        rows[right].append(left)
    return rows


def _connected(rows: list[list[int]], members: np.ndarray) -> bool:
    """Whether the nodes in ``members`` form one connected piece, by breadth-first search."""
    inside = {int(node) for node in members}
    start = int(members[0])
    seen = {start}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        for other in rows[node]:
            if other in inside and other not in seen:
                seen.add(other)
                queue.append(other)
    return seen == inside


@pytest.mark.analytic
@pytest.mark.parametrize("lattice", sorted(LATTICES))
def test_every_drawn_tiling_covers_the_lattice_in_connected_tiles(
    lattice: str,
) -> None:
    # Forty drawn tilings per lattice at tile counts from 2 to 12: every node
    # carries one tile, every tile is non-empty, and every tile is one
    # connected piece, which the tie rule guarantees.
    graph = LATTICES[lattice]
    rows = _neighbours(graph)
    rng = np.random.default_rng(1050)
    for _ in range(40):
        k = int(rng.integers(2, 13))
        tiles = tile_partition(graph, k, rng)

        assert tiles.shape == (graph.n_nodes,)
        assert set(np.unique(tiles)) == set(range(k))
        for tile in range(k):
            assert _connected(rows, np.flatnonzero(tiles == tile))


@pytest.mark.analytic
def test_a_node_between_two_centres_joins_the_one_drawn_first() -> None:
    # On a six-node chain default_rng(3) draws centres 0 and 4; node 2 is two
    # hops from each, and node 5 one hop from 4.
    chain = lattice_graph((6,), BoundaryCondition.OPEN, 1.0)

    tiles = tile_partition(chain, 2, np.random.default_rng(3))

    assert tiles.tolist() == [0, 0, 0, 1, 1, 1]


@pytest.mark.analytic
@pytest.mark.parametrize("params", [CI, RELEASE], ids=["ci", "release"])
def test_the_declared_field_is_its_strength_at_its_state_and_zero_elsewhere(
    params: SpatioTilingParams,
) -> None:
    # Entry by entry against the definition h[i, m] = s_t [m = a_t].
    rows = _neighbours(params.graph)
    for node in range(params.graph.n_nodes):
        tile = int(params.tiles[node])
        for state in range(params.n_states):
            expected = params.strengths[tile] if state == params.states[tile] else 0.0
            assert params.field[node, state] == expected
    for tile in range(params.n_tiles):
        assert _connected(rows, np.flatnonzero(params.tiles == tile))


@pytest.mark.smoke
@pytest.mark.snapshot
def test_the_declared_tilings_are_the_ones_their_files_describe() -> None:
    # The tile sizes the fixtures' comments state.
    assert np.bincount(CI.tiles).tolist() == [7, 3, 2]
    assert RELEASE.graph.n_nodes == 5041
    assert np.bincount(RELEASE.tiles).min() == 145
    assert np.bincount(RELEASE.tiles).max() == 665


@pytest.mark.smoke
def test_a_tiling_of_a_disconnected_graph_is_refused() -> None:
    two = PottsGraph(n_nodes=4, edges=((0, 1), (2, 3)), coupling=(1.0, 1.0))
    with pytest.raises(ValueError, match="not connected"):
        tile_partition(two, 1, np.random.default_rng(0))
    with pytest.raises(ValueError, match="k must be"):
        tile_partition(two, 5, np.random.default_rng(0))


@pytest.mark.smoke
def test_a_field_with_a_bad_state_or_strength_is_refused() -> None:
    tiles = np.array([0, 0, 1])
    with pytest.raises(ValueError, match="one state and one strength"):
        tiling_field(tiles, np.array([0, 1]), np.array([1.0]), 3)
    with pytest.raises(ValueError, match="favoured state must be"):
        tiling_field(tiles, np.array([0, 3]), np.array([1.0, 1.0]), 3)
    with pytest.raises(ValueError, match="strength must be positive"):
        tiling_field(tiles, np.array([0, 1]), np.array([1.0, 0.0]), 3)
    with pytest.raises(ValueError, match="tile must be in"):
        tiling_field(np.array([0, 2]), np.array([0, 1]), np.array([1.0, 1.0]), 3)
