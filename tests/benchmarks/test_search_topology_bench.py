"""Benchmarks for ``snakes_and_ladders.search.topology``'s NNI and SPR generators.

See ``tests/regression/test_search_topology.py`` for correctness. Runs at
``n = 10``, the module ``CLAUDE.md``'s topological-test size cap, on a
caterpillar topology (the shape that maximizes internal-edge chain length
for a given ``n``, so neither generator gets a shortcut from balance).
"""

from __future__ import annotations

from pytest_benchmark.fixture import BenchmarkFixture
from snakes_and_ladders.search.topology import Topology, nni_neighbours, spr_neighbours
from snakes_and_ladders.sim.tree import Node

N_TAXA = 10


def _caterpillar(n_taxa: int) -> Topology:
    leaves = [Node(name=f"t{i}", branch_length=None) for i in range(n_taxa)]
    tail: Node = leaves[-1]
    for leaf in reversed(leaves[2:-1]):
        tail = Node(name="i", branch_length=None, children=(leaf, tail))
    return Node(name="root", branch_length=None, children=(leaves[0], leaves[1], tail))


def test_nni_neighbours_benchmark(benchmark: BenchmarkFixture) -> None:
    topology = _caterpillar(N_TAXA)

    neighbours = benchmark(lambda: list(nni_neighbours(topology)))

    assert len(neighbours) == 2 * (N_TAXA - 3)


def test_spr_neighbours_benchmark(benchmark: BenchmarkFixture) -> None:
    topology = _caterpillar(N_TAXA)

    neighbours = benchmark(lambda: list(spr_neighbours(topology)))

    assert len(neighbours) == 2 * (N_TAXA - 3) * (2 * N_TAXA - 7)
