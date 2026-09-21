"""`FlowNetwork` keeps the contiguous form it built (issue #642).

`from_arcs` assembled the compiled consumer's arrays on its way to the list
store and then discarded them, so a network built from arcs and solved in Rust
made the round trip NumPy -> list -> NumPy for nothing. It keeps them now.

The claim is that keeping changes no number: `as_arrays` returns what it
returned before, bit for bit, whether the form was kept or derived. The second
claim is the harder one --- a kept form that outlives its store is a worse
defect than the derivation it replaced --- so every writer is pinned here to
drop it.
"""

from __future__ import annotations

import numpy as np
import pytest
from snakes_and_ladders.search.maxflow import FlowNetwork, max_flow


def _network(n_edges: int = 400, n_nodes: int = 64, seed: int = 3) -> FlowNetwork:
    rng = np.random.default_rng(seed)
    return FlowNetwork.from_arcs(
        n_nodes,
        rng.integers(0, n_nodes, n_edges),
        rng.integers(0, n_nodes, n_edges),
        rng.random(n_edges),
        rng.random(n_edges),
    )


@pytest.mark.smoke
@pytest.mark.critical
def test_the_kept_form_is_the_derived_form_bitwise() -> None:
    """Keeping is a shortcut to a value the lists would have produced, not a new one."""
    kept = _network()
    derived = _network()
    derived.forget_arrays()

    for left, right in zip(kept.as_arrays(), derived.as_arrays(), strict=True):
        assert np.array_equal(left, right)


@pytest.mark.smoke
@pytest.mark.critical
def test_every_writer_drops_the_kept_form() -> None:
    """A cache that outlives its store is the defect this would be trading for.

    `add_edge` appends to all three lists and `max_flow` turns capacities into
    residual capacities, so both have to forget. Asserted by reading
    `as_arrays` after each and comparing against the lists, which is what a
    stale form would disagree with.
    """
    network = _network()
    network.add_edge(0, 1, 0.5, 0.25)
    arcs, forward, backward = network.as_arrays()
    assert arcs.size == len(network.target)
    assert forward.size == backward.size == len(network.capacity) // 2

    solved = _network()
    before = solved.as_arrays().capacity.copy()
    max_flow(solved, 0, 1)
    after = solved.as_arrays().capacity
    # The residual capacities are what a second call sees, which is only true
    # if the solve dropped the form built from the originals.
    assert not np.array_equal(before, after)
    assert np.array_equal(
        after, np.asarray(solved.capacity, dtype=np.float64).reshape(-1, 2)[:, 0]
    )


@pytest.mark.smoke
def test_a_network_built_by_add_edge_derives_its_form() -> None:
    """The other construction path keeps nothing, and must still be correct."""
    built = FlowNetwork(n_nodes=3)
    built.add_edge(0, 1, 1.5)
    built.add_edge(1, 2, 2.5, 0.5)

    arcs, forward, backward = built.as_arrays()

    assert np.array_equal(arcs, np.array([0, 1, 1, 2]))
    assert np.array_equal(forward, np.array([1.5, 2.5]))
    assert np.array_equal(backward, np.array([0.0, 0.5]))
