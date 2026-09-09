"""The Swendsen-Wang labelling against the `rustworkx` front that lost to it (issues #322, #389).

The front is :mod:`snakes_and_ladders.sandbox.rustworkx_clusters`, kept there
because issue #389 declined it on two grounds, and both are tested here rather
than one of them recorded and the other assumed.

**The partition.** ``rustworkx.connected_components`` and the adopted
pointer-doubling roots must find the same clusters, or the front is not a
front at all. Compared as a set of blocks, on 200 seeded bond activations per
extent.

**The renumbering.** The adopted labelling gives every site its union-find
root, so the sweep recolours the clusters in ascending root order;
``connected_components`` numbers them by its own discovery order. The two
orders disagree, and the disagreement is *the stated reason for the decline*,
so it is pinned as a measured difference. Sorting it away --- comparing
partitions and stopping --- would destroy the finding, so the tests below
assert the label arrays differ, name the first cluster index at which the
visiting orders diverge, and show the two sweeps composing different chains
from one seed while drawing the same bonds.

The clock half of the decline (1.12x to 1.17x slower per sweep) is measured in
`tests/benchmarks/test_potts_mcmc_bench.py` and recorded in `STATUS.md` and
`docs/experiments/008-frameworks-on-three-hot-paths.md`; a test asserts no
wall time.

Skips without the ``frameworks`` extra.
"""

from __future__ import annotations

import numpy as np
import pytest
from snakes_and_ladders.search.potts_mcmc import (
    _Bonds,
    _bonds_of,
    _roots,
    _swendsen_wang_sweep,
    _union,
)
from snakes_and_ladders.sim.graph import BoundaryCondition, lattice_graph

# One home for the seeded bond activations, in the module that pins the
# labelling itself; building them again here would be a second fixture to
# keep in step with it.
from tests.regression.search.test_potts_mcmc_clusters import _seeded_bond_sets

pytest.importorskip("rustworkx")
from snakes_and_ladders.sandbox.rustworkx_clusters import (
    cluster_labels,
    swendsen_wang_sweep,
)

EXTENTS = (4, 8, 16)
N_DRAWS = 200
COUPLING = 0.6
FIELD = np.array([0.30, -0.10, -0.20])

#: The first cluster index at which the two visiting orders disagree, on the
#: extent-8 lattice at seed 8. Named so the renumbering is a number rather
#: than an adjective; see this module's docstring.
FIRST_DIVERGENCE = 15

#: Of :data:`N_DRAWS` seeded bond sets per extent, how many label arrays the
#: two labellings disagree on. Six of the extent-4 draws coincide, and every
#: one of those opened three bonds or fewer: with nothing but singletons and a
#: pair or two, the union-find roots are already ``arange`` and already in
#: ``rustworkx``'s discovery order. The measured counts are pinned rather than
#: rounded up to "all", because a labelling that coincided more often would be
#: a different finding.
RENUMBERED = {4: 194, 8: 200, 16: 200}


def _lattice(extent: int) -> tuple[int, _Bonds]:
    """The node count and bond arrays `_seeded_bond_sets` drew its activations on."""
    graph = lattice_graph((extent, extent), BoundaryCondition.OPEN, COUPLING)
    return graph.n_nodes, _bonds_of(graph)


def _partition(labels: np.ndarray) -> frozenset[frozenset[int]]:
    return frozenset(
        frozenset(np.flatnonzero(labels == value).tolist())
        for value in np.unique(labels)
    )


def _visiting_order(labels: np.ndarray) -> list[int]:
    """The lowest-indexed member of each cluster, in the order the sweep recolours them."""
    order = np.argsort(labels, kind="stable")
    grouped = labels[order]
    starts = np.flatnonzero(np.r_[True, grouped[1:] != grouped[:-1]])
    return [int(order[start]) for start in starts]


@pytest.mark.oracle
@pytest.mark.critical
@pytest.mark.parametrize("extent", EXTENTS)
def test_rustworkx_finds_the_partition_the_adopted_labelling_finds(extent: int) -> None:
    n_nodes, bonds = _lattice(extent)
    for parent, active in _seeded_bond_sets(extent, N_DRAWS):
        theirs = cluster_labels(n_nodes, bonds, active)

        assert _partition(_roots(parent.copy())) == _partition(theirs)


@pytest.mark.structural
@pytest.mark.parametrize("extent", EXTENTS)
def test_rustworkx_renumbers_the_clusters_the_roots_index_by_site(extent: int) -> None:
    # The difference the decline rests on, asserted rather than sorted away.
    # `_roots` labels a cluster by a *site index*, its union-find root, so its
    # labels are sparse in `[0, n_nodes)`; `connected_components` labels by a
    # counter dense in `[0, n_clusters)`. Equal arrays would mean the two
    # agree on the numbering as well as the partition, and the sweeps below
    # would then be the same chain.
    n_nodes, bonds = _lattice(extent)
    differed, coinciding_bonds = 0, []
    for parent, active in _seeded_bond_sets(extent, N_DRAWS):
        ours, theirs = _roots(parent.copy()), cluster_labels(n_nodes, bonds, active)
        assert set(theirs.tolist()) == set(range(len(np.unique(theirs))))
        if np.array_equal(ours, theirs):
            coinciding_bonds.append(int(active.sum()))
        else:
            differed += 1

    assert differed == RENUMBERED[extent]
    assert max(coinciding_bonds, default=0) <= 3


@pytest.mark.structural
def test_the_visiting_orders_diverge_at_a_named_cluster() -> None:
    # A number, not an adjective: the two orders agree on the first fifteen
    # clusters of this bond set and disagree from the sixteenth. A rustworkx
    # release that changed its scan would move this index, which is exactly
    # the event the conserved front exists to surface.
    n_nodes, bonds = _lattice(8)
    parent, active = _seeded_bond_sets(8, 1)[0]
    ours = _visiting_order(_roots(parent.copy()))
    theirs = _visiting_order(cluster_labels(n_nodes, bonds, active))
    assert len(ours) == len(theirs)
    divergence = next(
        index for index, (a, b) in enumerate(zip(ours, theirs, strict=True)) if a != b
    )

    assert divergence == FIRST_DIVERGENCE


@pytest.mark.oracle
@pytest.mark.parametrize("extent", [4, 8])
def test_the_front_s_sweep_draws_the_same_bonds_and_reaches_a_different_state(
    extent: int,
) -> None:
    # Both halves in one test, because either alone would mislead. The bond
    # draw is identical --- same generator, same first call --- so the two
    # sweeps partition the lattice identically; the recolouring then walks
    # those clusters in different orders and consumes the same draws for
    # different clusters, so the states diverge. That is the different Markov
    # chain the decline names, at one sweep.
    n_nodes, bonds = _lattice(extent)
    start = np.random.default_rng(extent).integers(0, FIELD.shape[0], size=n_nodes)

    ours, theirs = start.copy(), start.copy()
    _swendsen_wang_sweep(ours, bonds, FIELD, np.random.default_rng(7))
    swendsen_wang_sweep(theirs, bonds, FIELD, np.random.default_rng(7))

    bond_rng = np.random.default_rng(7)
    like = start[bonds.first] == start[bonds.second]
    active = like & (bond_rng.random(bonds.first.shape[0]) < bonds.activation)
    parent = np.arange(n_nodes)
    for edge in np.flatnonzero(active):
        _union(parent, int(bonds.first[edge]), int(bonds.second[edge]))
    assert _partition(_roots(parent)) == _partition(
        cluster_labels(n_nodes, bonds, active)
    )

    assert not np.array_equal(ours, theirs)


@pytest.mark.oracle
def test_sixty_sweeps_of_each_stay_valid_colourings_and_stay_apart() -> None:
    # The chains do not reconverge, and neither leaves the state space: every
    # site holds a colour the field is defined on. Sixty sweeps is what
    # experiment 001 measures autocorrelation over, which is the quantity the
    # renumbering would move.
    n_nodes, bonds = _lattice(8)
    start = np.random.default_rng(11).integers(0, FIELD.shape[0], size=n_nodes)
    ours, theirs = start.copy(), start.copy()
    mine, front = np.random.default_rng(3), np.random.default_rng(3)
    for _ in range(60):
        _swendsen_wang_sweep(ours, bonds, FIELD, mine)
        swendsen_wang_sweep(theirs, bonds, FIELD, front)

    assert set(ours.tolist()) <= set(range(FIELD.shape[0]))
    assert set(theirs.tolist()) <= set(range(FIELD.shape[0]))
    assert not np.array_equal(ours, theirs)


@pytest.mark.edge_case
def test_a_lattice_with_no_open_bond_is_every_site_its_own_cluster() -> None:
    # The corner the graph construction has to get right: an isolated site is
    # still a cluster and still gets recoloured, so the node set is added
    # whether or not a bond reaches it.
    n_nodes, bonds = _lattice(4)
    none_open = np.zeros(bonds.first.shape[0], dtype=bool)

    labels = cluster_labels(n_nodes, bonds, none_open)

    assert _partition(labels) == frozenset(frozenset({site}) for site in range(n_nodes))
