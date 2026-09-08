"""Neighbor joining is exact on additive distances and recovers the truth inside Atteson's radius.

Issue #364. On the path-length matrix of any tree the algorithm must return
that tree with every branch length exact --- the fixtures at four to eight
taxa and random trees at 20 and 50 --- which is the closed-form statement
a moment estimator is held to. On simulated alignments the guarantee is
Atteson's: the topology is recovered whenever every distance error is under
half the shortest branch, and the recovery rate rises with the site count.
"""

from __future__ import annotations

import numpy as np
import pytest
from snakes_and_ladders.likelihood.distance import distance_matrix, tree_distances
from snakes_and_ladders.search.neighbor_joining import (
    atteson_radius,
    four_point_violation,
    neighbor_joining,
    split_lengths,
)
from snakes_and_ladders.search.topology import leaf_bipartitions, random_topology
from snakes_and_ladders.sim.simulate import simulate_alignment
from snakes_and_ladders.sim.tree import Node

from tests._fixtures import EIGHT_TAXA, FOUR_TAXA, load_fixture

FIVE_TAXA = "tree_search/ci.yaml"
SIX_TAXA = "tree_search/stress.yaml"
HARD = "tree_search/release.yaml"
FIXTURES = (FOUR_TAXA, FIVE_TAXA, SIX_TAXA, HARD, EIGHT_TAXA)

#: Every branch length is recovered to this on an additive matrix.
EXACT = 1e-12


def random_tree(
    n_taxa: int, rng: np.random.Generator, lengths: tuple[float, float] = (0.02, 0.4)
) -> Node:
    """A random topology on ``n_taxa`` leaves with lengths uniform on ``lengths``."""
    topology = random_topology([f"t{index:02d}" for index in range(n_taxa)], rng)

    def with_lengths(node: Node, is_root: bool) -> Node:
        return Node(
            name=node.name,
            branch_length=None if is_root else float(rng.uniform(*lengths)),
            children=tuple(with_lengths(child, False) for child in node.children),
        )

    return with_lengths(topology, True)


def _assert_recovered(truth: Node, estimate: Node) -> None:
    assert leaf_bipartitions(estimate) == leaf_bipartitions(truth)
    lengths = split_lengths(truth)
    recovered = split_lengths(estimate)
    assert recovered.keys() == lengths.keys()
    for split, length in lengths.items():
        assert recovered[split] == pytest.approx(length, abs=EXACT)


@pytest.mark.mathematical
@pytest.mark.parametrize("name", FIXTURES)
def test_the_fixture_tree_is_recovered_exactly_from_its_path_lengths(name: str) -> None:
    """Topology and every branch length, to ``1e-12``, on the additive matrix.

    The rooted binary eight-taxon fixture recovers as its unrooted form,
    the two branches below the root as their sum.
    """
    params = load_fixture(name)
    names, distances = tree_distances(params.tau)

    _assert_recovered(params.tau, neighbor_joining(names, distances))


@pytest.mark.mathematical
@pytest.mark.parametrize("n_taxa", [20, 50])
def test_a_random_tree_is_recovered_exactly_from_its_path_lengths(n_taxa: int) -> None:
    """Ten random trees at each size, every one recovered to ``1e-12``."""
    for seed in range(10):
        truth = random_tree(n_taxa, np.random.default_rng([n_taxa, seed]))
        names, distances = tree_distances(truth)

        _assert_recovered(truth, neighbor_joining(names, distances))


@pytest.mark.mathematical
def test_the_four_point_condition_holds_on_additive_distances_and_fails_off_them() -> (
    None
):
    """Zero to floating point on a tree's path lengths; positive once one entry moves."""
    truth = random_tree(12, np.random.default_rng(3))
    _, distances = tree_distances(truth)

    assert four_point_violation(distances) < 1e-12

    perturbed = distances.copy()
    perturbed[0, 1] = perturbed[1, 0] = distances[0, 1] + 0.05
    assert four_point_violation(perturbed) == pytest.approx(0.05)


@pytest.mark.mathematical
def test_attesons_radius_is_half_the_shortest_branch() -> None:
    """On the six-taxon fixture the shortest branch is 0.06, so the radius is 0.03."""
    params = load_fixture(SIX_TAXA)

    assert atteson_radius(params.tau) == pytest.approx(0.03)
    assert atteson_radius(load_fixture(EIGHT_TAXA).tau) == pytest.approx(0.025)


@pytest.mark.simulated_truth
def test_recovery_rises_with_sites_and_is_certain_inside_attesons_radius() -> None:
    """The six-taxon fixture over 50 seeds at 100, 300, 1,500 and 10,000 sites.

    Two claims. The theorem: on every replicate whose largest distance error
    is under the radius, 0.03, the topology is the generating one --- no
    exception allowed. The trend: the recovery rate does not fall as the
    sites grow, and is 1 at the largest size. Realized recovery: 0.72, 0.96,
    1.00 and 1.00; replicates inside the radius: 0, 0, 2 and 46 of 50 --- the
    radius is a sufficient condition, and the topology is recovered on every
    replicate at 1,500 sites while only 2 sit inside it.
    """
    params = load_fixture(SIX_TAXA)
    radius = atteson_radius(params.tau)
    names, truth = tree_distances(params.tau)
    rates: list[float] = []
    inside_counts: list[int] = []
    for n_sites in (100, 300, 1500, 10_000):
        recovered = 0
        inside = 0
        for seed in range(50):
            dataset = simulate_alignment(
                params.tau,
                params.k,
                params.pi,
                np.random.default_rng([364, n_sites, seed]),
                n_sites,
            )
            estimated_names, distances, _ = distance_matrix(dataset.alignment, params.k)
            assert estimated_names == names
            hit = leaf_bipartitions(
                neighbor_joining(names, distances)
            ) == leaf_bipartitions(params.tau)
            if np.abs(distances - truth).max() < radius:
                inside += 1
                assert hit, (n_sites, seed)
            recovered += int(hit)
        rates.append(recovered / 50)
        inside_counts.append(inside)

    assert rates == sorted(rates), rates
    assert rates[0] < 1.0
    assert rates[-1] == 1.0
    assert inside_counts[-1] >= 40, inside_counts


@pytest.mark.edge_case
def test_unusable_matrices_are_refused() -> None:
    with pytest.raises(ValueError, match="at least 3 taxa"):
        neighbor_joining(["A", "B"], np.zeros((2, 2)))
    with pytest.raises(ValueError, match="distinct"):
        neighbor_joining(["A", "A", "B"], np.zeros((3, 3)))
    with pytest.raises(ValueError, match="shape"):
        neighbor_joining(["A", "B", "C"], np.zeros((2, 2)))
    asymmetric = np.array([[0.0, 1.0, 1.0], [2.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    with pytest.raises(ValueError, match="symmetric"):
        neighbor_joining(["A", "B", "C"], asymmetric)
    with pytest.raises(ValueError, match="at least 4 taxa"):
        four_point_violation(np.zeros((3, 3)))
