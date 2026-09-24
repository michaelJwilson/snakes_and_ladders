"""Neighbor joining is exact on additive distances and recovers the truth inside Atteson's radius.

Issue #364. On any tree's path-length matrix the tree returns with every length
exact (fixtures at four to eight taxa, random trees at 20 and 50). On
simulated alignments: Atteson's guarantee inside half the shortest branch,
and a recovery rate rising with sites. Against enumeration (issue #734) the
joined tree is the least-squares argmin at five to seven taxa, unique by 27
orders of magnitude.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest
from snakes_and_ladders.likelihood.distance import distance_matrix, tree_distances
from snakes_and_ladders.likelihood.surrogate import (
    least_squares_lengths,
    least_squares_residual,
)
from snakes_and_ladders.search.neighbor_joining import (
    atteson_radius,
    four_point_violation,
    neighbor_joining,
    split_lengths,
)
from snakes_and_ladders.sim.simulator import simulate_tree
from snakes_and_ladders.sim.topology import (
    enumerate_topologies,
    leaf_bipartitions,
    random_topology,
)
from snakes_and_ladders.sim.tree import Node

from tests._fixtures import EIGHT_TAXA, FOUR_TAXA, load_fixture
from tests._rows import every_value

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


@pytest.mark.analytic
@pytest.mark.parametrize("name", FIXTURES)
def test_the_fixture_tree_is_recovered_exactly_from_its_path_lengths(name: str) -> None:
    """Topology and every branch length, to ``1e-12``, on the additive matrix.

    The rooted eight-taxon fixture returns unrooted, its root pair summed.
    """
    params = load_fixture(name)
    names, distances = tree_distances(params.tau)

    _assert_recovered(params.tau, neighbor_joining(names, distances))


@pytest.mark.analytic
def test_a_random_tree_is_recovered_exactly_from_its_path_lengths() -> None:
    """Ten random trees at each size, every one recovered to ``1e-12``."""

    def check(n_taxa: int) -> None:
        for seed in range(10):
            truth = random_tree(n_taxa, np.random.default_rng([n_taxa, seed]))
            names, distances = tree_distances(truth)

            _assert_recovered(truth, neighbor_joining(names, distances))

    every_value([20, 50], check)


def _pairwise(
    names: Sequence[str], distances: np.ndarray
) -> dict[frozenset[str], float]:
    """The matrix as the mapping the least-squares fit reads, one entry per pair."""
    return {
        frozenset((first, second)): float(distances[row, column])
        for row, first in enumerate(names)
        for column, second in enumerate(names)
        if row < column
    }


@pytest.mark.oracle
@pytest.mark.critical
def test_the_joined_tree_is_the_least_squares_optimum_over_the_enumerated_topologies() -> (
    None
):
    # The rung below (#734): every topology (15, 105, 945) scored by the
    # least-squares fit of its path lengths (`least_squares_lengths`); on an
    # additive matrix exactly one attains zero. Over six trees: argmin residual
    # <= 3.3e-31 (1e-20), runner-up >= 1.9e-3 (1e-4).
    def check(n_taxa: int) -> None:
        for seed in range(2):
            truth = random_tree(n_taxa, np.random.default_rng([734, n_taxa, seed]))
            names, distances = tree_distances(truth)
            pairwise = _pairwise(names, distances)

            joined = neighbor_joining(names, distances)
            scored = sorted(
                (
                    least_squares_residual(
                        topology, pairwise, least_squares_lengths(topology, pairwise)
                    ),
                    index,
                    topology,
                )
                for index, topology in enumerate(enumerate_topologies(sorted(names)))
            )

            assert scored[0][0] < 1e-20
            assert scored[1][0] > 1e-4
            assert leaf_bipartitions(scored[0][2]) == leaf_bipartitions(joined)
            assert leaf_bipartitions(joined) == leaf_bipartitions(truth)

    every_value([5, 6, 7], check)


@pytest.mark.analytic
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


@pytest.mark.analytic
def test_attesons_radius_is_half_the_shortest_branch() -> None:
    """On the six-taxon fixture the shortest branch is 0.06, so the radius is 0.03."""
    params = load_fixture(SIX_TAXA)

    assert atteson_radius(params.tau) == pytest.approx(0.03)
    assert atteson_radius(load_fixture(EIGHT_TAXA).tau) == pytest.approx(0.025)


@pytest.mark.end2end
def test_recovery_rises_with_sites_and_is_certain_inside_attesons_radius() -> None:
    """The six-taxon fixture over 50 seeds at 100, 300, 1,500 and 10,000 sites.

    Inside radius 0.03 recovery is certain; the rate does not fall and is 1 at
    the largest. Recovery 0.72, 0.96, 1.00, 1.00; inside the radius 0, 0, 2, 46 of 50.
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
            dataset = simulate_tree(
                params, np.random.default_rng([364, n_sites, seed]), n_sites=n_sites
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


@pytest.mark.smoke
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
