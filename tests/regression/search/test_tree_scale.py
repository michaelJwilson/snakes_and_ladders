"""The leaf-count ladder, and the optimizations measured on it (issue #582).

Step 1 of #582 asked where a ladder over leaf count should stop, and answered
that it does not: neighbour joining recovers the generating topology exactly
at 8, 20, 50, 100 and 200 leaves, so leaf count is a *cost* axis and not a
difficulty axis. The fixtures under ``tests/regression/fixtures/tree_scale/``
are that ladder, kept for what it does measure --- the size the post-order
optimizations of this branch are paid for at --- with 20 leaves the rung a
pull request runs and 50 and 200 behind the stress and release gates.

What each case asserts is the simulated truth, which is what
``oracle: none`` commits the fixture to: the topology the file declares is
recovered, and the fit reaches the same maximum a full re-score does. There
is no enumeration at 20 leaves --- 2.2e20 unrooted topologies --- and root
`CLAUDE.md` admits recovery of the generating structure where no oracle is
affordable.
"""

from __future__ import annotations

import numpy as np
import pytest
from snakes_and_ladders.likelihood.distance import DistanceKind, distance_matrix
from snakes_and_ladders.search.infer import infer, score_topology
from snakes_and_ladders.search.neighbor_joining import neighbor_joining
from snakes_and_ladders.sim.params import SimulationParams
from snakes_and_ladders.sim.simulator import simulate_tree
from snakes_and_ladders.sim.topology import normalized_robinson_foulds
from snakes_and_ladders.sim.tree import Node, balanced_tree, preorder

from tests._scale import Fixture, at_fixture

#: Candidates a search may score at this rung. 12 rather than more because
#: the rung costs 3.8 s at 12 and 3.6 s at 6 -- the floor is the initial fit
#: and the final re-score, not the search, and `test_search_exhaustive.py` is
#: where the search itself is refereed against enumeration.
_BUDGET = 12

#: Candidates fitted per neighbourhood once the neighbourhood is ranked by
#: one likelihood evaluation each (issue #289). Measured at 6.29x the full
#: fit at 20 leaves, at an identical log-likelihood and an identical
#: topology, which is what brings this rung inside the per-test cap.
_LAZY_TOP = 4


def _dataset(
    instance: Fixture[SimulationParams],
) -> tuple[Node, dict[str, np.ndarray], int]:
    params = instance.params
    dataset = simulate_tree(params, np.random.default_rng(params.seed))
    return params.tau, dict(dataset.alignment), params.k


@pytest.mark.end2end
@at_fixture("instance", "tree_scale")
def test_neighbour_joining_recovers_the_declared_topology(
    instance: Fixture[SimulationParams],
) -> None:
    # The step 1 result, pinned rather than remembered: this is the
    # measurement that says leaf count is a cost axis, so if it ever stops
    # holding the ladder's premise has changed and the fixtures are wrong.
    truth, alignment, k = _dataset(instance)
    names, matrix, _ = distance_matrix(alignment, k, DistanceKind.JUKES_CANTOR)

    recovered = neighbor_joining(names, matrix)

    assert float(normalized_robinson_foulds(recovered, truth)) == 0.0


@pytest.mark.end2end
@at_fixture("instance", "tree_scale")
def test_a_budgeted_search_reaches_its_own_maximum_at_this_size(
    instance: Fixture[SimulationParams],
) -> None:
    # What the optimizations are exercised by. The leaf-partial cache and the
    # post-order's dropped allocations are always on, `lazy_top` is the knob
    # of issue #289, and every one of them is claimed bitwise or identical --
    # so the assertion here is the ordinary one, and a change that broke any
    # of them shows up as a fit that no longer maximizes.
    truth, alignment, _ = _dataset(instance)

    result = infer(
        alignment,
        4,
        topology=truth,
        max_evaluations=_BUDGET,
        lazy_top=_LAZY_TOP,
    )

    assert result.log_likelihood == pytest.approx(
        score_topology(result.topology, alignment, 4), rel=1e-8
    )
    assert result.evaluations <= _BUDGET


@pytest.mark.smoke
def test_the_declared_height_bounds_every_root_to_tip_path() -> None:
    # Why the ladder is expressible at all: holding the *edge* length fixed
    # grows the diameter with the leaf count, and by 20 leaves distant pairs
    # saturate -- a run that did so refused a pair differing on 0.7550 of
    # sites, past the Jukes-Cantor 0.75. Every rung shares one height, so the
    # rungs differ in leaf count and in nothing else.
    for n_taxa in (8, 20, 50, 200):
        tree = balanced_tree(n_taxa, 0.25)
        assert sum(1 for node in preorder(tree) if node.is_leaf) == n_taxa
        assert max(_root_to_tip(tree)) == pytest.approx(0.25)


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("n_taxa", "height", "match"),
    [
        (2, 0.25, "at least 3 leaves"),
        (20, 0.0, "height is positive"),
    ],
)
def test_an_unusable_declaration_is_refused(
    n_taxa: int, height: float, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        balanced_tree(n_taxa, height)


def _root_to_tip(node: Node, accumulated: float = 0.0) -> list[float]:
    if node.is_leaf:
        return [accumulated]
    return [
        length
        for child in node.children
        for length in _root_to_tip(child, accumulated + (child.branch_length or 0.0))
    ]
