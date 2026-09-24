"""Regression tests for the two search QA figures.

The sweeps behind these figures are a candidate fit per topology --- 105 for
the trajectory figure and, since issue #498 moved it to the 5-taxon fixture,
15 for the comparison --- which belongs
in the document build rather than the per-PR suite. What runs per
PR is what can be wrong cheaply: the caption, which must report the numbers
it was handed rather than numbers somebody typed, and the shape of what the
generating functions return. The end-to-end renders are release-gated.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from snakes_and_ladders.fixtures import load_params
from snakes_and_ladders.qa import search_topologies, search_trajectory
from snakes_and_ladders.sim.params import SimulationParams
from snakes_and_ladders.sim.tree import Node, preorder

from tests._fixtures import FIXTURES_DIR

# The manifest's instances since #498: 5 taxa enumerate the comparison's
# runner-up; the trajectory needs a climb, which 5 taxa end at once.
TRAJECTORY_FIXTURE = FIXTURES_DIR / "tree_search/stress.yaml"
COMPARISON_FIXTURE = FIXTURES_DIR / "tree_search/ci.yaml"


def _params(path: Path) -> SimulationParams:
    return load_params(path, SimulationParams)


# --- the trajectory figure ----------------------------------------------


@pytest.mark.smoke
def test_trajectory_caption_reports_the_environment_it_was_given() -> None:
    params = _params(TRAJECTORY_FIXTURE)
    trajectories = {
        "nni": [(0, -9000.0), (14, -8775.5)],
        "spr": [(0, -9000.0), (48, -8775.5)],
    }
    environment = np.array([-9000.0, -8900.0, -8817.1, -8775.5])

    _, caption = search_trajectory.build_figure(
        trajectories, -8775.5, environment, -8775.5, params
    )

    assert f"All {environment.size} unrooted topologies" in caption
    assert "41.6 log units" in caption
    assert str(params.seed) in caption


@pytest.mark.oracle
@pytest.mark.release
def test_the_environment_is_every_topology_sorted_ascending() -> None:
    # The oracle the figure rests on: 6 taxa is 105 unrooted topologies, and
    # the panel is meaningless if the sweep missed any.
    _, truth, environment, reached = search_trajectory.search_trajectories(
        _params(TRAJECTORY_FIXTURE)
    )

    assert environment.size == 105
    assert list(environment) == sorted(environment)
    assert reached == pytest.approx(environment[-1], abs=1e-5)
    assert truth == pytest.approx(environment[-1], abs=1e-5)


# --- the topology comparison --------------------------------------------


@pytest.mark.smoke
def test_comparison_caption_reports_both_scores_and_the_split() -> None:
    params = _params(COMPARISON_FIXTURE)
    found = Node(
        name="root",
        branch_length=None,
        children=(
            Node("A", 0.1),
            Node("B", 0.2),
            Node("i", 0.05, (Node("C", 0.1), Node("D", 0.1))),
        ),
    )
    other = Node(
        name="root",
        branch_length=None,
        children=(
            Node("A", 0.1),
            Node("C", 0.2),
            Node("j", 0.0, (Node("B", 0.1), Node("D", 0.1))),
        ),
    )

    _, caption = search_topologies.build_figure(
        found, -8775.5, other, -8817.1, frozenset({"C", "D"}), True, params
    )

    assert "-8775.5" in caption
    assert "-8817.1" in caption
    assert "41.6 log units" in caption
    assert "C D are a group" in caption
    assert "is the generating topology" in caption
    assert "fitted at 0.000" in caption
    assert not set(caption.replace("\\_", "")) & set("_%\\&#")


@pytest.mark.smoke
def test_comparison_caption_says_so_when_the_truth_was_not_found() -> None:
    # The flag has to be able to read both ways, or it is decoration.
    params = _params(COMPARISON_FIXTURE)
    tree = Node(
        name="root",
        branch_length=None,
        children=(
            Node("A", 0.1),
            Node("B", 0.2),
            Node("i", 0.05, (Node("C", 0.1), Node("D", 0.1))),
        ),
    )

    _, caption = search_topologies.build_figure(
        tree, -1.0, tree, -2.0, frozenset({"C"}), False, params
    )

    assert "is not the generating topology" in caption


@pytest.mark.end2end
@pytest.mark.release
def test_the_search_finds_the_generating_tree_and_rejects_a_worse_one() -> None:
    found, found_score, other, other_score, difference, recovered = (
        search_topologies.found_and_runner_up(_params(COMPARISON_FIXTURE))
    )

    assert recovered
    assert found_score > other_score
    assert difference
    for root in (found, other):
        for node in preorder(root):
            assert node is root or node.branch_length is not None
