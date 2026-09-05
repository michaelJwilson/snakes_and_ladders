"""Regression tests for the two search QA figures.

The sweeps behind these figures are 105 candidate fits apiece, which belongs
in the technical-document build rather than the per-PR suite. What runs per
PR is what can be wrong cheaply: the caption, which must report the numbers
it was handed rather than numbers somebody typed, and the shape of what the
generating functions return. The end-to-end renders are release-gated.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from phylo.qa import search_topologies, search_trajectory
from phylo.sim.params import SimulationParams, load_simulation_params
from phylo.sim.tree import Node, preorder

from tests._fixtures import FIXTURES_DIR

FIXTURE = FIXTURES_DIR / "simulation_params_6taxa.yaml"

# A 5-taxon tree is 15 unrooted topologies against the 6-taxon fixture's 105,
# so the same sweep runs in a few seconds. Written per test rather than added
# as a fixture file: nothing is asserted against its truth, and the
# repository's fixtures are for data that is.
_SMALL_PARAMS = """
seed: 20260906
n_sites: 400
tolerance: 0.01
k: 4
pi: [0.25, 0.25, 0.25, 0.25]
tau:
  name: root
  children:
    - name: A
      branch_length: 0.12
    - name: B
      branch_length: 0.28
    - name: ancestor_CDE
      branch_length: 0.06
      children:
        - name: C
          branch_length: 0.21
        - name: ancestor_DE
          branch_length: 0.09
          children:
            - name: D
              branch_length: 0.07
            - name: E
              branch_length: 0.33
"""


def _small_fixture(tmp_path: Path) -> Path:
    path = tmp_path / "simulation_params_5taxa.yaml"
    path.write_text(_SMALL_PARAMS)
    return path


def _params() -> SimulationParams:
    return load_simulation_params(FIXTURE)


# --- the trajectory figure ----------------------------------------------


@pytest.mark.structural
def test_trajectory_caption_reports_the_landscape_it_was_given() -> None:
    params = _params()
    trajectories = {
        "nni": [(0, -9000.0), (14, -8775.5)],
        "spr": [(0, -9000.0), (48, -8775.5)],
    }
    landscape = np.array([-9000.0, -8900.0, -8817.1, -8775.5])

    _, caption = search_trajectory.build_figure(
        trajectories, -8775.5, landscape, -8775.5, params
    )

    assert f"All {landscape.size} unrooted topologies" in caption
    assert "41.6 log units" in caption
    assert str(params.seed) in caption


@pytest.mark.structural
def test_trajectory_caption_is_latex_safe() -> None:
    params = _params()
    _, caption = search_trajectory.build_figure(
        {"nni": [(0, -1.0), (2, -0.5)]}, -0.5, np.array([-1.0, -0.5]), -0.5, params
    )

    # qa/CLAUDE.md: captions are plain text pulled into LaTeX verbatim. The
    # only escape permitted is \_, so removing those must leave no special
    # behind. write_qa_figure enforces this too; asserting it here keeps the
    # check on build_figure, which a caller could reach directly.
    assert not set(caption.replace("\\_", "")) & set("_%\\&#")


@pytest.mark.structural
def test_trajectory_main_writes_a_figure_and_caption(tmp_path: Path) -> None:
    # At 5 taxa, so the sweep is 15 fits rather than 105 and the whole
    # pipeline -- searches, enumeration, caption, render -- is still
    # exercised per PR.
    written = search_trajectory.main(
        [
            "--params",
            str(_small_fixture(tmp_path)),
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert written.figure_path.is_file()
    assert written.caption_path.read_text() == written.caption
    assert "All 15 unrooted topologies" in written.caption


@pytest.mark.oracle
@pytest.mark.release
def test_the_landscape_is_every_topology_sorted_ascending() -> None:
    # The oracle the figure rests on: 6 taxa is 105 unrooted topologies, and
    # the panel is meaningless if the sweep missed any.
    _, truth, landscape, reached = search_trajectory.search_trajectories(_params())

    assert landscape.size == 105
    assert list(landscape) == sorted(landscape)
    assert reached == pytest.approx(landscape[-1], abs=1e-5)
    assert truth == pytest.approx(landscape[-1], abs=1e-5)


# --- the topology comparison --------------------------------------------


@pytest.mark.structural
def test_comparison_caption_reports_both_scores_and_the_split() -> None:
    params = _params()
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


@pytest.mark.structural
def test_comparison_caption_says_so_when_the_truth_was_not_found() -> None:
    # The flag has to be able to read both ways, or it is decoration.
    params = _params()
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


@pytest.mark.structural
def test_comparison_main_writes_a_figure_and_caption(tmp_path: Path) -> None:
    written = search_topologies.main(
        [
            "--params",
            str(_small_fixture(tmp_path)),
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert written.figure_path.is_file()
    assert written.caption_path.read_text() == written.caption
    assert "log units" in written.caption


@pytest.mark.simulated_truth
@pytest.mark.release
def test_the_search_finds_the_generating_tree_and_rejects_a_worse_one() -> None:
    found, found_score, other, other_score, difference, recovered = (
        search_topologies.found_and_runner_up(_params())
    )

    assert recovered
    assert found_score > other_score
    assert difference
    for root in (found, other):
        for node in preorder(root):
            assert node is root or node.branch_length is not None
