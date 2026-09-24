"""Regression test for snakes_and_ladders.qa.sim_tree.

Pins the layout's numeric coordinates against branch lengths recomputed
independently, and the caption's content against the generating parameters
-- not just that the figure renders without raising (CLAUDE.md's
no-coverage-theatre rule).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest
from snakes_and_ladders.fixtures import load_params
from snakes_and_ladders.qa.figure import state_label
from snakes_and_ladders.qa.sim_tree import (
    SITES_SHOWN,
    render_sim_tree,
    tree_layout,
)
from snakes_and_ladders.sim.params import SimulationParams
from snakes_and_ladders.sim.simulate import simulate_alignment
from snakes_and_ladders.sim.tree import Node, preorder

from tests._fixtures import FIXTURES_DIR

PARAMS_PATH = FIXTURES_DIR / "tree_jc/release.yaml"


def _expected_depth(node: Node, parent_depth: float, target: str) -> float | None:
    depth = parent_depth + (node.branch_length or 0.0)
    if node.name == target:
        return depth
    for child in node.children:
        found = _expected_depth(child, depth, target)
        if found is not None:
            return found
    return None


@pytest.mark.oracle
def test_tree_layout_depths_match_branch_length_sums() -> None:
    params = load_params(FIXTURES_DIR / "tree_jc/release.yaml", SimulationParams)
    layout = tree_layout(params.tau)

    for node in preorder(params.tau):
        expected = _expected_depth(params.tau, 0.0, node.name)
        assert expected is not None
        depth, _y = layout[node.name]
        assert depth == expected


@pytest.mark.smoke
def test_the_drawn_axis_puts_the_first_taxon_at_the_top() -> None:
    # The layout counts leaves downward in traversal order and matplotlib
    # counts y upward, so the figure reads in the fixture's own order only if
    # the axis is inverted. Asserted on the drawn axes rather than on the
    # layout, which is deliberately left in tree coordinates.
    params = load_params(PARAMS_PATH, SimulationParams)
    _figure, ax = plt.subplots()
    try:
        layout = render_sim_tree(params.tau, ax)
        bottom, top = ax.get_ylim()
        first = next(node.name for node in preorder(params.tau) if node.is_leaf)
        last = [node.name for node in preorder(params.tau) if node.is_leaf][-1]
        assert bottom > top, "the y axis is not inverted"
        assert layout[first][1] < layout[last][1]
    finally:
        plt.close(_figure)


@pytest.mark.smoke
def test_tree_layout_gives_every_leaf_a_distinct_ordered_y() -> None:
    params = load_params(FIXTURES_DIR / "tree_jc/release.yaml", SimulationParams)
    layout = tree_layout(params.tau)
    leaves = [node.name for node in preorder(params.tau) if node.is_leaf]

    leaf_ys = [layout[name][1] for name in leaves]
    assert leaf_ys == list(range(len(leaves)))


@pytest.mark.smoke
def test_every_leaf_gets_its_own_sequence_aligned_to_its_row() -> None:
    # The figure's claim is that these sequences came from this tree, so the
    # check is that each leaf's text is its own simulated states, placed at
    # that leaf's y. Pinned against the alignment, not against the drawing.
    params = load_params(PARAMS_PATH, SimulationParams)
    dataset = simulate_alignment(
        tau=params.tau,
        k=params.k,
        pi=params.pi,
        rng=np.random.default_rng(params.seed),
        n_sites=SITES_SHOWN,
    )
    _, ax = plt.subplots()
    layout = render_sim_tree(params.tau, ax, alignment=dataset.alignment, k=params.k)

    drawn = {
        text.get_text(): text.get_position()[1]
        for text in ax.texts
        if text.get_fontfamily() == ["monospace"]
    }
    plt.close("all")

    leaves = [node.name for node in preorder(params.tau) if node.is_leaf]
    assert len(drawn) == len(leaves)
    for leaf in leaves:
        expected = "".join(
            state_label(int(state), params.k)
            for state in dataset.alignment[leaf][:SITES_SHOWN]
        )
        assert expected in drawn
        assert drawn[expected] == layout[leaf][1]


@pytest.mark.smoke
def test_no_sequences_are_drawn_when_no_alignment_is_given() -> None:
    # The parameter is optional, and a tree without an alignment must not
    # acquire an empty column of text.
    params = load_params(PARAMS_PATH, SimulationParams)
    _, ax = plt.subplots()
    render_sim_tree(params.tau, ax)

    monospace = [text for text in ax.texts if text.get_fontfamily() == ["monospace"]]
    plt.close("all")

    assert monospace == []
