"""Regression test for phylo.qa.sim_tree.

Pins the layout's numeric coordinates against branch lengths recomputed
independently, and the caption's content against the generating parameters
-- not just that the figure renders without raising (CLAUDE.md's
no-coverage-theatre rule).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pytest
from phylo.qa.figure import state_label
from phylo.qa.sim_tree import (
    SITES_SHOWN,
    build_caption,
    main,
    render_sim_tree,
    tree_layout,
)
from phylo.sim.params import load_simulation_params
from phylo.sim.simulate import simulate_alignment
from phylo.sim.tree import Node, preorder

from tests._fixtures import FIXTURES_DIR

PARAMS_PATH = FIXTURES_DIR / "simulation_params_8taxa.yaml"


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
    params = load_simulation_params(FIXTURES_DIR / "simulation_params_8taxa.yaml")
    layout = tree_layout(params.tau)

    for node in preorder(params.tau):
        expected = _expected_depth(params.tau, 0.0, node.name)
        assert expected is not None
        depth, _y = layout[node.name]
        assert depth == expected


@pytest.mark.structural
def test_tree_layout_gives_every_leaf_a_distinct_ordered_y() -> None:
    params = load_simulation_params(FIXTURES_DIR / "simulation_params_8taxa.yaml")
    layout = tree_layout(params.tau)
    leaves = [node.name for node in preorder(params.tau) if node.is_leaf]

    leaf_ys = [layout[name][1] for name in leaves]
    assert leaf_ys == list(range(len(leaves)))


@pytest.mark.structural
def test_main_writes_a_figure_and_caption_with_generating_truth(
    tmp_path: Path,
) -> None:
    params_path = FIXTURES_DIR / "simulation_params_8taxa.yaml"
    params = load_simulation_params(params_path)

    qa_figure = main(["--params", str(params_path), "--output-dir", str(tmp_path)])

    assert qa_figure.figure_path.is_file()
    assert qa_figure.figure_path.stat().st_size > 0
    assert qa_figure.caption == build_caption(params)
    assert str(params.seed) in qa_figure.caption
    assert "8 taxa" in qa_figure.caption
    assert "Jukes-Cantor" in qa_figure.caption


@pytest.mark.structural
def test_main_reads_sys_argv_when_no_argv_is_given(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    params_path = FIXTURES_DIR / "simulation_params_8taxa.yaml"
    monkeypatch.setattr(
        "sys.argv",
        ["sim_tree", "--params", str(params_path), "--output-dir", str(tmp_path)],
    )

    main()

    figure_path = tmp_path / "sim_tree.pdf"
    caption_path = tmp_path / "sim_tree_caption.txt"
    assert figure_path.is_file()
    assert caption_path.read_text() == build_caption(
        load_simulation_params(params_path)
    )
    captured = capsys.readouterr()
    assert str(figure_path) in captured.out
    assert str(caption_path) in captured.out


@pytest.mark.structural
def test_every_leaf_gets_its_own_sequence_aligned_to_its_row() -> None:
    # The figure's claim is that these sequences came from this tree, so the
    # check is that each leaf's text is its own simulated states, placed at
    # that leaf's y. Pinned against the alignment, not against the drawing.
    params = load_simulation_params(PARAMS_PATH)
    dataset = simulate_alignment(
        tau=params.tau,
        k=params.k,
        pi=params.pi,
        seed=params.seed,
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


@pytest.mark.edge_case
def test_no_sequences_are_drawn_when_no_alignment_is_given() -> None:
    # The parameter is optional, and a tree without an alignment must not
    # acquire an empty column of text.
    params = load_simulation_params(PARAMS_PATH)
    _, ax = plt.subplots()
    render_sim_tree(params.tau, ax)

    monospace = [text for text in ax.texts if text.get_fontfamily() == ["monospace"]]
    plt.close("all")

    assert monospace == []
