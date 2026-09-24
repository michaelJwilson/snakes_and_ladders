"""Regression test for snakes_and_ladders.qa.sim_example.

Pins the caption's content against the generating parameters and the
displayed tree against the fixture's, not just that the figure renders without
raising (CLAUDE.md's no-coverage-theatre rule). The script's `main` is a row
of `test_qa_runner.py::test_main_writes_its_output_and_caption`.
"""

from __future__ import annotations

import pytest
from snakes_and_ladders.fixtures import load_params
from snakes_and_ladders.qa.figure import state_label
from snakes_and_ladders.qa.sim_example import (
    build_caption,
    display_newick,
)
from snakes_and_ladders.sim.params import SimulationParams
from snakes_and_ladders.sim.tree import Node, preorder

from tests._fixtures import FIXTURES_DIR

PARAMS_PATH = FIXTURES_DIR / "tree_jc/stress.yaml"


@pytest.mark.smoke
def test_state_label_is_nucleotide_coded_for_k_four() -> None:
    assert [state_label(i, k=4) for i in range(4)] == ["A", "C", "G", "T"]


@pytest.mark.smoke
def test_state_label_falls_back_to_digit_for_other_k() -> None:
    assert state_label(2, k=3) == "2"


@pytest.mark.smoke
def test_n_sites_shown_is_capped_at_the_fixture_site_count() -> None:
    params = load_params(PARAMS_PATH, SimulationParams)
    caption = build_caption(params, n_sites_shown=params.n_sites + 1000)
    assert str(params.n_sites) in caption


@pytest.mark.smoke
def test_display_newick_uses_rho_for_the_root_and_greek_for_ancestors() -> None:
    # The raw serialization keeps names like "ancestor_CD"; the display form
    # replaces them, because an unescaped underscore is mathtext syntax and
    # the meaning is already visible from the tree.
    params = load_params(PARAMS_PATH, SimulationParams)

    rendered = display_newick(params.tau)

    assert rendered.endswith(r"\rho")
    assert r"\alpha" in rendered
    assert "ancestor" not in rendered
    # Leaf labels carry their branch length, escaped for mathtext.
    assert r"A\_0.1" in rendered
    assert r"D\_0.4" in rendered


@pytest.mark.smoke
def test_display_newick_names_every_leaf_exactly_once() -> None:
    params = load_params(PARAMS_PATH, SimulationParams)
    leaves = [node.name for node in preorder(params.tau) if node.is_leaf]

    rendered = display_newick(params.tau)

    for leaf in leaves:
        assert rendered.count(f"{leaf}\\_") == 1


@pytest.mark.smoke
def test_display_newick_refuses_a_tree_with_too_many_ancestors() -> None:
    # Guards the guard: the symbol table is finite, and running off its end
    # must say so rather than raise IndexError from inside a comprehension.
    deep: Node = Node(name="leaf", branch_length=0.1)
    for index in range(9):
        deep = Node(
            name=f"internal_{index}",
            branch_length=0.1,
            children=(deep, Node(name=f"leaf_{index}", branch_length=0.1)),
        )
    root = Node(name="root", branch_length=None, children=(deep,))

    with pytest.raises(ValueError, match="more than 8 internal ancestors"):
        display_newick(root)
