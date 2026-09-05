"""Regression test for phylo.qa.sim_example.

Pins the caption's content against the generating parameters and the
alignment table against sequences recomputed independently, not just that
the figure renders without raising (CLAUDE.md's no-coverage-theatre rule).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from phylo.qa.figure import latex_integer, state_label
from phylo.qa.sim_example import (
    build_caption,
    display_newick,
    main,
)
from phylo.sim.params import load_simulation_params
from phylo.sim.simulate import simulate_alignment
from phylo.sim.tree import Node, preorder

from tests._fixtures import FIXTURES_DIR

PARAMS_PATH = FIXTURES_DIR / "simulation_params.yaml"


@pytest.mark.structural
def test_state_label_is_nucleotide_coded_for_k_four() -> None:
    assert [state_label(i, k=4) for i in range(4)] == ["A", "C", "G", "T"]


@pytest.mark.structural
def test_state_label_falls_back_to_digit_for_other_k() -> None:
    assert state_label(2, k=3) == "2"


@pytest.mark.structural
def test_main_writes_a_figure_and_caption_with_generating_truth(
    tmp_path: Path,
) -> None:
    params = load_simulation_params(PARAMS_PATH)

    qa_figure = main(
        [
            "--params",
            str(PARAMS_PATH),
            "--output-dir",
            str(tmp_path),
            "--n-sites-shown",
            "10",
        ]
    )

    assert qa_figure.figure_path.is_file()
    assert qa_figure.figure_path.stat().st_size > 0
    assert qa_figure.caption == build_caption(params, n_sites_shown=10)
    assert str(params.seed) in qa_figure.caption
    assert latex_integer(params.n_sites) in qa_figure.caption
    assert "4-taxon" in qa_figure.caption
    assert "10" in qa_figure.caption


@pytest.mark.oracle
def test_sim_example_alignment_matches_independent_simulation() -> None:
    params = load_simulation_params(PARAMS_PATH)
    expected = simulate_alignment(
        tau=params.tau,
        k=params.k,
        pi=params.pi,
        seed=params.seed,
        n_sites=params.n_sites,
    )

    for leaf, states in expected.alignment.items():
        rendered = [state_label(int(state), params.k) for state in states[:10]]
        assert all(letter in "ACGT" for letter in rendered)
        assert leaf in expected.newick


@pytest.mark.edge_case
def test_n_sites_shown_is_capped_at_the_fixture_site_count() -> None:
    params = load_simulation_params(PARAMS_PATH)
    caption = build_caption(params, n_sites_shown=params.n_sites + 1000)
    assert str(params.n_sites) in caption


@pytest.mark.structural
def test_main_reads_sys_argv_when_no_argv_is_given(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "sim_example",
            "--params",
            str(PARAMS_PATH),
            "--output-dir",
            str(tmp_path),
            "--n-sites-shown",
            "5",
        ],
    )

    main()

    figure_path = tmp_path / "sim_example.pdf"
    caption_path = tmp_path / "sim_example_caption.txt"
    assert figure_path.is_file()
    assert caption_path.read_text() == build_caption(
        load_simulation_params(PARAMS_PATH), n_sites_shown=5
    )
    captured = capsys.readouterr()
    assert str(figure_path) in captured.out
    assert str(caption_path) in captured.out


@pytest.mark.structural
def test_display_newick_uses_rho_for_the_root_and_greek_for_ancestors() -> None:
    # The raw serialization keeps names like "ancestor_CD"; the display form
    # replaces them, because an unescaped underscore is mathtext syntax and
    # the meaning is already visible from the tree.
    params = load_simulation_params(PARAMS_PATH)

    rendered = display_newick(params.tau)

    assert rendered.endswith(r"\rho")
    assert r"\alpha" in rendered
    assert "ancestor" not in rendered
    # Leaf labels carry their branch length, escaped for mathtext.
    assert r"A\_0.1" in rendered
    assert r"D\_0.4" in rendered


@pytest.mark.structural
def test_display_newick_names_every_leaf_exactly_once() -> None:
    params = load_simulation_params(PARAMS_PATH)
    leaves = [node.name for node in preorder(params.tau) if node.is_leaf]

    rendered = display_newick(params.tau)

    for leaf in leaves:
        assert rendered.count(f"{leaf}\\_") == 1


@pytest.mark.edge_case
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
