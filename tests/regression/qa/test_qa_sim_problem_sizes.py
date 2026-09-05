"""Regression test for phylo.qa.sim_problem_sizes.

Pins the tabulated values against each fixture's yaml, read independently,
not just that the figure renders without raising (CLAUDE.md's
no-coverage-theatre rule).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from phylo.qa.sim_problem_sizes import (
    build_caption,
    main,
)
from phylo.sim.params import load_simulation_params
from phylo.sim.tree import preorder

from tests._fixtures import FIXTURES_DIR

FIXTURE_NAMES = (
    "simulation_params.yaml",
    "simulation_params_small_sites.yaml",
    "simulation_params_8taxa.yaml",
)
FIXTURE_PATHS = [FIXTURES_DIR / name for name in FIXTURE_NAMES]


def _argv(output_dir: Path) -> list[str]:
    """Build the argument vector naming every fixture, in row order.

    Returns
    -------
    list[str]
        The ``--params``/``--output-dir`` arguments for :func:`main`.
    """
    argv: list[str] = []
    for path in FIXTURE_PATHS:
        argv += ["--params", str(path)]
    return [*argv, "--output-dir", str(output_dir)]


@pytest.mark.structural
def test_main_writes_a_table_and_caption_naming_every_fixture(
    tmp_path: Path,
) -> None:
    qa_table = main(_argv(tmp_path))

    assert qa_table.table_path.is_file()
    assert qa_table.table_path.suffix == ".tex"
    assert qa_table.caption == build_caption(list(FIXTURE_NAMES))
    assert str(len(FIXTURE_NAMES)) in qa_table.caption


@pytest.mark.structural
def test_the_table_is_a_latex_tabular_not_an_image(tmp_path: Path) -> None:
    # The point of the change: main.tex \input's a typeset table instead of
    # \includegraphics-ing a matplotlib rendering of one.
    body = main(_argv(tmp_path)).table_path.read_text()

    assert body.startswith(r"\begin{tabular}")
    assert body.rstrip().endswith(r"\end{tabular}")
    assert r"\toprule" in body
    assert r"\bottomrule" in body
    # One header row plus one row per fixture.
    assert body.count(r"\\") == 1 + len(FIXTURE_NAMES)


@pytest.mark.structural
def test_site_counts_are_separated_and_seeds_are_not(tmp_path: Path) -> None:
    # Sites are a magnitude and read better separated; a seed is an
    # identifier, and separating 20260902 would disguise a date as a
    # quantity.
    body = main(_argv(tmp_path)).table_path.read_text()

    assert r"200\_000" in body
    assert r"20\_000" in body
    assert "20260902" in body
    assert r"20\_260\_902" not in body


@pytest.mark.structural
def test_underscores_in_fixture_names_are_escaped(tmp_path: Path) -> None:
    # An unescaped underscore in a filename is a LaTeX error, and these
    # filenames all contain them.
    body = main(_argv(tmp_path)).table_path.read_text()

    assert r"simulation\_params\_8taxa.yaml" in body
    for line in body.splitlines():
        stripped = line.replace(r"\_", "")
        assert "_" not in stripped, line


@pytest.mark.oracle
def test_problem_sizes_values_match_each_fixture_independently() -> None:
    for path in FIXTURE_PATHS:
        params = load_simulation_params(path)
        n_taxa = sum(1 for node in preorder(params.tau) if node.is_leaf)

        # Cross-check against direct knowledge of the fixtures rather than
        # re-deriving through the module under test.
        if path.name == "simulation_params.yaml":
            assert (n_taxa, params.n_sites, params.seed, params.tolerance) == (
                4,
                200000,
                20260902,
                0.01,
            )
        elif path.name == "simulation_params_small_sites.yaml":
            assert (n_taxa, params.n_sites, params.seed, params.tolerance) == (
                4,
                20000,
                20260903,
                0.03,
            )
        elif path.name == "simulation_params_8taxa.yaml":
            assert (n_taxa, params.n_sites, params.seed, params.tolerance) == (
                8,
                200000,
                20260904,
                0.01,
            )


@pytest.mark.structural
def test_main_reads_sys_argv_when_no_argv_is_given(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    argv = ["sim_problem_sizes"]
    for path in FIXTURE_PATHS:
        argv += ["--params", str(path)]
    argv += ["--output-dir", str(tmp_path)]
    monkeypatch.setattr("sys.argv", argv)

    main()

    table_path = tmp_path / "sim_problem_sizes.tex"
    caption_path = tmp_path / "sim_problem_sizes_caption.txt"
    assert table_path.is_file()
    assert caption_path.read_text() == build_caption(list(FIXTURE_NAMES))
    captured = capsys.readouterr()
    assert str(table_path) in captured.out
    assert str(caption_path) in captured.out
