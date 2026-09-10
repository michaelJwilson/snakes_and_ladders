"""Regression test for snakes_and_ladders.qa.sim_problem_sizes.

Pins the tabulated values against each fixture's yaml, read independently,
not just that the figure renders without raising (CLAUDE.md's
no-coverage-theatre rule).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from snakes_and_ladders.qa.manifest import FIGURES
from snakes_and_ladders.qa.sim_problem_sizes import (
    build_caption,
    main,
)
from snakes_and_ladders.sim.params import load_simulation_params
from snakes_and_ladders.sim.tree import preorder

from tests._fixtures import FIXTURES_DIR

FIXTURE_NAMES = (
    "tree_jc/stress.yaml",
    "tree_jc/ci.yaml",
    "tree_jc/release.yaml",
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
    # The point of the change: the document \input's a typeset table instead of
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
    # An unescaped underscore in a fixture's name is a LaTeX error, and
    # every problem name here contains one.
    body = main(_argv(tmp_path)).table_path.read_text()

    assert r"tree\_jc/release.yaml" in body
    for line in body.splitlines():
        stripped = line.replace(r"\_", "")
        assert "_" not in stripped, line


@pytest.mark.structural
def test_the_manifest_tabulates_these_three_fixtures_in_this_order() -> None:
    # The oracle below pins every cell against the yaml it is read from, but
    # only for the fixtures *this module* names. The committed table is
    # rendered from the manifest's arguments, and the caption's "3" counts
    # them, so a fourth `--params` added there would change both while every
    # assertion here still passed. Row order is part of it: the table is one
    # row per flag, in the order the flags are given.
    spec = next(spec for spec in FIGURES if spec.stem == "sim_problem_sizes")
    flags = [
        value
        for flag, value in zip(spec.arguments[::2], spec.arguments[1::2], strict=True)
        if flag == "--params"
    ]

    assert flags == [f"tests/regression/fixtures/{name}" for name in FIXTURE_NAMES]


@pytest.mark.oracle
def test_problem_sizes_values_match_each_fixture_independently() -> None:
    for path in FIXTURE_PATHS:
        params = load_simulation_params(path)
        n_taxa = sum(1 for node in preorder(params.tau) if node.is_leaf)

        # Cross-check against direct knowledge of the fixtures rather than
        # re-deriving through the module under test.
        if path.parts[-2:] == ("tree_jc", "stress.yaml"):
            assert (n_taxa, params.n_sites, params.seed, params.tolerance) == (
                4,
                200000,
                20260902,
                0.01,
            )
        elif path.parts[-2:] == ("tree_jc", "ci.yaml"):
            assert (n_taxa, params.n_sites, params.seed, params.tolerance) == (
                4,
                20000,
                20260903,
                0.03,
            )
        elif path.parts[-2:] == ("tree_jc", "release.yaml"):
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
    # The runner reports what it wrote through the run logger (issue #311),
    # which writes to stderr; nothing goes to stdout.
    captured = capsys.readouterr()
    written = captured.out + captured.err
    assert str(table_path) in written
    assert str(caption_path) in written
