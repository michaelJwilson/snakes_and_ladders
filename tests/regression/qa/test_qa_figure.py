"""Regression test for phylo.qa.figure.write_qa_figure.

Pins that the PDF it writes does not embed Type 3 fonts: GitHub's blob
viewer fails to render them ("Error rendering embedded code"), which is
what motivated scoping ``pdf.fonttype``/``ps.fonttype`` to 42 in
``write_qa_figure`` (see issue #76) rather than just checking the figure
renders without raising (CLAUDE.md's no-coverage-theatre rule).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pytest
from matplotlib.figure import Figure
from phylo.qa.figure import (
    check_latex_safe,
    latex_integer,
    write_qa_figure,
    write_qa_table,
)

mpl.use("Agg")


@pytest.mark.structural
def test_write_qa_figure_does_not_embed_type3_fonts(tmp_path: Path) -> None:
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    ax.set_title("regression fixture")

    qa_figure = write_qa_figure(tmp_path, "stem", fig, "caption text")

    pdf_bytes = qa_figure.figure_path.read_bytes()
    assert b"/Subtype/Type3" not in pdf_bytes
    assert b"/Subtype /Type3" not in pdf_bytes


@pytest.mark.structural
def test_write_qa_figure_does_not_mutate_global_font_rc(tmp_path: Path) -> None:
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])

    before = (mpl.rcParams["pdf.fonttype"], mpl.rcParams["ps.fonttype"])
    write_qa_figure(tmp_path, "stem", fig, "caption text")
    after = (mpl.rcParams["pdf.fonttype"], mpl.rcParams["ps.fonttype"])

    assert before == after


@pytest.mark.structural
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, "0"),
        (10, "10"),
        (9999, "9999"),
        (10_000, r"10\_000"),
        (200_000, r"200\_000"),
        (1_000_000, r"1\_000\_000"),
    ],
)
def test_latex_integer_separates_only_where_it_helps(value: int, expected: str) -> None:
    # Below ten thousand a separator makes a number harder to read, not
    # easier, so the threshold is deliberate rather than an accident of
    # formatting.
    assert latex_integer(value) == expected


@pytest.mark.structural
def test_latex_safe_accepts_a_separated_integer() -> None:
    check_latex_safe(f"simulated over {latex_integer(200_000)} sites")


@pytest.mark.edge_case
@pytest.mark.parametrize("special", ["_", "%", "&", "#", "\\"])
def test_an_unescaped_special_is_refused(special: str) -> None:
    # The check has to fail on something, or it is decoration. Every
    # character it claims to catch is exercised.
    with pytest.raises(ValueError, match="unescaped LaTeX special"):
        check_latex_safe(f"a caption containing {special} directly")


@pytest.mark.edge_case
def test_the_error_names_the_offending_characters() -> None:
    with pytest.raises(ValueError, match=r"\['#', '_'\]"):
        check_latex_safe("both _ and # are wrong")


@pytest.mark.edge_case
def test_write_qa_figure_refuses_an_unsafe_caption(tmp_path: Path) -> None:
    # Enforced at the point of writing, so a caption that would break the
    # LaTeX build fails in the QA script rather than in the document build.
    figure = Figure()
    with pytest.raises(ValueError, match="unescaped LaTeX special"):
        write_qa_figure(tmp_path, "unsafe", figure, "caption_with_underscore")


@pytest.mark.structural
def test_write_qa_table_writes_a_tex_fragment_and_its_caption(tmp_path: Path) -> None:
    body = r"\begin{tabular}{l}" + "\n" + r"  a \\" + "\n" + r"\end{tabular}"

    written = write_qa_table(tmp_path, "example", body, "A caption.")

    assert written.table_path == tmp_path / "example.tex"
    assert written.table_path.read_text() == body + "\n"
    assert written.caption_path.read_text() == "A caption."


@pytest.mark.edge_case
def test_write_qa_table_refuses_an_unsafe_caption(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unescaped LaTeX special"):
        write_qa_table(tmp_path, "unsafe", r"\begin{tabular}{l}\end{tabular}", "a_b")
