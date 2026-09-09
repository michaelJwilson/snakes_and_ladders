"""The citation check reports each way a citation stops resolving (issue #488).

`infra/check_citations.py` runs where the documents are built, over the real
`docs/tex/`, and that run is the check. What this pins is the guard's own
trigger: a check nothing has been seen to fail is a check that may not fail.
Four findings, one test each --- a figure cited under a name the tree does
not carry; a `\\ref` whose `\\label` the *other* document defines, which is
what `fig:turbo-waterfall` did (issue #249); a `\\cite` with no entry; and a
bibliography entry left unclosed, which a merge did during the 0.5.0 union.

The tree the fixture builds is the smallest one that carries all four, so a
finding is attributable to the defect written into it rather than to
anything `docs/tex/` happens to hold today. The real documents are checked
by the `documents` job on the push to `main` (`DEV.md`); repeating that here
would assert against a generated fragment this suite does not write.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "infra"))

import check_citations  # noqa: E402

CLEAN_BIBLIOGRAPHY = """\
@book{cited2020,
  author = {A. Author},
  title = {A Cited Work},
  year = {2020},
}

@book{after2021,
  author = {B. Author},
  title = {The Work After It},
  year = {2021},
}
"""

#: The same file with `cited2020`'s closing brace gone, which is the shape
#: the 0.5.0 union merge produced and a reader repaired by hand.
UNCLOSED_BIBLIOGRAPHY = CLEAN_BIBLIOGRAPHY.replace(
    "  year = {2020},\n}\n", "  year = {2020},\n"
)

PAPER = """\
\\documentclass{article}
\\begin{document}
\\input{shared}
\\includegraphics{figures/present}
Figure~\\ref{fig:present} and \\cite{cited2020}.
\\end{document}
"""

TEXTBOOK = """\
\\documentclass{article}
\\begin{document}
\\label{fig:textbook-only}
\\end{document}
"""

SHARED = "\\label{fig:present}\n"


def tex_tree(tmp_path: Path, bibliography: str = CLEAN_BIBLIOGRAPHY) -> Path:
    """A two-document `docs/tex/` that resolves, for a defect to be added to."""
    tex_dir = tmp_path / "tex"
    (tex_dir / "figures").mkdir(parents=True)
    (tex_dir / "figures" / "present.pdf").write_bytes(b"%PDF-1.4\n")
    (tex_dir / "references.bib").write_text(bibliography)
    (tex_dir / "shared.tex").write_text(SHARED)
    (tex_dir / "paper.tex").write_text(PAPER)
    (tex_dir / "textbook.tex").write_text(TEXTBOOK)
    return tex_dir


def check(tex_dir: Path) -> list[str]:
    """Every finding over both documents of ``tex_dir``."""
    documents = (tex_dir / "paper.tex", tex_dir / "textbook.tex")
    return check_citations.unresolved(documents, tex_dir=tex_dir)


@pytest.mark.structural
@pytest.mark.critical
def test_a_tree_that_resolves_reports_nothing(tmp_path: Path) -> None:
    # The half of the referee that keeps the check from being noise: the
    # four cases below are only evidence if this one is silent.
    assert check(tex_tree(tmp_path)) == []


@pytest.mark.structural
@pytest.mark.critical
def test_a_cited_figure_with_no_file_is_reported(tmp_path: Path) -> None:
    tex_dir = tex_tree(tmp_path)
    (tex_dir / "figures" / "present.pdf").unlink()

    out = check(tex_dir)

    assert [
        message for message in out if "no figure file for figures/present" in message
    ]
    assert len(out) == 1, out


@pytest.mark.structural
@pytest.mark.critical
def test_a_label_the_other_document_defines_does_not_resolve(tmp_path: Path) -> None:
    # `fig:turbo-waterfall`'s shape: the label exists, in the other document,
    # and the two build separately. Both documents' logs read as clean under a
    # grep that does not know which one defined what.
    tex_dir = tex_tree(tmp_path)
    paper = tex_dir / "paper.tex"
    paper.write_text(paper.read_text().replace("fig:present}", "fig:textbook-only}"))

    out = check(tex_dir)

    assert out == [
        "paper.tex: paper.tex:5: fig:textbook-only is cited and "
        "this document defines no such label"
    ]


@pytest.mark.structural
@pytest.mark.critical
def test_a_cite_with_no_bibliography_entry_is_reported(tmp_path: Path) -> None:
    tex_dir = tex_tree(tmp_path)
    paper = tex_dir / "paper.tex"
    paper.write_text(paper.read_text().replace("cited2020", "uncited1999"))

    out = check(tex_dir)

    assert out == [
        "paper.tex: paper.tex:5: uncited1999 is cited and references.bib has no such entry"
    ]


@pytest.mark.structural
@pytest.mark.critical
def test_a_bibliography_entry_that_does_not_close_is_named(tmp_path: Path) -> None:
    # The entry runs into the one after it, so it stops resolving while the
    # file still reads as text. Named, and the `\cite` to it reported with it.
    tex_dir = tex_tree(tmp_path, bibliography=UNCLOSED_BIBLIOGRAPHY)

    out = check(tex_dir)

    assert out == [
        "references.bib: entry cited2020 does not close its braces, "
        "so it runs into the entries after it",
        "paper.tex: paper.tex:5: cited2020 is cited and references.bib has no such entry",
    ]


@pytest.mark.structural
@pytest.mark.critical
def test_a_commented_out_citation_is_not_read_as_one(tmp_path: Path) -> None:
    # A check that read comments would fail on a citation the document does
    # not make, which is the noise this step is not allowed to add.
    tex_dir = tex_tree(tmp_path)
    paper = tex_dir / "paper.tex"
    paper.write_text(
        paper.read_text().replace("Figure~", "% \\ref{fig:nowhere}\nFigure~")
    )

    assert check(tex_dir) == []
