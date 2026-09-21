"""The per-section PDFs are cut from the book's pages by its own TOC (issue #844).

No TeX runs here: what is asserted is the reading of a ``.toc`` and the page
ranges it yields, on lines written the way LaTeX writes them.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import split_textbook

TOC = r"""\contentsline {part}{\numberline {I}The Problems}{2}{part.1}%
\contentsline {section}{\numberline {1}Notation}{3}{section.1}%
\contentsline {subsection}{\numberline {1.1}Symbols}{3}{subsection.1.1}%
\contentsline {section}{\numberline {2}Mixed Discrete and Continuous Optimization}{5}{section.2}%
\contentsline {section}{\numberline {3}The Gaussian Mixture, with $\log Z$}{5}{section.3}%
\contentsline {part}{\numberline {II}The Algorithms}{40}{part.2}%
\contentsline {section}{\numberline {4}Factor Graph \emph  {Optimization}}{41}{section.4}%
\contentsline {section}{\numberline {A}Reference Taxonomy}{60}{appendix.A}%
"""


@pytest.mark.infra
def test_every_section_is_read_with_its_number_title_and_page() -> None:
    found = split_textbook.sections(TOC)
    assert [(s.number, s.page) for s in found] == [
        ("1", 3),
        ("2", 5),
        ("3", 5),
        ("4", 41),
        ("A", 60),
    ]
    assert [s.slug for s in found] == [
        "01-notation",
        "02-mixed-discrete-and-continuous-optimization",
        "03-the-gaussian-mixture-with-z",
        "04-factor-graph-optimization",
        "a-reference-taxonomy",
    ]


@pytest.mark.infra
def test_the_ranges_cover_the_book_once_and_the_front_matter_first() -> None:
    cuts = split_textbook.ranges(split_textbook.sections(TOC))
    assert cuts == [
        ("00-front-matter", "1-2"),
        ("01-notation", "3-4"),
        ("02-mixed-discrete-and-continuous-optimization", "5"),
        ("03-the-gaussian-mixture-with-z", "5-40"),
        ("04-factor-graph-optimization", "41-59"),
        ("a-reference-taxonomy", "60-"),
    ]


@pytest.mark.infra
def test_the_drivers_are_written_in_book_order(tmp_path: Path) -> None:
    toc = tmp_path / "textbook.toc"
    toc.write_text(TOC)
    written = split_textbook.write_drivers(toc, tmp_path / "sections")
    assert [path.stem for path in written][:2] == ["00-front-matter", "01-notation"]
    assert (
        "\\includepdf[pages={60-},fitpaper=true]{../../../textbook.pdf}"
        in written[-1].read_text()
    )
    assert split_textbook.ranges([]) == []
