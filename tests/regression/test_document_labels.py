"""Code cites a label in the technical document, and every label it cites exists.

`ROADMAP.md` §1.3 binds the application logic to `docs/tex/`. A citation is
that binding made checkable --- and only if it names something the build
resolves. Before this test the code cited nine equations that had never
carried a `\\label`, nine sections by a title the next retitle would break,
and two equations by a *number* from a numbering that no longer existed
(issue #274). None of those ever failed anything.

Two things are asserted. Every label token in `python/`, `src/` and `tests/`
is defined in a document under `docs/tex/`; and no citation uses the
informal forms that cannot be resolved --- an equation named in parentheses
after the abbreviation, or a section quoted by its title.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCUMENTS = tuple(sorted((REPO_ROOT / "docs" / "tex").glob("*.tex")))
SOURCES = ("python", "src", "tests")
SUFFIXES = (".py", ".rs")

#: A label as written in a citation: kind, colon, name. `fig:` and `app:`
#: are included so a figure or appendix citation is held to the same rule.
LABEL = re.compile(r"\b(?:eq|alg|sec|fig|app):[a-z0-9][a-z0-9-]*")

#: The two informal forms: `eq.` or `Sec.` followed by a parenthesized name
#: or number, or `Sec.` followed by a quoted title. Built from parts so this
#: file does not itself contain the form it forbids.
INFORMAL = re.compile("|".join([r"\b(?:eq|alg)" + r"\. \(", r"\bSec" + r'\. "']))


def defined_labels(documents: tuple[Path, ...] = DOCUMENTS) -> set[str]:
    """Every ``\\label{...}`` in the documents."""
    labels: set[str] = set()
    for document in documents:
        labels.update(re.findall(r"\\label\{([^}]+)\}", document.read_text()))
    return labels


def _source_files() -> list[Path]:
    # This file is excluded: its guard-the-guard test below writes a label
    # that deliberately does not exist, and scanning it would report that
    # fixture as a defect.
    return [
        path
        for root in SOURCES
        for path in sorted((REPO_ROOT / root).rglob("*"))
        if path.suffix in SUFFIXES
        and "__pycache__" not in path.parts
        and path != Path(__file__).resolve()
    ]


def cited_labels(text: str) -> set[str]:
    """Every label token in ``text``."""
    return set(LABEL.findall(text))


def informal_citations(text: str) -> list[str]:
    """Every informal citation in ``text``, for the message."""
    return INFORMAL.findall(text)


def test_every_label_the_code_cites_is_defined_in_a_document() -> None:
    defined = defined_labels()
    assert defined, "no labels found under docs/tex/"

    dangling: dict[str, set[str]] = {}
    for path in _source_files():
        missing = cited_labels(path.read_text()) - defined
        if missing:
            dangling[str(path.relative_to(REPO_ROOT))] = missing

    assert dangling == {}, f"labels cited in code that no document defines: {dangling}"


def test_no_citation_uses_an_unresolvable_form() -> None:
    # A name in parentheses or a quoted title cannot be resolved by the
    # build, so it can go stale without anything noticing -- which is what
    # every one of them had done.
    offenders = {
        str(path.relative_to(REPO_ROOT)): informal_citations(path.read_text())
        for path in _source_files()
        if informal_citations(path.read_text())
    }

    assert offenders == {}, f"citations by name or title rather than label: {offenders}"


def test_the_code_actually_cites_the_document() -> None:
    # The two tests above pass on a tree with no citations at all. This one
    # pins that the binding ROADMAP §1.3 describes is in force: the pruning
    # recursion, the closed form and the estimator are each cited from code.
    cited: set[str] = set()
    for path in _source_files():
        cited |= cited_labels(path.read_text())

    assert {"eq:pruning", "eq:root", "eq:jc", "eq:reinforce", "eq:return"} <= cited


def test_the_guard_catches_a_dangling_label_and_an_informal_form(
    tmp_path: Path,
) -> None:
    # Guards the guard, per the repository's pattern: a check that cannot
    # fail reads as evidence while supplying none.
    document = tmp_path / "textbook.tex"
    document.write_text(r"\begin{equation} x \label{eq:real} \end{equation}")

    assert defined_labels((document,)) == {"eq:real"}
    assert cited_labels("see eq:real and eq:no-such-label") - {"eq:real"} == {
        "eq:no-such-label"
    }
    informal = "Implements eq" + ". (jc) of Sec" + '. "The model"'
    assert len(informal_citations(informal)) == 2
    assert informal_citations("Implements ``eq:jc`` of ``sec:phylo``") == []


@pytest.mark.parametrize("document", DOCUMENTS, ids=lambda path: path.name)
def test_no_label_is_defined_twice(document: Path) -> None:
    # `latexmk` exits zero on a multiply-defined label (`docs/CLAUDE.md`), so
    # the build's own check is a grep of its log; this is the same check
    # without the build.
    labels = re.findall(r"\\label\{([^}]+)\}", document.read_text())

    assert len(labels) == len(set(labels)), sorted(
        label for label in set(labels) if labels.count(label) > 1
    )
