"""Every problem statement in the textbook carries the five parts (issue #376).

The textbook's Notation section states the shape a problem statement takes:
the model, the sizes it is supported at, the model as a factor graph with a
sketch of that structure, the algorithms cited from the appendix, and the
validation. Before this guard one of the nine sections carried all five and
the rest carried between two and four, and nothing said so --- a section
missing its sizes or its validation reads as complete, because what is absent
leaves no mark on the page.

Five labels make each part checkable, and the convention is stated in
``docs/CLAUDE.md``: ``sec:<p>:model``, ``par:<p>:sizes``,
``sec:<p>:validation``, an ``\\input{<p>_figure}`` whose file exists and
defines ``fig:<p>:sketch``, and at least one ``\\ref{alg:...}`` in the
section's own text. A label is used rather than a heading because a heading
is broken by the next retitle, which is the rule ``docs/CLAUDE.md`` already
states for citations from code.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TEX = REPO_ROOT / "docs" / "tex"
TEXTBOOK = TEX / "textbook.tex"

#: The nine problem statements, by the stem their labels and figure file use.
#: A tenth problem class adds a row here and to ``PROBLEMS.md`` together.
PROBLEMS = (
    "phylo",
    "potts",
    "hmm",
    "coupled",
    "parsimony",
    "frustrated",
    "mixture",
    "testfunctions",
    "ldpc",
)

_SECTION = re.compile(r"^\\section\{", re.MULTILINE)
_ALGORITHM_CITATION = re.compile(r"\\ref\{alg:[a-z0-9-]+\}")


def problem_section(stem: str, text: str | None = None) -> str:
    """The text of one problem statement, from its ``\\label`` to the next section.

    Parameters
    ----------
    stem : str
        The problem's stem, as in :data:`PROBLEMS`.
    text : str | None
        The textbook source. ``None`` reads the committed file.

    Returns
    -------
    str
        Everything from the section's label to the start of the next
        ``\\section``, which is the span the five parts must sit inside.
    """
    source = TEXTBOOK.read_text() if text is None else text
    start = source.index(f"\\label{{sec:{stem}}}")
    following = _SECTION.search(source, start)
    return source[start : following.start() if following else len(source)]


@pytest.mark.critical
@pytest.mark.structural
@pytest.mark.parametrize("stem", PROBLEMS)
def test_each_problem_states_its_model_sizes_and_validation(stem: str) -> None:
    section = problem_section(stem)

    for part in (f"sec:{stem}:model", f"par:{stem}:sizes", f"sec:{stem}:validation"):
        assert f"\\label{{{part}}}" in section, f"{stem}: no \\label{{{part}}}"


@pytest.mark.critical
@pytest.mark.structural
@pytest.mark.parametrize("stem", PROBLEMS)
def test_each_problem_inputs_a_sketch_file_that_exists_and_labels_it(
    stem: str,
) -> None:
    # Hand-drawn, one file per problem, outside the figure cache and the
    # render cap: those govern rendered output, and a sketch is not rendered.
    section = problem_section(stem)
    assert f"\\input{{{stem}_figure}}" in section, f"{stem}: no \\input of its sketch"

    sketch = TEX / f"{stem}_figure.tex"
    assert sketch.is_file(), f"{stem}: {sketch.name} does not exist"
    assert f"\\label{{fig:{stem}:sketch}}" in sketch.read_text(), (
        f"{stem}: {sketch.name} defines no fig:{stem}:sketch"
    )


@pytest.mark.critical
@pytest.mark.structural
@pytest.mark.parametrize("stem", PROBLEMS)
def test_each_problem_cites_an_algorithm(stem: str) -> None:
    # The algorithms are stated in the appendix and cited from the section, so
    # a section citing none states a problem nothing solves.
    section = problem_section(stem)

    assert _ALGORITHM_CITATION.search(section), f"{stem}: cites no \\ref{{alg:...}}"


@pytest.mark.structural
def test_every_algorithm_the_textbook_defines_is_cited() -> None:
    # An algorithm environment nothing cites is a float that lands somewhere
    # arbitrary and is read by no one, which is the appendix's failure mode.
    text = TEXTBOOK.read_text()
    defined = set(re.findall(r"\\label\{(alg:[a-z0-9-]+)\}", text))
    cited = {token[5:-1] for token in _ALGORITHM_CITATION.findall(text)}

    assert defined, "the textbook defines no algorithm"
    assert defined <= cited, f"algorithms defined and never cited: {defined - cited}"


@pytest.mark.edge_case
def test_the_guard_reads_one_section_and_not_the_next() -> None:
    # Guards the guard: a section span that ran past its own \section would
    # let a missing part be satisfied by its neighbour's, which is the one
    # way this check can pass while the document is wrong.
    text = (
        "\\section{A}\n\\label{sec:phylo}\n\\label{par:phylo:sizes}\n"
        "\\section{B}\n\\label{sec:potts}\n\\label{par:potts:sizes}\n"
    )
    first = problem_section("phylo", text)

    assert "par:phylo:sizes" in first
    assert "par:potts:sizes" not in first
