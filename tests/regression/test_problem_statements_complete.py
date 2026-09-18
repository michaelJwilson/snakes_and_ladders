"""Every problem statement in the textbook carries the five parts (issue #376).

The textbook's Notation section states the shape a problem statement takes:
the model, the sizes it is supported at, the model as a factor graph with a
sketch of that structure, the algorithms cited from the appendix, and the
validation. Before this guard one of the sections carried all five and
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
CATALOGUE = REPO_ROOT / "PROBLEMS.md"
TEX = REPO_ROOT / "docs" / "tex"
TEXTBOOK = TEX / "textbook.tex"

#: What a catalogue key that **shares** a section must state of its own, as the
#: label of the sub-part stating it.
#:
#: The guard below checks one statement per stem, so a key sharing a stem with
#: another was invisible to it: `spatio_sequential_counts`,
#: `spatio_sequential_ragged` and `ragged_hmm` all resolve to a section whose
#: five parts were complete, while the model each declares was stated nowhere
#: (issue #681). A key here names the part that states it, so the next shared
#: key is caught by a failing test rather than by an audit.
#:
#: A key absent from this mapping is one whose section states it under the five
#: parts already --- `tree_jc` under `sec:phylo`, say --- which is why this is a
#: mapping and not a requirement on every key.
SHARED_KEYS: dict[str, str] = {
    "spatio_sequential_counts": "par:coupled:pairs",
    "spatio_sequential_ragged": "par:coupled:ragged",
    "ragged_hmm": "par:hmm:ragged",
}

#: Sections whose own text states every instance they carry, so a key sharing
#: one needs no sub-part: `sec:phylo` names the Jukes--Cantor and
#: general-time-reversible trees and the scaled instance together, `sec:potts`
#: its chain, lattice and per-site-field cases, `sec:ldpc` its three codes, and
#: `sec:frustrated` both the triangular antiferromagnet and the planted glass,
#: each at the sizes declared there.
STATED_IN_SIZES = frozenset({"sec:phylo", "sec:potts", "sec:ldpc", "sec:frustrated"})

#: The problem statements, by the stem their labels and figure file use. A
#: further problem class adds a row here and to ``PROBLEMS.md`` together.
PROBLEMS = (
    "phylo",
    "potts",
    "hmm",
    "coupled",
    "parsimony",
    "frustrated",
    "mixture",
    "emissionmixture",
    "testfunctions",
    "ldpc",
    "turbo",
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


@pytest.mark.critical
@pytest.mark.structural
@pytest.mark.parametrize(("key", "part"), sorted(SHARED_KEYS.items()))
def test_a_key_sharing_a_section_states_what_is_its_own(key: str, part: str) -> None:
    # The gap #681 names. `spatio_sequential_counts` and `sec:coupled`'s other
    # key resolve to one section, and a guard that checks one statement per
    # stem passes while the covariate contract and the unequal lengths are
    # stated nowhere. The part is asserted inside its own section's span, so a
    # label that drifted into a neighbouring section fails here too.
    stem = part.split(":")[1]
    section = problem_section(stem)

    assert f"\\label{{{part}}}" in section, f"{key}: no \\label{{{part}}} in sec:{stem}"


@pytest.mark.critical
@pytest.mark.structural
def test_every_shared_catalogue_key_is_covered() -> None:
    # The mapping above is a guard only if it covers what `PROBLEMS.md`
    # declares. Read from the catalogue rather than from a list, so a key added
    # there cannot drift away from the statement that has to state it: every
    # key sharing a section with another is either named in `SHARED_KEYS` or
    # sits in a section whose own sizes paragraph states its instances.
    by_statement: dict[str, list[str]] = {}
    for line in CATALOGUE.read_text().splitlines():
        cells = [cell.strip() for cell in line.split("|")]
        if len(cells) < 5 or not cells[2].startswith("`"):
            continue
        keys = [key.strip(" `") for key in cells[2].split(",")]
        by_statement.setdefault(cells[3].strip(" `"), []).extend(keys)

    assert by_statement, "no catalogue rows parsed from PROBLEMS.md"
    # The **principal** instance of a statement is the first key of its first
    # row: the one the section's five parts are about. The catalogue's own
    # reading gives this --- "a row with two keys is one problem declared at
    # two instances", and a second row on one statement is a second problem
    # sharing a section. Every other key declares something the five parts do
    # not, so it needs a sub-part or a section that states its instances.
    uncovered = {
        key: statement
        for statement, keys in by_statement.items()
        for key in keys[1:]
        if key not in SHARED_KEYS and statement not in STATED_IN_SIZES
    }

    assert not uncovered, (
        "these keys share a statement with another and are neither in "
        f"SHARED_KEYS nor in a section that states its instances: {uncovered}"
    )
