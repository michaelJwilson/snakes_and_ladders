"""Every problem statement in the textbook carries the five parts (issue #376).

The textbook's Notation section states them: the model, its supported sizes,
the factor graph with a sketch, the cited algorithms, and the validation. One
section carried all five and the rest two to four. Labels make each checkable
(``docs/CLAUDE.md``): ``sec:<p>:model``, ``par:<p>:sizes``,
``sec:<p>:validation``, an ``\\input{<p>_figure}`` defining ``fig:<p>:sketch``,
and a ``\\ref{alg:...}`` in the section's own text.
"""

from __future__ import annotations

import re

import catalogue
import pytest

from tests._paths import REPO_ROOT

TEX = REPO_ROOT / "docs" / "tex"
TEXTBOOK = TEX / "textbook.tex"

#: A catalogue key sharing a section, and the label of the sub-part stating
#: its own model: one check per stem missed three such keys (issue #681). A
#: key absent here is stated under the five parts already.
SHARED_KEYS: dict[str, str] = {
    "spatio_sequential_counts": "par:coupled:pairs",
    "spatio_sequential_ragged": "par:coupled:ragged",
    "ragged_hmm": "par:hmm:ragged",
}

#: Sections whose own text states every instance they carry at the declared
#: sizes (`sec:phylo`, `sec:potts`, `sec:ldpc`, `sec:frustrated`).
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

    ``text=None`` reads the committed textbook.
    """
    source = TEXTBOOK.read_text() if text is None else text
    start = source.index(f"\\label{{sec:{stem}}}")
    following = _SECTION.search(source, start)
    return source[start : following.start() if following else len(source)]


@pytest.mark.critical
@pytest.mark.infra
@pytest.mark.parametrize("stem", PROBLEMS)
def test_each_problem_states_its_model_sizes_and_validation(stem: str) -> None:
    section = problem_section(stem)

    for part in (f"sec:{stem}:model", f"par:{stem}:sizes", f"sec:{stem}:validation"):
        assert f"\\label{{{part}}}" in section, f"{stem}: no \\label{{{part}}}"


@pytest.mark.critical
@pytest.mark.infra
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
@pytest.mark.infra
@pytest.mark.parametrize("stem", PROBLEMS)
def test_each_problem_cites_an_algorithm(stem: str) -> None:
    # The algorithms are stated in the appendix and cited from the section, so
    # a section citing none states a problem nothing solves.
    section = problem_section(stem)

    assert _ALGORITHM_CITATION.search(section), f"{stem}: cites no \\ref{{alg:...}}"


@pytest.mark.infra
def test_every_algorithm_the_textbook_defines_is_cited() -> None:
    # An algorithm environment nothing cites is a float that lands somewhere
    # arbitrary and is read by no one, which is the appendix's failure mode.
    text = TEXTBOOK.read_text()
    defined = set(re.findall(r"\\label\{(alg:[a-z0-9-]+)\}", text))
    cited = {token[5:-1] for token in _ALGORITHM_CITATION.findall(text)}

    assert defined, "the textbook defines no algorithm"
    assert defined <= cited, f"algorithms defined and never cited: {defined - cited}"


@pytest.mark.smoke
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
@pytest.mark.infra
@pytest.mark.parametrize(("key", "part"), sorted(SHARED_KEYS.items()))
def test_a_key_sharing_a_section_states_what_is_its_own(key: str, part: str) -> None:
    # The gap #681 names, asserted inside the section's own span, so a label
    # drifted into a neighbouring section fails too.
    stem = part.split(":")[1]
    section = problem_section(stem)

    assert f"\\label{{{part}}}" in section, f"{key}: no \\label{{{part}}} in sec:{stem}"


@pytest.mark.critical
@pytest.mark.infra
def test_every_shared_catalogue_key_is_covered() -> None:
    # Read from the catalogue: every key sharing a section is in `SHARED_KEYS`
    # or in a section whose sizes paragraph states its instances.
    by_statement = catalogue.statements()

    assert by_statement, "no catalogue rows parsed from PROBLEMS.md"
    # The principal instance is the first key of a statement's first row;
    # every other key declares something the five parts do not.
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
