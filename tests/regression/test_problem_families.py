"""The superset claim of issue #622, stated per family and not per word.

The claim step 5 promised: for each problem, the derived selection covers what a
text search for that problem finds. Per word it is false --- `grep -rl mixture`
finds the emission mixture, `grep -rl potts` finds three keys and every guard
over this file --- so it is stated over the **family**: the set of catalogue keys
sharing one problem statement, `PROBLEMS.md`'s **Statement** column. Eleven
families over the 226 modules under ``tests/regression``, and the difference is
empty for nine of them; the two words that name no problem are in `BY_WORD`,
asserted rather than waived.

What counts as naming a family is narrowed three ways, each forced by a module
that means something else by the word and each stated where it is applied:
comments and docstrings are prose and not a selection input; a
`snakes_and_ladders` import is read by the catalogue, which is what
`tests/_problems.py` already does with it; and a stem matches as the head of a
name (``tree_likelihood``) rather than bare, because `ast.parse` returns a tree
and `sum_product` takes a ``"tree"`` schedule. A key still matches bare, which
is what `grep` is given.

The selection read here is the item's: the markers `tests/conftest.py` derives
through `fixtures_named_in`, and the problem markers an author writes by hand,
which `-m <key>` collects the same way --- `search/test_decoding.py` carries
``@pytest.mark.frustrated_lattice`` on two items and loads no fixture.
"""

from __future__ import annotations

import ast
import io
import re
import tokenize
from collections.abc import Mapping
from functools import cache
from pathlib import Path

import catalogue as catalogue_reader
import pytest

from tests._paths import REPO_ROOT
from tests._problems import fixtures_named_in, problem_names

TESTS = REPO_ROOT / "tests" / "regression"
CATALOGUE = catalogue_reader.CATALOGUE

#: Every problem key, which is every marker the hook can add.
DECLARED = frozenset(problem_names())

#: The three modules that name a family and exercise none, with the word and
#: what it means there. None is a gap: the first names notebooks in the map
#: `select_tests.py` is checked against --- ``docs/nb/hmm.ipynb`` is data, the
#: case `test_problem_markers.py` already meets in `QUOTED_CALLS` --- the
#: second passes ``parsimony_start=True`` to `search.infer`, shared machinery
#: that `PROBLEMS.md` says defines nothing, where the parsimony statement's own
#: tests import `likelihood.parsimony` and are selected; and the third names
#: `tree_messages` and the tree schedule, which are the graph-theoretic tree
#: `_stem_pattern` already excuses in its bare form --- the kernel is exact on
#: a chain and a Potts tree and touches no phylogeny, whose own tests import
#: `likelihood.pruning` and are selected. Asserted below, so an entry that
#: stops being a false match fails rather than hiding a gap.
BY_WORD: dict[str, str] = {
    "tests/regression/test_select_tests.py": "sec:hmm",
    "tests/regression/search/test_search_exhaustive.py": "sec:parsimony",
    "tests/regression/likelihood/test_message_passing_rust.py": "sec:phylo",
}

#: Spelled rather than written, for `test_problem_markers.py`'s reason: a
#: registry call in this file's source is a call `grep` counts and the parse
#: does not.
CALL = "fixture"

_EDGE = r"(?<![0-9A-Za-z_\-])"


def _key_pattern(key: str) -> re.Pattern[str]:
    """A key as a whole token, where ``-`` and ``_`` bind.

    ``-`` binds so that ``"emission-mixture"``, a model name, is one token and
    not a match for the key `mixture` of another family.
    """
    return re.compile(_EDGE + re.escape(key) + r"(?![0-9A-Za-z_\-])")


def _stem_pattern(stem: str) -> re.Pattern[str]:
    """A stem as the head of a name: ``tree_likelihood``, not ``tree``.

    The bare word is another thing in this tree --- an `ast` tree, a
    ``"tree"`` message schedule, a ``parsimony-start`` keyword --- and a stem
    is the family's word and not the repository's.
    """
    return re.compile(_EDGE + re.escape(stem) + r"_[0-9A-Za-z]")


@cache
def families(catalogue: Path = CATALOGUE) -> dict[str, frozenset[str]]:
    """``statement label -> the fixture keys stated by it``.

    Parameters
    ----------
    catalogue : Path
        `PROBLEMS.md`.

    Returns
    -------
    dict[str, frozenset[str]]
        A key may sit in two families --- `tree_jc` is the Jukes--Cantor tree
        of ``sec:phylo`` and the parsimony problem of ``sec:parsimony`` --- so
        a family is the section's key set and not a partition of the keys.

    Notes
    -----
    Read through `infra/catalogue.py`, which the textbook's tables read too.
    Two readings of the table used to be zipped ``strict`` here to hold them
    to the same rows; one reader is the stronger form of that check, so the
    zip went with the second reader (issue #863).
    """
    return {
        label: frozenset(keys)
        for label, keys in sorted(catalogue_reader.statements(catalogue).items())
    }


def stems(label: str, keys: frozenset[str]) -> frozenset[str]:
    """The family's word stems, derived from the statement and from its keys.

    Parameters
    ----------
    label : str
        The statement label, ``sec:phylo``.
    keys : frozenset[str]
        The family's keys.

    Returns
    -------
    frozenset[str]
        The statement's own short name, and the keys' common word prefix where
        they share one: ``sec:phylo`` gives ``phylo`` and ``tree``,
        ``sec:coupled`` gives ``coupled`` and ``spatio_sequential``. A key is
        not repeated as a stem, since a key is matched bare and a stem is not.
    """
    found = {label.split(":", 1)[-1]}
    prefix: list[str] = []
    for words in zip(*(key.split("_") for key in sorted(keys)), strict=False):
        if len(set(words)) != 1:
            break
        prefix.append(words[0])
    if prefix and len(keys) > 1:
        found.add("_".join(prefix))
    return frozenset(found - keys)


def code_of(path: Path) -> str:
    """One module's source without its comments, docstrings or package imports.

    Parameters
    ----------
    path : Path
        A Python source file.

    Returns
    -------
    str
        What is left is what the module *does*. Prose names a problem to
        explain a contrast --- "a gamma-Poisson mixture", "on a tree" --- and
        an import names one for the catalogue to read, which
        `tests/_problems.py` does; neither is a module exercising the problem.
    """
    source = path.read_text()
    dropped: set[int] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import | ast.ImportFrom) and node.end_lineno:
            dropped |= set(range(node.lineno, node.end_lineno + 1))
        elif isinstance(
            node, ast.Module | ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
        ):
            first = node.body[0] if node.body else None
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
                and first.end_lineno
            ):
                dropped |= set(range(first.lineno, first.end_lineno + 1))
    kept = "\n".join(
        line
        for number, line in enumerate(source.splitlines(), 1)
        if number not in dropped
    )
    tokens = tokenize.generate_tokens(io.StringIO(kept + "\n").readline)
    return "\n".join(token.string for token in tokens if token.type != tokenize.COMMENT)


def markers_of(path: Path) -> frozenset[str]:
    """The problem markers one module's items carry, derived and written.

    Parameters
    ----------
    path : Path
        A test module.

    Returns
    -------
    frozenset[str]
        `fixtures_named_in`, which is what `tests/conftest.py` adds, together
        with the problem markers the source applies by hand. Empty is the
        `infra` case: the module exercises no problem.

    Notes
    -----
    The written markers are read as decorators and not as text, since a
    module naming one in prose --- this one names
    ``pytest.mark.frustrated_lattice`` two paragraphs up --- would otherwise
    read as carrying it.
    """
    written: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            continue
        for decorator in node.decorator_list:
            applied = decorator.func if isinstance(decorator, ast.Call) else decorator
            if (
                isinstance(applied, ast.Attribute)
                and isinstance(applied.value, ast.Attribute)
                and applied.value.attr == "mark"
                and applied.attr in DECLARED
            ):
                written.add(applied.attr)
    return fixtures_named_in(path) | written


def unselected(
    keys: frozenset[str],
    family_stems: frozenset[str],
    modules: Mapping[str, tuple[frozenset[str], str]],
) -> list[str]:
    """The modules that name a family, exercise a problem, and are not selected.

    Parameters
    ----------
    keys : frozenset[str]
        The family's keys.
    family_stems : frozenset[str]
        Its stems, from `stems`.
    modules : Mapping[str, tuple[frozenset[str], str]]
        ``name -> (the problem markers it carries, its code)``.

    Returns
    -------
    list[str]
        Sorted. A module carrying no problem marker is `infra` --- a guard
        over this file, a document check, shared machinery --- and is excluded:
        it names the word without exercising the problem by construction, and
        `test_problem_markers.py` holds that claim from the other side.
    """
    patterns = [_key_pattern(key) for key in sorted(keys)]
    patterns += [_stem_pattern(stem) for stem in sorted(family_stems)]
    return sorted(
        name
        for name, (markers, code) in modules.items()
        if markers
        and not markers & keys
        and any(pattern.search(code) for pattern in patterns)
    )


@cache
def _tree() -> dict[str, tuple[frozenset[str], str]]:
    """Every collected module under ``tests/regression``, read once.

    ``tests/regression/docs/`` is excluded: those guards read the documents,
    which name every problem, and none of them runs a model.
    """
    return {
        str(path.relative_to(REPO_ROOT)): (markers_of(path), code_of(path))
        for path in sorted(TESTS.rglob("test_*.py"))
        if not path.is_relative_to(TESTS / "docs")
    }


@pytest.mark.critical
@pytest.mark.infra
@pytest.mark.parametrize("label", sorted(families()))
def test_the_selection_of_a_family_covers_every_module_naming_it(label: str) -> None:
    """The claim, per family: the selection is a superset of the text search.

    `search/test_maxflow.py` and `search/test_alpha_expansion.py` are in the
    `sec:potts` selection through their imports, which
    `test_problem_markers.py::test_the_two_modules_the_axis_was_opened_about_are_selected`
    asserts on its own; it is not repeated here.
    """
    keys = families()[label]
    missing = [
        name
        for name in unselected(keys, stems(label, keys), _tree())
        if BY_WORD.get(name) != label
    ]
    assert not missing, (
        f"{label} is stated for {sorted(keys)} and {missing} name it without being "
        f'selected: `-m "{" or ".join(sorted(keys))}"` would run a subset of the '
        "family's tests while looking like it ran all of them."
    )


@pytest.mark.critical
@pytest.mark.infra
def test_every_key_is_stated_and_one_key_is_stated_twice() -> None:
    """The family map is read from the catalogue, and this says it was read.

    An empty or partial parse leaves the guard above passing on nothing, which
    is the failure a derived map trades for a hand-written one. Both halves of
    the shape are pinned: every declared key sits in a family, and `tree_jc`
    sits in two, since a key is stated as a model and as a parsimony instance.
    """
    stated = families()
    assert frozenset().union(*stated.values()) == DECLARED
    carrying = sorted(label for label, keys in stated.items() if "tree_jc" in keys)
    assert carrying == ["sec:parsimony", "sec:phylo"]


@pytest.mark.critical
@pytest.mark.infra
def test_a_word_that_is_not_the_problem_is_still_a_word_and_not_the_problem() -> None:
    """`BY_WORD` is asserted, not waived: an entry that goes stale fails.

    Each entry is a module the text search reaches and the selection does not.
    If one gains a fixture call or an import of defining code the scan reads,
    it stops being an exception and this says so; if one stops naming the
    family, the entry is dead and this says that too.
    """
    tree = _tree()
    for name, label in BY_WORD.items():
        keys = families()[label]
        assert name in unselected(keys, stems(label, keys), tree), (
            f"{name} no longer names {label} without being selected for it, so its "
            "entry in BY_WORD is stale"
        )


@pytest.mark.infra
def test_the_word_alone_is_infra_and_the_call_is_selected(tmp_path: Path) -> None:
    """The exclusion carries weight, on two modules written for it.

    One mentions `potts_lattice` in a comment and in a constant and calls
    nothing: no marker, so `infra`, so excluded --- and the comment is not even
    read, since prose is stripped. The other makes the registry call and is
    selected for the family. The markers are derived by the same
    `fixtures_named_in` that collection uses, not asserted into place.
    """
    word = tmp_path / "test_word_only.py"
    word.write_text(
        "# potts_lattice in a comment, and in a constant below.\n"
        'NAME = "potts_lattice"\n\n\ndef test_word() -> None:\n    assert NAME\n'
    )
    call = tmp_path / "test_call.py"
    call.write_text(f'def test_call() -> None:\n    {CALL}("potts_chain", "ci")\n')

    assert markers_of(word) == frozenset()
    assert "potts_lattice" not in code_of(word).split("NAME")[0]
    assert "potts_chain" in markers_of(call)

    keys = families()["sec:potts"]
    modules = {str(path): (markers_of(path), code_of(path)) for path in (word, call)}
    assert unselected(keys, stems("sec:potts", keys), modules) == []
    # The `infra` marker set is the only thing excluding the first: give it a
    # problem it does not name and it is reported.
    modules[str(word)] = (frozenset({"hmm"}), code_of(word))
    assert unselected(keys, stems("sec:potts", keys), modules) == [str(word)]


@pytest.mark.infra
def test_the_checker_reports_a_module_that_names_a_family_and_is_not_selected() -> None:
    """The guard is not vacuous, on inputs it cannot have derived.

    Three modules, one of each kind the rule distinguishes: one naming the
    family and exercising another problem, which is the defect; one selected
    for a key of the family; one carrying no problem marker at all.
    """
    keys = families()["sec:potts"]
    modules = {
        "names_it.py": (frozenset({"hmm"}), 'LATTICE = "potts_lattice"\n'),
        "selected.py": (frozenset({"potts_chain"}), 'CHAIN = "potts_chain"\n'),
        "infra.py": (frozenset(), 'LATTICE = "potts_lattice"\n'),
    }
    assert unselected(keys, stems("sec:potts", keys), modules) == ["names_it.py"]
