"""The superset claim of issue #622, stated per family and not per word.

For each problem the derived selection covers what a text search finds. Per
word it is false (`grep -rl potts` finds three keys and every guard), so it is
stated per family: the keys sharing a `PROBLEMS.md` **Statement**. Eleven
families over 226 modules; the difference is empty for nine and the two words
naming no problem are asserted in `BY_WORD`. Narrowed three ways: prose is not
read, a package import is the catalogue's, and a stem matches only as a name's
head (``tree_likelihood``). The selection is the item's markers, derived and
hand-written (`search/test_decoding.py` writes two).
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

#: The three modules that name a family and exercise none, each asserted, so
#: an entry that stops being a false match fails: a notebook path as data, a
#: ``parsimony_start=True`` flag to shared machinery, and the graph-theoretic
#: tree of `tree_messages` (a chain and a Potts tree, no phylogeny).
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
    """A key as a whole token, where ``-`` and ``_`` bind (``"emission-mixture"``)."""
    return re.compile(_EDGE + re.escape(key) + r"(?![0-9A-Za-z_\-])")


def _stem_pattern(stem: str) -> re.Pattern[str]:
    """A stem as the head of a name: ``tree_likelihood``, not ``tree``.

    The bare word is an `ast` tree, a ``"tree"`` schedule, a keyword.
    """
    return re.compile(_EDGE + re.escape(stem) + r"_[0-9A-Za-z]")


@cache
def families(catalogue: Path = CATALOGUE) -> dict[str, frozenset[str]]:
    """``statement label -> the fixture keys stated by it``, via `infra/catalogue.py`.

    Not a partition: `tree_jc` sits in ``sec:phylo`` and ``sec:parsimony`` (#863).
    """
    return {
        label: frozenset(keys)
        for label, keys in sorted(catalogue_reader.statements(catalogue).items())
    }


def stems(label: str, keys: frozenset[str]) -> frozenset[str]:
    """The family's word stems, derived from the statement and from its keys.

    ``sec:phylo`` gives ``phylo`` and ``tree``; keys are not repeated as stems.
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

    What is left is what the module does; prose and imports are read elsewhere.
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

    Written markers are read as decorators, not text; empty means `infra`.
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

    Sorted; a module with no problem marker is `infra` and excluded.
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
    """Every collected module under ``tests/regression`` but ``docs/``, read once."""
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

    The two #614 modules are asserted in `test_problem_markers.py`, not here.
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

    Every declared key sits in a family, and `tree_jc` in two.
    """
    stated = families()
    assert frozenset().union(*stated.values()) == DECLARED
    carrying = sorted(label for label, keys in stated.items() if "tree_jc" in keys)
    assert carrying == ["sec:parsimony", "sec:phylo"]


@pytest.mark.critical
@pytest.mark.infra
def test_a_word_that_is_not_the_problem_is_still_a_word_and_not_the_problem() -> None:
    """`BY_WORD` is asserted, not waived: an entry that goes stale fails.

    Fails if an entry gains a selection, or stops naming the family.
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

    A word-only module is `infra`; a registry call is selected, by `fixtures_named_in`.
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

    Three modules: the defect, a selected one, and one with no problem marker.
    """
    keys = families()["sec:potts"]
    modules = {
        "names_it.py": (frozenset({"hmm"}), 'LATTICE = "potts_lattice"\n'),
        "selected.py": (frozenset({"potts_chain"}), 'CHAIN = "potts_chain"\n'),
        "infra.py": (frozenset(), 'LATTICE = "potts_lattice"\n'),
    }
    assert unselected(keys, stems("sec:potts", keys), modules) == ["names_it.py"]
