"""The map lists the surface, counted against a second walk of the tree.

Issue #576. A generated document fails quietly: the entries look complete
whatever the generator dropped, and a reader cannot tell a module that is
absent because it has no public names from one absent because its parse
raised. So what is asserted here is not that the generator ran but that its
count is the tree's, taken a second way --- `pathlib` and `ast` over the
package, with no import of the generator's own walk.

The two guards the ticket asks for are checked as guards: a public top-level
function with no summary is a blank the document would typeset, so `--write`
fails on one, and the suite asserts both that the tree has none today and that
the refusal fires on a tree that does.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGE = REPO_ROOT / "python" / "snakes_and_ladders"
GENERATED = REPO_ROOT / "docs" / "tex" / "generated" / "api_map.tex"


def _api_map() -> ModuleType:
    """`infra/api_map.py`, loaded by path: `infra/` is not an importable package."""
    spec = importlib.util.spec_from_file_location(
        "api_map", REPO_ROOT / "infra" / "api_map.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["api_map"] = module
    spec.loader.exec_module(module)
    return module


def _second_walk() -> dict[str, int]:
    """Modules, classes, functions and public methods, counted here from scratch.

    Deliberately a second implementation and not a call into the generator:
    two readings of one tree that must agree, which is what says neither
    dropped anything.
    """
    tally = {"module": 0, "class": 0, "function": 0, "method": 0}
    for path in sorted(PACKAGE.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text())
        tally["module"] += 1
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                if node.name.startswith("_"):
                    continue
                tally["class"] += 1
                for member in node.body:
                    if isinstance(
                        member, ast.FunctionDef | ast.AsyncFunctionDef
                    ) and not member.name.startswith("_"):
                        decorated = any(
                            getattr(decorator, "attr", getattr(decorator, "id", ""))
                            == "overload"
                            for decorator in member.decorator_list
                        )
                        if not decorated:
                            tally["method"] += 1
            elif isinstance(
                node, ast.FunctionDef | ast.AsyncFunctionDef
            ) and not node.name.startswith("_"):
                decorated = any(
                    getattr(decorator, "attr", getattr(decorator, "id", ""))
                    == "overload"
                    for decorator in node.decorator_list
                )
                if not decorated:
                    tally["function"] += 1
    return tally


@pytest.mark.infra
def test_the_map_counts_what_a_second_walk_finds() -> None:
    # The claim the document makes about itself, checked against a walk that
    # shares no line with the generator's. A module dropped for any reason --
    # a parse that raised, a filter that widened -- moves one of these four.
    generator = _api_map()

    assert generator.counts(generator.modules()) == _second_walk()


@pytest.mark.infra
def test_every_module_reaches_the_rendered_map() -> None:
    # Counting is not placement: a module could be found and still not
    # rendered. Each one's label is asserted in the output, so a module the
    # layout skipped fails here rather than being absent from a 64-page PDF
    # nobody reads end to end.
    generator = _api_map()
    found = generator.modules()

    rendered = generator.render(found)
    for module in found:
        label = f"\\label{{sec:api:{module.path.replace('/', ':')}}}"
        assert label in rendered, module.path


@pytest.mark.infra
def test_no_public_function_would_be_typeset_as_a_blank() -> None:
    # The guard the ticket asks for, in the direction that matters: the tree
    # has none today, and `--write` refuses one rather than printing an empty
    # cell in a document that states the surface.
    generator = _api_map()

    assert generator.without_summary(generator.modules()) == []


@pytest.mark.smoke
def test_the_refusal_fires_on_a_function_without_a_summary(tmp_path: Path) -> None:
    # Guarding the guard: the check above passes on a clean tree whether or
    # not the detector works, so the detector is shown a tree that is not.
    generator = _api_map()
    module = tmp_path / "blank.py"
    module.write_text('"""A module."""\n\n\ndef public():\n    pass\n')

    found = generator.modules(tmp_path)

    assert generator.without_summary(found) == ["blank/public"]


@pytest.mark.infra
def test_the_generated_map_is_not_committed() -> None:
    # `infra/build_documents.sh` writes it and the PDF is what the tree
    # carries, the arrangement `problems_tables.py` already uses (issue #425).
    # A committed copy is a machine-written merge participant for nothing.
    if not GENERATED.exists():
        return
    tracked = (REPO_ROOT / ".gitignore").read_text()
    assert "docs/tex/generated" in tracked
