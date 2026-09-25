"""The map lists the surface, counted against a second walk of the tree.

Issue #576. The referee is a second count of the tree --- `pathlib` and `ast`
over the package, with no import of the generator's walk --- so a dropped
module fails rather than reading as complete. A public top-level function with
no summary fails `--write`; the suite asserts the tree has none and that the
refusal fires on a tree that does.
"""

from __future__ import annotations

import ast
from pathlib import Path

import api_map
import pytest

from tests._paths import REPO_ROOT

PACKAGE = REPO_ROOT / "python" / "sal"
GENERATED = REPO_ROOT / "docs" / "tex" / "generated" / "api_map.tex"


def _second_walk() -> dict[str, int]:
    """Modules, classes, functions and public methods, counted here from scratch.

    A second implementation, not a call into the generator: two readings agree.
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
    assert api_map.counts(api_map.modules()) == _second_walk()


@pytest.mark.infra
def test_every_module_reaches_the_rendered_map() -> None:
    # Counting is not placement: a module could be found and still not
    # rendered. Each one's label is asserted in the output, so a module the
    # layout skipped fails here rather than being absent from a 64-page PDF
    # nobody reads end to end.
    found = api_map.modules()

    rendered = api_map.render(found)
    for module in found:
        label = f"\\label{{sec:api:{module.path.replace('/', ':')}}}"
        assert label in rendered, module.path


@pytest.mark.infra
def test_no_public_function_would_be_typeset_as_a_blank() -> None:
    # The guard the ticket asks for, in the direction that matters: the tree
    # has none today, and `--write` refuses one rather than printing an empty
    # cell in a document that states the surface.
    assert api_map.without_summary(api_map.modules()) == []


@pytest.mark.smoke
def test_the_refusal_fires_on_a_function_without_a_summary(tmp_path: Path) -> None:
    # Guarding the guard: the check above passes on a clean tree whether or
    # not the detector works, so the detector is shown a tree that is not.
    module = tmp_path / "blank.py"
    module.write_text('"""A module."""\n\n\ndef public():\n    pass\n')

    found = api_map.modules(tmp_path)

    assert api_map.without_summary(found) == ["blank/public"]


@pytest.mark.infra
def test_the_generated_map_is_not_committed() -> None:
    # `infra/build_documents.sh` writes it and the PDF is what the tree
    # carries, the arrangement `problems_tables.py` already uses (issue #425).
    # A committed copy is a machine-written merge participant for nothing.
    if not GENERATED.exists():
        return
    tracked = (REPO_ROOT / ".gitignore").read_text()
    assert "docs/tex/generated" in tracked
