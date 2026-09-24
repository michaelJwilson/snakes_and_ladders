"""`docs/source/index.rst` claims to cover every submodule. Nothing checked it.

`sphinx-build -W` fails on a broken entry and never on an absent one
(`docs/CLAUDE.md`). Repairs: #135 (fourteen entries), #154 (eighteen, all of
`snakes_and_ladders.qa`), and five more when this test was written.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[3]
INDEX = REPOSITORY / "docs" / "source" / "index.rst"
PACKAGE = REPOSITORY / "python"


def _listed() -> set[str]:
    return set(re.findall(r"automodule:: (\S+)", INDEX.read_text()))


def _present() -> set[str]:
    modules = set()
    for path in sorted((PACKAGE / "snakes_and_ladders").rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        relative = path.relative_to(PACKAGE)
        parts = (
            relative.parent.parts
            if path.name == "__init__.py"
            else relative.with_suffix("").parts
        )
        modules.add(".".join(parts))
    return modules


@pytest.mark.critical
@pytest.mark.infra
def test_every_module_has_an_entry() -> None:
    missing = sorted(_present() - _listed())

    assert not missing, (
        "modules absent from docs/source/index.rst: "
        + ", ".join(missing)
        + ". `sphinx-build -W` cannot catch this, which is why the invariant "
        "has been repaired three times."
    )


@pytest.mark.critical
@pytest.mark.infra
def test_every_entry_names_a_module_that_exists() -> None:
    # The other direction, which `sphinx-build -W` *does* catch -- but it
    # catches it in a job that takes a minute, and this takes milliseconds.
    stale = sorted(_listed() - _present())

    assert not stale, f"entries naming modules that no longer exist: {stale}"
