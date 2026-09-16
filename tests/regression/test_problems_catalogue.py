"""The problem catalogue names only what exists, and the checks ledger is writable.

`PROBLEMS.md` is a table of symbols; a symbol that stops resolving is a row
describing a problem the tree no longer supports the way the row says.
`CHECKS.md` is generated from the tests' own markers and is not committed
(issue #425), so what is checked here is what the generator writes: every
significant test appears in it exactly once, and the guard refuses a stale
file. That the tree carries no committed copy is
`tests/regression/test_review_gate_scripts.py`.
"""

from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "infra"))

import checks_ledger  # noqa: E402

PROBLEMS = REPO_ROOT / "PROBLEMS.md"
FIXTURES = REPO_ROOT / "tests" / "regression" / "fixtures"
PACKAGE = "snakes_and_ladders."
#: The two hand-written columns: a bare fixture key, and code under the
#: package named without its prefix.
KEY_CELL = re.compile(r"`([a-z_0-9]+)`")
DEFINES_CELL = re.compile(r"`([a-z_]+\.[A-Za-z0-9_.]+)`")


def _rows() -> list[tuple[str, list[str], list[str]]]:
    """``(problem, keys, defining names)`` per row of the catalogue."""
    found = []
    for line in PROBLEMS.read_text().splitlines():
        if not line.startswith("| ") or line.startswith("| Problem") or "---" in line:
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 4:
            continue
        found.append(
            (cells[0], KEY_CELL.findall(cells[1]), DEFINES_CELL.findall(cells[3]))
        )
    return found


def _resolves(symbol: str) -> bool:
    """A repository path, an importable module, or a symbol one defines.

    The module is tried first: `Defines` names a module wherever the module
    belongs to one problem, and a submodule is not an attribute of its package
    until something imports it, so the attribute reading alone refuses
    `learn.hmm` (issue #640).
    """
    if "/" in symbol or symbol.endswith((".md", ".yaml", ".ipynb")):
        return (REPO_ROOT / symbol).exists()
    try:
        importlib.import_module(symbol)
    except ImportError:
        pass
    else:
        return True
    module_path, _, attribute = symbol.rpartition(".")
    try:
        return hasattr(importlib.import_module(module_path), attribute)
    except ImportError:
        return False


@pytest.mark.structural
def test_every_defining_name_in_the_catalogue_resolves() -> None:
    # Read from the `Defines` column rather than from every backtick in the
    # file (issue #640): the minimal table backticks a fixture key, a LaTeX
    # label and a column name too, and none of those is a dotted symbol.
    names = [name for _, _, defines in _rows() for name in defines]
    assert len(names) > 40, "the catalogue lost its table"

    missing = sorted({name for name in names if not _resolves(f"{PACKAGE}{name}")})
    assert missing == [], f"PROBLEMS.md names code that does not exist: {missing}"


@pytest.mark.structural
def test_every_key_in_the_catalogue_declares_a_fixture() -> None:
    # A key is the fixture directory and the marker, which are one name. A key
    # naming no directory is a marker no test can carry.
    missing = sorted(
        {key for _, keys, _ in _rows() for key in keys if not (FIXTURES / key).is_dir()}
    )
    assert missing == [], f"PROBLEMS.md keys name no fixture: {missing}"


@pytest.mark.structural
def test_every_significant_test_appears_in_the_ledger_once() -> None:
    # Read from what the generator writes rather than from a file: the ledger
    # is not committed (issue #425), and the property is a property of the
    # rendering, not of whether someone remembered to re-run the tool.
    entries = checks_ledger.rows()
    names = [f"{Path(file).name}::{name}" for file, name, _, _ in entries]

    assert len(names) == len(set(names))
    text = checks_ledger.render(entries)
    for name in names:
        assert f"`{name}`" in text, name
    assert entries, "no oracle or simulated-truth test found"


@pytest.mark.structural
def test_the_ledger_guard_fails_on_a_stale_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Guards the guard, per the repository's pattern.
    stale = tmp_path / "CHECKS.md"
    stale.write_text(checks_ledger.render(checks_ledger.rows()) + "\nan edit by hand\n")
    monkeypatch.setattr(checks_ledger, "LEDGER", stale)

    assert checks_ledger.main(["--check"]) == 1
    assert checks_ledger.main(["--write"]) == 0
    assert checks_ledger.main(["--check"]) == 0
