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
SYMBOL = re.compile(r"`([^`]+)`")


def _resolves(symbol: str) -> bool:
    """A repository path, or an importable dotted name."""
    if "/" in symbol or symbol.endswith((".md", ".yaml", ".ipynb")):
        return (REPO_ROOT / symbol).exists()
    module_path, _, attribute = symbol.rpartition(".")
    try:
        return hasattr(importlib.import_module(module_path), attribute)
    except ModuleNotFoundError:
        return False


@pytest.mark.structural
def test_every_symbol_in_the_problem_catalogue_resolves() -> None:
    symbols = SYMBOL.findall(PROBLEMS.read_text())
    assert len(symbols) > 40, "the catalogue lost its table"

    missing = sorted({s for s in symbols if not _resolves(s)})
    assert missing == [], f"PROBLEMS.md names symbols that do not exist: {missing}"


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
