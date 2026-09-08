"""The textbook's applicability tables are what ``PROBLEMS.md`` generates (issue #358).

The same contract as ``CHECKS.md`` and a QA figure: the committed file is
what the tool writes, the textbook names no code, and a catalogue symbol the
generator cannot name fails rather than dropping out of the table.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "infra"))

import problems_tables  # noqa: E402


@pytest.mark.structural
def test_the_committed_tables_are_what_the_generator_writes() -> None:
    # Regenerate with `python infra/problems_tables.py --write`.
    assert problems_tables.GENERATED.read_text() == problems_tables.render()


@pytest.mark.structural
def test_the_generated_file_names_no_code() -> None:
    # The textbook inputs this file, and `docs/CLAUDE.md` forbids a module
    # path, a filename or a function call in the textbook.
    text = problems_tables.GENERATED.read_text()
    assert [
        needle
        for needle in ("snakes_and_ladders.", ".py", "\\texttt{")
        if needle in text
    ] == []


@pytest.mark.structural
def test_every_catalogue_row_has_an_algorithm_and_an_oracle() -> None:
    # A row with no oracle would be a problem nothing referees, which the
    # catalogue's own preamble forbids.
    text = problems_tables.GENERATED.read_text()
    for problem, symbols in problems_tables.rows():
        assert problems_tables.unnamed(symbols) == [], problem
        assert any(s in problems_tables.ORACLES for s in symbols), problem
        assert text.count(problems_tables._tex_text(problem)) == 2, problem


@pytest.mark.edge_case
def test_a_symbol_the_generator_cannot_name_is_refused(tmp_path: Path) -> None:
    # Guards the guard: the failure mode is a table quietly narrower than the
    # catalogue, so an unnamed symbol must raise rather than be skipped.
    catalogue = tmp_path / "PROBLEMS.md"
    catalogue.write_text(
        "| Problem | Simulate |\n| --- | --- |\n"
        "| Made up | `snakes_and_ladders.no_such.symbol` |\n"
    )
    with pytest.raises(problems_tables.UnnamedSymbolError, match="no_such.symbol"):
        problems_tables.render(catalogue)
    assert problems_tables.rows(catalogue) == [("Made up", ["no_such.symbol"])]
