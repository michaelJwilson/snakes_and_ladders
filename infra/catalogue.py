"""One reader of ``PROBLEMS.md``, for everything that reads it.

Eight modules split the table on ``|``, each with its own filter for which
lines are rows: two spellings of the header test, two of the separator test,
three of the cell regexes, and one --- `tests/regression/test_fixture_registry.py`
--- with no short-row guard at all, so a row losing a column would have been
read as a row with a missing key rather than refused (issue #863,
design-audit row R20).

The filter is the property, not the spelling: a row is a line that starts a
cell, is not the header, is not the separator, and carries the four columns
``PROBLEMS.md`` declares. A row short of them is skipped here and counted
nowhere, which is what the guard over the table is for.

`tests/regression/test_problem_markers.py` keeps a reader of its own, and
says in its docstring why: a guard that imports the thing it checks agrees
with it by construction. That is one deliberate second reading, not eight.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from _paths import REPO_ROOT

#: The table this module reads.
CATALOGUE = REPO_ROOT / "PROBLEMS.md"

#: A fixture key in the **Key** column: a bare directory under
#: ``tests/regression/fixtures/``, which is also the marker its tests carry.
KEY_CELL = re.compile(r"`([a-z_0-9]+)`")

#: A defining name in the **Defines** column: code under the package, written
#: without the package prefix, so it carries a dot.
DEFINES_CELL = re.compile(r"`([a-z_]+\.[A-Za-z0-9_.]+)`")

#: A LaTeX label in the **Statement** column.
STATEMENT_CELL = re.compile(r"`([^`]+)`")

#: The columns a row declares: problem, key, statement, defines.
COLUMNS = 4


@dataclass(frozen=True)
class Row:
    """One row of the catalogue, by column.

    Parameters
    ----------
    title : str
        The **Problem** cell, as written.
    keys : tuple[str, ...]
        The **Key** cells, in the order the row writes them. A row with two
        keys is one problem declared at two instances.
    statements : tuple[str, ...]
        The **Statement** labels.
    defines : tuple[str, ...]
        The **Defines** names, rooted at ``snakes_and_ladders`` and written
        without it.
    """

    title: str
    keys: tuple[str, ...]
    statements: tuple[str, ...]
    defines: tuple[str, ...]


def cells(line: str) -> list[str]:
    """One Markdown table line as its stripped cells.

    Shared with the two readers of `DEV.md`'s tables, which filter their rows
    on their own anchors and split them the same way.

    Returns
    -------
    list[str]
    """
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


@cache
def rows(catalogue: Path = CATALOGUE) -> tuple[Row, ...]:
    """Every declared row of the catalogue, in file order.

    Parameters
    ----------
    catalogue : Path
        ``PROBLEMS.md``.

    Returns
    -------
    tuple[Row, ...]
    """
    found: list[Row] = []
    for line in catalogue.read_text().splitlines():
        if not line.startswith("| ") or line.startswith("| Problem") or "---" in line:
            continue
        cell = cells(line)
        if len(cell) < COLUMNS:
            continue
        found.append(
            Row(
                title=cell[0],
                keys=tuple(KEY_CELL.findall(cell[1])),
                statements=tuple(STATEMENT_CELL.findall(cell[2])),
                defines=tuple(DEFINES_CELL.findall(cell[3])),
            )
        )
    return tuple(found)


def keys(catalogue: Path = CATALOGUE) -> dict[str, tuple[str, ...]]:
    """``row title -> its fixture keys``.

    Returns
    -------
    dict[str, tuple[str, ...]]
    """
    return {row.title: row.keys for row in rows(catalogue)}


def defines(catalogue: Path = CATALOGUE) -> dict[str, frozenset[str]]:
    """``defining name -> the fixture keys of every row naming it``.

    Returns
    -------
    dict[str, frozenset[str]]
        A name appearing in two rows carries both keys: ``sec:phylo`` is
        stated once and declared at two instances.
    """
    found: dict[str, set[str]] = {}
    for row in rows(catalogue):
        for name in row.defines:
            found.setdefault(name, set()).update(row.keys)
    return {name: frozenset(held) for name, held in sorted(found.items())}


def statements(catalogue: Path = CATALOGUE) -> dict[str, tuple[str, ...]]:
    """``statement label -> the fixture keys stated by it``, in row order.

    Returns
    -------
    dict[str, tuple[str, ...]]
        A key may sit under two labels --- `tree_jc` is the Jukes--Cantor
        tree of ``sec:phylo`` and the parsimony problem of ``sec:parsimony``
        --- so this is not a partition of the keys. Order is the table's,
        which is what makes the first key of a label its principal instance.
    """
    found: dict[str, list[str]] = {}
    for row in rows(catalogue):
        for label in row.statements:
            found.setdefault(label, []).extend(row.keys)
    return {label: tuple(held) for label, held in found.items()}
