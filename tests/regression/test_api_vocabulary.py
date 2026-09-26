"""The package's public signatures use `DEV.md`'s API vocabulary (issue #1091).

Read from the source tree, not imported: a public function's parameter and a
public class's field are the names a caller writes, so a stray synonym there
is the drift root `CLAUDE.md`'s API conventions forbid.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[2] / "python" / "sal"

#: Names retired from public signatures, and the vocabulary's name for each.
RETIRED = {"k": "n_states"}


def _public_names(path: Path) -> list[tuple[int, str]]:
    """Every public function's parameter and public class's field, by line."""
    found: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            if node.name.startswith("_") and node.name != "__init__":
                continue
            arguments = node.args
            found.extend(
                (argument.lineno, argument.arg)
                for argument in (
                    arguments.posonlyargs + arguments.args + arguments.kwonlyargs
                )
            )
        elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            found.extend(
                (statement.lineno, statement.target.id)
                for statement in node.body
                if isinstance(statement, ast.AnnAssign)
                and isinstance(statement.target, ast.Name)
            )
    return found


@pytest.mark.critical
@pytest.mark.infra
def test_no_public_signature_uses_a_retired_name() -> None:
    retired = [
        f"{path.relative_to(PACKAGE)}:{line} {name} -> {RETIRED[name]}"
        for path in sorted(PACKAGE.rglob("*.py"))
        for line, name in _public_names(path)
        if name in RETIRED
    ]

    assert retired == [], f"{len(retired)} public names off the vocabulary: {retired}"
