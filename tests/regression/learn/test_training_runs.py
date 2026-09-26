"""Every trainer's result is one `TrainingRun` (issue #1090).

Referee: the guard reads the package, so a new trainer's result outside
`TrainingRun` fails here.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[3] / "python" / "sal"


@pytest.mark.critical
@pytest.mark.infra
def test_every_training_result_is_a_training_run() -> None:
    outside = [
        f"{path.relative_to(PACKAGE)}:{node.lineno} {node.name}"
        for path in sorted((PACKAGE / "learn").rglob("*.py"))
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.ClassDef)
        and node.name.endswith("Training")
        and not any("TrainingRun" in ast.unparse(base) for base in node.bases)
    ]

    assert outside == [], f"training results outside `TrainingRun`: {outside}"
