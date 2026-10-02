"""No module of the package constructs a retired name (issue #1189).

Referee: the package source, parsed. The six per-family HMM objectives are
deprecated constructors kept one release; only the module that defines them
mentions them as calls.
"""

from __future__ import annotations

import ast

import pytest

from tests._paths import REPO_ROOT

#: The retired names.
RETIRED = (
    "HmmObjective",
    "GaussianHmmObjective",
    "PoissonHmmObjective",
    "BinomialHmmObjective",
    "BetaBinomialHmmObjective",
    "NegativeBinomialHmmObjective",
)

#: The one module that defines them.
HOME = REPO_ROOT / "python" / "sal" / "opt" / "hmm" / "objectives.py"


@pytest.mark.infra
def test_no_module_of_the_package_constructs_a_retired_name() -> None:
    package = REPO_ROOT / "python" / "sal"
    callers = []
    for path in sorted(package.rglob("*.py")):
        if path == HOME:
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name | ast.Attribute)
                and (
                    node.func.id if isinstance(node.func, ast.Name) else node.func.attr
                )
                in RETIRED
            ):
                callers.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
    assert callers == []
