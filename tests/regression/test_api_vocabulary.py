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
#: A key is read three ways: ``name`` is any public parameter or field;
#: ``function(name)`` is one function's parameter, where ``name`` keeps another
#: meaning elsewhere; ``.name`` is a method of any class, public or not.
#: ``emissions`` stays a package and a parameters field (issue #1163). A
#: ``__call__`` is public, as ``__init__`` is: a protocol's call is its
#: signature (issue #1176).
RETIRED = {
    "k": "n_states",
    "baum_welch_family(emissions)": "components",
    "baum_welch_rectangular(emissions)": "components",
    ".emissions": "components",
    "viterbi(emissions)": "components",
    "hmm_log_likelihood(emissions)": "components",
    "compiled_family(emissions)": "components",
    "normalizer(emissions)": "components",
    "__call__(emissions)": "components",
}

#: The dunder methods a caller writes the parameters of.
_PUBLIC_DUNDERS = ("__init__", "__call__")


def _public_names(source: str) -> list[tuple[int, str]]:
    """Every key `RETIRED` can match in ``source``, by line."""
    found: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            if node.name.startswith("_") and node.name not in _PUBLIC_DUNDERS:
                continue
            arguments = node.args
            for argument in (
                arguments.posonlyargs + arguments.args + arguments.kwonlyargs
            ):
                found.append((argument.lineno, argument.arg))
                found.append((argument.lineno, f"{node.name}({argument.arg})"))
        elif isinstance(node, ast.ClassDef):
            found.extend(
                (statement.lineno, f".{statement.name}")
                for statement in node.body
                if isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef)
            )
            if node.name.startswith("_"):
                continue
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
        for line, name in _public_names(path.read_text())
        if name in RETIRED
    ]

    assert retired == [], f"{len(retired)} public names off the vocabulary: {retired}"


@pytest.mark.infra
def test_the_guard_reads_each_form_of_a_retired_key() -> None:
    """Each key form catches its own site and no other (issue #1163)."""
    source = """
def baum_welch_family(observations, emissions): ...
def simulate(emissions, k): ...
class _HmmObjective:
    def emissions(self, theta): ...
class HmmParams:
    emissions: object
class CovariateUpdate:
    def __call__(self, emissions, posterior, covariate): ...
"""
    caught = sorted(
        (line, name) for line, name in _public_names(source) if name in RETIRED
    )

    # The function-scoped argument, the bare name and the method are caught;
    # `simulate(emissions)` and the `emissions` field are the other meaning;
    # a protocol's `__call__` is public (issue #1176).
    assert caught == [
        (2, "baum_welch_family(emissions)"),
        (3, "k"),
        (5, ".emissions"),
        (9, "__call__(emissions)"),
    ]
