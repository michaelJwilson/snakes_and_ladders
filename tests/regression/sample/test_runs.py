"""Every annealer's result is one `Annealed`, every tempering's one `Tempered` (issue #1090).

Referees: the guard reads the package's classes, so a new annealer's result
that is not an `Annealed` fails here; and each subclass carries the shared
fields a caller compares two annealers on.
"""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

import pytest
from sal.sample.gibbs import AnnealedLabelling, AnnealedTopology
from sal.sample.hmc import AnnealedTheta
from sal.sample.hmc import Tempered as HmcTempered
from sal.sample.potts_mcmc import AnnealedPotts, ClusterTempered, TemperedChains
from sal.sample.schedule import Annealed, Tempered

PACKAGE = Path(__file__).resolve().parents[3] / "python" / "sal"

#: Annealed results that are not an `Annealed`, and why.
EXEMPT = {
    # Its best is a mixture, weights and components, so it takes the EM
    # fit's field set of #1090's fifth finding rather than `best`.
    "AnnealedAssignments": "sample/mixture_anneal.py",
    # An annealed EM fit: it wraps the fit, and the fit is its result.
    "AnnealedFit": "sandbox/annealed_em.py",
}


@pytest.mark.critical
@pytest.mark.infra
def test_every_annealed_result_is_an_annealed() -> None:
    outside = [
        f"{path.relative_to(PACKAGE)}:{node.lineno} {node.name}"
        for path in sorted(PACKAGE.rglob("*.py"))
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.ClassDef)
        and node.name.startswith("Annealed")
        and node.name != "Annealed"
        and not any("Annealed[" in ast.unparse(base) for base in node.bases)
        and EXEMPT.get(node.name) != str(path.relative_to(PACKAGE))
    ]

    assert outside == [], f"annealed results outside `Annealed`: {outside}"


@pytest.mark.smoke
@pytest.mark.parametrize(
    "result", [AnnealedPotts, AnnealedLabelling, AnnealedTheta, AnnealedTopology]
)
def test_each_annealed_result_carries_the_shared_fields(result: type) -> None:
    names = {field.name for field in dataclasses.fields(result)}
    shared = {field.name for field in dataclasses.fields(Annealed)}

    assert shared <= names
    assert issubclass(result, Annealed)


#: Tempered dataclasses that are not a `Tempered`, and why.
TEMPERED_EXEMPT = {
    # Samples from the ladder, not a best point found on it.
    "TemperedEnsemble": "sample/tempered.py",
    "SimulatedTempered": "sample/annealed.py",
    "TemperedSupport": "search/support.py",
    # An input: the model a tempering runs on.
    "TemperedModel": "sample/potts_mcmc/chains.py",
}


@pytest.mark.critical
@pytest.mark.infra
def test_every_tempered_result_is_a_tempered() -> None:
    outside = [
        f"{path.relative_to(PACKAGE)}:{node.lineno} {node.name}"
        for path in sorted(PACKAGE.rglob("*.py"))
        if path.relative_to(PACKAGE) != Path("sample/schedule.py")
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.ClassDef)
        and "Tempered" in node.name
        and any(
            isinstance(decorator, ast.Call)
            and ast.unparse(decorator.func) == "dataclass"
            for decorator in node.decorator_list
        )
        and not any("Tempered" in ast.unparse(base) for base in node.bases)
        and TEMPERED_EXEMPT.get(node.name) != str(path.relative_to(PACKAGE))
    ]

    assert outside == [], f"tempered results outside `Tempered`: {outside}"


@pytest.mark.smoke
@pytest.mark.parametrize("result", [TemperedChains, ClusterTempered, HmcTempered])
def test_each_tempered_result_carries_the_shared_fields(result: type) -> None:
    names = {field.name for field in dataclasses.fields(result)}
    shared = {field.name for field in dataclasses.fields(Tempered)}

    assert shared <= names
    assert issubclass(result, Tempered)
