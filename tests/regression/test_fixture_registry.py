"""The fixture registry and the problem catalogue name each other (issue #382).

A supported problem's instance is a file, and `PROBLEMS.md` says which file.
Two ways that can rot: a row naming a fixture that no longer loads, and a
fixture no row names --- an instance nothing is claimed about, which is how
the catalogue would come to describe less than the tree carries. Both fail
here.

The registry's oracle vocabulary is held to the tables the textbook typesets,
so a fixture cannot state an oracle the applicability tables have no column
for; and the two fixtures that restate a canonical constructor are pinned
against it, since a file that has drifted from the instance it copies is a
second truth (``sim/CLAUDE.md``).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pytest

from snakes_and_ladders.emissions import CategoricalEmission
from snakes_and_ladders.fixtures import Scale
from snakes_and_ladders.sim.canonical import frustrated_triangular_lattice
from snakes_and_ladders.sim.fixtures import (
    FIXTURES_DIR,
    LOADERS,
    ORACLES,
    fixture,
    fixtures,
    problems,
    tiers,
)
from snakes_and_ladders.sim.spatio_sequential import canonical_spatio_sequential

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "infra"))

import problems_tables  # noqa: E402

CATALOGUE = REPO_ROOT / "PROBLEMS.md"
_FIXTURE_PATH = re.compile(r"`(tests/regression/fixtures/[a-z_0-9]+/[a-z]+\.yaml)`")

#: The applicability table column each stated oracle fills. ``none`` fills
#: none, which is the point of stating it: the size is past every oracle.
ORACLE_COLUMNS = {
    "enumeration": "enumeration or brute force",
    "transfer-matrix": "independent algorithm",
    "closed-form": "closed form or published value",
}


def _catalogue_fixtures() -> dict[str, list[str]]:
    """``row title -> fixture paths`` for every row of the catalogue."""
    found: dict[str, list[str]] = {}
    for line in CATALOGUE.read_text().splitlines():
        if not line.startswith("| ") or line.startswith("| Problem") or "---" in line:
            continue
        title = line.strip().strip("|").split("|")[0].strip()
        found[title] = _FIXTURE_PATH.findall(line)
    return found


@pytest.mark.structural
def test_every_catalogue_row_names_a_ci_fixture_that_loads() -> None:
    # The claim the column makes: this problem has an instance, at the size
    # the per-pull-request suite runs. A row that names none is a problem
    # nothing can be applied to without inventing one.
    rows = _catalogue_fixtures()
    assert len(rows) > 8, "the catalogue lost its table"

    for title, paths in rows.items():
        assert paths, f"{title} names no fixture"
        assert any(path.endswith("/ci.yaml") for path in paths), (
            f"{title} names no ci fixture: {paths}"
        )
        for path in paths:
            problem, tier = Path(path).parent.name, Path(path).stem
            assert fixture(problem, tier).path == REPO_ROOT / path


@pytest.mark.structural
def test_every_fixture_is_named_by_the_catalogue() -> None:
    # The other direction: an instance the catalogue does not claim is one
    # no row is answerable for.
    named = {
        Path(path).parent.name
        for paths in _catalogue_fixtures().values()
        for path in paths
    }

    assert set(problems()) == named


@pytest.mark.structural
def test_every_fixture_loads_and_states_an_oracle_the_tables_know() -> None:
    loaded = [
        fixture(problem, tier) for problem in problems() for tier in tiers(problem)
    ]
    assert len(loaded) == len(list(FIXTURES_DIR.rglob("*.yaml")))

    for entry in loaded:
        assert entry.oracle in ORACLES
        assert entry.model in LOADERS
        assert entry.params is not None
        column = ORACLE_COLUMNS.get(entry.oracle)
        assert column is None or column in problems_tables.ORACLE_COLUMNS


@pytest.mark.structural
def test_every_problem_declares_the_ci_tier() -> None:
    # `fixtures("ci")` is what a test parametrizing over the class of
    # problems iterates; a problem missing from it is silently untested
    # there rather than failing.
    assert {entry.problem for entry in fixtures(Scale.CI)} == set(problems())


@pytest.mark.oracle
def test_the_coupled_fixture_is_the_canonical_instance() -> None:
    # The file restates `canonical_spatio_sequential`, whose enumerable size
    # is the reason the instance exists. Drift between them would leave two
    # instances under one name.
    declared = fixture("spatio_sequential", Scale.CI).params
    canonical = canonical_spatio_sequential()

    assert declared.graph == canonical.graph
    assert (declared.n_classes, declared.n_states, declared.n_positions) == (
        canonical.n_classes,
        canonical.n_states,
        canonical.n_positions,
    )
    assert (declared.beta, declared.self_transition) == (
        canonical.beta,
        canonical.self_transition,
    )
    np.testing.assert_allclose(declared.initial, canonical.initial)
    for declared_family, canonical_family in zip(
        declared.emissions, canonical.emissions, strict=True
    ):
        assert isinstance(declared_family, CategoricalEmission)
        assert isinstance(canonical_family, CategoricalEmission)
        np.testing.assert_allclose(
            declared_family.matrix.numpy(), canonical_family.matrix.numpy()
        )


@pytest.mark.oracle
def test_the_frustrated_fixture_builds_the_lattice_with_the_known_ground_state() -> (
    None
):
    # The 3x3 periodic triangular antiferromagnet: the counting argument
    # fixes one agreeing edge in three, which is what makes this instance
    # worth declaring at all.
    params = fixture("frustrated_lattice", Scale.CI).params

    assert params.lattice() == frustrated_triangular_lattice(
        params.shape, params.boundary, params.coupling
    )
