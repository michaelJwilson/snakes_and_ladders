"""The fixture registry and the problem catalogue name each other (issue #382).

A supported problem's instance is a file, and `PROBLEMS.md` says which file.
Two ways that can rot: a row naming a fixture that no longer loads, and a
fixture no row names --- an instance nothing is claimed about, which is how
the catalogue would come to describe less than the tree carries. Both fail
here.

The rule ``sim/CLAUDE.md`` states --- a supported instance is a fixture,
never a literal --- is enforced here for the two consumers that cannot import
the suite's helpers: the QA scripts and the notebooks. Neither may construct a
problem's declared truth or call a canonical constructor.

The registry's oracle vocabulary is held to the tables the textbook typesets,
so a fixture cannot state an oracle the applicability tables have no column
for; and the two fixtures that restate a canonical constructor are pinned
against it, since a file that has drifted from the instance it copies is a
second truth (``sim/CLAUDE.md``).

The baseline records of issue #401 are held to the one property that makes
reading a cached number safe: the reader returns what the writer wrote, and
raises where the tree has moved under it. Both directions are asserted here,
the second against a copy of a fixture directory with one byte changed, since
the committed fixtures must not be edited to prove it.
"""

from __future__ import annotations

import ast
import json
import re
import shutil
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
    StaleBaselineError,
    baseline,
    baseline_path,
    baselines,
    fixture,
    fixtures,
    problems,
    read_baseline,
    tiers,
)
from snakes_and_ladders.sim.spatio_sequential import canonical_spatio_sequential

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "infra"))

import baselines as baseline_script  # noqa: E402
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


# --- the baseline records (issue #401) ---------------------------------------

#: The cheapest record to recompute, so the determinism check pays 2 s rather
#: than the 28 s the whole set costs.
CHEAPEST = "potts_chain/ci"


@pytest.mark.structural
def test_every_committed_baseline_reads_back_against_the_current_tree() -> None:
    # The round trip the readers depend on: what `infra/baselines.py --write`
    # wrote is what `baseline()` returns, and its digest is this tree's. A
    # record left behind by a change to the code it measures fails here, on
    # the pull request that made the change, without recomputing a number.
    recorded = baselines()
    assert recorded, "no baseline record is committed"

    for problem, tier in recorded:
        record = baseline(problem, tier)
        assert (record.problem, record.tier) == (problem, tier)
        assert record.path == baseline_path(problem, tier)
        assert record.measurements, f"{record.path} records no measurement"
        for name, measurement in record.measurements.items():
            assert measurement.algorithm, f"{record.path}: {name} names no algorithm"
        assert read_baseline(record.path).digest == record.digest


@pytest.mark.edge_case
def test_a_mutated_fixture_makes_its_baseline_unreadable(tmp_path: Path) -> None:
    # The property that makes a cached number safe to read. Against a *copy*
    # of the fixture directory, so the committed instance is untouched: one
    # changed seed and the record beside it is refused rather than served.
    copied = tmp_path / "fixtures"
    shutil.copytree(FIXTURES_DIR, copied)
    assert baseline("tree_search", Scale.RELEASE, copied).value("greedy_rate") == 0.48

    fixture_file = copied / "tree_search" / "release.yaml"
    fixture_file.write_text(
        fixture_file.read_text().replace("seed: 20260907", "seed: 20260908")
    )

    with pytest.raises(StaleBaselineError, match="different tree"):
        baseline("tree_search", Scale.RELEASE, copied)


@pytest.mark.edge_case
def test_an_edited_budget_makes_its_baseline_unreadable(tmp_path: Path) -> None:
    # The other half of the digest's job. The recorded value is an output and
    # is not hashed --- the release gate recomputes it --- but the budget that
    # says what the value means is an input, so a record whose restart count
    # was edited to match a test is refused rather than believed.
    copied = tmp_path / "fixtures"
    shutil.copytree(FIXTURES_DIR, copied)
    record = copied / "tree_search" / "release.baseline.json"
    record.write_text(record.read_text().replace('"starts": 50', '"starts": 20'))

    with pytest.raises(StaleBaselineError, match="different tree"):
        baseline("tree_search", Scale.RELEASE, copied)


@pytest.mark.structural
def test_the_same_baseline_computed_twice_is_the_same_record() -> None:
    # A reference algorithm whose answer moved between two runs of the same
    # tree would make every committed record a snapshot rather than a fact,
    # and the release gate would fail at random. Checked on the cheapest
    # record; the rest are recomputed at the release gate.
    (spec,) = baseline_script.selected([CHEAPEST])
    first, second = baseline_script.compute(spec), baseline_script.compute(spec)

    assert baseline_script.differences(first, second) == []
    assert first.digest == second.digest
    assert baseline_script.differences(first, read_baseline(first.path)) == []


@pytest.mark.edge_case
def test_the_writer_refuses_a_record_it_does_not_know() -> None:
    with pytest.raises(ValueError, match="no baseline spec"):
        baseline_script.selected(["tree_search/nonexistent"])


# --- a supported instance is a fixture, never a literal ----------------------

#: What a figure script or a notebook may not build for itself: the declared
#: truth of a supported problem, and the canonical constructors that stand for
#: one. Each has a fixture file, so a caller that constructs one is declaring
#: a second instance under the same name.
FORBIDDEN = frozenset(
    {
        "SimulationParams",
        "PottsParams",
        "PottsLatticeParams",
        "HmmParams",
        "MixtureParams",
        "SpatioSequentialParams",
        "LdpcParams",
        "FrustratedLatticeParams",
        "TestFunctionParams",
        "TestFunctionSuite",
        "frustrated_triangular_lattice",
        "planted_spin_glass",
        "canonical_spatio_sequential",
        "ambiguous_hmm",
        "gallager_code",
    }
)

QA = REPO_ROOT / "python" / "snakes_and_ladders" / "qa"
NOTEBOOKS = REPO_ROOT / "docs" / "nb"


def _constructions(source: str) -> set[str]:
    """The forbidden names ``source`` *calls*, ignoring imports and annotations."""
    called = set()
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        name = node.func.id if isinstance(node.func, ast.Name) else None
        if isinstance(node.func, ast.Attribute):
            name = node.func.attr
        if name in FORBIDDEN:
            called.add(name)
    return called


def _notebook_code(path: Path) -> str:
    """Every code cell of a notebook, magics and shell escapes dropped."""
    cells = json.loads(path.read_text())["cells"]
    source = "\n".join(
        "".join(cell["source"]) for cell in cells if cell["cell_type"] == "code"
    )
    return "\n".join(
        line for line in source.splitlines() if not line.lstrip().startswith(("%", "!"))
    )


@pytest.mark.structural
def test_no_qa_script_builds_its_own_instance() -> None:
    # A figure whose instance is typed into the module is a figure whose
    # inputs the stamp cannot see, and a figure the catalogue cannot claim.
    offenders = {
        path.name: sorted(_constructions(path.read_text()))
        for path in sorted(QA.glob("*.py"))
        if _constructions(path.read_text())
    }

    assert offenders == {}, (
        "a QA script builds a problem instance; take it from "
        "snakes_and_ladders.sim.fixtures and name the file in the manifest"
    )


@pytest.mark.structural
def test_no_notebook_builds_its_own_instance() -> None:
    offenders = {
        path.name: sorted(_constructions(_notebook_code(path)))
        for path in sorted(NOTEBOOKS.glob("*.ipynb"))
        if _constructions(_notebook_code(path))
    }

    assert offenders == {}, (
        "a notebook builds a problem instance; read the fixture instead, so "
        "the notebook and the suite run on the same one"
    )


@pytest.mark.edge_case
def test_the_guard_catches_a_constructed_instance() -> None:
    # Guards the guard, per the repository's pattern: an import or an
    # annotation is not a construction, and a call is.
    assert (
        _constructions("from snakes_and_ladders.sim.mixture import MixtureParams")
        == set()
    )
    assert _constructions("def f(x: MixtureParams) -> None: ...") == set()
    assert _constructions("p = MixtureParams(w, c, 10, 1, 1e-12)") == {"MixtureParams"}
    assert _constructions("code = ldpc.gallager_code(12, 3, 6, rng)") == {
        "gallager_code"
    }
