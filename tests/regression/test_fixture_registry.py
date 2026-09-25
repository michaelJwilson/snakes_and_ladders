"""The fixture registry and the problem catalogue name each other (issue #382).

`PROBLEMS.md` names each supported instance's file; a row naming no loadable
fixture, or a fixture no row names, fails. ``sim/CLAUDE.md``'s rule (an
instance is a fixture, never a literal) is enforced on the QA scripts and
notebooks; oracle names are held to the textbook's tables; the fixtures that
restate a canonical constructor are pinned against it. Issue #401's baseline
records: read back as written, refused under other library versions, and
caught when the tree no longer produces them (issue #460), with the selection
of records to recompute asserted. Proofs run on a copy of the fixture directory.
"""

from __future__ import annotations

import ast
import json
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

import baselines as baseline_script
import catalogue
import numpy as np
import problems_tables
import pytest
from sal.emissions import CategoricalEmission
from sal.fixtures import Scale
from sal.sim.canonical import frustrated_triangular_lattice
from sal.sim.fixtures import (
    FIXTURES_DIR,
    ORACLES,
    PARAMS,
    Baseline,
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
from sal.sim.spatio_sequential import canonical_spatio_sequential

from tests._paths import REPO_ROOT

#: The applicability table column each stated oracle fills. ``none`` fills
#: none, which is the point of stating it: the size is past every oracle.
ORACLE_COLUMNS = {
    "enumeration": "enumeration or brute force",
    "transfer-matrix": "independent algorithm",
    "closed-form": "closed form or published value",
}


@pytest.mark.smoke
def test_every_catalogue_row_names_a_ci_fixture_that_loads() -> None:
    # The claim the column makes: this problem has an instance, at the size
    # the per-pull-request suite runs.
    rows = catalogue.keys()
    assert len(rows) > 8, "the catalogue lost its table"

    for title, keys in rows.items():
        assert keys, f"{title} names no fixture"
        for key in keys:
            assert "ci" in tiers(key), f"{title} names no ci fixture: {key}"
            assert fixture(key, "ci").path == (FIXTURES_DIR / key / "ci.yaml"), (
                f"{title}'s {key} does not resolve to its own directory"
            )


@pytest.mark.smoke
def test_every_fixture_is_named_by_the_catalogue() -> None:
    # The other direction: an instance the catalogue does not claim is one
    # no row is answerable for.
    named = {key for keys in catalogue.keys().values() for key in keys}

    assert set(problems()) == named


@pytest.mark.smoke
def test_every_fixture_loads_and_states_an_oracle_the_tables_know() -> None:
    loaded = [
        fixture(problem, tier) for problem in problems() for tier in tiers(problem)
    ]
    assert len(loaded) == len(list(FIXTURES_DIR.rglob("*.yaml")))

    for entry in loaded:
        assert entry.oracle in ORACLES
        assert entry.model in PARAMS
        assert entry.params is not None
        column = ORACLE_COLUMNS.get(entry.oracle)
        assert column is None or column in problems_tables.ORACLE_COLUMNS


@pytest.mark.smoke
def test_every_problem_declares_the_ci_tier() -> None:
    # `fixtures("ci")` is what a test parametrizing over the class of
    # problems iterates; a problem missing from it is silently untested
    # there rather than failing.
    assert {entry.problem for entry in fixtures(Scale.CI)} == set(problems())


@pytest.mark.smoke
def test_the_count_release_fixture_is_the_stress_model_on_a_smaller_lattice() -> None:
    # `spatio_sequential_counts/release` states that it carries the stress
    # file's classes, states and emission ladders on a 10x10 lattice over
    # 1,000 positions (issue #891). A ladder edited in one file and not the
    # other would leave the notebook reading a model no other file declares.
    release = fixture("spatio_sequential_counts", Scale.RELEASE).params.model
    stress = fixture("spatio_sequential_counts", Scale.STRESS).params.model

    assert release.graph.n_nodes == 100
    assert (release.n_classes, release.n_states, release.n_positions) == (10, 10, 1000)
    assert (release.n_classes, release.n_states) == (stress.n_classes, stress.n_states)
    assert (release.beta, release.self_transition) == (
        stress.beta,
        stress.self_transition,
    )
    np.testing.assert_array_equal(release.initial, stress.initial)
    for ours, theirs in zip(release.emissions, stress.emissions, strict=True):
        for name, value in ours.named_parameters().items():
            np.testing.assert_array_equal(value, theirs.named_parameters()[name])


@pytest.mark.oracle
def test_the_coupled_fixture_is_the_canonical_instance() -> None:
    # The file restates `canonical_spatio_sequential`, whose enumerable size is
    # why the instance exists. Drift would leave two instances under one name.
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
    # The 3x3 periodic triangular antiferromagnet: the counting argument fixes
    # one agreeing edge in three, which is what makes it worth declaring.
    params = fixture("frustrated_lattice", Scale.CI).params

    assert params.lattice() == frustrated_triangular_lattice(
        params.shape, params.boundary, params.coupling
    )


# --- the baseline records (issue #401) ---------------------------------------

#: The cheapest record to recompute, so the determinism check pays 2 s rather
#: than the 28 s the whole set costs.
CHEAPEST = "potts_chain/ci"


@pytest.fixture(scope="module")
def recomputed_cheapest() -> tuple[baseline_script.BaselineSpec, Baseline]:
    """The CHEAPEST spec and one recomputation of it, shared by the module."""
    (spec,) = baseline_script.selected([CHEAPEST])
    return spec, baseline_script.compute(spec)


@pytest.mark.smoke
def test_every_committed_baseline_reads_back_against_the_current_tree() -> None:
    # `infra/baselines.py --write` round-trips through `baseline()` under the
    # installed libraries; a code change is caught by `infra/baselines.py`.
    recorded = baselines()
    assert recorded, "no baseline record is committed"

    for problem, tier in recorded:
        record = baseline(problem, tier)
        assert (record.problem, record.tier) == (problem, tier)
        assert record.path == baseline_path(problem, tier)
        assert record.measurements, f"{record.path} records no measurement"
        for name, measurement in record.measurements.items():
            assert measurement.algorithm, f"{record.path}: {name} names no algorithm"
        assert read_baseline(record.path).libraries == record.libraries


@pytest.mark.smoke
def test_a_mutated_fixture_makes_its_baseline_fail_recomputation(
    tmp_path: Path,
    recomputed_cheapest: tuple[baseline_script.BaselineSpec, Baseline],
) -> None:
    # A record whose instance moved (in a copy) fails recomputation, naming
    # both values (issue #460).
    root = tmp_path / "tree"
    (root / "tests" / "regression").mkdir(parents=True)
    shutil.copytree(FIXTURES_DIR, root / "tests" / "regression" / "fixtures")

    spec, unmutated = recomputed_cheapest
    committed = read_baseline(baseline_path(spec.problem, spec.tier))
    assert baseline_script.differences(unmutated, committed) == []

    fixture_file = (
        root / "tests" / "regression" / "fixtures" / "potts_chain" / "ci.yaml"
    )
    fixture_file.write_text(
        fixture_file.read_text().replace("coupling: 0.75", "coupling: 1.25")
    )
    found = baseline_script.differences(baseline_script.compute(spec, root), committed)
    assert found, "a changed coupling left every recomputed number where it was"
    assert any("enumerated_optimum" in line for line in found), found


@pytest.mark.smoke
def test_an_edited_budget_is_a_disagreement_the_recomputation_reports(
    tmp_path: Path,
) -> None:
    # The budget says what a value means, so `differences` reports it beside
    # a moved value: an edited restart count cannot pass.
    copied = tmp_path / "release.baseline.json"
    original = baseline_path("tree_search", Scale.RELEASE)
    copied.write_text(original.read_text().replace('"starts": 50', '"starts": 20'))

    found = baseline_script.differences(read_baseline(original), read_baseline(copied))
    assert any("budget" in line for line in found), found


@pytest.mark.smoke
def test_a_record_computed_against_another_library_is_refused(tmp_path: Path) -> None:
    # The one input to a number that a recomputation elsewhere cannot check,
    # because it is a fact about this machine and not about the tree. It is
    # free on the read, so the read is where it is asserted (issue #460).
    copied = tmp_path / "fixtures"
    shutil.copytree(FIXTURES_DIR, copied)
    assert baseline("tree_search", Scale.RELEASE, copied).value("greedy_rate") == 0.48

    record = copied / "tree_search" / "release.baseline.json"
    held = json.loads(record.read_text())
    held["libraries"] = ["numpy==0.0.1", *held["libraries"][1:]]
    record.write_text(json.dumps(held, indent=2, sort_keys=True) + "\n")

    with pytest.raises(StaleBaselineError, match="numpy==0.0.1"):
        baseline("tree_search", Scale.RELEASE, copied)


@pytest.mark.smoke
def test_a_change_is_recomputed_against_the_records_it_reaches() -> None:
    # A record is a function of its fixture and its import closure, so a change
    # to neither cannot move it; a missed record is a check that did not run.
    every = {f"{spec.problem}/{spec.tier}" for spec in baseline_script.SPECS}

    def reached(*paths: str) -> set[str]:
        return {
            f"{spec.problem}/{spec.tier}" for spec in baseline_script.reached_by(paths)
        }

    assert reached("README.md", "docs/tex/paper.tex") == set()
    assert reached("python/sal/learn/ranking.py") == {"tree_search/ci"}
    assert reached("tests/regression/fixtures/potts_chain/ci.yaml") == {
        "potts_chain/ci"
    }
    assert reached("tests/regression/fixtures/potts_chain/ci.baseline.json") == {
        "potts_chain/ci"
    }
    # The environment and this script itself reach every record: neither is in
    # any closure, and both decide every number.
    assert reached("uv.lock") == every
    assert reached("infra/baselines.py") == every


@pytest.mark.smoke
def test_the_same_baseline_computed_twice_is_the_same_record(
    recomputed_cheapest: tuple[baseline_script.BaselineSpec, Baseline],
) -> None:
    # A reference algorithm whose answer moved between two runs of the same
    # tree would make every committed record a snapshot rather than a fact.
    # Checked on the cheapest record; the rest run at the release gate.
    spec, first = recomputed_cheapest
    second = baseline_script.compute(spec)

    assert baseline_script.differences(first, second) == []
    assert baseline_script.differences(first, read_baseline(first.path)) == []


#: The GitHub runner's `tree_search/ci` `maximized_log_likelihood` values, the
#: 45 fits of run 34549737349, job 103109927613, on a comment-only PR: the
#: second host's arithmetic the tolerance is derived from.
RUNNER_FITS = (
    -963.7650335864475,
    -963.9244927034631,
    -963.9244927031195,
    -951.5175416461561,
    -963.7650335864462,
    -961.263999074587,
    -963.9244927031189,
    -963.9244927037184,
    -951.5175416455179,
    -961.2639990750451,
    -953.4215895855352,
    -963.9244927033953,
    -963.9244927032872,
    -943.4779537057238,
    -953.4215895854888,
    -1009.5623583470353,
    -1009.8274432207423,
    -1009.8172093168685,
    -996.7995751190404,
    -1009.562358347034,
    -1009.5384934704672,
    -1009.8274432208141,
    -1009.8172093168362,
    -996.7995751177407,
    -1009.5384934704649,
    -994.0621025395212,
    -1009.8131656429082,
    -1009.8131656429207,
    -987.2694223590945,
    -994.0621025393559,
    -1044.2960556364872,
    -1044.2960556367857,
    -1041.1823818576054,
    -1031.2909204183081,
    -1044.296055636489,
    -1043.8788414559262,
    -1044.2960556364903,
    -1041.1823818574146,
    -1031.290920418392,
    -1043.878841454717,
    -1021.277786372497,
    -1043.2139003592315,
    -1043.2139003601483,
    -1016.6076660892631,
    -1019.9514425628881,
)

#: A relative move the comparison has to catch, measured rather than picked:
#: loosening the fit's own convergence test from 1e-8 to 1e-7 relative moves
#: 14 of these 45 values, by 1.434e-11 relative at the widest. A bound that
#: admitted that would hide a regression instead of admitting noise.
REAL_CHANGE = 1e-11


def _moved(record: Baseline, name: str, **fields: Any) -> Baseline:
    """``record`` with one measurement's fields replaced."""
    held = dict(record.measurements)
    held[name] = replace(record.measurement(name), **fields)
    return replace(record, measurements=held)


@pytest.mark.smoke
def test_a_fit_is_compared_within_its_declared_tolerance_and_not_bitwise() -> None:
    # Issue #527: the runner moved 26 of these 45 fitted values, by 4.365e-15
    # relative at the widest; the bound admits that and catches a move three
    # orders above it.
    committed = read_baseline(baseline_path("tree_search", Scale.CI))
    recorded = committed.values("maximized_log_likelihood")
    deviation = max(
        abs(fresh - stored) / abs(stored)
        for fresh, stored in zip(RUNNER_FITS, recorded, strict=True)
    )
    assert sum(a != b for a, b in zip(RUNNER_FITS, recorded, strict=True)) == 26
    assert deviation == pytest.approx(4.365e-15, rel=1e-3)
    assert deviation < baseline_script.FIT_RTOL < REAL_CHANGE

    runner = _moved(committed, "maximized_log_likelihood", value=RUNNER_FITS)
    assert baseline_script.differences(runner, committed) == []

    moved = _moved(
        committed,
        "maximized_log_likelihood",
        value=(recorded[0] * (1.0 + REAL_CHANGE), *recorded[1:]),
    )
    found = baseline_script.differences(moved, committed)
    assert len(found) == 1, found
    assert "maximized_log_likelihood: 1 of 45 values moved" in found[0], found[0]
    assert "index 0" in found[0], found[0]


@pytest.mark.smoke
def test_a_value_that_declares_no_tolerance_is_still_compared_exactly() -> None:
    # Counted or enumerated values reproduce bit for bit; one ulp fails.
    committed = read_baseline(baseline_path("potts_chain", Scale.CI))
    assert committed.measurement("enumerated_optimum").rtol is None

    one_ulp = float(np.nextafter(committed.value("enumerated_optimum"), 0.0))
    found = baseline_script.differences(
        _moved(committed, "enumerated_optimum", value=one_ulp), committed
    )
    assert any("enumerated_optimum" in line for line in found), found


@pytest.mark.smoke
def test_a_record_cannot_loosen_the_tolerance_it_is_checked_at() -> None:
    # The stricter of the two declared tolerances is taken, so a record edited
    # to pass (#527) reports the moved value and the edited tolerance.
    original = read_baseline(baseline_path("tree_search", Scale.CI))
    recorded = original.values("maximized_log_likelihood")
    loosened = _moved(original, "maximized_log_likelihood", rtol=1e-3)
    moved = _moved(
        loosened,
        "maximized_log_likelihood",
        value=(recorded[0] * (1.0 + REAL_CHANGE), *recorded[1:]),
    )

    found = baseline_script.differences(moved, original)
    assert any("maximized_log_likelihood: 1 of 45 values moved" in x for x in found)
    assert any("rtol 0.001" in line for line in found), found


@pytest.mark.smoke
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
        "BicycleParams",
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

QA = REPO_ROOT / "python" / "sal" / "qa"
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


@pytest.mark.smoke
def test_no_qa_script_builds_its_own_instance() -> None:
    # A figure whose instance is typed into the module has inputs the stamp
    # cannot see, and the catalogue cannot claim it.
    offenders = {
        path.name: sorted(_constructions(path.read_text()))
        for path in sorted(QA.glob("*.py"))
        if _constructions(path.read_text())
    }

    assert offenders == {}, (
        "a QA script builds a problem instance; take it from "
        "sal.sim.fixtures and name the file in the manifest"
    )


@pytest.mark.smoke
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


@pytest.mark.smoke
def test_the_guard_catches_a_constructed_instance() -> None:
    # Guards the guard, per the repository's pattern: an import or an
    # annotation is not a construction, and a call is.
    assert _constructions("from sal.sim.mixture import MixtureParams") == set()
    assert _constructions("def f(x: MixtureParams) -> None: ...") == set()
    assert _constructions("p = MixtureParams(w, c, 10, 1, 1e-12)") == {"MixtureParams"}
    assert _constructions("code = ldpc.gallager_code(12, 3, 6, rng)") == {
        "gallager_code"
    }


@pytest.mark.smoke
def test_a_fixture_file_that_is_not_a_mapping_is_refused_as_one(
    tmp_path: Path,
) -> None:
    # One parse, `load_declared`'s: a list-valued file is refused naming the
    # file and what it parsed to, not an `AttributeError` (issue #864).
    directory = tmp_path / "fixtures"
    (directory / "listed").mkdir(parents=True)
    (directory / "listed" / "ci.yaml").write_text(
        "- model: jukes-cantor\n- oracle: enumeration\n"
    )

    with pytest.raises(ValueError, match="expected a mapping of fields, got list"):
        fixture("listed", Scale.CI, directory)
