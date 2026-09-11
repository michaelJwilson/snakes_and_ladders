"""The fixture registry and the problem catalogue name each other (issue #382).

A supported problem's instance is a file, and `PROBLEMS.md` says which file.
Two ways that can rot: a row naming a fixture that no longer loads, and a
fixture no row names --- an instance nothing is claimed about. Both fail here.

The rule ``sim/CLAUDE.md`` states --- a supported instance is a fixture,
never a literal --- is enforced here for the two consumers that cannot import
the suite's helpers: the QA scripts and the notebooks. Neither may construct a
problem's declared truth or call a canonical constructor.

The registry's oracle vocabulary is held to the tables the textbook typesets,
so a fixture cannot state an oracle the applicability tables have no column
for; and the two fixtures that restate a canonical constructor are pinned
against it, since a file that has drifted from the instance it copies is a
second truth (``sim/CLAUDE.md``).

The baseline records of issue #401 are held to what makes reading a cached
number safe. The reader returns what the writer wrote; a record computed
against another ``numpy``, ``scipy`` or ``torch`` is refused rather than
served; and a record the tree no longer produces is caught by recomputing it,
which replaced the committed digest (issue #460). The selection deciding which
records a change is recomputed against is asserted here too, since a selection
that misses a record is a check that silently did not run.

Every proof runs against a *copy* of the fixture directory, since the
committed fixtures and records must not be edited to make one.
"""

from __future__ import annotations

import ast
import json
import re
import shutil
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from snakes_and_ladders.emissions import CategoricalEmission
from snakes_and_ladders.fixtures import Scale
from snakes_and_ladders.sim.canonical import frustrated_triangular_lattice
from snakes_and_ladders.sim.fixtures import (
    FIXTURES_DIR,
    LOADERS,
    ORACLES,
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
    # the per-pull-request suite runs.
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


@pytest.mark.structural
def test_every_committed_baseline_reads_back_against_the_current_tree() -> None:
    # The round trip the readers depend on: what `infra/baselines.py --write`
    # wrote is what `baseline()` returns, computed against the libraries
    # installed here. A record left behind by a library upgrade fails here
    # without recomputing a number; one left behind by a change to the code
    # it measures fails in `infra/baselines.py`, which recomputes.
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


@pytest.mark.edge_case
def test_a_mutated_fixture_makes_its_baseline_fail_recomputation(
    tmp_path: Path,
) -> None:
    # The property that makes a cached number safe: a record whose instance
    # moved under it does not survive being recomputed. Against a *copy* of
    # the fixture directory, so the committed instance is untouched. The
    # removed digest said the tree had moved; this says the number did, and
    # names both values (issue #460).
    root = tmp_path / "tree"
    (root / "tests" / "regression").mkdir(parents=True)
    shutil.copytree(FIXTURES_DIR, root / "tests" / "regression" / "fixtures")

    (spec,) = baseline_script.selected([CHEAPEST])
    committed = read_baseline(baseline_path(spec.problem, spec.tier))
    assert (
        baseline_script.differences(baseline_script.compute(spec, root), committed)
        == []
    )

    fixture_file = (
        root / "tests" / "regression" / "fixtures" / "potts_chain" / "ci.yaml"
    )
    fixture_file.write_text(
        fixture_file.read_text().replace("coupling: 0.75", "coupling: 1.25")
    )
    found = baseline_script.differences(baseline_script.compute(spec, root), committed)
    assert found, "a changed coupling left every recomputed number where it was"
    assert any("enumerated_optimum" in line for line in found), found


@pytest.mark.edge_case
def test_an_edited_budget_is_a_disagreement_the_recomputation_reports(
    tmp_path: Path,
) -> None:
    # The budget says what a value means, so a record whose restart count was
    # edited to match a test describes a measurement other than the one it
    # holds. `differences` reports the budget beside a moved value; reporting
    # the value alone would let the edit pass wherever the number was
    # reproduced.
    copied = tmp_path / "release.baseline.json"
    original = baseline_path("tree_search", Scale.RELEASE)
    copied.write_text(original.read_text().replace('"starts": 50', '"starts": 20'))

    found = baseline_script.differences(read_baseline(original), read_baseline(copied))
    assert any("budget" in line for line in found), found


@pytest.mark.edge_case
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


@pytest.mark.structural
def test_a_change_is_recomputed_against_the_records_it_reaches() -> None:
    # The selection that replaced the digest, and why it is safe to recompute
    # less than everything: a record's numbers are a function of its fixture
    # and of the import closure of the modules that computed them, so a change
    # to neither can move them. A missed record is a check that silently did
    # not run, the failure mode `infra/CLAUDE.md` names for `select_tests.py`.
    every = {f"{spec.problem}/{spec.tier}" for spec in baseline_script.SPECS}

    def reached(*paths: str) -> set[str]:
        return {
            f"{spec.problem}/{spec.tier}" for spec in baseline_script.reached_by(paths)
        }

    assert reached("README.md", "docs/tex/paper.tex") == set()
    assert reached("python/snakes_and_ladders/search/surrogate.py") == {
        "tree_search/ci"
    }
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


@pytest.mark.structural
def test_the_same_baseline_computed_twice_is_the_same_record() -> None:
    # A reference algorithm whose answer moved between two runs of the same
    # tree would make every committed record a snapshot rather than a fact.
    # Checked on the cheapest record; the rest run at the release gate.
    (spec,) = baseline_script.selected([CHEAPEST])
    first, second = baseline_script.compute(spec), baseline_script.compute(spec)

    assert baseline_script.differences(first, second) == []
    assert baseline_script.differences(first, read_baseline(first.path)) == []


#: What the GitHub runner computed for `tree_search/ci`'s
#: `maximized_log_likelihood` where the committed record is what this
#: repository's 4-core host computes: the 45 fits of run 34549737349, job
#: 103109927613, on a pull request that changed only docstrings, comments and
#: LaTeX. Kept as data because it is the observation the tolerance is derived
#: from --- a second host's arithmetic, which no single-host run reproduces.
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
    """``record`` with one measurement's fields replaced.

    Returns
    -------
    Baseline
    """
    held = dict(record.measurements)
    held[name] = replace(record.measurement(name), **fields)
    return replace(record, measurements=held)


@pytest.mark.structural
@pytest.mark.edge_case
def test_a_fit_is_compared_within_its_declared_tolerance_and_not_bitwise() -> None:
    # The comparison issue #527 is about, against the observation that raised
    # it. A recorded maximum-likelihood fit is an iterative optimiser over a
    # floating-point reduction, so a host whose BLAS orders that reduction
    # differently reproduces it to a tolerance and not bit for bit: the
    # runner moved 26 of these 45 values, by 4.365e-15 relative at the
    # widest. The bound has to admit that and still catch a move three orders
    # of magnitude above it, or it hides a regression rather than admitting
    # noise.
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


@pytest.mark.structural
@pytest.mark.edge_case
def test_a_value_that_declares_no_tolerance_is_still_compared_exactly() -> None:
    # The other half of the rule: an enumerated optimum, a ground-state
    # energy and a rate over seeded rollouts are counted or enumerated, not
    # fitted, so they reproduce bit for bit and a tolerance on them would
    # admit a change nothing else catches. One ulp is the smallest move there
    # is, and it fails.
    committed = read_baseline(baseline_path("potts_chain", Scale.CI))
    assert committed.measurement("enumerated_optimum").rtol is None

    one_ulp = float(np.nextafter(committed.value("enumerated_optimum"), 0.0))
    found = baseline_script.differences(
        _moved(committed, "enumerated_optimum", value=one_ulp), committed
    )
    assert any("enumerated_optimum" in line for line in found), found


@pytest.mark.edge_case
def test_a_record_cannot_loosen_the_tolerance_it_is_checked_at() -> None:
    # A tolerance is a declaration, and the record is not what gets to relax
    # the check it is caught by --- the failure #527 names is a record edited
    # to make a check pass. The comparison takes the stricter of the two
    # declarations, so the edit reports the moved value *and* the edited
    # tolerance.
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
    # A figure whose instance is typed into the module has inputs the stamp
    # cannot see, and the catalogue cannot claim it.
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
