"""Every test module lands in exactly one attribution status (issue #622).

`infra/attribute_problems.py` names the work step 1 of the plan exists to
name: which problem each module without a fixture-registry call exercises, and
whether the registry declares an instance at the shape the module builds. The
plan's stated validation is that the statuses *partition* the tree --- no
module in two, none in none --- because a module in neither list is a
conversion nobody scheduled and a module in both is two conversions of one
file.

The second claim here is that `unattributed` means what it says. The plan
forbids guessing, so the guard asserts the shape of the evidence rather than
the answer: an attributed module resolves to exactly one declared problem, and
an unattributed one resolves to none or to more than one, never to a single
problem the script declined to report.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "infra"))

import attribute_problems  # noqa: E402


@pytest.fixture(scope="module")
def rows() -> tuple[attribute_problems.Attribution, ...]:
    """Every test module's attribution, read once for the module."""
    return attribute_problems.attributions(REPO_ROOT)


@pytest.mark.structural
@pytest.mark.critical
def test_every_module_lands_in_exactly_one_status(
    rows: tuple[attribute_problems.Attribution, ...],
) -> None:
    """One status each, and the statuses account for every module."""
    modules = attribute_problems.modules(REPO_ROOT)
    reported = [row.module for row in rows]
    assert sorted(reported) == sorted(
        str(path.relative_to(REPO_ROOT)) for path in modules
    )
    assert len(set(reported)) == len(reported), "a module attributed twice"
    unknown = sorted({row.status for row in rows} - set(attribute_problems.STATUSES))
    assert unknown == [], f"statuses outside the declared partition: {unknown}"
    assert sum(attribute_problems.counts(rows).values()) == len(modules)


@pytest.mark.structural
def test_an_attribution_names_one_declared_problem(
    rows: tuple[attribute_problems.Attribution, ...],
) -> None:
    """`attributed` is one declared problem; `unattributed` is never one."""
    declared = set(attribute_problems.catalogue().problems)
    for row in rows:
        if row.status == "attributed":
            assert len(row.problems) == 1, f"{row.module}: {row.problems}"
            assert set(row.problems) <= declared, f"{row.module}: {row.problems}"
        elif row.status == "unattributed":
            assert len(row.problems) != 1, (
                f"{row.module} resolved to one problem and was not attributed"
            )
        elif row.status == "infra":
            assert row.problems == (), f"{row.module}: infra naming {row.problems}"


@pytest.mark.structural
def test_infra_modules_name_no_catalogued_symbol(
    rows: tuple[attribute_problems.Attribution, ...],
) -> None:
    """`infra` is a claim about the imports, and it is checked against them."""
    index = attribute_problems.catalogue()
    for row in rows:
        if row.status != "infra":
            continue
        reading = attribute_problems.read(REPO_ROOT / row.module, index.problems)
        score, _ = attribute_problems.rank(reading, index.rows)
        assert score == 0, f"{row.module} imports catalogued symbols but reads as infra"


@pytest.mark.structural
def test_the_catalogue_names_only_declared_fixtures() -> None:
    """The join the readings rest on: every row's fixture is in the registry."""
    index = attribute_problems.catalogue()
    declared = set(index.problems)
    for row in index.rows:
        assert set(row.declared) <= declared, f"{row.title}: {row.declared}"
    assert any(row.declared for row in index.rows), "no row declares a fixture"
