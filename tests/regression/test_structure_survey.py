"""What `infra/appraise_structures.py` claims, held to the tree (issue #586).

The survey exists because the two it joins --- `infra/duplication_survey.py`
and `infra/seams_survey.py` --- carry hand-written lists, so they re-count
what is known and discover nothing. A derived survey earns that only if it is
held to what the hand lists already found *and* shown to report something they
do not, which is what this file asserts.

The false-positive cases are the important half. A survey of shapes that
matches on substrings measures names: the first draft of the layout classifier
filed `restarts` as a compressed layout because it contains "starts", and
`evaluated_columns` and `columns` as coordinate lists because they contain
"col". Those three are pinned below, because a survey that flags them is worse
than no survey --- it spends a reviewer's attention to say nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "infra"))

import appraise_structures  # noqa: E402
import seams_survey  # noqa: E402


@pytest.fixture(scope="module")
def found() -> list[appraise_structures.Structure]:
    return appraise_structures.structures()


@pytest.fixture(scope="module")
def grouped(
    found: list[appraise_structures.Structure],
) -> list[appraise_structures.Cluster]:
    return appraise_structures.clusters(found)


@pytest.mark.structural
def test_the_walk_recovers_every_contract_the_hand_list_names(
    found: list[appraise_structures.Structure],
) -> None:
    # The gate on replacing a hand list: the derivation finds at least what the
    # list did. Protocols are not here and are not meant to be -- they name
    # behaviour, and this walk is over types that carry state -- so the
    # comparison is against `CONTRACTS`, which is the state-carrying half.
    named = {f"{seam.module}.{seam.name}" for seam in seams_survey.CONTRACTS}
    assert named, "seams_survey declares no contracts; this test lost its subject"

    derived = {structure.qualified for structure in found}

    assert named <= derived, f"the walk missed {sorted(named - derived)}"


@pytest.mark.structural
def test_the_incidence_cluster_holds_the_three_representations(
    grouped: list[appraise_structures.Cluster],
) -> None:
    # The finding the ticket rests on: one relation, three memory layouts.
    incidence = next(c for c in grouped if c.key == "role:incidence")
    members = set(incidence.members)

    assert "sim.ldpc.ParityCheck" in members
    assert "sim.graph.PottsGraph" in members
    assert "sim.factor_graph.FactorGraph" in members
    assert incidence.layouts["sim.ldpc.ParityCheck"] == "csr"
    assert incidence.layouts["sim.graph.PottsGraph"] == "coo"
    assert incidence.layouts["sim.factor_graph.FactorGraph"] == "object-graph"


@pytest.mark.structural
def test_the_two_layout_findings_are_reported(
    grouped: list[appraise_structures.Cluster],
) -> None:
    # Both are real, both were found by the walk rather than by a reader, and
    # both name a rule in root `CLAUDE.md`'s Runtime Optimization
    # Opportunities. They are pinned so a fix has to delete the finding.
    findings = " ".join(f for cluster in grouped for f in cluster.findings)

    assert "search.maxflow.FlowNetwork" in findings
    assert "list of lists" in findings
    assert "sim.graph.PottsGraph" in findings
    assert "derives a compressed layout per call" in findings


@pytest.mark.structural
@pytest.mark.parametrize(
    "qualified",
    [
        "opt.testfunctions.TestFunctionParams",
        "likelihood.blocks.Interval",
        "likelihood.patterns.SitePatterns",
    ],
)
def test_a_name_that_merely_reads_like_a_layout_is_not_one(
    found: list[appraise_structures.Structure], qualified: str
) -> None:
    # `restarts` is not `offsets`; `evaluated_columns` and `columns` are not a
    # coordinate list. Each of these was misfiled by a substring match, and
    # each is a structure whose fields say it carries no incidence at all.
    structure = next(s for s in found if s.qualified == qualified)

    assert structure.layout == "none", (
        f"{qualified} is filed as {structure.layout} on its field names alone"
    )


@pytest.mark.structural
def test_the_survey_reports_a_near_duplicate_it_has_never_seen(
    tmp_path: Path,
) -> None:
    # The self-test the duplication guards already carry, in the form this
    # survey needs: a shape planted in a tree the walk has never read must be
    # reported. Without it the survey is only asserted to agree with what it
    # was written against.
    package = tmp_path / "planted"
    package.mkdir()
    for index in range(3):
        (package / f"module_{index}.py").write_text(
            "from dataclasses import dataclass\n"
            "import numpy as np\n\n\n"
            "@dataclass(frozen=True)\n"
            f"class Shape{index}:\n"
            "    offsets: np.ndarray\n"
            "    indices: np.ndarray\n"
            "    weights: np.ndarray\n"
        )

    planted = appraise_structures.structures(package)
    clusters = appraise_structures.clusters(planted)

    assert {s.qualified for s in planted} == {f"module_{i}.Shape{i}" for i in range(3)}
    assert all(s.layout == "csr" for s in planted)
    assert any(c.key == "role:incidence" and len(c.members) == 3 for c in clusters)


@pytest.mark.edge_case
def test_a_shape_below_the_rule_is_not_a_cluster(tmp_path: Path) -> None:
    # Two classes sharing a shape are a pair, not a pattern. The rule is three,
    # the same count root `CLAUDE.md` puts on a seam, and a survey that
    # reported pairs would bury the clusters that matter.
    package = tmp_path / "pair"
    package.mkdir()
    for index in range(appraise_structures.CLUSTER_RULE - 1):
        (package / f"module_{index}.py").write_text(
            "from dataclasses import dataclass\n"
            "import numpy as np\n\n\n"
            "@dataclass(frozen=True)\n"
            f"class Shape{index}:\n"
            "    offsets: np.ndarray\n"
            "    indices: np.ndarray\n"
        )

    assert appraise_structures.clusters(appraise_structures.structures(package)) == []
