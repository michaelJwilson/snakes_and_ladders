"""What `infra/appraise_structures.py` claims, held to the tree (issue #586).

The survey exists because the two it joins --- `infra/duplication_survey.py`
and the deleted `infra/seams_survey.py` --- carried hand-written lists, so they
re-counted what was known and discovered nothing. A derived survey earns that
only if it is held to what the hand lists already found *and* shown to report
something they do not, which is what this file asserts.

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

#: The four data contracts `infra/seams_survey.py` declared by hand before
#: issue #586 deleted it. Written out here rather than imported, because the
#: hand list is exactly what that ticket removed and this assertion is the
#: record that the derived walk still covers everything it named.
HAND_LISTED_CONTRACTS = (
    "sim.factor_graph.FactorGraph",
    "sim.hmm.HmmParams",
    "opt.potts.PottsParams",
    "sim.spatio_sequential.SpatioSequentialParams",
)


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
    # comparison is against the contracts, which are the state-carrying half.
    named = set(HAND_LISTED_CONTRACTS)

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
def test_the_one_structure_the_three_now_hold_is_in_the_cluster(
    grouped: list[appraise_structures.Cluster],
) -> None:
    # The three above still store what their problems need -- a Potts graph
    # is an edge list, a factor graph is tables -- and each now derives its
    # incidence through one structure rather than three. The survey should
    # place that structure in the cluster it was lifted from, and classify it
    # by the same rule, or it has stopped measuring the tree.
    incidence = next(c for c in grouped if c.key == "role:incidence")

    assert "incidence.SparseIncidence" in incidence.members
    assert incidence.layouts["incidence.SparseIncidence"] == "csr"


@pytest.mark.structural
def test_the_per_call_derivation_finding_is_gone_for_the_graph_that_fixed_it(
    grouped: list[appraise_structures.Cluster],
) -> None:
    # The survey reported `PottsGraph.compressed_adjacency` deriving the
    # layout per call; it is derived once now, and the finding has to go with
    # it -- a survey whose findings outlive their fixes is a survey nobody
    # reads. Stated as the absence, because that is what the fix changed.
    #
    # The absence is asserted of `PottsGraph` and not of the sentence. The
    # sentence is the survey's standing vocabulary for the finding, so any
    # class that later derives a layout per call raises it again --- and one
    # does: `search.maxflow.FlowNetwork`, open as #642. A substring match
    # over every cluster reads that new finding as this fix regressing, which
    # is the opposite of what a survey is for.
    findings = " ".join(f for cluster in grouped for f in cluster.findings)

    assert "PottsGraph" not in findings
    assert "sim.graph.PottsGraph: compressed_adjacency" not in findings


@pytest.mark.structural
def test_the_open_per_call_derivation_finding_names_its_ticket(
    grouped: list[appraise_structures.Cluster],
) -> None:
    # The survey's live finding, and the reason the test above is narrow: the
    # same sentence is raised about `FlowNetwork`, where the layout is still
    # derived per call. It stands until #642 lands.
    findings = " ".join(f for cluster in grouped for f in cluster.findings)

    assert "search.maxflow.FlowNetwork: from_arcs derives a compressed layout" in (
        findings
    )


@pytest.mark.structural
def test_the_list_of_lists_finding_carries_its_measurement(
    grouped: list[appraise_structures.Cluster],
) -> None:
    # The other finding was measured rather than fixed: converting Dinic's
    # adjacency would have made it slower, so the survey reports the layout
    # and the numbers instead of asking again. A finding answered with a
    # benchmark still shows, so the reader can re-run it.
    findings = " ".join(f for cluster in grouped for f in cluster.findings)

    assert "search.maxflow.FlowNetwork" in findings
    assert "list of lists kept" in findings
    assert "25.63 ms as a NumPy slice" in findings


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
