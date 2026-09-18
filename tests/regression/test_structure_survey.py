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

import ast
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


@pytest.mark.infra
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


@pytest.mark.infra
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


@pytest.mark.infra
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


@pytest.mark.infra
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


#: A class with a genuine per-call derivation over a store that is not
#: compressed: `rows` is an accessor, asked twice it pays twice, and storing
#: the result would remove the second payment. This is what the finding is for.
_PER_CALL_ACCESSOR = """
class Example:
    def rows(self):
        offsets = numpy.cumsum(self.degrees)
        return offsets
"""

#: The same derivation reached through a constructor. A `classmethod` building
#: an instance runs once per instance, so there is no second payment to remove
#: and no rewrite that clears the finding short of deleting the constructor.
_CONSTRUCTOR = """
class Example:
    @classmethod
    def from_arcs(cls, degrees):
        offsets = numpy.cumsum(degrees)
        return cls(offsets)
"""


def _findings(source: str) -> tuple[str, ...]:
    """The survey's notes for one class, read from source rather than the tree."""
    node = ast.parse(source).body[0]
    assert isinstance(node, ast.ClassDef)
    return appraise_structures._notes("example.Example", node, ("degrees",))


@pytest.mark.infra
def test_a_per_call_derivation_is_still_reported(
    grouped: list[appraise_structures.Cluster],
) -> None:
    """The positive control for the narrowing #642 made.

    That change stops the finding being raised about a constructor, which is
    the whole of what it removes. A narrowing is only as good as the proof it
    did not go further, so the two halves are asserted against each other here:
    the accessor is reported and the constructor is not, from sources this file
    owns rather than from whatever the tree happens to contain.

    Without this, the day the detector stops firing for a real accessor it goes
    quiet instead of failing, and a survey nobody can trust to fire is the
    thing `infra/appraise_structures.py` exists to not be.
    """
    accessor = _findings(_PER_CALL_ACCESSOR)
    constructor = _findings(_CONSTRUCTOR)

    assert any("rows derives a compressed layout per call" in f for f in accessor)
    assert not any("per call" in f for f in constructor)

    # And the tree itself: `FlowNetwork.from_arcs` was the one constructor
    # carrying it, so no per-call finding stands anywhere now (#642).
    live = " ".join(f for cluster in grouped for f in cluster.findings)
    assert "derives a compressed layout per call" not in live


@pytest.mark.infra
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


@pytest.mark.infra
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


@pytest.mark.infra
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


@pytest.mark.infra
def test_the_implicit_offsets_store_is_in_the_incidence_cluster(
    grouped: list[appraise_structures.Cluster],
) -> None:
    # The ticket's headline (#677). `ragged.Ragged` stores `values` and
    # `lengths` and derives the offsets, which is the third way this package
    # writes one relation -- and the classifier knew the other two only, so the
    # largest addition since the survey was written arrived invisible. The
    # cluster is where it belongs, beside `SparseIncidence` and `_EdgeLayout`.
    incidence = next(c for c in grouped if c.key == "role:incidence")

    assert "ragged.Ragged" in incidence.members
    assert incidence.layouts["ragged.Ragged"] == "csr"
    assert incidence.layouts["sim.hmm.SimulatedHmmDataset"] == "csr"


@pytest.mark.infra
def test_a_declaration_of_lengths_is_not_a_layout(
    found: list[appraise_structures.Structure],
) -> None:
    # The corroboration rule, and why the payload partner is required rather
    # than nice to have. `HmmParams.lengths` declares the shape of a batch and
    # addresses nothing; a params filed as a compressed layout would put
    # sixteen parameter bundles in the incidence cluster and bury it.
    params = next(s for s in found if s.qualified == "sim.hmm.HmmParams")

    assert "lengths" in params.fields
    assert params.layout == "none"


@pytest.mark.infra
@pytest.mark.parametrize(
    "qualified",
    [
        "search.potts_mcmc.ClusterCounter",
        "search.ground_state.Rung",
        "sim.potts.SpatioOnlyParams",
    ],
)
def test_sizes_is_not_a_segmentation(
    found: list[appraise_structures.Structure], qualified: str
) -> None:
    # `sizes` is a histogram of cluster sizes in one of these and a set of
    # problem sizes in the other two. Admitting the name as a length field
    # misfiles all three, which is the `restarts`-for-`starts` failure the
    # survey exists to avoid, one spelling later.
    structure = next(s for s in found if s.qualified == qualified)

    assert structure.layout == "none"


@pytest.mark.infra
def test_reading_the_offsets_is_not_deriving_them(
    found: list[appraise_structures.Structure],
) -> None:
    # `Ragged.segments` reads `self.offsets` once and walks the views. Counting
    # it as a second derivation reports two costs where the source pays one,
    # and the fix for the pair would then look like a fix for one of them.
    ragged = next(s for s in found if s.qualified == "ragged.Ragged")

    assert not any("segments derives" in note for note in ragged.notes)
    assert any("39.3 us" in note for note in ragged.notes)


@pytest.mark.infra
def test_the_offsets_finding_carries_its_measurement(
    grouped: list[appraise_structures.Cluster],
) -> None:
    # Measured rather than fixed, and the numbers are why: 39.3 us against
    # 335.11 ms is 0.012% of the iteration that encloses it, and the NumPy
    # cumsum that would replace the scan saves 6 us of that. A ratio with no
    # effect size is what root `CLAUDE.md` leaves alone, so the survey reports
    # the layout and stops calling it a cost.
    findings = " ".join(f for cluster in grouped for f in cluster.findings)

    assert "ragged.Ragged: offsets rebuilt per call kept" in findings
    assert "0.012%" in findings


@pytest.mark.infra
def test_a_finding_on_an_unclustered_class_is_reported(
    found: list[appraise_structures.Structure],
    grouped: list[appraise_structures.Cluster],
) -> None:
    # The second blindness (#677). Findings printed only inside a cluster, so a
    # cost on a class sharing its shape with nobody was derived and dropped:
    # nine of them, on `likelihood.schedule.Layout`, `search.gibbs._Indexed`,
    # `learn.surrogate.Examples` and the rest. What a structure costs does not
    # depend on how many others look like it.
    loose = appraise_structures.unclustered_findings(found, grouped)
    rendered = appraise_structures.render(found, grouped)

    assert "likelihood.schedule.Layout" in " ".join(loose)
    assert "findings outside every cluster" in rendered
    assert all(finding in rendered for finding in loose)


@pytest.mark.infra
def test_a_planted_segmented_store_is_reported_without_a_partner_being_guessed(
    tmp_path: Path,
) -> None:
    # The planted control for the new spelling, the form the CSR one already
    # has: a lengths-plus-payload store the walk has never read is reported,
    # and a lengths-only one beside it is not.
    package = tmp_path / "segmented"
    package.mkdir()
    (package / "addressed.py").write_text(
        "from dataclasses import dataclass\n"
        "import numpy as np\n\n\n"
        "@dataclass(frozen=True)\n"
        "class Addressed:\n"
        "    values: np.ndarray\n"
        "    lengths: tuple[int, ...]\n"
    )
    (package / "declared.py").write_text(
        "from dataclasses import dataclass\n\n\n"
        "@dataclass(frozen=True)\n"
        "class Declared:\n"
        "    lengths: tuple[int, ...]\n"
        "    tolerance: float\n"
    )

    planted = {s.name: s for s in appraise_structures.structures(package)}

    assert planted["Addressed"].layout == "csr"
    assert planted["Declared"].layout == "none"


@pytest.mark.infra
def test_every_class_added_since_the_survey_was_written_is_in_it() -> None:
    # #677's own acceptance test, and it reads `git log` rather than memory:
    # these eight classes are every `class` added under the package since #586
    # merged, the commit that deleted the hand-written inventories. Seven carry
    # state and are in the survey; `CovariateNotSupportedError` is an exception
    # with no fields, excluded by the definition of state-carrying rather than
    # missed by the walk, which is why it is named here rather than left out.
    added = {
        "BetaBinomialEmission",
        "HmmEnvironment",
        "NegativeBinomialEmission",
        "PottsEnvironment",
        "Ragged",
        "_SolvedBetaBinomial",
        "_SolvedDispersion",
    }
    stateless = {"CovariateNotSupportedError"}

    names = {structure.name for structure in appraise_structures.structures()}

    assert added <= names
    assert not (stateless & names)
