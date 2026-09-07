"""The accuracy figure, and the distance it reports.

The claim is that the search recovers the generating topology given enough
sites, measured against the bound `ROADMAP.md` states. That rests on the
distance being the standard one, so it is pinned first: normalized by
internal splits, which is what makes the bound mean the same thing at any
taxon count.

The sweep is release-gated. It runs a full search per replicate per site
count; what runs per PR is the distance itself, plus the figure at one size.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose
from phylo.qa import topology_accuracy
from phylo.qa.topology_accuracy import REPLICATES, REQUIREMENT, SITE_COUNTS
from phylo.search.topology import (
    enumerate_topologies,
    normalized_robinson_foulds,
    robinson_foulds,
)
from phylo.sim.params import load_simulation_params

from tests._fixtures import FIXTURES_DIR

FIXTURE = FIXTURES_DIR / "simulation_params_6taxa.yaml"
FIVE_TAXA = FIXTURES_DIR / "simulation_params_5taxa.yaml"


# --- the distance ---------------------------------------------------------


@pytest.mark.mathematical
def test_a_topology_is_at_distance_zero_from_itself() -> None:
    for topology in enumerate_topologies(list("ABCDE")):
        assert robinson_foulds(topology, topology) == 0
        assert normalized_robinson_foulds(topology, topology) == 0.0


@pytest.mark.oracle
def test_the_distance_is_the_symmetric_difference_of_the_splits() -> None:
    # Against the definition, computed here independently of the
    # implementation: splits in one tree and not the other, both ways.
    from phylo.search.topology import leaf_bipartitions

    topologies = list(enumerate_topologies(list("ABCDE")))
    for first in topologies[:4]:
        for second in topologies:
            expected = len(
                leaf_bipartitions(first).symmetric_difference(leaf_bipartitions(second))
            )
            assert robinson_foulds(first, second) == expected


@pytest.mark.mathematical
def test_the_normalizer_is_the_internal_split_count() -> None:
    # 2(n - 3) for two binary unrooted trees on n leaves. Trivial splits are
    # excluded deliberately: every tree over the same leaves induces all of
    # them, so counting them would shrink every distance by a factor that
    # depends on the taxon count, and would weaken the bound without saying so.
    for names in (list("ABCDE"), list("ABCDEF")):
        topologies = list(enumerate_topologies(names))
        worst = max(
            normalized_robinson_foulds(topologies[0], other) for other in topologies
        )
        assert_allclose(worst, 1.0, atol=1e-12)
        raw = max(robinson_foulds(topologies[0], other) for other in topologies)
        assert raw == 2 * (len(names) - 3)


@pytest.mark.edge_case
def test_a_tree_with_no_internal_edge_scores_zero() -> None:
    # Below four taxa there is no internal split, so the normalizer is zero
    # and the ratio undefined. Reporting 0.0 is right -- three leaves admit
    # one unrooted topology, so any two such trees are the same tree.
    star = next(iter(enumerate_topologies(list("ABC"))))
    assert robinson_foulds(star, star) == 0
    assert normalized_robinson_foulds(star, star) == 0.0


@pytest.mark.edge_case
def test_the_distance_refuses_trees_over_different_leaves() -> None:
    first = next(iter(enumerate_topologies(list("ABCDE"))))
    second = next(iter(enumerate_topologies(list("ABCDF"))))
    with pytest.raises(ValueError, match="different leaf sets"):
        robinson_foulds(first, second)


# --- the figure -----------------------------------------------------------


@pytest.mark.structural
def test_the_sweep_covers_sizes_on_both_sides_of_the_requirement() -> None:
    # A sweep that only covers sizes already known to work locates no margin,
    # which is the whole point of plotting against the site count.
    assert min(SITE_COUNTS) < 250 < max(SITE_COUNTS)
    assert REPLICATES > 1
    assert REQUIREMENT == 0.05


@pytest.mark.structural
def test_main_writes_a_figure_and_caption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # At 5 taxa and a two-by-two sweep, so the whole pipeline -- sweep,
    # search, distance, caption, render -- runs per PR rather than only behind
    # the release gate.
    #
    # The sweep is cut down deliberately (issue #154). At its committed size
    # this test ran 6 site counts x 8 replicates = 48 searches, 27.7 s, 23% of
    # the whole per-PR suite, to assert four things about a caption. The
    # figure at full size is rendered once, by the technical-document build,
    # and the scientific claim it makes is the release-gated test below. What
    # is left here is what only this test checks: that the pipeline runs end
    # to end and the caption reports the sweep it was handed.
    monkeypatch.setattr(topology_accuracy, "SITE_COUNTS", (min(SITE_COUNTS), 250))
    monkeypatch.setattr(topology_accuracy, "REPLICATES", 2)

    written = topology_accuracy.main(
        ["--params", str(FIVE_TAXA), "--output-dir", str(tmp_path)]
    )
    assert written.figure_path.is_file()
    assert written.caption == written.caption_path.read_text()
    assert "Robinson-Foulds" in written.caption
    assert str(REQUIREMENT) in written.caption
    # The caption reports the sweep that actually ran, not the module's
    # defaults -- which is what makes the reduced size safe to assert on.
    assert "2" in written.caption
    assert str(min(SITE_COUNTS)) in written.caption


@pytest.mark.simulated_truth
@pytest.mark.release
def test_more_sites_recover_the_topology_more_often() -> None:
    # The claim the figure makes. Asserted as a comparison between the ends of
    # the sweep rather than as a threshold at any one size: the rate at a
    # given site count is a property of this fixture, the monotone trend is a
    # property of the method.
    measured = topology_accuracy.accuracy(load_simulation_params(FIXTURE))
    smallest = float(np.mean(measured[min(SITE_COUNTS)]))
    largest = float(np.mean(measured[max(SITE_COUNTS)]))
    assert largest <= REQUIREMENT
    assert largest < smallest
