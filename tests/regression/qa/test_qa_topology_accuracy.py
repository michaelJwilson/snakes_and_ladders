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

import numpy as np
import pytest
from numpy.testing import assert_allclose
from sal.fixtures import load_params
from sal.qa import topology_accuracy
from sal.qa.manifest import FIGURES
from sal.qa.topology_accuracy import REPLICATES, REQUIREMENT, SITE_COUNTS
from sal.sim.params import SimulationParams
from sal.sim.topology import (
    enumerate_topologies,
    normalized_robinson_foulds,
    robinson_foulds,
)

from tests._fixtures import FIXTURES_DIR

FIXTURE = FIXTURES_DIR / "tree_search/stress.yaml"
FIVE_TAXA = FIXTURES_DIR / "tree_search/ci.yaml"


# --- the distance ---------------------------------------------------------


@pytest.mark.analytic
def test_a_topology_is_at_distance_zero_from_itself() -> None:
    for topology in enumerate_topologies(list("ABCDE")):
        assert robinson_foulds(topology, topology) == 0
        assert normalized_robinson_foulds(topology, topology) == 0.0


@pytest.mark.oracle
def test_the_distance_is_the_symmetric_difference_of_the_splits() -> None:
    # Against the definition, computed here independently of the
    # implementation: splits in one tree and not the other, both ways.
    from sal.sim.topology import leaf_bipartitions

    topologies = list(enumerate_topologies(list("ABCDE")))
    for first in topologies[:4]:
        for second in topologies:
            expected = len(
                leaf_bipartitions(first).symmetric_difference(leaf_bipartitions(second))
            )
            assert robinson_foulds(first, second) == expected


@pytest.mark.analytic
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


@pytest.mark.smoke
def test_a_tree_with_no_internal_edge_scores_zero() -> None:
    # Below four taxa there is no internal split, so the normalizer is zero
    # and the ratio undefined. Reporting 0.0 is right -- three leaves admit
    # one unrooted topology, so any two such trees are the same tree.
    star = next(iter(enumerate_topologies(list("ABC"))))
    assert robinson_foulds(star, star) == 0
    assert normalized_robinson_foulds(star, star) == 0.0


@pytest.mark.smoke
def test_the_distance_refuses_trees_over_different_leaves() -> None:
    first = next(iter(enumerate_topologies(list("ABCDE"))))
    second = next(iter(enumerate_topologies(list("ABCDF"))))
    with pytest.raises(ValueError, match="different leaf sets"):
        robinson_foulds(first, second)


# --- the figure -----------------------------------------------------------


@pytest.mark.smoke
def test_the_sweep_covers_sizes_on_both_sides_of_the_requirement() -> None:
    # A sweep that only covers sizes already known to work locates no margin,
    # which is the whole point of plotting against the site count.
    assert min(SITE_COUNTS) < 250 < max(SITE_COUNTS)
    assert REPLICATES > 1
    assert REQUIREMENT == 0.05


@pytest.mark.smoke
def test_the_caption_states_the_sweep_this_module_declares() -> None:
    # Every number the committed caption asserts, pinned. The caption reads
    # them from the module and the fixture rather than restating them, so
    # what this holds is the source they are read from: the replicate count,
    # the six site counts, and their two ends, which the caption names.
    assert REPLICATES == 8
    assert SITE_COUNTS == (60, 125, 250, 500, 1000, 2000)
    assert len(SITE_COUNTS) == 6


@pytest.mark.oracle
def test_the_manifest_renders_this_figure_from_the_fixture_the_caption_names() -> None:
    # The caption's seed and state count come from the fixture the manifest
    # names, so the committed caption's 20260905 and "4-state" are only
    # pinned if that argument is. Read from the yaml independently of the
    # renderer.
    spec = next(spec for spec in FIGURES if spec.stem == "topology_accuracy")
    assert spec.arguments == (
        "--params",
        "tests/regression/fixtures/tree_search/stress.yaml",
    )

    params = load_params(FIXTURE, SimulationParams)
    assert params.seed == 20260905
    assert len(params.pi) == 4


@pytest.mark.end2end
@pytest.mark.release
def test_more_sites_recover_the_topology_more_often() -> None:
    # The caption's sentence since #492: the mean meets the requirement at
    # the longest alignment and misses it at the shortest, compared end to
    # end; mid-sweep rates are the fixture's and discontinuous over 48 fits
    # (`docs/CLAUDE.md`).
    measured = topology_accuracy.accuracy(load_params(FIXTURE, SimulationParams))
    smallest = float(np.mean(measured[min(SITE_COUNTS)]))
    largest = float(np.mean(measured[max(SITE_COUNTS)]))
    assert largest <= REQUIREMENT
    assert smallest > REQUIREMENT
    assert largest < smallest
