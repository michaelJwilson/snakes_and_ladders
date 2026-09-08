"""Support for a discrete search result, pinned where enumeration reaches.

The neighbourhood weight is held to the enumerated one where they coincide
(four taxa under NNI, where the neighbourhood is every other topology) and
bounded by it where they do not; the bootstrap is held to its definition;
and the enumerated weight, which is exact, is calibrated on simulated data:
the fraction of returned trees equal to the generating one rises with the
support reported (issue #270).
"""

from __future__ import annotations

import numpy as np
import pytest
from snakes_and_ladders.likelihood.parsimony import fitch_score
from snakes_and_ladders.search.infer import MoveSet, infer
from snakes_and_ladders.search.support import (
    Support,
    SupportKind,
    bootstrap_support,
    enumerated_support,
    internal_splits,
    neighbourhood_support,
    pattern_support,
    split_pattern_support,
)
from snakes_and_ladders.search.topology import (
    enumerate_topologies,
    leaf_bipartitions,
    random_topology,
)
from snakes_and_ladders.sim.params import SimulationParams, load_simulation_params
from snakes_and_ladders.sim.simulate import simulate_alignment

from tests._fixtures import FOUR_TAXA, fixture_path, load_fixture

FIVE_TAXA = "simulation_params_5taxa.yaml"


def _alignment(
    params: SimulationParams, seed: int, n_sites: int
) -> dict[str, np.ndarray]:
    dataset = simulate_alignment(
        params.tau, params.k, params.pi, np.random.default_rng(seed), n_sites
    )
    return dict(dataset.alignment)


@pytest.mark.oracle
def test_the_nni_neighbourhood_of_four_taxa_is_the_whole_space_so_the_two_supports_agree() -> (
    None
):
    params = load_fixture(FOUR_TAXA)
    alignment = _alignment(params, 1, 200)
    topology = random_topology(sorted(alignment), np.random.default_rng(1))

    neighbourhood = neighbourhood_support(topology, alignment, params.k)
    enumerated = enumerated_support(topology, alignment, params.k)

    assert neighbourhood.n_candidates == enumerated.n_candidates == 3
    assert neighbourhood.kind is SupportKind.NEIGHBOURHOOD
    assert enumerated.kind is SupportKind.ENUMERATED
    # The same three fits reached from two rootings: equal to the optimizer's
    # tolerance, not bitwise.
    assert abs(neighbourhood.weight - enumerated.weight) < 1e-9
    assert abs(neighbourhood.margin - enumerated.margin) < 1e-9 * abs(enumerated.margin)


@pytest.mark.mathematical
def test_the_neighbourhood_weight_bounds_the_enumerated_one_and_the_best_tree_has_a_positive_margin() -> (
    None
):
    # A neighbourhood is a subset of the space, so its denominator is smaller
    # and its weight larger; and the topology that beats every topology beats
    # every neighbour, so its NNI margin equals its enumerated margin.
    params = load_simulation_params(fixture_path(FIVE_TAXA))
    alignment = _alignment(params, 2, 300)
    topologies = list(enumerate_topologies(sorted(alignment)))
    assert len(topologies) == 15

    neighbourhood = [neighbourhood_support(t, alignment, params.k) for t in topologies]
    enumerated = [enumerated_support(t, alignment, params.k) for t in topologies]

    for near, exact in zip(neighbourhood, enumerated, strict=True):
        assert near.weight >= exact.weight - 1e-12
        assert near.n_candidates < exact.n_candidates == 15
    best = max(range(15), key=lambda i: enumerated[i].weight)
    assert enumerated[best].margin > 0.0
    assert abs(neighbourhood[best].margin - enumerated[best].margin) < 1e-9
    assert abs(sum(support.weight for support in enumerated) - 1.0) < 1e-9


@pytest.mark.structural
def test_bootstrap_support_is_a_frequency_over_the_returned_topology_s_internal_splits() -> (
    None
):
    params = load_simulation_params(fixture_path(FIVE_TAXA))
    alignment = _alignment(params, 3, 300)
    topology = infer(alignment, params.k, rng=np.random.default_rng(3)).topology

    support = bootstrap_support(
        topology, alignment, params.k, np.random.default_rng(4), n_replicates=8
    )
    again = bootstrap_support(
        topology, alignment, params.k, np.random.default_rng(4), n_replicates=8
    )

    assert set(support) == set(internal_splits(topology))
    assert len(support) == 2  # five taxa: two internal edges
    assert all(0.0 <= value <= 1.0 for value in support.values())
    assert all(round(value * 8) == value * 8 for value in support.values())
    assert support == again


@pytest.mark.simulated_truth
def test_the_generating_splits_have_full_bootstrap_support_at_many_sites() -> None:
    params = load_simulation_params(fixture_path(FIVE_TAXA))
    alignment = _alignment(params, 5, 1000)
    truth = params.tau

    support = bootstrap_support(
        truth, alignment, params.k, np.random.default_rng(6), n_replicates=10
    )

    assert set(support) == set(internal_splits(truth))
    assert all(value >= 0.8 for value in support.values()), support


@pytest.mark.simulated_truth
def test_the_enumerated_support_is_calibrated_on_simulated_data() -> None:
    # The test the ticket names: bin the returned trees by the support they
    # report and the fraction that equal the generating topology must not
    # fall as the support rises. Site counts from 30 to 300 spread the runs
    # across the bins; six seeds each.
    params = load_simulation_params(fixture_path(FIVE_TAXA))
    truth = leaf_bipartitions(params.tau)
    edges = (0.0, 0.5, 0.9, 1.0 + 1e-12)
    hits = np.zeros(3)
    counts = np.zeros(3)
    for n_sites in (30, 60, 120, 300):
        for seed in range(6):
            alignment = _alignment(params, 100 * n_sites + seed, n_sites)
            returned = infer(
                alignment, params.k, moves=MoveSet.NNI, rng=np.random.default_rng(seed)
            ).topology
            support = enumerated_support(returned, alignment, params.k)
            bin_index = int(np.searchsorted(edges, support.weight, side="right") - 1)
            counts[bin_index] += 1
            hits[bin_index] += leaf_bipartitions(returned) == truth

    populated = counts > 0
    rates = hits[populated] / counts[populated]
    assert populated.sum() >= 2, counts
    assert bool((np.diff(rates) >= 0).all()), (rates, counts)


@pytest.mark.edge_case
def test_an_enumeration_past_the_limit_and_an_empty_bootstrap_are_refused() -> None:
    params = load_fixture(FOUR_TAXA)
    alignment = _alignment(params, 1, 50)
    topology = random_topology(sorted(alignment), np.random.default_rng(1))

    with pytest.raises(ValueError, match="topologies on 4 taxa"):
        enumerated_support(topology, alignment, params.k, max_topologies=2)
    with pytest.raises(ValueError, match="at least one replicate"):
        bootstrap_support(topology, alignment, params.k, np.random.default_rng(0), 0)


@pytest.mark.edge_case
def test_a_topology_with_no_competitor_has_all_the_weight() -> None:
    # Three taxa have one unrooted topology, so the neighbourhood is empty.
    params = load_fixture(FOUR_TAXA)
    alignment = _alignment(params, 1, 50)
    three = {name: alignment[name] for name in sorted(alignment)[:3]}
    topology = random_topology(sorted(three), np.random.default_rng(1))

    support = neighbourhood_support(topology, three, params.k)

    assert isinstance(support, Support)
    assert support.weight == 1.0
    assert support.n_candidates == 1
    assert support.margin == np.inf


# --- pattern support (issue #328) -----------------------------------------


@pytest.mark.oracle
def test_pattern_support_is_the_fraction_of_sites_some_tree_with_the_split_fits_in_the_fewest_changes() -> (
    None
):
    # A site is compatible with a split when a resolved tree carrying that
    # split explains it in one change per extra state, the fewest any tree
    # can (Felsenstein, *Inferring Phylogenies*, ch. 8). Enumerated per site
    # as the minimum Fitch score over every five-taxon topology containing
    # the split, which shares nothing with the straddling-state count under
    # test. The split tree with polytomies is not the oracle: it charges a
    # change for two unresolved leaves that share a state.
    params = load_simulation_params(fixture_path(FIVE_TAXA))
    alignment = _alignment(params, 8, 120)
    taxa = sorted(alignment)
    n_sites = next(iter(alignment.values())).shape[0]
    topologies = list(enumerate_topologies(taxa))
    checked: set[frozenset[str]] = set()
    for topology in topologies:
        support = pattern_support(topology, alignment, params.k)
        assert set(support) == set(internal_splits(topology))
        for split, value in support.items():
            if split in checked:
                continue
            checked.add(split)
            containing = [t for t in topologies if split in leaf_bipartitions(t)]
            assert len(containing) == 3
            compatible = 0
            for site in range(n_sites):
                column = {
                    name: states[site : site + 1] for name, states in alignment.items()
                }
                distinct = len({int(states[0]) for states in column.values()})
                fewest = min(fitch_score(t, column, params.k) for t in containing)
                assert fewest >= distinct - 1
                compatible += fewest == distinct - 1
            assert value == compatible / n_sites, split
    # Every 2|3 bipartition of five taxa, C(5, 2) = 10 of them, each met
    # under one canonical side.
    assert len(checked) == 10


@pytest.mark.oracle
def test_the_four_taxon_support_is_one_minus_the_frequency_of_the_two_conflicting_patterns() -> (
    None
):
    # Closed form at four taxa: a site conflicts with AB|CD exactly when it
    # reads xyxy or xyyx, so the support is one minus those two frequencies,
    # counted here directly on the columns.
    params = load_simulation_params(fixture_path(FOUR_TAXA))
    alignment = _alignment(params, 9, 400)
    a, b, c, d = (alignment[name] for name in ("A", "B", "C", "D"))
    conflicting = ((a == c) & (b == d) & (a != b)) | ((a == d) & (b == c) & (a != b))
    expected = 1.0 - float(np.mean(conflicting))
    assert split_pattern_support(frozenset({"C", "D"}), alignment, params.k) == expected
    assert split_pattern_support(frozenset({"A", "B"}), alignment, params.k) == expected
    assert 0.0 < expected < 1.0


@pytest.mark.simulated_truth
def test_pattern_support_ranks_the_generating_split_first_where_the_bootstrap_returns_it_always() -> (
    None
):
    # What pattern support says about the bootstrap frequency is a finding:
    # at 1000 sites on four taxa, the bootstrap returns the generating split
    # in every replicate, and the pattern support of that split exceeds the
    # pattern support of both alternatives.
    params = load_simulation_params(fixture_path(FOUR_TAXA))
    alignment = _alignment(params, 10, 1000)
    truth = params.tau
    (true_split,) = internal_splits(truth)
    bootstrap = bootstrap_support(
        truth, alignment, params.k, np.random.default_rng(11), n_replicates=5
    )
    assert bootstrap == {true_split: 1.0}
    alternatives = [frozenset({"B", "C"}), frozenset({"B", "D"})]
    assert true_split not in alternatives
    winner = split_pattern_support(true_split, alignment, params.k)
    for split in alternatives:
        assert split_pattern_support(split, alignment, params.k) < winner


@pytest.mark.edge_case
def test_a_split_that_is_not_a_bipartition_of_the_alignment_is_refused() -> None:
    params = load_simulation_params(fixture_path(FOUR_TAXA))
    alignment = _alignment(params, 12, 20)
    with pytest.raises(ValueError, match="lacks"):
        split_pattern_support(frozenset({"A", "Z"}), alignment, params.k)
    with pytest.raises(ValueError, match="each side"):
        split_pattern_support(frozenset(alignment), alignment, params.k)
    # A trivial split cannot be crossed, so every site is compatible.
    assert split_pattern_support(frozenset({"A"}), alignment, params.k) == 1.0
