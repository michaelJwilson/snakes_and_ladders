"""Support for a discrete search result, pinned where enumeration reaches.

The neighbourhood weight is held to the enumerated one where they coincide
(four taxa under NNI, where the neighbourhood is every other topology) and
bounded by it where they do not; the bootstrap is held to its definition;
and the enumerated weight, which is exact, is calibrated on simulated data:
the fraction of returned trees equal to the generating one rises with the
support reported (issue #270). The same two weights over the labellings of
a factor graph are held to `enumerate_potts` at `beta = 1` and to the
enumerated path posterior, and the calibration is re-measured at seven and
eight taxa behind the release gate (issue #331).
"""

from __future__ import annotations

import numpy as np
import pytest
from snakes_and_ladders.likelihood.hmm_paths import (
    emission_log_density,
    enumerate_hidden_paths,
    path_log_probability,
)
from snakes_and_ladders.likelihood.parsimony import fitch_score
from snakes_and_ladders.likelihood.potts import enumerate_potts, log_weights
from snakes_and_ladders.search.infer import MoveSet, infer
from snakes_and_ladders.search.support import (
    Support,
    SupportKind,
    bootstrap_support,
    enumerated_labelling_support,
    enumerated_support,
    internal_splits,
    neighbourhood_labelling_support,
    neighbourhood_support,
    pattern_support,
    split_pattern_support,
)
from snakes_and_ladders.search.topology import (
    enumerate_topologies,
    leaf_bipartitions,
    random_topology,
)
from snakes_and_ladders.sim.canonical import AMBIGUOUS_OBSERVATIONS, ambiguous_hmm
from snakes_and_ladders.sim.factor_graph import (
    Factor,
    FactorGraph,
    from_hmm,
    from_potts,
)
from snakes_and_ladders.sim.graph import BoundaryCondition, lattice_graph
from snakes_and_ladders.sim.params import SimulationParams, load_simulation_params
from snakes_and_ladders.sim.simulate import simulate_alignment

from tests._fixtures import EIGHT_TAXA, FOUR_TAXA, fixture_path, load_fixture

FIVE_TAXA = "tree_search/ci.yaml"
SEVEN_TAXA = "tree_search/release.yaml"
FIELD = np.array([0.6, -0.4, 0.1])


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
@pytest.mark.release
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
@pytest.mark.release
def test_bootstrap_support_is_a_frequency_over_the_returned_topology_s_internal_splits() -> (
    None
):
    params = load_simulation_params(fixture_path(FIVE_TAXA))
    alignment = _alignment(params, 3, 300)
    topology = infer(alignment, params.k, rng=np.random.default_rng(3)).topology

    support = bootstrap_support(
        topology,
        alignment,
        params.k,
        np.random.default_rng(4),
        n_replicates=8,
        workers=1,
    )
    again = bootstrap_support(
        topology,
        alignment,
        params.k,
        np.random.default_rng(4),
        n_replicates=8,
        workers=1,
    )

    assert set(support) == set(internal_splits(topology))
    assert len(support) == 2  # five taxa: two internal edges
    assert all(0.0 <= value <= 1.0 for value in support.values())
    assert all(round(value * 8) == value * 8 for value in support.values())
    assert support == again


@pytest.mark.simulated_truth
@pytest.mark.release
def test_the_generating_splits_have_full_bootstrap_support_at_many_sites() -> None:
    params = load_simulation_params(fixture_path(FIVE_TAXA))
    alignment = _alignment(params, 5, 1000)
    truth = params.tau

    support = bootstrap_support(
        truth,
        alignment,
        params.k,
        np.random.default_rng(6),
        n_replicates=10,
        workers=1,
    )

    assert set(support) == set(internal_splits(truth))
    assert all(value >= 0.8 for value in support.values()), support


@pytest.mark.simulated_truth
@pytest.mark.release
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


def _calibration_bins(
    supports: list[float], correct: list[bool]
) -> tuple[np.ndarray, np.ndarray]:
    """Per bin of support ``(0, 0.5], (0.5, 0.9], (0.9, 1]``: runs, and the fraction correct."""
    edges = (0.0, 0.5, 0.9, 1.0 + 1e-12)
    hits = np.zeros(3)
    counts = np.zeros(3)
    for support, hit in zip(supports, correct, strict=True):
        bin_index = int(np.searchsorted(edges, support, side="right") - 1)
        counts[bin_index] += 1
        hits[bin_index] += hit
    populated = counts > 0
    return counts, hits[populated] / counts[populated]


@pytest.mark.simulated_truth
@pytest.mark.release
@pytest.mark.parametrize(("fixture", "n_taxa"), [(SEVEN_TAXA, 7), (EIGHT_TAXA, 8)])
def test_the_neighbourhood_and_bootstrap_supports_are_calibrated_at_seven_and_eight_taxa(
    fixture: str, n_taxa: int
) -> None:
    # The five-taxon calibration above, at the sizes where the enumerated
    # weight is no longer affordable (945 fits per tree at seven taxa;
    # refused at eight), so the quantities a search there can report are
    # the ones binned: the NNI neighbourhood weight, and the bootstrap as a
    # tree-level statistic -- the smallest support over the tree's internal
    # splits, since the tree is right only if every split is. Four site
    # counts, four seeds, eight replicates: 16 runs per taxon count, about
    # 10 min at seven taxa and 30 at eight. The realized tables are in
    # `STATUS.md`; the assertion is that neither fraction falls from one
    # bin to the next.
    params = load_simulation_params(fixture_path(fixture))
    assert len(list(internal_splits(params.tau))) == n_taxa - 3
    truth = leaf_bipartitions(params.tau)
    neighbourhood: list[float] = []
    bootstrap: list[float] = []
    correct: list[bool] = []
    for n_sites in (50, 100, 200, 400):
        for seed in range(4):
            alignment = _alignment(params, 100 * n_sites + seed, n_sites)
            returned = infer(
                alignment, params.k, moves=MoveSet.NNI, rng=np.random.default_rng(seed)
            ).topology
            neighbourhood.append(
                neighbourhood_support(returned, alignment, params.k).weight
            )
            bootstrap.append(
                min(
                    bootstrap_support(
                        returned,
                        alignment,
                        params.k,
                        np.random.default_rng(seed),
                        n_replicates=8,
                        workers=1,
                    ).values()
                )
            )
            correct.append(leaf_bipartitions(returned) == truth)

    for name, supports in (("neighbourhood", neighbourhood), ("bootstrap", bootstrap)):
        counts, rates = _calibration_bins(supports, correct)
        assert counts.sum() == 16
        assert (counts > 0).sum() >= 2, (name, counts)
        assert bool((np.diff(rates) >= 0).all()), (name, rates, counts)


# --- labellings and decodings (issue #331) -------------------------------


def _two_site_chain(clamp: int | None) -> FactorGraph:
    """Two sites, three states, one bond; ``clamp`` fixes the second site's state."""
    graph = from_potts(lattice_graph((2,), BoundaryCondition.OPEN, 0.8), FIELD)
    if clamp is None:
        return graph
    indicator = np.full(3, -np.inf)
    indicator[clamp] = 0.0
    return FactorGraph(
        graph.variables, [*graph.factors, Factor("x", ("s1",), indicator)]
    )


@pytest.mark.oracle
def test_the_single_site_neighbourhood_is_the_whole_space_only_where_one_site_is_free() -> (
    None
):
    # On a two-site chain the single-site flips reach 5 of the 9 labellings:
    # the four that change both sites are two moves away, so the
    # neighbourhood weight is the enumerated one renormalized over that
    # ball, and larger. Clamp one site by an indicator factor, as a leaf's
    # observed state clamps a tree's, and the ball is every labelling with
    # weight, so the two agree to 1e-12.
    labelling = np.array([1, 0])
    free = _two_site_chain(None)
    near, exact = (
        neighbourhood_labelling_support(free, labelling),
        enumerated_labelling_support(free, labelling),
    )
    assert (near.n_candidates, exact.n_candidates) == (5, 9)
    ball = [labelling.copy() for _ in range(5)]
    for index, (site, state) in enumerate(((0, 0), (0, 2), (1, 1), (1, 2))):
        ball[index + 1][site] = state
    over_ball = sum(enumerated_labelling_support(free, b).weight for b in ball)
    assert abs(near.weight - exact.weight / over_ball) < 1e-12
    assert near.weight > exact.weight

    clamped = _two_site_chain(0)
    near, exact = (
        neighbourhood_labelling_support(clamped, labelling),
        enumerated_labelling_support(clamped, labelling),
    )
    assert near.kind is SupportKind.NEIGHBOURHOOD
    assert exact.kind is SupportKind.ENUMERATED
    assert abs(near.weight - exact.weight) < 1e-12
    assert abs(near.margin - exact.margin) < 1e-12
    assert near.log_score == exact.log_score == clamped.log_density({"s0": 1, "s1": 0})


@pytest.mark.oracle
def test_the_enumerated_labelling_weight_is_enumerate_potts_s_boltzmann_weight_at_beta_one() -> (
    None
):
    # Every labelling of a 3 x 2 lattice, three states: `exp(log_weights -
    # log Z)` from the Potts oracle, which shares no code with the factor
    # graph's own log-density, equals the enumerated support to 1e-12 and
    # the 729 weights sum to one; the single-site neighbourhood never gives
    # a labelling less than its enumerated weight.
    graph = lattice_graph((3, 2), BoundaryCondition.OPEN, 0.8)
    factor_graph = from_potts(graph, FIELD)
    exact = enumerate_potts(graph, FIELD)
    labellings = np.array(list(np.ndindex((3,) * graph.n_nodes)), dtype=np.int64)
    boltzmann = np.exp(log_weights(graph, FIELD, labellings) - exact.log_partition)

    weights = np.array(
        [enumerated_labelling_support(factor_graph, s).weight for s in labellings]
    )

    assert np.abs(weights - boltzmann).max() < 1e-12
    assert abs(weights.sum() - 1.0) < 1e-12
    for labelling in labellings[::73]:
        near = neighbourhood_labelling_support(factor_graph, labelling)
        assert near.n_candidates == 13
        assert near.weight >= boltzmann[int(np.ravel_multi_index(labelling, (3,) * 6))]


@pytest.mark.oracle
def test_the_enumerated_decoding_weight_is_the_path_posterior_and_pins_the_ambiguous_fixture() -> (
    None
):
    # A decoding is a labelling of the chain's factor graph. On the fixture
    # whose two decoders disagree, the Viterbi path's enumerated weight is
    # `exp(log P(path, x) - log P(x))` from the path enumeration, its margin
    # is the 0.3033 nats `ambiguous_hmm` documents, and the posterior-decoded
    # path sits 0.6066 nats below it -- a negative margin, which is what a
    # decoding that is no maximum over paths looks like.
    params = ambiguous_hmm()
    observations = AMBIGUOUS_OBSERVATIONS
    graph = from_hmm(
        np.log(params.initial),
        np.log(params.transition),
        emission_log_density(params, observations),
    )
    paths = enumerate_hidden_paths(params, observations)

    viterbi = enumerated_labelling_support(graph, paths.viterbi)
    posterior = enumerated_labelling_support(graph, paths.posterior_path)

    for support, path in ((viterbi, paths.viterbi), (posterior, paths.posterior_path)):
        expected = np.exp(
            path_log_probability(params, path, observations) - paths.log_likelihood
        )
        assert abs(support.weight - expected) < 1e-12
        assert support.n_candidates == 32
    assert abs(viterbi.margin - 0.3033) < 5e-5
    assert abs(posterior.margin + 0.6066) < 5e-5
    # The runner-up to the Viterbi path is three changes away, so its
    # single-change margin is at least the enumerated one.
    near = neighbourhood_labelling_support(graph, paths.viterbi)
    assert near.n_candidates == 6
    assert near.margin >= viterbi.margin - 1e-12


@pytest.mark.edge_case
def test_a_labelling_outside_its_domain_or_past_the_cap_is_refused() -> None:
    graph = _two_site_chain(None)
    with pytest.raises(ValueError, match="inside its cardinality"):
        neighbourhood_labelling_support(graph, np.array([0, 3]))
    with pytest.raises(ValueError, match="inside its cardinality"):
        enumerated_labelling_support(graph, np.array([0]))
    with pytest.raises(ValueError, match="9 labellings of 2 variables"):
        enumerated_labelling_support(graph, np.array([0, 0]), max_configurations=8)


@pytest.mark.edge_case
def test_an_enumeration_past_the_limit_and_an_empty_bootstrap_are_refused() -> None:
    params = load_fixture(FOUR_TAXA)
    alignment = _alignment(params, 1, 50)
    topology = random_topology(sorted(alignment), np.random.default_rng(1))

    with pytest.raises(ValueError, match="topologies on 4 taxa"):
        enumerated_support(topology, alignment, params.k, max_topologies=2)
    with pytest.raises(ValueError, match="at least one replicate"):
        bootstrap_support(
            topology, alignment, params.k, np.random.default_rng(0), 0, workers=1
        )


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
        truth, alignment, params.k, np.random.default_rng(11), n_replicates=5, workers=1
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


@pytest.mark.structural
@pytest.mark.release
def test_four_workers_report_the_bootstrap_one_worker_reports() -> None:
    """Replicate ``i`` draws the stream spawned for it, whichever worker runs it (issue #344).

    The frequencies are equal exactly, not within a Monte Carlo tolerance,
    because the resample and the search of each replicate come from the
    generator spawned for that replicate and from nothing scheduled.
    """
    params = load_simulation_params(fixture_path(FIVE_TAXA))
    alignment = _alignment(params, 3, 300)
    topology = infer(alignment, params.k, rng=np.random.default_rng(3)).topology

    serial = bootstrap_support(
        topology, alignment, params.k, np.random.default_rng(4), 8, workers=1
    )
    pooled = bootstrap_support(
        topology, alignment, params.k, np.random.default_rng(4), 8, workers=4
    )

    assert pooled == serial


@pytest.mark.structural
def test_a_replicate_is_the_search_of_the_resample_its_spawned_generator_draws() -> (
    None
):
    """The stream a replicate sees is stated, so a reader can reproduce one by hand.

    Replicate ``i`` gets ``rng.spawn(n_replicates)[i]``, draws its columns
    from it, then searches with the same generator. Pinned against that
    construction rather than against a recorded frequency, because the
    frequency would also hold under a stream nobody could name.
    """
    params = load_simulation_params(fixture_path(FIVE_TAXA))
    alignment = _alignment(params, 3, 300)
    topology = infer(alignment, params.k, rng=np.random.default_rng(3)).topology
    n_sites = 300

    support = bootstrap_support(
        topology, alignment, params.k, np.random.default_rng(9), 2, workers=1
    )

    counts = dict.fromkeys(internal_splits(topology), 0)
    for child in np.random.default_rng(9).spawn(2):
        columns = child.integers(0, n_sites, size=n_sites)
        resampled = {name: states[columns] for name, states in alignment.items()}
        found = infer(resampled, params.k, rng=child).topology
        for split in internal_splits(found):
            if split in counts:
                counts[split] += 1
    assert support == {split: count / 2 for split, count in counts.items()}
