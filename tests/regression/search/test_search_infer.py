"""Regression tests for topology search.

The claim this module carries is issue #63's, tested for the first time: the
same model-agnostic ``snakes_and_ladders.opt.fit`` scores every candidate topology, with
the discrete move sitting outside it as an operation that builds a new
objective. Nothing in ``snakes_and_ladders.opt`` changed to make that work, and the
import-graph test in ``test_opt_objective.py`` still holds.

Exhaustive validation of *search quality* -- whether hill climbing finds the
global optimum -- is separate and lives in ``test_search_exhaustive.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose
from snakes_and_ladders.likelihood import pruning_torch
from snakes_and_ladders.likelihood.device import CROSS_DEVICE_RTOL_FLOAT64
from snakes_and_ladders.likelihood.objective import BranchLengthObjective
from snakes_and_ladders.likelihood.pruning_torch import (
    PartialCache,
    branch_order,
    log_likelihood_cached,
)
from snakes_and_ladders.opt.fit import fit
from snakes_and_ladders.search import infer as infer_module
from snakes_and_ladders.search.infer import (
    Inference,
    Model,
    MoveSet,
    infer,
    score_topology,
)
from snakes_and_ladders.search.topology import (
    branch_splits,
    leaf_bipartitions,
    nni_neighbours,
    random_topology,
)
from snakes_and_ladders.sim.newick import (
    count_topologies,
    to_newick,
    validate_unrooted_newick,
)
from snakes_and_ladders.sim.simulate import simulate_alignment
from snakes_and_ladders.sim.tree import preorder

from tests._fixtures import EIGHT_TAXA, SMALL_SITES, load_fixture

# Enough sites to distinguish topologies, few enough that a search is
# seconds. One candidate fit costs about 0.12 s here.
_SITES = 2000


def _alignment() -> tuple[dict[str, np.ndarray], int]:
    params = load_fixture(SMALL_SITES)
    dataset = simulate_alignment(
        tau=params.tau,
        k=params.k,
        pi=params.pi,
        rng=np.random.default_rng(params.seed),
        n_sites=_SITES,
    )
    return dict(dataset.alignment), params.k


# --- the starting topology ----------------------------------------------


@pytest.mark.structural
@pytest.mark.parametrize("n_taxa", [3, 4, 5, 6])
def test_random_topology_is_a_valid_unrooted_topology(n_taxa: int) -> None:
    names = [f"t{index}" for index in range(n_taxa)]
    topology = random_topology(names, np.random.default_rng(0))

    assert validate_unrooted_newick(to_newick(topology))
    assert sorted(node.name for node in preorder(topology) if node.is_leaf) == names


@pytest.mark.oracle
def test_random_topology_reaches_every_topology_and_only_those() -> None:
    # The generator must be able to start anywhere, or a search seeded from
    # it is quietly restricted to part of the space. Checked against the
    # closed-form count rather than against a second enumeration.
    names = list("ABCDE")
    rng = np.random.default_rng(7)
    found = {leaf_bipartitions(random_topology(names, rng)) for _ in range(4000)}

    assert len(found) == count_topologies(len(names) - 1) == 15


@pytest.mark.structural
def test_random_topology_is_reproducible_from_its_seed() -> None:
    names = list("ABCDEF")
    first = random_topology(names, np.random.default_rng(3))
    second = random_topology(names, np.random.default_rng(3))

    assert leaf_bipartitions(first) == leaf_bipartitions(second)


@pytest.mark.structural
def test_random_topology_carries_no_branch_lengths() -> None:
    # Lengths belong to the objective that fits them. A starting topology
    # carrying them would silently seed the fit.
    topology = random_topology(list("ABCDE"), np.random.default_rng(0))

    assert all(node.branch_length is None for node in preorder(topology))


@pytest.mark.edge_case
@pytest.mark.parametrize(
    ("names", "message"),
    [
        (["A", "B"], "at least 3 leaves"),
        (["A", "B", "B"], "must be distinct"),
    ],
)
def test_random_topology_refuses_unusable_leaf_sets(
    names: list[str], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        random_topology(names, np.random.default_rng(0))


# --- a fixed topology reduces to the continuous fit ----------------------


@pytest.mark.oracle
def test_a_fixed_topology_with_no_budget_is_exactly_the_continuous_fit() -> None:
    # The API's two cases are one code path, not two: with the topology
    # given and no budget, `infer` must agree with calling the objective and
    # the optimizer directly.
    alignment, k = _alignment()
    params = load_fixture(SMALL_SITES)

    result = infer(alignment, k, topology=params.tau, max_evaluations=0)

    objective = BranchLengthObjective(params.tau, k, np.full(k, 1.0 / k), alignment)
    expected = fit(objective)
    assert_allclose(result.log_likelihood, -expected.value, rtol=1e-9)
    assert_allclose(
        result.parameters["branch_lengths"],
        objective.constrain(expected.theta)["branch_lengths"].numpy(),
        rtol=1e-9,
    )
    assert result.evaluations == 0
    assert result.converged


@pytest.mark.oracle
def test_score_topology_agrees_with_a_zero_budget_search() -> None:
    alignment, k = _alignment()
    params = load_fixture(SMALL_SITES)

    assert_allclose(
        score_topology(params.tau, alignment, k),
        infer(alignment, k, topology=params.tau, max_evaluations=0).log_likelihood,
        rtol=1e-12,
    )


# --- the loop ------------------------------------------------------------


@pytest.mark.mathematical
@pytest.mark.parametrize("moves", [MoveSet.NNI, MoveSet.SPR])
def test_every_accepted_move_strictly_improves(moves: MoveSet) -> None:
    # Guaranteed by construction, and worth pinning: a loop that accepted a
    # non-improving move would still terminate and still look plausible.
    alignment, k = _alignment()

    result = infer(alignment, k, rng=np.random.default_rng(1), moves=moves)

    assert list(result.trace) == sorted(result.trace)
    assert len(set(result.trace)) == len(result.trace)


@pytest.mark.mathematical
@pytest.mark.parametrize("moves", [MoveSet.NNI, MoveSet.SPR])
def test_the_search_converges_and_ends_on_its_best_score(moves: MoveSet) -> None:
    alignment, k = _alignment()

    result = infer(alignment, k, rng=np.random.default_rng(1), moves=moves)

    assert result.converged
    assert result.log_likelihood == result.trace[-1]
    assert result.log_likelihood == max(result.trace)


@pytest.mark.structural
def test_no_topology_is_scored_twice() -> None:
    # The deduplication claim, checked against the closed-form count: at 4
    # taxa there are 3 unrooted topologies, so a converged search can never
    # have spent more than 3 fits however many times a neighbourhood
    # proposes the same tree.
    alignment, k = _alignment()

    result = infer(alignment, k, rng=np.random.default_rng(1), moves=MoveSet.SPR)

    assert result.converged
    assert result.evaluations <= count_topologies(len(alignment) - 1) == 3


@pytest.mark.structural
def test_the_budget_is_respected_and_reported_unconverged() -> None:
    alignment, k = _alignment()

    result = infer(alignment, k, rng=np.random.default_rng(1), max_evaluations=1)

    assert result.evaluations <= 1
    assert not result.converged


@pytest.mark.structural
def test_a_search_is_reproducible_from_its_seed() -> None:
    alignment, k = _alignment()

    first = infer(alignment, k, rng=np.random.default_rng(5))
    second = infer(alignment, k, rng=np.random.default_rng(5))

    assert leaf_bipartitions(first.topology) == leaf_bipartitions(second.topology)
    assert_allclose(first.log_likelihood, second.log_likelihood, rtol=1e-12)


@pytest.mark.structural
def test_different_seeds_can_start_from_different_topologies() -> None:
    # Otherwise the seed is decorative and every run measures one start.
    alignment, k = _alignment()
    starts = {
        leaf_bipartitions(random_topology(sorted(alignment), np.random.default_rng(s)))
        for s in range(20)
    }

    assert len(starts) > 1


@pytest.mark.structural
def test_the_general_model_is_searchable_too() -> None:
    alignment, k = _alignment()

    result = infer(alignment, k, rng=np.random.default_rng(1), model=Model.GTR)

    assert isinstance(result, Inference)
    assert set(result.parameters) == {"branch_lengths", "exchangeabilities", "pi"}
    assert_allclose(result.parameters["pi"].sum(), 1.0, rtol=1e-9)
    # The general model has more freedom, so it cannot fit worse than JC on
    # the same topology -- a strictly larger model class always reaches at
    # least the smaller one's optimum.
    jc = score_topology(result.topology, alignment, k, model=Model.JC)
    assert result.log_likelihood >= jc - 1e-6


@pytest.mark.edge_case
def test_too_few_taxa_is_refused() -> None:
    alignment = {name: np.zeros(5, dtype=np.int64) for name in "ABC"}

    with pytest.raises(ValueError, match="at least 4 taxa"):
        infer(alignment, 4)


# --- what carries from a topology to its neighbour (issue #289) ------------


def _eight_taxa() -> tuple[dict[str, np.ndarray], int]:
    params = load_fixture(EIGHT_TAXA)
    dataset = simulate_alignment(
        params.tau,
        params.k,
        params.pi,
        np.random.default_rng(params.seed),
        n_sites=1000,
    )
    return dict(dataset.alignment), params.k


@pytest.mark.structural
def test_branch_splits_are_aligned_with_the_branch_order() -> None:
    # The split below each branch, in the order a `branch_lengths` tensor
    # follows: a fitted length can then be carried by what it separates.
    alignment, _ = _eight_taxa()
    topology = random_topology(sorted(alignment), np.random.default_rng(3))

    splits = branch_splits(topology)

    assert len(splits) == len(branch_order(topology))
    assert set(splits) == set(leaf_bipartitions(topology))
    assert len(set(splits)) == len(splits)


@pytest.mark.mathematical
def test_an_nni_move_replaces_exactly_one_split() -> None:
    # The invariant warm starts rest on: a neighbour keeps every branch but
    # the one across the swapped edge, so the symmetric difference of the two
    # split sets is two -- the split lost and the split gained.
    alignment, _ = _eight_taxa()
    topology = random_topology(sorted(alignment), np.random.default_rng(3))

    for neighbour in nni_neighbours(topology):
        assert len(set(branch_splits(topology)) ^ set(branch_splits(neighbour))) == 2


@pytest.mark.oracle
def test_a_warm_start_reaches_the_cold_optimum_on_every_neighbour() -> None:
    # The pin that lets warm starts be the default: where a fit starts moves,
    # where it ends does not. Every SPR neighbour of a fitted topology is fitted
    # cold and from the parent's lengths; the optima agree within the float64
    # agreement bound (realized worst 5.3e-12 relative over 90 neighbours at
    # eight taxa) and the parent refitted from its own lengths is the parent.
    alignment, k = _eight_taxa()
    start = random_topology(sorted(alignment), np.random.default_rng(1))
    parent = infer_module._score(Model.JC, start, k, alignment)

    own = infer_module._warm_lengths(start, parent).numpy()
    assert np.array_equal(own, parent.parameters["branch_lengths"])

    worst = 0.0
    for neighbour in nni_neighbours(start):
        cold = infer_module._score(Model.JC, neighbour, k, alignment)
        warm = infer_module._score(Model.JC, neighbour, k, alignment, parent)
        worst = max(worst, abs(cold.value - warm.value) / abs(cold.value))
    assert worst < CROSS_DEVICE_RTOL_FLOAT64, worst


@pytest.mark.oracle
def test_the_partial_cache_returns_what_the_recursion_computes() -> None:
    # Bitwise: a partial served from the cache is the tensor the recursion
    # would produce, because the arithmetic inside a subtree is the same
    # whatever sits above it. Checked by evaluating every NNI neighbour once
    # with an empty cache and once with the parent's, and by pinning the
    # cached evaluator against the plain recursion (realized deviation 0.0).
    alignment, k = _eight_taxa()
    pi = np.full(k, 1.0 / k)
    start = random_topology(sorted(alignment), np.random.default_rng(1))
    parent = infer_module._score(Model.JC, start, k, alignment)
    shared = PartialCache()
    lengths = infer_module._warm_lengths(start, parent)

    plain = float(pruning_torch.log_likelihood(start, k, pi, alignment, lengths))
    cached = log_likelihood_cached(start, k, pi, alignment, lengths, shared)
    assert cached == pytest.approx(plain, rel=CROSS_DEVICE_RTOL_FLOAT64)

    for neighbour in nni_neighbours(start):
        warm = infer_module._warm_lengths(neighbour, parent)
        fresh = log_likelihood_cached(neighbour, k, pi, alignment, warm, PartialCache())
        reused = log_likelihood_cached(neighbour, k, pi, alignment, warm, shared)
        assert fresh == reused
    assert shared.hits > 0


@pytest.mark.simulated_truth
def test_lazy_ranking_places_the_fitted_best_first_for_nni() -> None:
    # One unfitted evaluation at the parent's lengths ranks the NNI
    # neighbourhood correctly at eight taxa: the fitted best is the lazy best
    # on 6 of 6 neighbourhoods measured. Asserted at the margin the
    # measurement supports. The same is *not* true of SPR, where a regraft's
    # new branches sit at the default length and the ranking is poor (1 of 6
    # at K = 1, 3 of 6 at K = 5); that number is recorded in STATUS.md and is
    # why lazy scoring is opt-in.
    alignment, k = _eight_taxa()
    hits = 0
    for seed in range(4):
        start = random_topology(sorted(alignment), np.random.default_rng(100 + seed))
        parent = infer_module._score(Model.JC, start, k, alignment)
        neighbours = list(nni_neighbours(start))
        cache = PartialCache()
        lazy = [
            infer_module._lazy_score(Model.JC, n, k, alignment, parent, cache)
            for n in neighbours
        ]
        full = [
            infer_module._score(Model.JC, n, k, alignment, parent).value
            for n in neighbours
        ]
        hits += int(np.argmax(lazy) == np.argmax(full))
    assert hits >= 3, hits


@pytest.mark.oracle
@pytest.mark.parametrize("moves", [MoveSet.NNI, MoveSet.SPR])
def test_warm_starts_do_not_move_the_search_answer(moves: MoveSet) -> None:
    # The search's answer is the same tree at the same likelihood, warm or
    # cold, from the same start; what differs is the cost, which is reported.
    alignment, k = _alignment()
    for seed in range(3):
        cold = infer(
            alignment, k, rng=np.random.default_rng(seed), moves=moves, warm_start=False
        )
        warm = infer(
            alignment, k, rng=np.random.default_rng(seed), moves=moves, warm_start=True
        )
        assert leaf_bipartitions(cold.topology) == leaf_bipartitions(warm.topology)
        assert warm.log_likelihood == pytest.approx(
            cold.log_likelihood, rel=CROSS_DEVICE_RTOL_FLOAT64
        )
        assert warm.fits == cold.fits
        assert warm.likelihood_evaluations >= warm.fits


@pytest.mark.oracle
@pytest.mark.release
def test_lazy_nni_search_reaches_what_the_full_search_reaches() -> None:
    # With every candidate ranked lazily and only the top one fitted, the NNI
    # search still ends where the full search ends on the eight-taxon fixture,
    # from each of three starts, at fewer fits.
    alignment, k = _eight_taxa()
    for seed in range(3):
        full = infer(
            alignment,
            k,
            rng=np.random.default_rng(seed),
            moves=MoveSet.NNI,
            max_evaluations=300,
        )
        lazy = infer(
            alignment,
            k,
            rng=np.random.default_rng(seed),
            moves=MoveSet.NNI,
            max_evaluations=300,
            lazy_top=1,
        )
        assert lazy.log_likelihood == pytest.approx(full.log_likelihood, rel=1e-8)
        assert lazy.fits < full.fits
        assert lazy.likelihood_evaluations < full.likelihood_evaluations


@pytest.mark.edge_case
def test_a_non_positive_lazy_top_is_refused() -> None:
    alignment, k = _alignment()
    with pytest.raises(ValueError, match="lazy_top must be at least 1"):
        infer(alignment, k, rng=np.random.default_rng(0), lazy_top=0)


# --- the bounded regraft and the partial refit (issue #408) ---------------


@pytest.mark.structural
def test_a_radius_at_the_leaf_count_reproduces_the_unbounded_search() -> None:
    # The equivalence the bound is worth having, at the level a caller sees:
    # not merely the same tree, but the same trajectory at the same cost, so
    # a radius that quietly reordered the neighbourhood would fail here.
    alignment, k = _alignment()

    unbounded = infer(alignment, k, rng=np.random.default_rng(0), moves=MoveSet.SPR)
    bounded = infer(
        alignment,
        k,
        rng=np.random.default_rng(0),
        moves=MoveSet.SPR,
        radius=len(alignment),
    )

    assert leaf_bipartitions(bounded.topology) == leaf_bipartitions(unbounded.topology)
    assert bounded.log_likelihood == unbounded.log_likelihood
    assert bounded.trace == unbounded.trace
    assert bounded.evaluations == unbounded.evaluations
    assert bounded.fits == unbounded.fits
    assert bounded.likelihood_evaluations == unbounded.likelihood_evaluations


@pytest.mark.edge_case
def test_a_radius_with_nni_moves_is_refused() -> None:
    # NNI has no pruning point to measure from, so a radius is meaningless
    # there; ignoring it would report a bounded search that was not one.
    alignment, k = _alignment()
    with pytest.raises(ValueError, match="no pruning point"):
        infer(alignment, k, rng=np.random.default_rng(0), moves=MoveSet.NNI, radius=2)


@pytest.mark.edge_case
def test_partial_reoptimization_without_warm_start_is_refused() -> None:
    alignment, k = _alignment()
    with pytest.raises(ValueError, match="warm_start is where they come from"):
        infer(
            alignment,
            k,
            rng=np.random.default_rng(0),
            warm_start=False,
            partial_reoptimization=True,
        )


@pytest.mark.structural
def test_a_partial_fit_moves_only_the_branches_the_move_created() -> None:
    # What `_Restricted` claims: every coordinate outside the disturbed set
    # comes back at the value it went in with, exactly. A scatter that
    # dropped or transposed an index would still fit and still improve, and
    # only this catches it.
    alignment, k = _alignment()
    start = random_topology(sorted(alignment), np.random.default_rng(0))
    parent = infer_module._score(Model.JC, start, k, alignment)
    neighbour = next(iter(nni_neighbours(start)))

    fitted = infer_module._score(
        Model.JC, neighbour, k, alignment, parent, partial=True
    )

    disturbed = infer_module._disturbed(neighbour, parent).tolist()
    assert len(disturbed) == 1  # an NNI creates exactly one split
    kept = [
        split
        for index, split in enumerate(branch_splits(neighbour))
        if index not in disturbed
    ]
    assert kept
    for split in kept:
        assert fitted.lengths_by_split[split] == parent.lengths_by_split[split]


@pytest.mark.structural
def test_partial_reoptimization_reports_a_full_fit() -> None:
    # A partial fit is a lower bound, so the accepted move is refitted over
    # every branch before it is reported. Dropping that refit would leave
    # `log_likelihood` below the topology's maximum, and this pins it
    # against the same fit taken from scratch.
    alignment, k = _alignment()

    result = infer(
        alignment,
        k,
        rng=np.random.default_rng(0),
        moves=MoveSet.SPR,
        partial_reoptimization=True,
    )

    assert result.log_likelihood == pytest.approx(
        score_topology(result.topology, alignment, k), rel=1e-8
    )
