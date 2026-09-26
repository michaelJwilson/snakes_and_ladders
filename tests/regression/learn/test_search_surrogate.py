"""Surrogates in the topology search (issue #308).

Analytic bounds and learned predictors rank a neighbourhood the search fits
only the top of; measured are cost, misses, and whether the answer moves.
Learned models are scored on alignments they never saw. 45 fits at 200 sites
cost 7.5 s against 0.1 s for the surrogates, so the per-PR test reads them
from the ``tree_search/ci`` baseline (#401); the release run fits its own.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pytest
import torch
from sal.bound import Bound
from sal.fixtures import Scale
from sal.learn.ranking import (
    LearnedTreeSurrogate,
    TreeTarget,
    maximized_target,
    shuffle_children,
    tree_examples,
)
from sal.learn.surrogate import (
    LinearSurrogate,
    SetSurrogate,
    argmax_agreement,
    calibrate,
    fit_surrogate,
    r_squared,
    split_by_group,
)
from sal.likelihood.surrogate import (
    ParsimonyUpperBound,
    PlugInLikelihood,
)
from sal.search.infer import infer
from sal.sim.fixtures import baseline
from sal.sim.simulator import simulate_tree
from sal.sim.topology import (
    MoveSet,
    Topology,
    enumerate_topologies,
    leaf_bipartitions,
)

from tests._fixtures import SMALL_SITES, load_fixture

FIVE_TAXA = "tree_search/ci.yaml"
N_SITES = 200


def _alignments(n: int) -> tuple[list[dict[str, np.ndarray]], int, np.ndarray]:
    params = load_fixture(FIVE_TAXA)
    alignments = [
        dict(
            simulate_tree(
                params, np.random.default_rng(1000 + seed), n_sites=N_SITES
            ).alignment
        )
        for seed in range(n)
    ]
    return alignments, params.n_states, np.asarray(params.pi)


def _recorded_target(problem: str, tier: str) -> TreeTarget:
    """The committed maximized log-likelihoods, replayed in the order they were taken.

    A cursor, not a key: a misordering is a wrong number the bound check catches.
    """
    remaining = iter(baseline(problem, tier).values("maximized_log_likelihood"))

    def target(_topology: Topology, _alignment: Mapping[str, np.ndarray]) -> float:
        return next(remaining)

    return target


@pytest.mark.end2end
@pytest.mark.release
def test_learned_surrogates_rank_held_out_neighbourhoods() -> None:
    # Six alignments, 90 fits at 200 sites, split 3/2/1 by alignment. On the
    # gap above the plug-in bound: R^2 0.86 (linear) and 0.9 (set) over 15
    # held-out values spanning 4 nats; the set model ignores child order.
    alignments, k, pi = _alignments(6)
    topologies = [list(enumerate_topologies(sorted(a))) for a in alignments]
    examples = tree_examples(alignments, topologies, k, pi, maximized_target(k))
    split = split_by_group(examples.groups, (0.5, 0.25, 0.25))
    train, validation = examples.subset(split.train), examples.subset(split.validation)
    test = examples.subset(split.test)
    n_features, n_token_features = (
        examples.features.shape[1],
        examples.tokens[0].shape[1],
    )
    for model in (
        LinearSurrogate(n_features),
        SetSurrogate(n_features, n_token_features),
    ):
        fitted = fit_surrogate(
            model, train, validation, generator=torch.Generator().manual_seed(0)
        )
        predicted = fitted.predict(test)
        assert r_squared(predicted, test.targets) > 0.8
        assert argmax_agreement(predicted, test.targets, test.groups) == 1.0
    surrogate = LearnedTreeSurrogate(fitted, k, pi)
    rng = np.random.default_rng(1)
    topology, alignment = topologies[-1][0], alignments[-1]
    assert float(surrogate(topology, alignment)) == pytest.approx(
        float(surrogate(shuffle_children(topology, rng), alignment)), abs=1e-9
    )
    assert surrogate.kind is Bound.POINT
    bound = LearnedTreeSurrogate(calibrate(fitted, validation, Bound.LOWER, 0.8), k, pi)
    assert bound.kind is Bound.LOWER
    assert float(bound(topology, alignment)) < float(surrogate(topology, alignment))


@pytest.mark.end2end
def test_learned_surrogates_rank_a_held_out_alignment_from_the_recorded_fits() -> None:
    # The per-PR sibling (#401): three alignments, the 45 fits read from the
    # baseline (7.5 s of 12.2 s), each checked against the plug-in bound.
    # Measured: R^2 0.912 (linear), 0.953 (set), both ranking the best first.
    alignments, k, pi = _alignments(3)
    topologies = [list(enumerate_topologies(sorted(a))) for a in alignments]
    examples = tree_examples(
        alignments, topologies, k, pi, _recorded_target("tree_search", Scale.CI)
    )
    offset = examples.offset
    assert offset is not None
    assert torch.all(examples.targets >= offset)

    split = split_by_group(examples.groups, (1 / 3, 1 / 3, 1 / 3))
    train, validation = examples.subset(split.train), examples.subset(split.validation)
    test = examples.subset(split.test)
    n_features, n_token_features = (
        examples.features.shape[1],
        examples.tokens[0].shape[1],
    )
    for model in (
        LinearSurrogate(n_features),
        SetSurrogate(n_features, n_token_features),
    ):
        fitted = fit_surrogate(
            model, train, validation, generator=torch.Generator().manual_seed(0)
        )
        predicted = fitted.predict(test)
        assert r_squared(predicted, test.targets) > 0.8
        assert argmax_agreement(predicted, test.targets, test.groups) == 1.0


@pytest.mark.oracle
def test_analytic_bounds_rank_the_fitted_best_first_on_fresh_alignments() -> None:
    # Against a full fit of every topology on three alignments: the plug-in
    # bound and the parsimony bound each put the fitted best first, at a
    # hundredth (2.4 ms against 250 ms) and a thousandth of the cost.
    alignments, k, pi = _alignments(3)
    for alignment in alignments:
        topologies = list(enumerate_topologies(sorted(alignment)))
        exact = [maximized_target(k)(t, alignment) for t in topologies]
        for surrogate in (PlugInLikelihood(k, pi), ParsimonyUpperBound(k, pi)):
            values = [float(surrogate(t, alignment)) for t in topologies]
            assert int(np.argmax(values)) == int(np.argmax(exact))


def _spearman(first: np.ndarray, second: np.ndarray) -> float:
    """Rank correlation, Pearson on the ranks; no ties arise on these values."""
    ranks = [np.argsort(np.argsort(values)).astype(float) for values in (first, second)]
    return float(np.corrcoef(ranks[0], ranks[1])[0, 1])


@pytest.mark.oracle
@pytest.mark.critical
def test_the_learned_surrogate_ranks_the_topologies_the_plug_in_bound_ranks() -> None:
    # The rung below (#734): `PlugInLikelihood`, whose bound offsets the
    # target. Held-out 15 topologies against the recorded fits: rank
    # correlation 0.896 between the two (0.8 declared), 10 of 105 pairs
    # differ, both put the best first; learned 0.893 to the truth, bound 0.804.
    alignments, k, pi = _alignments(3)
    topologies = [list(enumerate_topologies(sorted(a))) for a in alignments]
    examples = tree_examples(
        alignments, topologies, k, pi, _recorded_target("tree_search", Scale.CI)
    )
    split = split_by_group(examples.groups, (1 / 3, 1 / 3, 1 / 3))
    train, validation = examples.subset(split.train), examples.subset(split.validation)
    test = examples.subset(split.test)
    fitted = fit_surrogate(
        LinearSurrogate(examples.features.shape[1]),
        train,
        validation,
        generator=torch.Generator().manual_seed(0),
    )

    held_out = int(np.unique(examples.groups[split.test])[0])
    alignment, candidates = alignments[held_out], topologies[held_out]
    learned = LearnedTreeSurrogate(fitted, k, pi)
    bound = PlugInLikelihood(k, pi)
    ranked = np.array([float(learned(topology, alignment)) for topology in candidates])
    analytic = np.array([float(bound(topology, alignment)) for topology in candidates])
    maximized = test.targets.numpy()

    assert _spearman(ranked, analytic) > 0.8
    assert _spearman(ranked, analytic) < 1.0
    assert int(np.argmax(ranked)) == int(np.argmax(analytic))
    assert int(np.argmax(ranked)) == int(np.argmax(maximized))
    assert _spearman(ranked, maximized) > _spearman(analytic, maximized)


@pytest.mark.oracle
def test_surrogate_ranked_search_reaches_what_the_full_search_reaches() -> None:
    # Lazy ranking by the plug-in bound, fitting one candidate per
    # neighbourhood, lands on the full search's optimum from three starts on
    # the small-sites fixture with fewer fits and no lazy evaluations.
    params = load_fixture(SMALL_SITES)
    alignment = dict(
        simulate_tree(
            params, np.random.default_rng(params.seed), n_sites=2000
        ).alignment
    )
    k, pi = params.n_states, np.asarray(params.pi)
    for seed in range(3):
        full = infer(alignment, k, rng=np.random.default_rng(seed), moves=MoveSet.NNI)
        ranked = infer(
            alignment,
            k,
            rng=np.random.default_rng(seed),
            moves=MoveSet.NNI,
            lazy_top=1,
            surrogate=PlugInLikelihood(k, pi),
        )
        assert leaf_bipartitions(ranked.topology) == leaf_bipartitions(full.topology)
        assert ranked.log_likelihood == pytest.approx(full.log_likelihood, rel=1e-8)
        assert ranked.fits < full.fits
        assert ranked.likelihood_evaluations < full.likelihood_evaluations


@pytest.mark.smoke
def test_a_surrogate_without_lazy_top_is_refused() -> None:
    alignments, k, pi = _alignments(1)
    with pytest.raises(ValueError, match="give both"):
        infer(
            alignments[0],
            k,
            rng=np.random.default_rng(0),
            surrogate=PlugInLikelihood(k, pi),
        )
