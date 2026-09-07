"""Surrogates in the topology search (issue #308).

The analytic bounds and the learned predictors rank a neighbourhood the
search then fits only the top of; what they cost, what they miss, and
whether the search's answer moves are the measurements. The learned models
are trained on alignments whose every topology has been fitted and scored
on alignments they never saw.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from snakes_and_ladders.bound import Bound
from snakes_and_ladders.learn.surrogate import (
    LinearSurrogate,
    SetSurrogate,
    argmax_agreement,
    calibrate,
    fit_surrogate,
    r_squared,
    split_by_group,
)
from snakes_and_ladders.likelihood.surrogate import (
    ParsimonyUpperBound,
    PlugInLikelihood,
)
from snakes_and_ladders.search.infer import MoveSet, infer
from snakes_and_ladders.search.surrogate import (
    LearnedTreeSurrogate,
    maximized_target,
    shuffle_children,
    tree_examples,
)
from snakes_and_ladders.search.topology import enumerate_topologies, leaf_bipartitions
from snakes_and_ladders.sim.simulate import simulate_alignment

from tests._fixtures import SMALL_SITES, load_fixture

FIVE_TAXA = "simulation_params_5taxa.yaml"
N_SITES = 200


def _alignments(n: int) -> tuple[list[dict[str, np.ndarray]], int, np.ndarray]:
    params = load_fixture(FIVE_TAXA)
    alignments = [
        dict(
            simulate_alignment(
                params.tau,
                params.k,
                params.pi,
                np.random.default_rng(1000 + seed),
                N_SITES,
            ).alignment
        )
        for seed in range(n)
    ]
    return alignments, params.k, np.asarray(params.pi)


@pytest.mark.simulated_truth
def test_learned_surrogates_rank_held_out_neighbourhoods() -> None:
    # Six alignments, every topology of each fitted (90 fits at 200 sites),
    # split by alignment: three to train, two to validate, one held out. The
    # models learn the gap above the plug-in bound, so the linear model on
    # the bound features and the set model on the branch tokens both explain
    # the held-out maximized log-likelihood (R^2 0.86 and 0.9 measured, the
    # 15 held-out values spanning 4 nats) and rank its best first; the set
    # model gives the same value for a shuffled spelling of a tree.
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


@pytest.mark.oracle
def test_surrogate_ranked_search_reaches_what_the_full_search_reaches() -> None:
    # Lazy ranking by the plug-in bound, fitting one candidate per
    # neighbourhood, lands on the full search's optimum from three starts on
    # the small-sites fixture with fewer fits and no lazy evaluations.
    params = load_fixture(SMALL_SITES)
    alignment = dict(
        simulate_alignment(
            params.tau, params.k, params.pi, np.random.default_rng(params.seed), 2000
        ).alignment
    )
    k, pi = params.k, np.asarray(params.pi)
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


@pytest.mark.edge_case
def test_a_surrogate_without_lazy_top_is_refused() -> None:
    alignments, k, pi = _alignments(1)
    with pytest.raises(ValueError, match="give both"):
        infer(
            alignments[0],
            k,
            rng=np.random.default_rng(0),
            surrogate=PlugInLikelihood(k, pi),
        )
