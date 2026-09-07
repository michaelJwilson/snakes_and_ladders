"""Regression tests for topology search.

The claim this module carries is issue #63's, tested for the first time: the
same model-agnostic ``phylo.opt.fit`` scores every candidate topology, with
the discrete move sitting outside it as an operation that builds a new
objective. Nothing in ``phylo.opt`` changed to make that work, and the
import-graph test in ``test_opt_objective.py`` still holds.

Exhaustive validation of *search quality* -- whether hill climbing finds the
global optimum -- is separate and lives in ``test_search_exhaustive.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose
from phylo.likelihood.objective import BranchLengthObjective
from phylo.opt.fit import fit
from phylo.search.infer import Inference, Model, MoveSet, infer, score_topology
from phylo.search.topology import leaf_bipartitions, random_topology
from phylo.sim.newick import count_topologies, to_newick, validate_unrooted_newick
from phylo.sim.simulate import simulate_alignment
from phylo.sim.tree import preorder

from tests._fixtures import SMALL_SITES, load_fixture

# Enough sites to distinguish topologies, few enough that a search is
# seconds. One candidate fit costs about 0.12 s here.
_SITES = 2000


def _alignment() -> tuple[dict[str, np.ndarray], int]:
    params = load_fixture(SMALL_SITES)
    dataset = simulate_alignment(
        tau=params.tau, k=params.k, pi=params.pi, seed=params.seed, n_sites=_SITES
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

    result = infer(alignment, k, seed=1, moves=moves)

    assert list(result.trace) == sorted(result.trace)
    assert len(set(result.trace)) == len(result.trace)


@pytest.mark.mathematical
@pytest.mark.parametrize("moves", [MoveSet.NNI, MoveSet.SPR])
def test_the_search_converges_and_ends_on_its_best_score(moves: MoveSet) -> None:
    alignment, k = _alignment()

    result = infer(alignment, k, seed=1, moves=moves)

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

    result = infer(alignment, k, seed=1, moves=MoveSet.SPR)

    assert result.converged
    assert result.evaluations <= count_topologies(len(alignment) - 1) == 3


@pytest.mark.structural
def test_the_budget_is_respected_and_reported_unconverged() -> None:
    alignment, k = _alignment()

    result = infer(alignment, k, seed=1, max_evaluations=1)

    assert result.evaluations <= 1
    assert not result.converged


@pytest.mark.structural
def test_a_search_is_reproducible_from_its_seed() -> None:
    alignment, k = _alignment()

    first = infer(alignment, k, seed=5)
    second = infer(alignment, k, seed=5)

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

    result = infer(alignment, k, seed=1, model=Model.GTR)

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
