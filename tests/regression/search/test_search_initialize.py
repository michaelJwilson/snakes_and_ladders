"""Data-driven starts for the tree, and what they buy at equal evaluations (issue #364).

`FromDistances` and `FromHadamard` are held to the `Initializer` protocol
exactly as the mixture's k-means++ is: they take an `Objective`, refuse one
whose parameters they cannot interpret, and return one point in its
unconstrained coordinates. The measurement is the part that matters, and it
is two comparisons through `opt.budget.compare`. For the branch-length fit
on the generating topology every start reaches the same optimum, so what
separates them is the evaluations spent reaching it. For the topology
search the start is a topology, so the random start is the baseline and the
two estimators are the candidates, scored by whether the climb reaches the
enumerated optimum and the candidates it scores on the way.

The per-pull-request tier pins the direction on the five-taxon fixture over
three datasets; the release-gated tier is the measurement
`docs/experiments/005` reports, and ``pytest -s -m release -k initializers``
reproduces its tables.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pytest
import torch
from snakes_and_ladders.likelihood.distance import DistanceKind
from snakes_and_ladders.likelihood.objective import (
    BranchLengthObjective,
    SubstitutionModelObjective,
)
from snakes_and_ladders.opt.budget import Budget, Comparison, Outcome, compare
from snakes_and_ladders.opt.fit import fit
from snakes_and_ladders.opt.initialize import (
    FromObjective,
    Initializer,
    Perturbed,
    RandomRestart,
)
from snakes_and_ladders.opt.testfunctions import Rosenbrock
from snakes_and_ladders.search.infer import MoveSet, infer, score_topology
from snakes_and_ladders.search.initialize import FromDistances, FromHadamard
from snakes_and_ladders.search.neighbor_joining import split_lengths
from snakes_and_ladders.search.topology import (
    Topology,
    enumerate_topologies,
    leaf_bipartitions,
)
from snakes_and_ladders.sim.simulate import simulate_alignment
from snakes_and_ladders.sim.tree import Node

from tests._fixtures import EIGHT_TAXA, load_fixture
from tests._objective_checks import Counted
from tests.regression.search.test_neighbor_joining import random_tree

FIVE_TAXA = "simulation_params_5taxa.yaml"
SIX_TAXA = "simulation_params_6taxa.yaml"

#: Likelihood evaluations one branch-length fit may spend; every start
#: converges well inside it, and a fit that did not would be refused by the
#: comparison rather than rounded.
FIT_BUDGET = Budget("evaluations", 2000)
#: Candidates one topology search may score.
SEARCH_BUDGET = Budget("candidates", 60)
#: Sites of the twenty-taxon instance, where nothing is enumerated, and its
#: branch-length range: with twenty taxa a path crosses up to a dozen
#: branches, and at ``[0.02, 0.4]`` some pairs saturate at 1,000 sites ---
#: three quarters of sites differing, where no finite distance exists and
#: the estimator refuses. The range keeps every path inside the model.
TWENTY_TAXA_SITES = 1000
TWENTY_TAXA_LENGTHS = (0.02, 0.15)
#: Two fits of one surface agree to the optimizer's convergence, not to
#: machine precision (`tests/regression/search/test_search_exhaustive.py`).
TOLERANCE = 1e-6


@pytest.fixture(autouse=True)
def _single_thread() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@dataclass(frozen=True)
class Instance:
    """One simulated dataset with the tree that generated it."""

    alignment: Mapping[str, np.ndarray]
    k: int
    pi: np.ndarray
    truth: Node


def _instances(name: str | None, seeds: range, n_taxa: int = 20) -> list[Instance]:
    """One dataset per seed, from a fixture or from a random tree at ``n_taxa``."""
    found: list[Instance] = []
    for seed in seeds:
        if name is None:
            truth = random_tree(
                n_taxa, np.random.default_rng([364, n_taxa]), TWENTY_TAXA_LENGTHS
            )
            k, pi, n_sites = 4, np.full(4, 0.25), TWENTY_TAXA_SITES
        else:
            params = load_fixture(name)
            truth, k, pi, n_sites = params.tau, params.k, params.pi, params.n_sites
        dataset = simulate_alignment(
            truth, k, pi, np.random.default_rng([364, seed]), n_sites
        )
        found.append(Instance(dict(dataset.alignment), k, pi, truth))
    return found


class Start(StrEnum):
    """The starts compared, in the order the tables list them."""

    OBJECTIVE = "FromObjective"
    PERTURBED = "Perturbed"
    RANDOM = "RandomRestart"
    DISTANCES = "FromDistances"
    HADAMARD = "FromHadamard"


def _initializer(start: Start, rng: np.random.Generator) -> Initializer:
    if start is Start.OBJECTIVE:
        return FromObjective()
    if start is Start.PERTURBED:
        return Perturbed()
    if start is Start.RANDOM:
        return RandomRestart(1, scale=1.0, rng=rng, include_nominal=False)
    if start is Start.DISTANCES:
        return FromDistances()
    return FromHadamard()


@dataclass(frozen=True)
class FitFrom:
    """A branch-length fit on the generating topology from one start; spends evaluations."""

    start: Start

    def __call__(
        self, instance: Instance, budget: Budget, rng: np.random.Generator
    ) -> Outcome:
        objective = BranchLengthObjective(
            instance.truth, instance.k, instance.pi, instance.alignment
        )
        counted = Counted(objective)
        theta0 = _initializer(self.start, rng).starts(objective)[0]
        result = fit(counted, theta0=theta0)
        assert result.converged
        assert counted.calls <= budget.size
        return Outcome(result.value, counted.calls)


@dataclass(frozen=True)
class SearchFrom:
    """An NNI hill climb from one start topology; spends candidates scored."""

    start: Start

    def __call__(
        self, instance: Instance, budget: Budget, rng: np.random.Generator
    ) -> Outcome:
        topology: Topology | None = None
        if self.start is Start.DISTANCES:
            topology = FromDistances().tree(instance.alignment, instance.k)
        elif self.start is Start.HADAMARD:
            topology = FromHadamard().tree(instance.alignment, instance.k)
        inference = infer(
            instance.alignment,
            instance.k,
            topology=topology,
            moves=MoveSet.NNI,
            max_evaluations=budget.size,
            rng=rng,
        )
        return Outcome(-inference.log_likelihood, inference.evaluations)


def _enumerated_optimum(instance: Instance) -> float:
    """The negative log-likelihood of the best of every topology, fitted."""
    return -max(
        score_topology(topology, instance.alignment, instance.k)
        for topology in enumerate_topologies(sorted(instance.alignment))
    )


def measure_fits(instances: list[Instance], starts: tuple[Start, ...]) -> Comparison:
    """Every start's fit on every instance, scored against the best value found."""
    return compare(
        {start.value: FitFrom(start) for start in starts},
        instances,
        FIT_BUDGET,
        seeds=(0,),
        workers=1,
    )


def measure_search(
    instances: list[Instance], starts: tuple[Start, ...], known: list[float] | None
) -> Comparison:
    """Every start's climb on every instance, against the enumerated optimum where given."""
    return compare(
        {start.value: SearchFrom(start) for start in starts},
        instances,
        SEARCH_BUDGET,
        seeds=(0,),
        workers=1,
        known=known,
    )


def _spent_table(comparison: Comparison, unit: str) -> str:
    """Hits and the spend per method: mean, and its extremes, over the instances."""
    n_instances = comparison.reference.shape[0]
    hits = comparison.hits(TOLERANCE, relative=True)
    lines = [
        f"| start | reached the optimum (of {n_instances}) | mean {unit} | min | max |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row, name in enumerate(comparison.methods):
        spent = comparison.spent[row]
        lines.append(
            f"| {name} | {hits[name]} | {spent.mean():.1f} | {spent.min()} | {spent.max()} |"
        )
    return "\n".join(lines)


# --- the protocol ----------------------------------------------------------


@pytest.mark.structural
@pytest.mark.parametrize("initializer", [FromDistances(), FromHadamard()])
def test_the_start_is_one_point_in_the_objective_s_coordinates(
    initializer: Initializer,
) -> None:
    """One start, of the objective's dimension, whose lengths are the estimate's.

    On the five-taxon fixture the objective is unrooted, so every branch is
    estimable and the constrained start is the estimate's length on each
    split, floored at the initializer's minimum.
    """
    params = load_fixture(FIVE_TAXA)
    dataset = simulate_alignment(
        params.tau,
        params.k,
        params.pi,
        np.random.default_rng(params.seed),
        params.n_sites,
    )
    objective = BranchLengthObjective(
        params.tau, params.k, params.pi, dataset.alignment
    )
    assert isinstance(initializer, Initializer)

    starts = initializer.starts(objective)

    assert len(starts) == 1
    assert starts[0].shape == objective.initial().shape
    fitted = objective.fitted_tree(starts[0])
    assert leaf_bipartitions(fitted) == leaf_bipartitions(params.tau)
    for length in split_lengths(fitted).values():
        assert length >= 1e-4
    assert float(objective(starts[0])) < float(objective(objective.initial()))


@pytest.mark.structural
def test_the_general_model_start_carries_the_log_det_lengths_and_jukes_cantor_rates() -> (
    None
):
    """On a `SubstitutionModelObjective` only the branch block moves off `initial()`."""
    params = load_fixture(FIVE_TAXA)
    dataset = simulate_alignment(
        params.tau,
        params.k,
        params.pi,
        np.random.default_rng(params.seed),
        params.n_sites,
    )
    objective = SubstitutionModelObjective(params.tau, params.k, dataset.alignment)

    start = FromDistances(kind=DistanceKind.LOG_DET).starts(objective)[0]

    n_branches = len(objective.parameter_names) - (6 - 1) - (params.k - 1)
    assert torch.equal(start[n_branches:], objective.initial()[n_branches:])
    named = objective.constrain(start)
    assert float(named["branch_lengths"].min()) >= 1e-4
    assert float(objective(start)) < float(objective(objective.initial()))


@pytest.mark.structural
def test_a_rooted_binary_topology_places_the_root_pair_as_its_sum() -> None:
    """The eight-taxon fixture roots at degree 2: the estimable sum carries the split's length."""
    params = load_fixture(EIGHT_TAXA)
    dataset = simulate_alignment(
        params.tau, params.k, params.pi, np.random.default_rng(params.seed), 2000
    )
    objective = BranchLengthObjective(
        params.tau, params.k, params.pi, dataset.alignment
    )
    initializer = FromDistances()

    start = initializer.starts(objective)[0]

    estimate = split_lengths(initializer.tree(dataset.alignment, params.k))
    placed = split_lengths(objective.fitted_tree(start))
    assert placed.keys() == estimate.keys()
    for split, length in estimate.items():
        assert placed[split] == pytest.approx(length)


@pytest.mark.edge_case
@pytest.mark.parametrize("initializer", [FromDistances(), FromHadamard()])
def test_an_objective_that_is_not_a_tree_is_refused(initializer: Initializer) -> None:
    with pytest.raises(TypeError, match="does not know"):
        initializer.starts(Rosenbrock())


@pytest.mark.edge_case
def test_a_non_positive_floor_is_refused() -> None:
    with pytest.raises(ValueError, match="positive"):
        FromDistances(minimum_length=0.0)
    with pytest.raises(ValueError, match="positive"):
        FromHadamard(minimum_length=-1.0)


# --- the measurement -------------------------------------------------------


@pytest.mark.simulated_truth
def test_every_start_reaches_the_fit_optimum_and_none_reaches_it_cheaper() -> None:
    """Five-taxon fixture, three datasets: all five starts converge to one optimum at one cost.

    The finding, and the opposite of the plausible guess. The neighbor-joining
    start begins within a few nats of the optimum and the objective's own
    constant 200 nats above it, yet L-BFGS spends the same 21 to 26
    evaluations from either: the stop is the gradient relative to the
    objective, and the line search and curvature pairs cost the same from
    anywhere in the basin. Realized evaluations per dataset: FromObjective
    22, 22, 21; Perturbed 25, 25, 21; RandomRestart 21, 24, 23;
    FromDistances 22, 26, 23; FromHadamard 24, 23, 20.
    """
    comparison = measure_fits(_instances(FIVE_TAXA, range(3)), tuple(Start))
    hits = comparison.hits(TOLERANCE, relative=True)

    assert all(count == 3 for count in hits.values()), hits
    spent = comparison.spent
    assert spent.min() >= 20, spent
    assert spent.max() <= 30, spent
    means = spent.mean(axis=1)
    assert means.max() / means.min() < 1.25, means


@pytest.mark.simulated_truth
@pytest.mark.release
def test_the_estimator_starts_reach_the_enumerated_optimum_at_five_taxa() -> None:
    """Five-taxon fixture, three datasets: the climb from either estimator reaches the enumerated optimum.

    The random start reaches it too on these datasets; what the estimators
    buy is the spend, since a start at the optimum scores one neighbourhood
    and stops. Realized candidates scored: RandomRestart 15.3, FromDistances
    4.0, FromHadamard 4.0.
    """
    instances = _instances(FIVE_TAXA, range(3))
    known = [_enumerated_optimum(instance) for instance in instances]

    comparison = measure_search(
        instances, (Start.RANDOM, Start.DISTANCES, Start.HADAMARD), known
    )
    hits = comparison.hits(TOLERANCE, relative=True)

    assert hits[Start.DISTANCES] == 3, hits
    assert hits[Start.HADAMARD] == 3, hits
    rows = {name: comparison.spent[row] for row, name in enumerate(comparison.methods)}
    assert rows[Start.DISTANCES].mean() < rows[Start.RANDOM].mean(), rows


@pytest.mark.release
@pytest.mark.simulated_truth
@pytest.mark.parametrize(
    ("name", "n_seeds"),
    [
        pytest.param(FIVE_TAXA, 20, id="five"),
        pytest.param(SIX_TAXA, 10, id="six"),
        pytest.param(None, 5, id="twenty"),
    ],
)
def test_initializers_at_equal_evaluations(name: str | None, n_seeds: int) -> None:
    """The measurement `docs/experiments/005` reports; ``pytest -s`` prints its tables.

    Twenty datasets of the five-taxon fixture, ten of the six-taxon one and
    five of a random twenty-taxon tree at 1,000 sites --- the counts that
    keep each case inside ten minutes on one core, since the six-taxon
    referee is 105 fits per dataset and a twenty-taxon climb is 60. The fit
    comparison runs every start; the search comparison runs the three that
    name a topology, the Hadamard start only where the taxon count admits
    it, against the enumerated optimum at five and six taxa and the best
    found at twenty.
    """
    print()
    label = name or "random tree, 20 taxa"
    instances = _instances(name, range(n_seeds))
    fits = measure_fits(instances, tuple(Start) if name else tuple(Start)[:-1])
    print(f"fit on the generating topology, {label}, {n_seeds} datasets")
    print(_spent_table(fits, "evaluations"))
    assert all(
        count == n_seeds for count in fits.hits(TOLERANCE, relative=True).values()
    )

    known = [_enumerated_optimum(instance) for instance in instances] if name else None
    starts = (Start.RANDOM, Start.DISTANCES) + ((Start.HADAMARD,) if name else ())
    search = measure_search(instances, starts, known)
    print(f"NNI search, {label}, {n_seeds} datasets, {SEARCH_BUDGET.size} candidates")
    print(_spent_table(search, "candidates"))
    p_value = search.paired_p(Start.DISTANCES, Start.RANDOM, TOLERANCE, relative=True)
    print(f"McNemar FromDistances against RandomRestart: p = {p_value:.3f}")
    hits = search.hits(TOLERANCE, relative=True)
    if name:
        assert hits[Start.DISTANCES] >= hits[Start.RANDOM], hits
