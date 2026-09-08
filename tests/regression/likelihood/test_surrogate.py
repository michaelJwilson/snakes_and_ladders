"""The analytic surrogates against the exact evaluations they bound (issue #308).

A bound is certified over every structure an oracle can score: all 15
topologies of the five-taxon fixture against the fitted log-likelihood, and
small lattices against enumeration of ``log Z``. The parsimony bound's
proof rests on a vertex argument, and the vertices are enumerated here so
the argument is checked rather than trusted. The features a learned
surrogate reads are functions of the unrooted tree, and a spelling of the
same tree with its children swapped must give the same features.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable

import numpy as np
import pytest
import torch
from snakes_and_ladders.bound import Bound, BoundViolation, Certificate, certify
from snakes_and_ladders.likelihood.features import (
    TREE_FEATURE_NAMES,
    lattice_features,
    tree_features,
    tree_tokens,
)
from snakes_and_ladders.likelihood.parsimony import fitch_score
from snakes_and_ladders.likelihood.potts import enumerate_potts, log_weights
from snakes_and_ladders.likelihood.pruning_torch import branch_order, log_likelihood
from snakes_and_ladders.likelihood.surrogate import (
    ParsimonyUpperBound,
    PlugInLikelihood,
    ground_state_energy_bounds,
    jc_distances,
    least_squares_lengths,
    least_squares_residual,
    mean_field_log_partition,
    prune_with_matrices,
    site_fitch_scores,
    spanning_tree_log_partition,
)
from snakes_and_ladders.search.infer import score_topology
from snakes_and_ladders.search.surrogate import shuffle_children
from snakes_and_ladders.search.topology import enumerate_topologies
from snakes_and_ladders.sim.graph import BoundaryCondition, PottsGraph, lattice_graph
from snakes_and_ladders.sim.simulate import simulate_alignment
from snakes_and_ladders.sim.tree import Node

from tests._fixtures import FIXTURES_DIR, load_fixture

FIVE_TAXA = "simulation_params_5taxa.yaml"
N_SITES = 200
FIELD = np.array([0.3, -0.2, 0.1])


def _alignment(n_sites: int = N_SITES) -> tuple[dict[str, np.ndarray], int, np.ndarray]:
    params = load_fixture(FIVE_TAXA)
    dataset = simulate_alignment(
        params.tau, params.k, params.pi, np.random.default_rng(params.seed), n_sites
    )
    return dict(dataset.alignment), params.k, np.asarray(params.pi)


@pytest.fixture(scope="module")
def scored() -> Scored:
    """Every topology of the five-taxon fixture, fitted once for the module."""
    alignment, k, pi = _alignment()
    topologies = list(enumerate_topologies(sorted(alignment)))
    exact = [score_topology(topology, alignment, k) for topology in topologies]
    return alignment, k, pi, topologies, exact


Scored = tuple[dict[str, np.ndarray], int, np.ndarray, list[Node], list[float]]


def _fitted(k: int) -> Callable[[object, object], float]:
    """``score_topology`` under the seam's untyped signature."""

    def exact(structure: object, data: object) -> float:
        assert isinstance(structure, Node)
        assert isinstance(data, dict)
        return score_topology(structure, data, k)

    return exact


def _identity(structure: object, data: object) -> float:
    del data
    return float(structure)  # type: ignore[arg-type]


def _random_lattice(shape: tuple[int, int], rng: np.random.Generator) -> PottsGraph:
    graph = lattice_graph(shape, BoundaryCondition.OPEN, 1.0)
    couplings = rng.uniform(0.2, 1.0, len(graph.edges))
    return PottsGraph(graph.n_nodes, graph.edges, tuple(float(c) for c in couplings))


# --- trees --------------------------------------------------------------------


@pytest.mark.oracle
def test_plug_in_bound_is_below_every_fitted_likelihood(scored: Scored) -> None:
    # One pruning evaluation at least-squares lengths against a full fit, on
    # all 15 topologies: never above, within 8 nats at 200 sites (6.7 was the
    # worst gap measured at 1,200), and ranking the fitted best first.
    alignment, k, pi, topologies, exact = scored
    surrogate = PlugInLikelihood(k, pi)
    certificate = certify(surrogate, _fitted(k), topologies, alignment)
    assert certificate == Certificate(
        15, 0, certificate.worst_gap, certificate.mean_gap
    )
    assert certificate.worst_gap < 8.0
    values = [float(surrogate(topology, alignment)) for topology in topologies]
    assert int(np.argmax(values)) == int(np.argmax(exact))


@pytest.mark.oracle
def test_parsimony_bound_is_above_every_fitted_likelihood(scored: Scored) -> None:
    alignment, k, pi, topologies, exact = scored
    surrogate = ParsimonyUpperBound(k, pi)
    certificate = certify(surrogate, _fitted(k), topologies, alignment)
    assert certificate.violations == 0
    values = [float(surrogate(topology, alignment)) for topology in topologies]
    assert int(np.argmax(values)) == int(np.argmax(exact))


@pytest.mark.mathematical
def test_parsimony_bound_is_the_maximum_over_the_vertices() -> None:
    # The site likelihood is multilinear in one variable per branch under
    # Jukes--Cantor, so its maximum over lengths is at a vertex where each
    # branch is the identity or the uniform matrix. Enumerating the 2^8
    # vertices: the per-site maximum equals pi(x_1) k^{-F_s} on every site,
    # and pruning at random lengths never exceeds it.
    alignment, k, pi = _alignment(40)
    rng = np.random.default_rng(3)
    identity, uniform = np.eye(k), np.full((k, k), 1.0 / k)
    first = alignment[sorted(alignment)[0]]
    for topology in list(enumerate_topologies(sorted(alignment)))[:4]:
        names = branch_order(topology)
        best = np.full(40, -np.inf)
        for choice in itertools.product([identity, uniform], repeat=len(names)):
            matrices = dict(zip(names, choice, strict=True))
            best = np.maximum(
                best, _site_log_likelihoods(topology, k, pi, alignment, matrices)
            )
        bound = np.log(pi[first]) - site_fitch_scores(topology, alignment) * np.log(k)
        np.testing.assert_allclose(best, bound, atol=1e-12)
        for _ in range(3):
            lengths = torch.as_tensor(rng.exponential(0.3, len(names)))
            value = float(log_likelihood(topology, k, pi, alignment, lengths))
            assert value <= float(best.sum()) + 1e-9


def _site_log_likelihoods(
    tau: Node,
    k: int,
    pi: np.ndarray,
    alignment: dict[str, np.ndarray],
    matrices: dict[str, np.ndarray],
) -> np.ndarray:
    """``prune_with_matrices`` per site, for the vertex enumeration above."""
    n_sites = next(iter(alignment.values())).shape[0]
    values = np.empty(n_sites)
    for site in range(n_sites):
        column = {name: states[site : site + 1] for name, states in alignment.items()}
        values[site] = prune_with_matrices(tau, k, pi, column, matrices)
    return values


@pytest.mark.oracle
def test_site_fitch_scores_sum_to_the_fitch_score() -> None:
    alignment, k, _ = _alignment()
    for topology in list(enumerate_topologies(sorted(alignment)))[:5]:
        assert int(site_fitch_scores(topology, alignment).sum()) == fitch_score(
            topology, alignment, k
        )


@pytest.mark.mathematical
def test_least_squares_lengths_are_feasible_and_fit_the_distances() -> None:
    # Non-negative, which is what makes the plug-in value a feasible point of
    # the fit and hence a bound, and no worse a fit than a uniform length.
    alignment, k, _ = _alignment()
    distances = jc_distances(alignment, k)
    topology = next(enumerate_topologies(sorted(alignment)))
    lengths = least_squares_lengths(topology, distances)
    assert bool((lengths >= 0).all())
    uniform = torch.full_like(lengths, 0.1)
    assert least_squares_residual(topology, distances, lengths) < (
        least_squares_residual(topology, distances, uniform)
    )


@pytest.mark.mathematical
def test_tree_features_are_invariant_to_child_order() -> None:
    # Every feature is a function of the unrooted tree, so the same tree with
    # its children swapped at every node is the same feature vector, and the
    # same set of branch tokens.
    alignment, k, pi = _alignment()
    rng = np.random.default_rng(7)
    for topology in list(enumerate_topologies(sorted(alignment)))[:6]:
        shuffled = shuffle_children(topology, rng)
        torch.testing.assert_close(
            tree_features(topology, alignment, k, pi),
            tree_features(shuffled, alignment, k, pi),
        )
        # Rows compared as a sorted multiset: two branches at the same clamped
        # length tie, so a sort on one column alone would not fix the order.
        original = sorted(
            map(tuple, np.round(tree_tokens(topology, alignment, k).numpy(), 9))
        )
        after = sorted(
            map(tuple, np.round(tree_tokens(shuffled, alignment, k).numpy(), 9))
        )
        assert original == after
    assert len(TREE_FEATURE_NAMES) == tree_features(topology, alignment, k, pi).shape[0]


# --- lattices -----------------------------------------------------------------


@pytest.mark.oracle
@pytest.mark.parametrize("shape", [(2, 3), (3, 3)])
def test_mean_field_and_spanning_tree_bounds_sandwich_log_z(
    shape: tuple[int, int],
) -> None:
    # Against enumeration on random-coupling open lattices: the mean-field
    # value never above log Z, the spanning-tree value never below, and each
    # within 0.1 nats per node (0.078 and 0.052 measured over 40 lattices).
    rng = np.random.default_rng(11)
    for _ in range(4):
        graph = _random_lattice(shape, rng)
        field = rng.normal(0.0, 0.4, 3)
        exact = enumerate_potts(graph, field).log_partition
        lower = float(mean_field_log_partition(graph, torch.as_tensor(field)))
        upper = float(spanning_tree_log_partition(graph, torch.as_tensor(field)))
        assert lower <= exact + 1e-9 <= upper + 1e-9
        assert (exact - lower) / graph.n_nodes < 0.1
        assert (upper - exact) / graph.n_nodes < 0.1


@pytest.mark.mathematical
def test_spanning_tree_bound_is_exact_on_a_tree_and_mean_field_without_couplings() -> (
    None
):
    # Where the graph is its own spanning tree the Jensen bound is an equality;
    # where nothing couples, the product distribution is exact.
    chain = lattice_graph((6,), BoundaryCondition.OPEN, 0.8)
    exact = enumerate_potts(chain, FIELD).log_partition
    assert float(spanning_tree_log_partition(chain, torch.as_tensor(FIELD))) == (
        pytest.approx(exact, abs=1e-9)
    )
    independent = PottsGraph(chain.n_nodes, chain.edges, (0.0,) * len(chain.edges))
    exact = enumerate_potts(independent, FIELD).log_partition
    assert float(mean_field_log_partition(independent, torch.as_tensor(FIELD))) == (
        pytest.approx(exact, abs=1e-9)
    )


@pytest.mark.mathematical
def test_lattice_bounds_differentiate_like_finite_differences() -> None:
    graph = lattice_graph((2, 3), BoundaryCondition.PERIODIC, 0.5)
    for bound in (mean_field_log_partition, spanning_tree_log_partition):
        field = torch.tensor(FIELD, dtype=torch.float64, requires_grad=True)
        bound(graph, field).backward()  # type: ignore[no-untyped-call]
        assert field.grad is not None
        for i in range(3):
            step = torch.zeros(3, dtype=torch.float64)
            step[i] = 1e-5
            central = (
                float(bound(graph, field.detach() + step))
                - float(bound(graph, field.detach() - step))
            ) / 2e-5
            assert float(field.grad[i]) == pytest.approx(central, rel=1e-6, abs=1e-8)


@pytest.mark.oracle
def test_ground_state_bracket_contains_the_enumerated_minimum() -> None:
    rng = np.random.default_rng(5)
    for shape in [(2, 3), (3, 3)]:
        graph = _random_lattice(shape, rng)
        field = rng.normal(0.0, 0.4, 3)
        lower, upper = ground_state_energy_bounds(graph, field, 3.0)
        configurations = np.array(
            list(itertools.product(range(3), repeat=graph.n_nodes))
        )
        minimum = -float(log_weights(graph, field, configurations).max())
        assert lower - 1e-9 <= minimum <= upper + 1e-9
        assert (minimum - lower) / graph.n_nodes < 0.1
    features = lattice_features(graph, field)
    assert features[3] * features[4] == pytest.approx(minimum, abs=1e-9)


# --- the seam -----------------------------------------------------------------


class _WrongSide:
    kind = Bound.LOWER

    def __call__(self, structure: object, data: object) -> torch.Tensor:
        return torch.tensor(_identity(structure, data) + 1.0)


@pytest.mark.edge_case
def test_certify_refuses_a_bound_on_the_wrong_side() -> None:
    with pytest.raises(BoundViolation, match="lower bound violated on 3 of 3"):
        certify(_WrongSide(), _identity, [1.0, 2.0, 3.0], None)
    with pytest.raises(ValueError, match="at least one structure"):
        certify(_WrongSide(), _identity, [], None)


@pytest.mark.edge_case
def test_certify_allows_the_stated_violation_rate() -> None:
    # A calibrated bound claims a rate: one violation in four is inside 0.3.
    class _Mostly:
        kind = Bound.LOWER

        def __call__(self, structure: object, data: object) -> torch.Tensor:
            value = _identity(structure, data)
            return torch.tensor(value + 1.0 if value == 4.0 else value - 1.0)

    certificate = certify(
        _Mostly(),
        _identity,
        [1.0, 2.0, 3.0, 4.0],
        None,
        allowed_violation_rate=0.3,
    )
    assert certificate.violations == 1
    with pytest.raises(BoundViolation):
        certify(_Mostly(), _identity, [1.0, 2.0, 3.0, 4.0], None)


@pytest.mark.edge_case
def test_tree_and_lattice_surrogates_refuse_the_wrong_structure() -> None:
    with pytest.raises(TypeError, match="topology and an alignment"):
        PlugInLikelihood(4, np.full(4, 0.25))(object(), {})
    with pytest.raises(ValueError, match="beta must be positive"):
        ground_state_energy_bounds(
            lattice_graph((2, 2), BoundaryCondition.OPEN, 1.0), FIELD, 0.0
        )
    assert FIXTURES_DIR.is_dir()
