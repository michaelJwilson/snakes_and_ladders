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
from dataclasses import replace

import numpy as np
import pytest
import torch
from sal.bound import Bound, BoundViolation, Certificate, certify
from sal.learn.ranking import shuffle_children
from sal.likelihood.brute_force import brute_force_log_likelihood
from sal.likelihood.features import (
    TREE_FEATURE_NAMES,
    tree_features,
    tree_tokens,
)
from sal.likelihood.parsimony import fitch_score
from sal.likelihood.potts import enumerate_potts, log_weights
from sal.likelihood.pruning.torch import branch_order, log_likelihood
from sal.likelihood.surrogate import (
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
from sal.search.alpha_expansion import alpha_expansion
from sal.search.infer import score_topology
from sal.sim.graph import BoundaryCondition, PottsGraph, lattice_graph
from sal.sim.jc import jc_transition_probabilities
from sal.sim.simulator import simulate_tree
from sal.sim.topology import enumerate_topologies
from sal.sim.tree import Node, preorder

from tests._fixtures import FIXTURES_DIR, load_fixture
from tests._rows import every_value

FIVE_TAXA = "tree_search/ci.yaml"
N_SITES = 200
FIELD = np.array([0.3, -0.2, 0.1])


def _alignment(n_sites: int = N_SITES) -> tuple[dict[str, np.ndarray], int, np.ndarray]:
    params = load_fixture(FIVE_TAXA)
    dataset = simulate_tree(params, np.random.default_rng(params.seed), n_sites=n_sites)
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


@pytest.mark.oracle
@pytest.mark.analytic
def test_parsimony_bound_is_the_maximum_over_the_vertices() -> None:
    # Multilinear per branch under JC, so the maximum over lengths is at a
    # vertex: over the 2^8 vertices it is pi(x_1) k^{-F_s} per site, and
    # random lengths never exceed it.
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


def _scaled(tau: Node, factor: float) -> Node:
    """``tau`` with every branch length multiplied by ``factor``."""
    return replace(
        tau,
        branch_length=(
            None if tau.branch_length is None else tau.branch_length * factor
        ),
        children=tuple(_scaled(child, factor) for child in tau.children),
    )


@pytest.mark.oracle
@pytest.mark.critical
def test_prune_with_matrices_is_the_enumeration_at_the_jukes_cantor_matrices() -> None:
    """Handed `P(t)` per branch, the arbitrary-matrix recursion is the JC likelihood.

    `brute_force_log_likelihood`, 3 scalings: 3.29e-16, 1.87e-16, 0.0 (1e-12; #717).
    """
    alignment, k, pi = _alignment()
    params = load_fixture(FIVE_TAXA)
    short = {name: states[:60] for name, states in alignment.items()}
    for factor in (0.25, 1.0, 3.0):
        tau = _scaled(params.tau, factor)
        matrices = {
            node.name: jc_transition_probabilities(node.branch_length, k)
            for node in preorder(tau)
            if node.branch_length is not None
        }
        assert prune_with_matrices(tau, k, pi, short, matrices) == pytest.approx(
            brute_force_log_likelihood(tau, k, pi, short), rel=1e-12
        )


@pytest.mark.critical
@pytest.mark.oracle
def test_site_fitch_scores_sum_to_the_fitch_score() -> None:
    alignment, k, _ = _alignment()
    for topology in list(enumerate_topologies(sorted(alignment)))[:5]:
        assert int(site_fitch_scores(topology, alignment).sum()) == fitch_score(
            topology, alignment, k
        )


@pytest.mark.analytic
def test_least_squares_lengths_are_feasible_and_fit_the_distances() -> None:
    # Non-negative, which is what makes the plug-in value a feasible point of
    # the fit and hence a bound, and no worse a fit than a uniform length.
    alignment, k, _ = _alignment()
    distances = jc_distances(alignment, k)
    topology = next(enumerate_topologies(sorted(alignment)))
    lengths = least_squares_lengths(topology, distances)
    assert bool((lengths >= 0).all())
    uniform = np.full_like(lengths, 0.1)
    assert least_squares_residual(topology, distances, lengths) < (
        least_squares_residual(topology, distances, uniform)
    )


@pytest.mark.bug
@pytest.mark.smoke
def test_least_squares_lengths_are_bitwise_across_repeated_calls() -> None:
    # MKL `gelsy` gave 42-ulp-different solutions in 1,000 calls and moved 9
    # of 15 topologies (#1026); NumPy `gelsd` gives one answer in 100 repeats.
    alignment, k, pi = _alignment()
    distances = jc_distances(alignment, k)
    bound = PlugInLikelihood(k, pi)
    for topology in enumerate_topologies(sorted(alignment)):
        lengths = {
            least_squares_lengths(topology, distances).tobytes() for _ in range(100)
        }
        assert len(lengths) == 1
        values = {float(bound(topology, alignment)) for _ in range(10)}
        assert len(values) == 1


@pytest.mark.analytic
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
            map(tuple, np.round(tree_tokens(topology, alignment, k), 9).tolist())
        )
        after = sorted(
            map(tuple, np.round(tree_tokens(shuffled, alignment, k), 9).tolist())
        )
        assert original == after
    assert len(TREE_FEATURE_NAMES) == tree_features(topology, alignment, k, pi).shape[0]


# --- lattices -----------------------------------------------------------------


@pytest.mark.critical
@pytest.mark.oracle
def test_mean_field_and_spanning_tree_bounds_sandwich_log_z() -> None:
    # Against enumeration on random-coupling open lattices: the mean-field
    # value never above log Z, the spanning-tree value never below, and each
    # within 0.1 nats per node (0.078 and 0.052 measured over 40 lattices).
    def check(shape: tuple[int, int]) -> None:
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

    every_value([(2, 3), (3, 3)], check)


@pytest.mark.oracle
@pytest.mark.analytic
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


@pytest.mark.analytic
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


@pytest.mark.critical
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
    expansion = alpha_expansion(
        graph,
        np.tile(field, (graph.n_nodes, 1)),
        3,
        start=np.zeros(graph.n_nodes, dtype=np.int64),
    )
    assert expansion.energy == pytest.approx(minimum, abs=1e-9)


# --- the seam -----------------------------------------------------------------


class _WrongSide:
    kind = Bound.LOWER

    def __call__(self, structure: object, data: object) -> float:
        return _identity(structure, data) + 1.0


@pytest.mark.smoke
def test_certify_refuses_a_bound_on_the_wrong_side() -> None:
    with pytest.raises(BoundViolation, match="lower bound violated on 3 of 3"):
        certify(_WrongSide(), _identity, [1.0, 2.0, 3.0], None)
    with pytest.raises(ValueError, match="at least one structure"):
        certify(_WrongSide(), _identity, [], None)


@pytest.mark.smoke
def test_certify_allows_the_stated_violation_rate() -> None:
    # A calibrated bound claims a rate: one violation in four is inside 0.3.
    class _Mostly:
        kind = Bound.LOWER

        def __call__(self, structure: object, data: object) -> float:
            value = _identity(structure, data)
            return value + 1.0 if value == 4.0 else value - 1.0

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


@pytest.mark.smoke
def test_tree_and_lattice_surrogates_refuse_the_wrong_structure() -> None:
    with pytest.raises(TypeError, match="topology and an alignment"):
        PlugInLikelihood(4, np.full(4, 0.25))(object(), {})
    with pytest.raises(ValueError, match="beta must be positive"):
        ground_state_energy_bounds(
            lattice_graph((2, 2), BoundaryCondition.OPEN, 1.0), FIELD, 0.0
        )
    assert FIXTURES_DIR.is_dir()
