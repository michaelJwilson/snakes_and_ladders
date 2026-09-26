"""Max-Cut, and a certificate that is not enumeration.

References: a complete bipartite graph (the cut is every edge, at any size);
enumeration where it fits; and the relaxation's value as a bound past it. The
certificate is weaker than the theorem: Goemans-Williamson assumes an exact
SDP, and Burer-Monteiro ascent solves it approximately (no SDP solver is
carried), so on a bipartite graph the ratio comes out slightly above 1,
asserted.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
import torch
from sal.search.alpha_expansion import alpha_expansion
from sal.search.max_cut import (
    GOEMANS_WILLIAMSON_RATIO,
    complete_bipartite,
    cut_value,
    enumerate_max_cut,
    goemans_williamson,
)
from sal.sim.canonical import frustrated_triangular_lattice
from sal.sim.graph import BoundaryCondition, PottsGraph, lattice_graph
from sal.sim.potts import energy

from tests._rows import every_row, every_value


def _random_graph(n_nodes: int, density: float, seed: int) -> PottsGraph:
    """An Erdos-Renyi-shaped instance, which is *not* bipartite.

    On a bipartite lattice any colour-separating solver is optimal.
    """
    rng = np.random.default_rng(seed)
    edges = tuple(
        (first, second)
        for first in range(n_nodes)
        for second in range(first + 1, n_nodes)
        if rng.random() < density
    )
    return PottsGraph(n_nodes=n_nodes, edges=edges, coupling=(1.0,) * len(edges))


@pytest.mark.oracle
def test_a_complete_bipartite_graph_has_every_edge_in_its_maximum_cut() -> None:
    # Known without solving anything: every edge joins the two parts, so
    # separating them cuts all of them and nothing can do better.
    def check(first: int, second: int) -> None:
        graph = complete_bipartite(first, second)

        result = goemans_williamson(graph, generator=torch.Generator().manual_seed(1))

        assert result.value == pytest.approx(float(len(graph.edges)))

    every_row([(3, 4), (5, 5)], check)


@pytest.mark.oracle
def test_the_rounded_cut_reaches_the_enumerated_optimum_on_a_lattice() -> None:
    # A lattice is bipartite, so this is the easy regime and reaching the
    # optimum is expected. It is here to catch a solver that is broken rather
    # than to distinguish a good one -- the random graphs below do that.
    def check(shape: tuple[int, int]) -> None:
        graph = lattice_graph(shape, BoundaryCondition.OPEN, 1.0)

        result = goemans_williamson(graph, generator=torch.Generator().manual_seed(3))
        _, optimum = enumerate_max_cut(graph)

        assert result.value == pytest.approx(optimum)

    every_value([(3, 3), (4, 3), (4, 4)], check)


@pytest.mark.analytic
def test_the_realized_ratio_beats_the_bound_on_a_graph_with_triangles() -> None:
    # The measurement that matters, against the true optimum rather than
    # against the relaxation. Measured: the rounded cut reached the optimum on
    # every one of these instances, so the realized ratio is 1.0000 against a
    # guarantee of 0.87856 -- the bound is not tight and is not meant to be.
    def check(n_nodes: int, density: float, seed: int) -> None:
        graph = _random_graph(n_nodes, density, seed)

        result = goemans_williamson(graph, generator=torch.Generator().manual_seed(5))
        _, optimum = enumerate_max_cut(graph)

        assert result.value / optimum >= GOEMANS_WILLIAMSON_RATIO
        assert result.value <= optimum + 1e-9

    every_row([(12, 0.4, 1), (16, 0.3, 2), (18, 0.25, 3)], check)


@pytest.mark.analytic
def test_the_certificate_holds_where_the_optimum_is_unknown() -> None:
    # The point of the relaxation: `value / relaxation` is computable without
    # knowing the optimum, so it certifies a run at a size enumeration cannot
    # reach. Measured on these instances: 0.95 to 0.98.
    def check(n_nodes: int, density: float, seed: int) -> None:
        graph = _random_graph(n_nodes, density, seed)

        result = goemans_williamson(graph, generator=torch.Generator().manual_seed(5))

        assert result.ratio >= GOEMANS_WILLIAMSON_RATIO

    every_row([(12, 0.4, 1), (16, 0.3, 2), (18, 0.25, 3)], check)


@pytest.mark.smoke
def test_the_relaxation_is_solved_approximately_and_says_so() -> None:
    # An exact relaxation bounds the optimum `|E|`, so a ratio above 1 shows the
    # ascent stopped short: the rounding is certified against the relaxation
    # computed, not its optimum.
    graph = complete_bipartite(8, 6)

    result = goemans_williamson(graph, generator=torch.Generator().manual_seed(1))

    assert result.value == pytest.approx(float(len(graph.edges)))
    assert result.ratio > 1.0 - 1e-6
    assert result.ratio < 1.001


@pytest.mark.oracle
def test_max_cut_is_the_antiferromagnetic_ising_ground_state() -> None:
    # All-negative, no field: minimum energy = total weight - max cut. Not
    # bipartite, where it degenerates to `0 = 0`.
    def check(weight: float) -> None:
        positive = _random_graph(10, 0.45, 4)
        positive = PottsGraph(
            n_nodes=positive.n_nodes,
            edges=positive.edges,
            coupling=(weight,) * len(positive.edges),
        )
        negative = PottsGraph(
            n_nodes=positive.n_nodes,
            edges=positive.edges,
            coupling=(-weight,) * len(positive.edges),
        )
        field = np.zeros((positive.n_nodes, 2))

        minimum = min(
            energy(negative, field, np.array(assignment, dtype=np.int64))
            for assignment in itertools.product(range(2), repeat=positive.n_nodes)
        )
        _, maximum_cut = enumerate_max_cut(positive)

        assert minimum == pytest.approx(
            weight * len(positive.edges) - maximum_cut, abs=1e-12
        )
        # And the instance is genuinely non-bipartite, so the relation is not
        # degenerate: some edge is uncut at the optimum.
        assert maximum_cut < weight * len(positive.edges)

    every_value([1.0, 2.5, 0.4], check)


def _bipartite(left: int, right: int, seed: int) -> tuple[PottsGraph, np.ndarray]:
    """A connected bipartite graph with drawn per-edge weights, and each node's side."""
    rng = np.random.default_rng(seed)
    edges, weights = [], []
    for first in range(left):
        for second in range(left, left + right):
            if rng.random() < 0.6 or first == 0 or second == left:
                edges.append((first, second))
                weights.append(float(rng.uniform(0.5, 2.0)))
    graph = PottsGraph(
        n_nodes=left + right, edges=tuple(edges), coupling=tuple(weights)
    )
    return graph, np.array([0] * left + [1] * right, dtype=np.int64)


@pytest.mark.oracle
@pytest.mark.critical
def test_the_rounded_cut_is_the_gauged_alpha_expansion_optimum_at_two_labels() -> None:
    # The rung below (#734). Max-Cut is the antiferromagnetic ground state,
    # which `alpha_expansion` refuses (asserted); on a bipartite graph a gauge
    # flip makes it ferromagnetic. Realized: expansion -17.09743549305819,
    # rounded cut 17.09743549305819, 0.0 (1e-12), equal up to the global flip.
    # Frustrated: the rounded cut is the enumerated 18.0, ratio 0.8898 against
    # the guaranteed 0.87856.
    graph, side = _bipartite(4, 5, seed=17)
    total = float(graph.edge_coupling.sum())
    start = np.random.default_rng(3).integers(0, 2, size=graph.n_nodes)

    expansion = alpha_expansion(graph, np.zeros(2), start=start, n_states=2)
    rounded = goemans_williamson(graph, generator=torch.Generator().manual_seed(5))

    # The gauge: the expansion agrees across every edge, and flipping one side
    # of the bipartition turns that labelling into a cut of the same weight.
    gauged = expansion.labelling.astype(np.int64) ^ side
    assert expansion.energy == pytest.approx(-total, abs=1e-12)
    assert cut_value(graph, gauged) == pytest.approx(-expansion.energy, abs=1e-12)
    assert rounded.value == pytest.approx(-expansion.energy, abs=1e-12)
    assert np.array_equal(rounded.assignment, gauged) or np.array_equal(
        1 - rounded.assignment, gauged
    )

    # The boundary: the frustrated instance the gauge cannot reach.
    frustrated = frustrated_triangular_lattice((3, 3), BoundaryCondition.PERIODIC, -1.0)
    with pytest.raises(ValueError, match="every coupling must be non-negative"):
        alpha_expansion(frustrated, np.zeros(2), n_states=2)

    positive = PottsGraph(
        n_nodes=frustrated.n_nodes,
        edges=frustrated.edges,
        coupling=tuple(abs(value) for value in frustrated.coupling),
    )
    certified = goemans_williamson(positive, generator=torch.Generator().manual_seed(5))
    _, optimum = enumerate_max_cut(positive)

    assert certified.value == pytest.approx(optimum)
    assert certified.ratio > GOEMANS_WILLIAMSON_RATIO


@pytest.mark.analytic
def test_the_cut_does_not_depend_on_the_sign_of_the_coupling() -> None:
    # A graph written with the antiferromagnetic sign the problem corresponds
    # to and one written with positive weights are the same instance.
    positive = lattice_graph((3, 3), BoundaryCondition.OPEN, 1.0)
    negative = PottsGraph(
        n_nodes=positive.n_nodes,
        edges=positive.edges,
        coupling=(-1.0,) * len(positive.edges),
    )
    assignment = np.array(
        [node % 2 for node in range(positive.n_nodes)], dtype=np.int64
    )

    assert cut_value(positive, assignment) == cut_value(negative, assignment)


@pytest.mark.smoke
def test_enumeration_refuses_a_size_it_cannot_do() -> None:
    graph = lattice_graph((5, 5), BoundaryCondition.OPEN, 1.0)

    with pytest.raises(
        ValueError, match=r"refusing to enumerate .*2\*\*25 assignments"
    ):
        enumerate_max_cut(graph)


@pytest.mark.oracle
def test_enumeration_fixes_one_side_without_losing_the_optimum() -> None:
    # Complementing an assignment gives the same cut, so half the search space
    # is redundant. Fixing the last node is exact rather than a heuristic --
    # checked against the unrestricted search on a small instance.
    graph = _random_graph(9, 0.5, 6)

    _, restricted = enumerate_max_cut(graph)
    unrestricted = max(
        cut_value(graph, np.array(assignment, dtype=np.int64))
        for assignment in itertools.product(range(2), repeat=graph.n_nodes)
    )

    assert restricted == pytest.approx(unrestricted)


@pytest.mark.smoke
def test_a_graph_with_no_edges_has_an_empty_cut() -> None:
    graph = PottsGraph(n_nodes=4, edges=(), coupling=())

    result = goemans_williamson(graph, generator=torch.Generator().manual_seed(1))

    assert result.value == pytest.approx(0.0)
    assert result.ratio == pytest.approx(1.0)
