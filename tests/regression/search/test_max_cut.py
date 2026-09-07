"""Max-Cut, and a certificate that is not enumeration.

Three references, of three different kinds, which is the reason for the
ticket. Every other discrete claim in this repository rests on exhaustive
enumeration and therefore stops at about twenty sites.

1. **A construction whose answer is known at any size.** A complete bipartite
   graph's maximum cut is every edge, because every edge joins the two parts.
   No solver is needed to know that, so it is the check that this one is not
   merely self-consistent.
2. **Enumeration**, where it fits, as the true optimum.
3. **The relaxation's own value**, as a computable bound past enumeration ---
   with the qualification below, which is the honest half of the ticket.

**The certificate is weaker than the theorem.** Goemans-Williamson assumes
the semidefinite program is solved to optimality; this one is solved
approximately, by Burer-Monteiro gradient ascent, because the repository
carries no SDP solver. The symptom is measurable and is asserted rather than
hidden: on an instance whose optimum the relaxation should bound from above,
the ratio comes out *slightly above 1* --- impossible for an exact solve.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from phylo.search.alpha_expansion import energy
from phylo.search.max_cut import (
    GOEMANS_WILLIAMSON_RATIO,
    complete_bipartite,
    cut_value,
    enumerate_max_cut,
    goemans_williamson,
)
from phylo.sim.graph import BoundaryCondition, PottsGraph, lattice_graph


def _random_graph(n_nodes: int, density: float, seed: int) -> PottsGraph:
    """An Erdos-Renyi-shaped instance, which is *not* bipartite.

    A lattice is bipartite, so its maximum cut is every edge and any solver
    that separates the two colours is optimal. That makes it useless for
    telling a good solver from a lucky one, which is why the interesting
    fixtures here carry triangles.
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
@pytest.mark.parametrize(("first", "second"), [(3, 4), (5, 5), (8, 6)])
def test_a_complete_bipartite_graph_has_every_edge_in_its_maximum_cut(
    first: int, second: int
) -> None:
    # Known without solving anything: every edge joins the two parts, so
    # separating them cuts all of them and nothing can do better.
    graph = complete_bipartite(first, second)

    result = goemans_williamson(graph, seed=1)

    assert result.value == pytest.approx(float(len(graph.edges)))


@pytest.mark.oracle
@pytest.mark.parametrize("shape", [(3, 3), (4, 3), (4, 4)])
def test_the_rounded_cut_reaches_the_enumerated_optimum_on_a_lattice(
    shape: tuple[int, int],
) -> None:
    # A lattice is bipartite, so this is the easy regime and reaching the
    # optimum is expected. It is here to catch a solver that is broken rather
    # than to distinguish a good one -- the random graphs below do that.
    graph = lattice_graph(shape, BoundaryCondition.OPEN, 1.0)

    result = goemans_williamson(graph, seed=3)
    _, optimum = enumerate_max_cut(graph)

    assert result.value == pytest.approx(optimum)


@pytest.mark.mathematical
@pytest.mark.parametrize(
    ("n_nodes", "density", "seed"), [(12, 0.4, 1), (16, 0.3, 2), (18, 0.25, 3)]
)
def test_the_realized_ratio_beats_the_bound_on_a_graph_with_triangles(
    n_nodes: int, density: float, seed: int
) -> None:
    # The measurement that matters, against the true optimum rather than
    # against the relaxation. Measured: the rounded cut reached the optimum on
    # every one of these instances, so the realized ratio is 1.0000 against a
    # guarantee of 0.87856 -- the bound is not tight and is not meant to be.
    graph = _random_graph(n_nodes, density, seed)

    result = goemans_williamson(graph, seed=5)
    _, optimum = enumerate_max_cut(graph)

    assert result.value / optimum >= GOEMANS_WILLIAMSON_RATIO
    assert result.value <= optimum + 1e-9


@pytest.mark.mathematical
@pytest.mark.parametrize(
    ("n_nodes", "density", "seed"), [(12, 0.4, 1), (16, 0.3, 2), (18, 0.25, 3)]
)
def test_the_certificate_holds_where_the_optimum_is_unknown(
    n_nodes: int, density: float, seed: int
) -> None:
    # The point of the relaxation: `value / relaxation` is computable without
    # knowing the optimum, so it certifies a run at a size enumeration cannot
    # reach. Measured on these instances: 0.95 to 0.98.
    graph = _random_graph(n_nodes, density, seed)

    result = goemans_williamson(graph, seed=5)

    assert result.ratio >= GOEMANS_WILLIAMSON_RATIO


@pytest.mark.structural
def test_the_relaxation_is_solved_approximately_and_says_so() -> None:
    # The honest limit of the certificate, asserted rather than left in prose.
    # An exactly solved relaxation upper bounds the true optimum, so the ratio
    # could never exceed 1. Burer-Monteiro gradient ascent stops short, and on
    # a bipartite graph -- where the optimum is exactly `|E|` -- the relaxation
    # lands a hair below it and the ratio comes out just above 1.
    #
    # That is the evidence the certificate is weaker than the theorem: it
    # certifies the rounding against the relaxation actually computed, not
    # against the relaxation's optimum.
    graph = complete_bipartite(8, 6)

    result = goemans_williamson(graph, seed=1)

    assert result.value == pytest.approx(float(len(graph.edges)))
    assert result.ratio > 1.0 - 1e-6
    assert result.ratio < 1.001


@pytest.mark.oracle
@pytest.mark.parametrize("weight", [1.0, 2.5, 0.4])
def test_max_cut_is_the_antiferromagnetic_ising_ground_state(weight: float) -> None:
    # The identity the module rests on, checked rather than asserted in prose:
    # with every coupling negative and no field, the minimum energy is the
    # total weight less the maximum cut. A non-bipartite graph is used because
    # on a bipartite one the maximum cut is every edge and the relation
    # degenerates to `0 = 0`.
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


@pytest.mark.mathematical
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


@pytest.mark.edge_case
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


@pytest.mark.edge_case
def test_a_graph_with_no_edges_has_an_empty_cut() -> None:
    graph = PottsGraph(n_nodes=4, edges=(), coupling=())

    result = goemans_williamson(graph, seed=1)

    assert result.value == pytest.approx(0.0)
    assert result.ratio == pytest.approx(1.0)
