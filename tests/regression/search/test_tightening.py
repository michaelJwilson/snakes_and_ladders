"""The dual bound, asserted where it must hold and measured where it is loose.

Issue #696. Two kinds of claim, and conflating them is the mistake this file
avoids.

**Validity is structural and is asserted at every iteration**, including after
one sweep, because the bound follows from the decomposition and not from
convergence: the shares sum to the energy, each is maximized independently, so
nothing can beat the total. A test that only checked the converged bound would
pass on an implementation that is wrong until it converges.

**Tightness is a measurement.** How much the triangles close is a property of
the instance, so what is asserted is the ordering --- tightened is at least as
tight as pairwise, and neither exceeds the enumerated ground state --- with the
fraction reported in `STATUS.md`.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
from snakes_and_ladders.likelihood.message_passing import (
    MessageScheduleName,
    max_product,
    sum_product,
)
from snakes_and_ladders.search.tightening import dual_bound
from snakes_and_ladders.sim.factor_graph import from_potts
from snakes_and_ladders.sim.graph import (
    BoundaryCondition,
    PottsGraph,
    lattice_graph,
    triangular_lattice_graph,
)
from snakes_and_ladders.sim.potts import energy

FIELD = np.log(np.array([0.5, 0.35, 0.15]))

#: The tree `test_message_passing.py` and `test_belief_propagation.py` both
#: run on: six nodes, five edges, couplings of mixed sign.
TREE = PottsGraph(
    n_nodes=6,
    edges=((0, 1), (0, 2), (1, 3), (1, 4), (2, 5)),
    coupling=(0.8, -0.4, 1.2, 0.3, 0.9),
)


def _ground_state(graph: PottsGraph) -> float:
    """The exact minimum, by exhaustive enumeration of every labelling."""
    return min(
        energy(graph, FIELD, np.asarray(labelling))
        for labelling in itertools.product(range(3), repeat=graph.n_nodes)
    )


def _triangles(graph: PottsGraph) -> tuple[tuple[int, ...], ...]:
    adjacency: dict[int, set[int]] = {i: set() for i in range(graph.n_nodes)}
    for left, right in graph.edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    return tuple(
        sorted(
            {
                tuple(sorted((first, second, third)))
                for first in range(graph.n_nodes)
                for second in adjacency[first]
                for third in adjacency[first] & adjacency[second]
            }
        )
    )


@pytest.mark.oracle
@pytest.mark.potts_lattice
@pytest.mark.parametrize("coupling", [0.6, -0.8])
@pytest.mark.parametrize("iterations", [1, 3, 20])
def test_the_bound_never_exceeds_the_enumerated_ground_state(
    coupling: float, iterations: int
) -> None:
    # Validity, at one sweep as at twenty: it follows from the decomposition,
    # not from convergence. Both signs of coupling, because the repulsive one
    # is where this earns its place -- minimum cut cannot take it.
    graph = lattice_graph((3, 3), BoundaryCondition.OPEN, coupling)

    certificate = dual_bound(graph, FIELD, iterations=iterations)

    assert certificate.bound <= _ground_state(graph) + 1e-9


@pytest.mark.oracle
@pytest.mark.potts_lattice
@pytest.mark.parametrize("coupling", [0.6, -0.8])
def test_the_decoded_labelling_is_certified_optimal_on_the_square_lattice(
    coupling: float,
) -> None:
    # The whole point of a bound rather than an approximation: the gap closes
    # and the labelling is *proved* optimal, with no oracle consulted. The
    # enumeration here checks that claim rather than supplying it.
    graph = lattice_graph((3, 3), BoundaryCondition.OPEN, coupling)

    certificate = dual_bound(graph, FIELD, iterations=200)

    assert certificate.optimal
    assert certificate.energy == pytest.approx(_ground_state(graph), abs=1e-9)


def _cooled(graph: PottsGraph, beta: float) -> PottsGraph:
    """The same model at inverse temperature ``beta``: every coupling scaled."""
    return PottsGraph(
        n_nodes=graph.n_nodes,
        edges=graph.edges,
        coupling=tuple(beta * value for value in graph.coupling),
    )


@pytest.mark.oracle
@pytest.mark.critical
@pytest.mark.potts_lattice
def test_the_dual_bound_is_the_zero_temperature_belief_propagation_energy() -> None:
    # The rung below (issue #734). Sum-product is exact on a tree and the
    # pairwise relaxation is tight there, so the two meet at zero temperature
    # and the relation pinned is an equality; on the loopy lattice neither is
    # exact and what survives is the inequality the bound is for.
    #
    # Three statements, in that order.
    #
    # 1. On the tree the bound is the max-product MAP energy -- belief
    #    propagation's ground state, exact where the graph is a tree.
    #    Realized |bound - energy| 8.9e-16 against the 1e-9 declared, which is
    #    the slack `Certificate.optimal` itself reads.
    # 2. The same number as a limit rather than an argmax: `-log Z / beta`
    #    from sum-product on the cooled model rises to the bound from below,
    #    and cannot be further below it than the entropy `log(k ** n) / beta`.
    #    Realized gaps 3.7e-2 at beta = 10 and 4.5e-7 at beta = 100, inside
    #    bounds of 0.659 and 0.066.
    # 3. On the 3x3 lattice sum-product is the Bethe approximation and its
    #    decoded labelling is a labelling like any other, so the bound is
    #    below its energy. Realized slack 0.0 at J = 0.4 -- the decoding is
    #    optimal there and the bound certifies it -- and 9.0 at J = -0.8,
    #    where the frustrated marginals decode to a labelling the bound
    #    rejects.
    assignment, _ = max_product(from_potts(TREE, FIELD))
    decoded = np.array(
        [assignment[f"s{site}"] for site in range(TREE.n_nodes)], dtype=np.int64
    )
    map_energy = energy(TREE, FIELD, decoded)

    certificate = dual_bound(TREE, FIELD, iterations=200)

    assert certificate.optimal
    assert certificate.bound == pytest.approx(map_energy, abs=1e-9)

    entropy = math.log(FIELD.shape[0] ** TREE.n_nodes)
    for beta in (10.0, 100.0):
        cooled = sum_product(from_potts(_cooled(TREE, beta), beta * FIELD))
        free = -cooled.log_partition / beta
        assert cooled.exact
        assert free <= certificate.bound + 1e-9
        assert free >= certificate.bound - entropy / beta

    for coupling in (0.4, -0.8):
        loopy = lattice_graph((3, 3), BoundaryCondition.OPEN, coupling)
        marginals = sum_product(
            from_potts(loopy, FIELD),
            schedule=MessageScheduleName.FLOODING,
            tolerance=1e-12,
        )
        beliefs = np.stack(
            [marginals.variable[f"s{site}"] for site in range(loopy.n_nodes)]
        )
        bethe = np.asarray(beliefs.argmax(axis=1), dtype=np.int64)

        assert not marginals.exact
        assert (
            dual_bound(loopy, FIELD, iterations=200).bound
            <= energy(loopy, FIELD, bethe) + 1e-9
        )


@pytest.mark.analytic
@pytest.mark.potts_lattice
def test_a_decoded_labelling_is_never_better_than_the_bound() -> None:
    # The two sides of the interval, in the right order: the energy of a
    # labelling that exists is an upper bound on the minimum, the dual a lower
    # one, and a gap that came out negative would mean one of them is wrong.
    graph = triangular_lattice_graph((3, 3), BoundaryCondition.OPEN, -0.8)

    for iterations in (1, 5, 50):
        certificate = dual_bound(graph, FIELD, iterations=iterations)
        assert certificate.gap >= -1e-9


@pytest.mark.oracle
@pytest.mark.frustrated_lattice
@pytest.mark.parametrize("coupling", [-0.8, -1.5])
def test_triangles_tighten_what_the_pairwise_relaxation_cannot_see(
    coupling: float,
) -> None:
    # The ticket's thesis, on the instance that motivates it. The pairwise
    # relaxation believes every edge can be satisfied at every site's preferred
    # label, which no labelling achieves -- so its bound does not even depend
    # on the coupling. Adding the triangles is what sees the odd cycle.
    graph = triangular_lattice_graph((3, 3), BoundaryCondition.OPEN, coupling)
    ground = _ground_state(graph)

    pairwise = dual_bound(graph, FIELD, iterations=300)
    tightened = dual_bound(graph, FIELD, iterations=300, plaquettes=_triangles(graph))

    assert pairwise.bound <= ground + 1e-9
    assert tightened.bound <= ground + 1e-9
    assert tightened.bound > pairwise.bound


@pytest.mark.analytic
@pytest.mark.frustrated_lattice
def test_the_pairwise_bound_does_not_depend_on_the_coupling_here() -> None:
    # Stated as its own test because it is the clearest statement of what the
    # relaxation misses, and because it is a prediction rather than an
    # observation: with three states every triangle is 3-colourable, so the
    # relaxation can satisfy every edge at no cost whatever the coupling, and
    # only a constraint over the triangle can charge for the frustration.
    weak = triangular_lattice_graph((3, 3), BoundaryCondition.OPEN, -0.8)
    strong = triangular_lattice_graph((3, 3), BoundaryCondition.OPEN, -1.5)

    assert dual_bound(weak, FIELD, iterations=300).bound == pytest.approx(
        dual_bound(strong, FIELD, iterations=300).bound, rel=1e-12
    )
