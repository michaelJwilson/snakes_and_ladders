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

import numpy as np
import pytest
from snakes_and_ladders.search.tightening import dual_bound
from snakes_and_ladders.sim.graph import (
    BoundaryCondition,
    PottsGraph,
    lattice_graph,
    triangular_lattice_graph,
)
from snakes_and_ladders.sim.potts import energy

FIELD = np.log(np.array([0.5, 0.35, 0.15]))


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


@pytest.mark.mathematical
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


@pytest.mark.mathematical
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
