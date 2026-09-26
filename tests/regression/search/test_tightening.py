"""The dual bound, asserted where it must hold and measured where it is loose.

Issue #696. Validity is structural and asserted at every iteration: the shares
sum to the energy and are maximized independently. Tightness is a
measurement: tightened is at least as tight as pairwise and neither exceeds
the enumerated ground state; the fraction is in `STATUS.md`.
"""

from __future__ import annotations

import itertools
import math
from itertools import product

import numpy as np
import pytest
from sal.likelihood.message_passing import (
    MessageScheduleName,
    max_product,
    sum_product,
)
from sal.search.tightening import dual_bound
from sal.sim.factor_graph import from_potts
from sal.sim.graph import (
    BoundaryCondition,
    PottsGraph,
    lattice_graph,
    triangular_lattice_graph,
)
from sal.sim.potts import energy

from tests._rows import every_row, every_value

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
def test_the_bound_never_exceeds_the_enumerated_ground_state() -> None:
    # Validity, at one sweep as at twenty: it follows from the decomposition,
    # not from convergence. Both signs of coupling, because the repulsive one
    # is where this earns its place -- minimum cut cannot take it.
    def check(coupling: float, iterations: int) -> None:
        graph = lattice_graph((3, 3), BoundaryCondition.OPEN, coupling)

        certificate = dual_bound(graph, FIELD, max_iterations=iterations)

        assert certificate.bound <= _ground_state(graph) + 1e-9

    every_row(product([0.6, -0.8], [1, 3, 20]), check)


@pytest.mark.oracle
@pytest.mark.potts_lattice
def test_the_decoded_labelling_is_certified_optimal_on_the_square_lattice() -> None:
    # The whole point of a bound rather than an approximation: the gap closes
    # and the labelling is *proved* optimal, with no oracle consulted. The
    # enumeration here checks that claim rather than supplying it.
    def check(coupling: float) -> None:
        graph = lattice_graph((3, 3), BoundaryCondition.OPEN, coupling)

        certificate = dual_bound(graph, FIELD, max_iterations=200)

        assert certificate.optimal
        assert certificate.energy == pytest.approx(_ground_state(graph), abs=1e-9)

    every_value([0.6, -0.8], check)


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
    # The rung below (#734). On the tree the bound is max-product's MAP energy:
    # 8.9e-16 (1e-9, `Certificate.optimal`'s slack). `-log Z / beta` rises to
    # it within `log(k ** n) / beta`: gaps 3.7e-2 at beta 10, 4.5e-7 at 100
    # (bounds 0.659, 0.066). On the loopy 3x3 the bound is below the Bethe
    # decoding's energy: slack 0.0 at J = 0.4, 9.0 at J = -0.8.
    assignment, _ = max_product(from_potts(TREE, FIELD))
    decoded = np.array(
        [assignment[f"s{site}"] for site in range(TREE.n_nodes)], dtype=np.int64
    )
    map_energy = energy(TREE, FIELD, decoded)

    certificate = dual_bound(TREE, FIELD, max_iterations=200)

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
            dual_bound(loopy, FIELD, max_iterations=200).bound
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
        certificate = dual_bound(graph, FIELD, max_iterations=iterations)
        assert certificate.gap >= -1e-9


@pytest.mark.analytic
@pytest.mark.potts_lattice
def test_the_tolerance_is_the_one_it_hard_coded_and_a_looser_one_settles_sooner() -> (
    None
):
    # Issue #1091 lifted the 1e-12 out of the loop: at that value the run is
    # the default's, bitwise; a looser one settles in no more sweeps, and
    # the bound it certifies stays a lower bound.
    graph = triangular_lattice_graph((3, 3), BoundaryCondition.OPEN, -0.8)
    ground = _ground_state(graph)

    default = dual_bound(graph, FIELD, max_iterations=300)
    spelled = dual_bound(graph, FIELD, max_iterations=300, tolerance=1e-12)
    loose = dual_bound(graph, FIELD, max_iterations=300, tolerance=1e-3)

    assert spelled.bound == default.bound
    assert spelled.iterations == default.iterations
    assert loose.iterations <= default.iterations
    assert loose.bound <= ground + 1e-9


@pytest.mark.oracle
@pytest.mark.frustrated_lattice
def test_triangles_tighten_what_the_pairwise_relaxation_cannot_see() -> None:
    # The ticket's thesis, on the instance that motivates it. The pairwise
    # relaxation believes every edge can be satisfied at every site's preferred
    # label, which no labelling achieves -- so its bound does not even depend
    # on the coupling. Adding the triangles is what sees the odd cycle.
    def check(coupling: float) -> None:
        graph = triangular_lattice_graph((3, 3), BoundaryCondition.OPEN, coupling)
        ground = _ground_state(graph)

        pairwise = dual_bound(graph, FIELD, max_iterations=300)
        tightened = dual_bound(
            graph, FIELD, max_iterations=300, plaquettes=_triangles(graph)
        )

        assert pairwise.bound <= ground + 1e-9
        assert tightened.bound <= ground + 1e-9
        assert tightened.bound > pairwise.bound

    every_value([-0.8, -1.5], check)


@pytest.mark.analytic
@pytest.mark.frustrated_lattice
def test_the_pairwise_bound_does_not_depend_on_the_coupling_here() -> None:
    # A prediction: three states color every triangle, so pairwise pays
    # nothing; only a triangle constraint charges the frustration.
    weak = triangular_lattice_graph((3, 3), BoundaryCondition.OPEN, -0.8)
    strong = triangular_lattice_graph((3, 3), BoundaryCondition.OPEN, -1.5)

    assert dual_bound(weak, FIELD, max_iterations=300).bound == pytest.approx(
        dual_bound(strong, FIELD, max_iterations=300).bound, rel=1e-12
    )
