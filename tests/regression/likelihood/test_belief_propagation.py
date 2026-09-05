"""Belief propagation, asserted where it is exact and measured where it is not.

Two regimes, two different kinds of test, and conflating them is the mistake
this file exists to avoid.

On a tree BP is exact, so equality against exhaustive enumeration is the right
assertion and carries the correctness claim: a wrong sign in the Bethe free
energy, an off-by-one in the degree correction, or a message that failed to
exclude its own reverse all fail here.

On a loopy lattice BP is approximate. Asserting agreement would assert
something false, and asserting only that it ran would be coverage theatre
(root `CLAUDE.md`). What is asserted instead is the structure the physics
requires --- exact at zero coupling, and a deviation that grows away from it
--- with the size of the error reported as a measurement against the exact
strip transfer matrix.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from phylo.likelihood.belief_propagation import (
    ConvergenceError,
    belief_propagation,
)
from phylo.likelihood.potts import enumerate_potts, strip_log_partition
from phylo.sim.graph import BoundaryCondition, PottsGraph, lattice_graph

RELATIVE_TOLERANCE = 1e-11

FIELD = np.array([0.3, -0.7, 0.15])

# A tree, so that BP is exact on it: six nodes, five edges, no cycle, and
# couplings of mixed sign so that a test passing by symmetry cannot.
TREE = PottsGraph(
    n_nodes=6,
    edges=((0, 1), (0, 2), (1, 3), (1, 4), (2, 5)),
    coupling=(0.8, -0.4, 1.2, 0.3, 0.9),
)


def _relative(realized: float, reference: float) -> float:
    return abs(realized - reference) / abs(reference)


@pytest.mark.oracle
def test_the_bethe_free_energy_is_exact_on_a_tree() -> None:
    exact = enumerate_potts(TREE, FIELD)

    result = belief_propagation(TREE, FIELD)

    assert _relative(result.bethe_log_partition, exact.log_partition) < 1e-14


@pytest.mark.oracle
def test_the_beliefs_are_the_exact_marginals_on_a_tree() -> None:
    exact = enumerate_potts(TREE, FIELD)

    result = belief_propagation(TREE, FIELD)

    np.testing.assert_allclose(
        result.single_site, exact.single_site, rtol=RELATIVE_TOLERANCE, atol=1e-12
    )
    np.testing.assert_allclose(
        result.pairwise, exact.pairwise, rtol=RELATIVE_TOLERANCE, atol=1e-12
    )


@pytest.mark.mathematical
def test_the_pairwise_beliefs_reduce_to_the_single_site_ones_on_a_tree() -> None:
    # A consistency the beliefs owe each other wherever BP is exact. On a loop
    # it holds too, by construction of the pairwise belief, but there it is
    # consistency between two approximations rather than evidence of either.
    result = belief_propagation(TREE, FIELD)

    for position, (first, second) in enumerate(TREE.edges):
        np.testing.assert_allclose(
            result.pairwise[position].sum(axis=1),
            result.single_site[first],
            atol=1e-12,
        )
        np.testing.assert_allclose(
            result.pairwise[position].sum(axis=0),
            result.single_site[second],
            atol=1e-12,
        )


@pytest.mark.oracle
def test_a_zero_coupling_lattice_is_exact_despite_its_loops() -> None:
    # The loops are still there; what is gone is the coupling that makes them
    # matter. Separating "loopy" from "approximate" this way is what shows the
    # error measured below comes from the cycles carrying correlation, not
    # from the lattice geometry alone.
    shape = (6, 4)
    graph = lattice_graph(shape, BoundaryCondition.OPEN, 0.0)
    exact = strip_log_partition(shape, BoundaryCondition.OPEN, 0.0, FIELD)

    result = belief_propagation(graph, FIELD)

    assert _relative(result.bethe_log_partition, exact) < RELATIVE_TOLERANCE


# Measured on a 6x4 open strip against `strip_log_partition`, 3 states,
# field (0.3, -0.7, 0.15). The exact q-state Potts transition on a square
# lattice is at J_c = ln(1 + sqrt(q)) = 1.005 for q = 3.
_DEVIATION_CURVE = (
    (0.000, 1.7e-15, 2),
    (0.125, 4.4e-06, 40),
    (0.250, 7.1e-05, 50),
    (0.500, 1.1e-03, 85),
    (0.750, 4.2e-03, 158),
    (0.875, 5.2e-03, 169),
    (1.000, 4.5e-03, 140),
    (1.500, 8.8e-04, 79),
    (2.000, 3.7e-04, 65),
)


@pytest.mark.oracle
@pytest.mark.parametrize(("coupling", "expected", "sweeps"), _DEVIATION_CURVE)
def test_the_bethe_deviation_is_the_measured_size(
    coupling: float, expected: float, sweeps: int
) -> None:
    # Pinned to the order of magnitude, not the digit: the deliverable is how
    # far the approximation sits from exact, and a bound an order of magnitude
    # either side of the measurement still fails if the estimator changes.
    shape = (6, 4)
    graph = lattice_graph(shape, BoundaryCondition.OPEN, coupling)
    exact = strip_log_partition(shape, BoundaryCondition.OPEN, coupling, FIELD)

    result = belief_propagation(graph, FIELD)
    realized = _relative(result.bethe_log_partition, exact)

    assert realized == pytest.approx(expected, rel=0.1) or (
        expected < 1e-12 and realized < 1e-12
    )
    # Reported beside the deviation, per the ticket, so a point that only just
    # converged is visible rather than inferred. Banded rather than pinned:
    # the count is a threshold crossing on a float residual, so a machine
    # summing the messages in a different order can land a sweep either side.
    assert result.iterations == pytest.approx(sweeps, rel=0.2)


@pytest.mark.mathematical
def test_the_deviation_grows_with_coupling_below_the_transition() -> None:
    # Monotone on the weak-coupling arm only. It is *not* monotone in J
    # overall: the curve above peaks at J = 0.875 and falls away, because deep
    # in the ordered phase the sites agree and the correlations the Bethe
    # approximation neglects are short-ranged again. Asserting monotone growth
    # across the whole range would be asserting something false.
    shape = (6, 4)
    deviations = []
    for coupling in (0.0, 0.125, 0.25, 0.5, 0.75):
        graph = lattice_graph(shape, BoundaryCondition.OPEN, coupling)
        exact = strip_log_partition(shape, BoundaryCondition.OPEN, coupling, FIELD)
        result = belief_propagation(graph, FIELD)
        deviations.append(_relative(result.bethe_log_partition, exact))

    assert deviations == sorted(deviations)


@pytest.mark.mathematical
def test_the_deviation_peaks_in_the_neighbourhood_of_the_transition() -> None:
    # The reason to have the curve rather than one number. `J_c` is a closed
    # form, pinned here rather than chosen: the peak is a prediction this
    # test checks, not a coupling picked because it flattered the result.
    shape = (6, 4)
    transition = math.log(1.0 + math.sqrt(3.0))
    couplings = [round(0.125 * step, 3) for step in range(17)]

    peak = max(
        couplings,
        key=lambda coupling: _relative(
            belief_propagation(
                lattice_graph(shape, BoundaryCondition.OPEN, coupling), FIELD
            ).bethe_log_partition,
            strip_log_partition(shape, BoundaryCondition.OPEN, coupling, FIELD),
        ),
    )

    # Within one grid step of the closed form. A 6x4 open strip is small and
    # its boundary is free, both of which shift an effective transition down
    # from the bulk value, so the peak is expected below `J_c` rather than on
    # it -- which is why this is a bracket and not an equality.
    assert transition - 0.25 <= peak <= transition + 0.125


@pytest.mark.edge_case
def test_a_frustrated_lattice_that_does_not_settle_is_refused() -> None:
    # The loudest way BP fails: strong antiferromagnetic coupling on a
    # periodic lattice, where the messages orbit rather than converge. A Bethe
    # free energy read off orbiting messages is not an estimate of anything,
    # and a caller cannot tell it from one that is.
    graph = lattice_graph((4, 4), BoundaryCondition.PERIODIC, -3.0)

    with pytest.raises(ConvergenceError, match="did not converge in 500"):
        belief_propagation(graph, FIELD, damping=0.0, max_iterations=500)


@pytest.mark.edge_case
def test_the_refusal_carries_the_residual_it_stopped_at() -> None:
    graph = lattice_graph((4, 4), BoundaryCondition.PERIODIC, -3.0)

    with pytest.raises(ConvergenceError) as failure:
        belief_propagation(graph, FIELD, damping=0.0, max_iterations=500)

    assert failure.value.iterations == 500
    assert failure.value.residual > failure.value.tolerance


@pytest.mark.mathematical
def test_undamped_updates_are_permitted_and_converge_on_a_tree() -> None:
    # Damping is a default, not a requirement. A tree needs none, and pinning
    # that keeps the damped path from being load-bearing for correctness.
    exact = enumerate_potts(TREE, FIELD)

    result = belief_propagation(TREE, FIELD, damping=0.0)

    assert _relative(result.bethe_log_partition, exact.log_partition) < 1e-14


@pytest.mark.edge_case
def test_damping_of_one_is_refused() -> None:
    # At 1 no message ever updates, so the residual is zero on the first sweep
    # and every graph "converges" immediately to the uniform initialization.
    # That is a wrong answer reported as a converged one.
    with pytest.raises(ValueError, match=r"damping must be in \[0, 1\)"):
        belief_propagation(TREE, FIELD, damping=1.0)


@pytest.mark.edge_case
@pytest.mark.oracle
def test_an_edgeless_graph_is_exactly_its_independent_sites() -> None:
    # No messages exist, so there is no residual to converge; handled as its
    # own case rather than as a loop that runs zero times and then claims to
    # have converged.
    graph = PottsGraph(n_nodes=4, edges=(), coupling=())
    single = float(np.log(np.exp(FIELD).sum()))

    result = belief_propagation(graph, FIELD)

    assert _relative(result.bethe_log_partition, 4 * single) < RELATIVE_TOLERANCE
    independent = np.tile(np.exp(FIELD) / np.exp(FIELD).sum(), (graph.n_nodes, 1))
    np.testing.assert_allclose(result.single_site, independent, atol=1e-12)
