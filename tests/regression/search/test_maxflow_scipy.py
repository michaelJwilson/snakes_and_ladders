"""The ground-state minimum cut against the scipy front that lost to it (issues #322, #388).

The front is :mod:`snakes_and_ladders.sandbox.scipy_mincut`, kept there
because issue #388 declined it: ``scipy.sparse.csgraph.maximum_flow`` beats
the Python Dinic 3.5x to 8.8x and loses to the Rust Dinic already fronting
this path by 2.3x to 4.2x. These tests are what stop that decline from
becoming an assertion about a version nobody is holding.

A ground state is a combinatorial minimum, so agreement is **exact on the
energy** and never a tolerance. The *configuration* may legitimately differ
where the minimum is degenerate, which is why the energy is what is compared.

**The capacity scaling is what needs a test, and it is where writing one
changed the finding.** scipy takes ``int32`` capacities and the reduction is
real-valued, so every capacity is scaled and rounded, and rounding can select
a different cut. Issue #388 swept 1e2 to 1e8 at extent 16 on one field and
recorded the energy unchanged. Over twenty fields it is not: at **1e2** the
rounding picks a strictly worse cut on 3 of 20 --- seeds 1, 3 and 5, above the
minimum by 7.18e-4, 1.92e-3 and 4.31e-3 --- and from **1e3** upward all twenty
agree exactly. :data:`SAFE_SCALES` is the part of the sweep that survived, and
:func:`test_the_coarsest_scale_swept_does_select_a_worse_cut` pins the part
that did not, since a range quietly narrowed is a finding deleted.

**No bound relating the rounding error to the gap between the best and
second-best cut is derived**, here or in the experiment. Twenty fields on one
lattice at one coupling is what is claimed and all that is claimed; the 1e2
result is what that absence looks like when it bites.

scipy is a core dependency (issue #383), so nothing here skips.
"""

from __future__ import annotations

import itertools
import sys

import numpy as np
import pytest
from snakes_and_ladders.sandbox.scipy_mincut import DEFAULT_SCALE
from snakes_and_ladders.sandbox.scipy_mincut import (
    ising_ground_state as scipy_ground_state,
)
from snakes_and_ladders.search import maxflow_rust
from snakes_and_ladders.search.maxflow import energy, ising_ground_state
from snakes_and_ladders.sim.graph import BoundaryCondition, PottsGraph, lattice_graph

# The Python blocking flow recurses to the depth of the level graph, and the
# extent-16 lattice below reaches it.
sys.setrecursionlimit(50_000)

#: The scales the energy is unchanged at over :data:`SEEDS`, at extent 16.
SAFE_SCALES = (1.0e3, 1.0e4, 1.0e6, 1.0e8)

#: The coarse end of issue #388's sweep, and the seeds it selects a worse cut
#: on, with how far above the minimum each lands.
COARSE_SCALE = 1.0e2
COARSE_EXCESS = {
    1: 7.178051667438012e-04,
    3: 1.917592616962338e-03,
    5: 4.309857453790755e-03,
}

#: The fields the scale sweep is checked over. One lattice, twenty draws.
SEEDS = range(20)
COUPLING = 0.6


def _random_field(graph: PottsGraph, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(graph.n_nodes, 2))


def _enumerated_minimum(graph: PottsGraph, field_values: np.ndarray) -> float:
    configurations = np.array(
        list(itertools.product(range(2), repeat=graph.n_nodes)), dtype=np.int64
    )
    return float(energy(graph, field_values, configurations).min())


@pytest.mark.oracle
@pytest.mark.critical
@pytest.mark.parametrize("scale", SAFE_SCALES)
def test_the_scaled_int32_cut_reports_both_dinics_energies(scale: float) -> None:
    # Extent 16, which is where the scale sweep was measured, against both
    # implementations: the Python Dinic that is the oracle and the Rust Dinic
    # that actually fronts the path and that the decline is measured against.
    # Exact equality --- the energy of the chosen configuration is scored in
    # real arithmetic by `maxflow.energy` on all three sides, so a difference
    # here is a *different cut* selected by the rounding, not a rounded number.
    graph = lattice_graph((16, 16), BoundaryCondition.OPEN, COUPLING)
    for seed in SEEDS:
        field_values = _random_field(graph, seed)
        _, ours = ising_ground_state(graph, field_values)
        _, rust = maxflow_rust.ising_ground_state(graph, field_values)
        _, theirs = scipy_ground_state(graph, field_values, scale=scale)

        assert (theirs, rust) == (ours, ours)


@pytest.mark.oracle
def test_the_coarsest_scale_swept_does_select_a_worse_cut() -> None:
    # The finding writing this test produced, pinned rather than dropped from
    # the range. At 1e2 the rounded capacities are minimised by a cut that is
    # not the real-valued minimum, on 3 of 20 fields; the excess is above zero
    # by construction, since a cut worse than the minimum is worse in the
    # energy the minimum is defined in.
    graph = lattice_graph((16, 16), BoundaryCondition.OPEN, COUPLING)
    excess: dict[int, float] = {}
    for seed in SEEDS:
        field_values = _random_field(graph, seed)
        _, ours = ising_ground_state(graph, field_values)
        _, theirs = scipy_ground_state(graph, field_values, scale=COARSE_SCALE)
        if theirs != ours:
            excess[seed] = theirs - ours

    assert excess.keys() == COARSE_EXCESS.keys()
    for seed, gap in COARSE_EXCESS.items():
        assert excess[seed] == pytest.approx(gap, rel=1e-9)


@pytest.mark.oracle
@pytest.mark.parametrize("shape", [(2, 2), (3, 3), (4, 3), (4, 4)])
@pytest.mark.parametrize("coupling", [0.0, 0.4, 1.2])
def test_the_scipy_cut_finds_the_enumerated_minimum(
    shape: tuple[int, int], coupling: float
) -> None:
    # The reference the Dinic path is held to, applied to the front: a
    # combinatorial minimum checked against every configuration, where the
    # size allows it. `abs=1e-12` is the enumeration's own summation order,
    # not a rounding allowance for the scaling.
    rng = np.random.default_rng(0)
    graph = lattice_graph(shape, BoundaryCondition.OPEN, coupling)
    for _ in range(3):
        field_values = rng.normal(size=(graph.n_nodes, 2))
        _, realized = scipy_ground_state(graph, field_values)

        assert realized == pytest.approx(
            _enumerated_minimum(graph, field_values), abs=1e-12
        )


@pytest.mark.mathematical
def test_a_zero_coupling_ground_state_follows_the_field_site_by_site() -> None:
    # The corner with an answer written down: with no coupling the cut
    # decomposes and every site independently takes its better state, so the
    # front is checked against arithmetic rather than against another solver.
    graph = lattice_graph((6, 6), BoundaryCondition.OPEN, 0.0)
    field_values = _random_field(graph, 5)

    configuration, realized = scipy_ground_state(graph, field_values)

    assert np.array_equal(configuration, field_values.argmax(axis=1))
    assert realized == pytest.approx(-field_values.max(axis=1).sum(), abs=1e-12)


@pytest.mark.mathematical
def test_a_zero_field_ground_state_is_aligned_at_the_analytic_energy() -> None:
    # The other corner: with no field every coupling favours agreement, so an
    # aligned configuration is optimal at `-J |E|`.
    graph = lattice_graph((8, 8), BoundaryCondition.OPEN, COUPLING)

    configuration, realized = scipy_ground_state(graph, np.zeros(2))

    assert len(set(configuration.tolist())) == 1
    assert realized == pytest.approx(-COUPLING * len(graph.edges), abs=1e-9)


@pytest.mark.edge_case
def test_a_capacity_that_overflows_int32_is_refused_rather_than_wrapped() -> None:
    # The boundary the `int32` capacities put on the scaling. A wrapped
    # capacity poses a different flow problem and would return a
    # lattice-shaped wrong answer, so the scale is refused by name.
    graph = lattice_graph((4, 4), BoundaryCondition.OPEN, COUPLING)

    with pytest.raises(ValueError, match="overflows int32"):
        scipy_ground_state(graph, _random_field(graph, 0), scale=1.0e12)


@pytest.mark.edge_case
def test_a_negative_coupling_is_refused_as_the_dinic_path_refuses_it() -> None:
    # The submodularity boundary. The front must refuse what it fronts
    # refuses, or a caller comparing the two gets an answer from one and an
    # exception from the other.
    graph = PottsGraph(n_nodes=2, edges=((0, 1),), coupling=(-0.5,))
    field_values = np.zeros((2, 2))

    with pytest.raises(ValueError, match="non-submodular"):
        ising_ground_state(graph, field_values)
    with pytest.raises(ValueError, match="non-submodular"):
        scipy_ground_state(graph, field_values)


@pytest.mark.edge_case
def test_a_non_positive_scale_is_refused() -> None:
    graph = lattice_graph((3, 3), BoundaryCondition.OPEN, COUPLING)

    with pytest.raises(ValueError, match="scale must be positive"):
        scipy_ground_state(graph, _random_field(graph, 0), scale=0.0)


@pytest.mark.structural
def test_the_default_scale_is_one_the_energy_was_measured_unchanged_at() -> None:
    # The default is not an unmeasured fourth scale, and after the result
    # above it must not be the coarse one either: a caller who passes nothing
    # gets a scale this module has run twenty fields through.
    assert DEFAULT_SCALE in SAFE_SCALES
