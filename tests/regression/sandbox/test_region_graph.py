"""Region graphs and the Kikuchi free energy, refereed by the case they generalize.

Issue #689. The free energy here is asserted where an exact answer exists and
where an existing implementation already computes the same number:

* On a **tree**, ``-F_K`` over the Bethe region graph is ``log Z``, which
  `message_passing.sum_product` computes exactly under the tree schedule.
* On a **loopy** graph, the same construction must reproduce that module's
  Bethe value --- the general expression evaluated at the special case *is*
  the special case, so a sign error in the entropy term or an off-by-one in a
  counting number fails here rather than at a size where nothing can check it.
* The counting numbers themselves are asserted against the closed form the
  pairwise case has, ``1 - d`` on a variable of degree ``d``.

What is not asserted is that Kikuchi beats Bethe: that is a measurement, it
belongs in `STATUS.md`, and the algorithm that would produce it is the next
step of this ticket.
"""

from __future__ import annotations

import numpy as np
import pytest
from snakes_and_ladders.likelihood.belief_propagation import ConvergenceError
from snakes_and_ladders.likelihood.message_passing import Marginals, sum_product
from snakes_and_ladders.likelihood.potts import enumerate_potts
from snakes_and_ladders.sandbox.region_graph import (
    RegionGraph,
    bethe_region_graph,
    generalized_belief_propagation,
    kikuchi_free_energy,
    lattice_plaquettes,
    region_graph,
)
from snakes_and_ladders.sim.factor_graph import FactorGraph, from_potts
from snakes_and_ladders.sim.graph import BoundaryCondition, lattice_graph

FIELD = np.log(np.array([0.5, 0.3, 0.2]))


def _graph(shape: tuple[int, ...], coupling: float) -> FactorGraph:
    return from_potts(lattice_graph(shape, BoundaryCondition.OPEN, coupling), FIELD)


def _beliefs(
    regions: RegionGraph, marginals: Marginals
) -> dict[tuple[str, ...], np.ndarray]:
    """The beliefs message passing already computed, on each region's axes."""
    out: dict[tuple[str, ...], np.ndarray] = {}
    for region in regions.regions:
        key = region.variables
        if len(key) == 1:
            out[key] = np.asarray(marginals.variable[key[0]])
            continue
        for factor in regions.graph.factors:
            if tuple(sorted(factor.variables)) == key:
                belief = np.asarray(marginals.factor[factor.name])
                order = [list(factor.variables).index(name) for name in key]
                out[key] = np.transpose(belief, order)
                break
    return out


@pytest.mark.oracle
def test_the_free_energy_is_the_exact_log_partition_on_a_tree() -> None:
    # The strongest statement available: on a chain the Bethe region graph's
    # free energy is not an approximation of anything, so equality against the
    # exact `log Z` is the right assertion and machine precision the right
    # tolerance.
    graph = _graph((6,), 0.4)
    exact = sum_product(graph, schedule="tree")
    regions = bethe_region_graph(graph)

    energy = kikuchi_free_energy(regions, _beliefs(regions, exact))

    assert -energy == pytest.approx(exact.log_partition, rel=1e-14)


@pytest.mark.oracle
@pytest.mark.parametrize("shape", [(3, 3), (4, 4)])
def test_the_free_energy_reproduces_the_bethe_value_on_a_loop(
    shape: tuple[int, int],
) -> None:
    # The reduction, which is what makes the general expression checkable
    # before any of it is iterated: evaluated at the Bethe region graph, on the
    # beliefs message passing settled at, `-F_K` is the number that module
    # reports. The tolerance is the flooding schedule's own convergence, not a
    # property of this expression.
    graph = _graph(shape, 0.6)
    loopy = sum_product(graph, schedule="flooding")
    regions = bethe_region_graph(graph)

    energy = kikuchi_free_energy(regions, _beliefs(regions, loopy))

    assert -energy == pytest.approx(loopy.log_partition, rel=1e-9)


@pytest.mark.mathematical
def test_the_counting_numbers_are_one_minus_the_degree() -> None:
    # The closed form the pairwise case carries, and the reason the Bethe
    # region graph is built by the general construction rather than written
    # down: a corner of a 3x3 lattice has degree 2, an edge 3, the centre 4,
    # and Mobius inversion has to produce -1, -2 and -3 without being told.
    graph = _graph((3, 3), 0.6)
    regions = bethe_region_graph(graph)

    degree = {variable.name: 0 for variable in graph.variables}
    for factor in graph.factors:
        if len(factor.variables) == 2:
            for name in factor.variables:
                degree[name] += 1
    singletons = {
        region.variables[0]: region.counting
        for region in regions.regions
        if len(region.variables) == 1
    }

    assert singletons == {name: 1 - d for name, d in degree.items()}
    assert sorted(set(singletons.values())) == [-3, -2, -1]


@pytest.mark.mathematical
@pytest.mark.parametrize(("shape", "expected"), [((3, 3), 9), ((4, 4), 25)])
def test_the_plaquette_closure_counts_everything_once(
    shape: tuple[int, int], expected: int
) -> None:
    # The region graph #689 is for. The closure of the unit cells under
    # intersection gives the cells, the edges they share and the sites shared
    # by four of them; `region_graph` checks the counting, so what is asserted
    # here is the shape of the result and that the check is what passed.
    graph = _graph(shape, 0.6)

    regions = region_graph(graph, lattice_plaquettes(shape))

    assert len(regions.regions) == expected
    assert set(regions.counts.values()) == {1}
    assert set(regions.factor_counts.values()) == {1}
    assert {len(r.variables) for r in regions.regions} == {4, 2, 1}


@pytest.mark.edge_case
def test_a_cluster_that_leaves_a_factor_outside_is_refused() -> None:
    # A factor whose scope fits in no cluster has its energy dropped, not
    # approximated, and the free energy would still return a number. Refused
    # with the factor named, because the caller who chose the clusters is who
    # can widen one.
    graph = _graph((3, 3), 0.6)
    too_small = [frozenset({"s0", "s1"})]

    with pytest.raises(ValueError, match="lies inside no cluster"):
        region_graph(graph, too_small)


@pytest.mark.edge_case
def test_a_belief_that_is_not_a_distribution_is_refused() -> None:
    # A caller who has not normalized has not converged, so renormalizing here
    # would hide an unconverged run inside a plausible number.
    graph = _graph((3,), 0.4)
    regions = bethe_region_graph(graph)
    beliefs = _beliefs(regions, sum_product(graph, schedule="tree"))
    first = next(iter(beliefs))
    beliefs[first] = beliefs[first] * 2.0

    with pytest.raises(ValueError, match="sums to"):
        kikuchi_free_energy(regions, beliefs)


@pytest.mark.mathematical
def test_a_region_graph_that_counts_a_variable_twice_is_refused() -> None:
    # The validity condition, triggered rather than described: two overlapping
    # clusters whose intersection is not among the regions count the shared
    # variables twice. The closure is what prevents it, so the test builds the
    # unclosed collection by hand.
    from snakes_and_ladders.sandbox.region_graph import Region

    graph = _graph((2, 2), 0.6)
    both = tuple(sorted(v.name for v in graph.variables))
    twice = RegionGraph(
        graph=graph,
        regions=(
            Region(variables=both, factors=(), counting=1),
            Region(variables=both, factors=(), counting=1),
        ),
        parents=((), ()),
    )

    with pytest.raises(ValueError, match="counts every variable exactly once"):
        twice.check()


@pytest.mark.oracle
def test_the_updates_find_the_exact_beliefs_on_a_tree() -> None:
    # On a chain the region graph is a tree and the fixed point is the exact
    # one, so the beliefs are marginals and equality against message passing
    # under the tree schedule is the right assertion.
    graph = _graph((6,), 0.4)
    exact = sum_product(graph, schedule="tree")

    settled = generalized_belief_propagation(bethe_region_graph(graph))

    for name, marginal in exact.variable.items():
        assert settled.beliefs[(name,)] == pytest.approx(
            np.asarray(marginal), abs=1e-11
        )
    assert settled.kikuchi_log_partition == pytest.approx(
        exact.log_partition, rel=1e-12
    )


@pytest.mark.oracle
def test_the_updates_find_the_bethe_fixed_point_on_a_loop() -> None:
    # The referee for the algorithm, as the reduction is for the energy: on the
    # Bethe region graph these updates and `message_passing`'s flooding
    # schedule are two routes to one fixed point, so they must agree. A wrong
    # message set that still converged would land somewhere else.
    graph = _graph((3, 3), 0.6)
    loopy = sum_product(graph, schedule="flooding")

    settled = generalized_belief_propagation(bethe_region_graph(graph))

    worst = max(
        float(np.abs(settled.beliefs[(name,)] - np.asarray(marginal)).max())
        for name, marginal in loopy.variable.items()
    )
    assert worst < 1e-8
    assert settled.kikuchi_log_partition == pytest.approx(loopy.log_partition, rel=1e-9)


@pytest.mark.oracle
def test_the_plaquette_regions_are_nearer_the_truth_than_the_pairwise_ones() -> None:
    # The claim the ticket exists to test, against exhaustive enumeration of
    # all 3**9 configurations. Asserted as an ordering rather than pinned to a
    # digit: what is established is that seeing the 4-cycles helps, and by how
    # much is a measurement `STATUS.md` carries, since it moves with the
    # coupling -- 23,000x at J = 0.3 and 130x at J = 1.2 on this instance.
    coupling = 0.6
    lattice = lattice_graph((3, 3), BoundaryCondition.OPEN, coupling)
    gauge = FIELD - np.log(float(np.exp(FIELD).sum()))
    graph = from_potts(lattice, gauge)
    exact = enumerate_potts(lattice, gauge)

    bethe = generalized_belief_propagation(bethe_region_graph(graph))
    kikuchi = generalized_belief_propagation(
        region_graph(graph, lattice_plaquettes((3, 3))), damping=0.7
    )

    pairwise_error = abs(bethe.kikuchi_log_partition - exact.log_partition)
    plaquette_error = abs(kikuchi.kikuchi_log_partition - exact.log_partition)
    assert plaquette_error < pairwise_error / 100.0
    assert kikuchi.iterations > bethe.iterations


@pytest.mark.mathematical
def test_at_size_the_plaquette_graph_settles_where_the_pairwise_one_always_does() -> (
    None
):
    # The binding constraint at size is convergence, not accuracy, and it is
    # asserted in the direction the measurement found. On the 6x4 strip the
    # pairwise region graph settles at every coupling measured; the plaquette
    # one settles at J = 0.25 and does not at J = 0.5 -- damping 0.7 through
    # 0.98 and 20,000 sweeps all refuse, with the residual falling from 0.377
    # to 0.020 as the damping rises, so more damping buys a slower approach
    # and not a fixed point (`STATUS.md`).
    #
    # The cap here is 200 sweeps rather than 20,000: what is asserted is that
    # the two graphs part on this instance, and the fuller sweep is a
    # measurement recorded rather than a test run per pull request.
    shape, cap = (6, 4), 200
    for coupling in (0.25, 0.5):
        pairwise = generalized_belief_propagation(
            bethe_region_graph(_graph(shape, coupling)), max_iterations=cap
        )
        assert pairwise.residual <= 1e-12

    plaquette = region_graph(_graph(shape, 0.5), lattice_plaquettes(shape))
    with pytest.raises(ConvergenceError):
        generalized_belief_propagation(plaquette, damping=0.7, max_iterations=cap)


@pytest.mark.edge_case
def test_damping_that_freezes_every_message_is_refused() -> None:
    # At damping 1 no message moves, so the residual is zero on the first sweep
    # and every graph "converges" -- a silent wrong answer rather than a loud
    # one, which is the same refusal `belief_propagation` carries.
    regions = bethe_region_graph(_graph((3, 3), 0.6))

    with pytest.raises(ValueError, match=r"damping must be in \[0, 1\)"):
        generalized_belief_propagation(regions, damping=1.0)


@pytest.mark.edge_case
def test_a_run_that_does_not_settle_raises_rather_than_reporting() -> None:
    # A free energy read off messages that never settled estimates nothing, and
    # the caller cannot tell it from one that did.
    regions = bethe_region_graph(_graph((4, 4), 1.5))

    with pytest.raises(ConvergenceError):
        generalized_belief_propagation(regions, damping=0.0, max_iterations=2)
