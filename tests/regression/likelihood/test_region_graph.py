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
from snakes_and_ladders.likelihood.message_passing import Marginals, sum_product
from snakes_and_ladders.likelihood.region_graph import (
    RegionGraph,
    bethe_region_graph,
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
@pytest.mark.potts_chain
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
@pytest.mark.potts_lattice
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
@pytest.mark.potts_lattice
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
@pytest.mark.potts_lattice
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
@pytest.mark.potts_lattice
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
@pytest.mark.potts_lattice
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
@pytest.mark.potts_lattice
def test_a_region_graph_that_counts_a_variable_twice_is_refused() -> None:
    # The validity condition, triggered rather than described: two overlapping
    # clusters whose intersection is not among the regions count the shared
    # variables twice. The closure is what prevents it, so the test builds the
    # unclosed collection by hand.
    from snakes_and_ladders.likelihood.region_graph import Region

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
