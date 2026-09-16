"""Regions of variables, their counting numbers, and the Kikuchi free energy.

Issue #689. Belief propagation is exact on a tree and the Bethe approximation
on a loop, and the Bethe free energy is the region-based free energy of the
smallest interesting region graph: one region per factor, one per variable,
and a counting number that subtracts the over-counted variables. Kikuchi's
free energy is the same expression over *larger* regions --- on a square
lattice, the plaquettes, which are exactly the 4-cycles Bethe cannot see
(`sim/graph.py`, issue #172).

**This module is the structure and the energy, not the algorithm.** What a
region graph is, which ones are valid, and what a set of beliefs over them
costs; the parent-to-child updates that look for a stationary point come next.
That split is deliberate: the free energy has an oracle --- at the Bethe
region graph it *is* ``eq:bethe-factor``, which
:mod:`snakes_and_ladders.likelihood.message_passing` already computes --- so
it can be refereed before anything iterates.

**Kikuchi is not a bound, and nothing here may be read as one.** Mean field
bounds ``log Z`` from below; Bethe and Kikuchi are stationary points of a
non-convex functional and may fall either side of the truth. What is claimed
is accuracy against an exact referee, never a certificate
(``likelihood/CLAUDE.md``: an approximate evaluator states which regime
carries its correctness).

**The validity condition is arithmetic, not taste.** A region graph is valid
when every variable and every factor is counted exactly once:
``sum over regions containing it of c_R == 1``. The counting numbers follow by
Mobius inversion over the inclusion order, ``c_R = 1 - sum of c_A`` over the
strict ancestors of ``R``, so the condition is a theorem for the graphs built
here and a check for any built elsewhere --- and it is asserted rather than
assumed, because a region graph that counts a variable twice returns a number
that looks like a free energy and is not one.

See Yedidia, Freeman and Weiss (2005) for the region-based derivation, and
``sec:potts`` and ``app:bethe`` of ``docs/tex/textbook.tex`` for the pairwise
case this generalizes.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from snakes_and_ladders.sim.factor_graph import FactorGraph


@dataclass(frozen=True)
class Region:
    """A set of variables, the factors it owns, and how often it is counted.

    Parameters
    ----------
    variables : tuple[str, ...]
        Variable names, sorted, so two regions over the same variables are one
        region whatever order built them.
    factors : tuple[str, ...]
        Every factor whose scope fits inside this region, sorted --- not a
        partition. A factor belongs to each region containing it, and the
        counting numbers are what make the total one: a unary factor on a
        degree-``d`` variable sits in ``d`` pairwise regions at ``c = 1`` and
        in the variable's own at ``c = 1 - d``, which sums to one. Assigning
        each factor to a single region instead counts it ``c_R`` times, which
        is ``1 - d`` for a unary factor and was the first way this was
        written.
    counting : int
        ``c_R`` of the region-based free energy. One for a region no other
        contains; below that, whatever Mobius inversion makes it, which is
        negative as often as not.
    """

    variables: tuple[str, ...]
    factors: tuple[str, ...]
    counting: int


@dataclass(frozen=True)
class RegionGraph:
    """Regions over one factor graph, ordered by inclusion.

    Parameters
    ----------
    graph : FactorGraph
        The model the regions are over.
    regions : tuple[Region, ...]
        Ordered largest first, so a parent always precedes its children.
    parents : tuple[tuple[int, ...], ...]
        Per region, the indices of its *direct* parents --- those containing
        it with nothing strictly between.
    """

    graph: FactorGraph
    regions: tuple[Region, ...]
    parents: tuple[tuple[int, ...], ...]

    @property
    def counts(self) -> dict[str, int]:
        """How often each variable is counted, which a valid graph makes one."""
        totals: dict[str, int] = {v.name: 0 for v in self.graph.variables}
        for region in self.regions:
            for name in region.variables:
                totals[name] += region.counting
        return totals

    @property
    def factor_counts(self) -> dict[str, int]:
        """How often each factor's energy is counted."""
        totals: dict[str, int] = {f.name: 0 for f in self.graph.factors}
        for region in self.regions:
            for name in region.factors:
                totals[name] += region.counting
        return totals

    def check(self) -> None:
        """Refuse a region graph that counts anything other than once.

        Raises
        ------
        ValueError
            If a variable or a factor is counted other than exactly once, or
            if a factor is owned by no region. The message names what, because
            the caller who built the regions is who can fix it.
        """
        wrong = {name: n for name, n in self.counts.items() if n != 1}
        if wrong:
            msg = (
                "a valid region graph counts every variable exactly once; "
                f"these are counted otherwise: {wrong}"
            )
            raise ValueError(msg)
        wrong = {name: n for name, n in self.factor_counts.items() if n != 1}
        if wrong:
            msg = (
                "a valid region graph counts every factor exactly once; "
                f"these are counted otherwise: {wrong}"
            )
            raise ValueError(msg)


def _ancestors(variable_sets: list[frozenset[str]], index: int) -> list[int]:
    """Every region strictly containing region ``index``."""
    mine = variable_sets[index]
    return [
        other
        for other, theirs in enumerate(variable_sets)
        if other != index and mine < theirs
    ]


def region_graph(graph: FactorGraph, clusters: list[frozenset[str]]) -> RegionGraph:
    """The cluster-variation region graph over ``clusters``, closed and counted.

    The construction is Kikuchi's: take the given clusters as the top regions,
    close the collection under intersection, order by inclusion, and read the
    counting numbers off the order by Mobius inversion. Every factor is
    assigned to one region --- the smallest top region whose variables contain
    its scope --- so an energy term is never split across regions.

    Parameters
    ----------
    graph : FactorGraph
        The model.
    clusters : list[frozenset[str]]
        The top regions, as sets of variable names. A cluster containing
        another is kept and the contained one becomes its descendant, which is
        the caller's choice to make rather than this function's to silently
        drop.

    Returns
    -------
    RegionGraph
        Checked: the returned graph counts every variable and factor once.

    Raises
    ------
    ValueError
        If a cluster names a variable the graph does not carry, or if some
        factor's scope lies inside no cluster --- which would drop its energy
        from the free energy entirely rather than approximate it.
    """
    known = {variable.name for variable in graph.variables}
    for cluster in clusters:
        unknown = set(cluster) - known
        if unknown:
            msg = f"cluster names variables the graph does not carry: {sorted(unknown)}"
            raise ValueError(msg)

    # Close under intersection. A region graph whose regions are not closed
    # has a variable shared by two regions and counted by neither correction,
    # so the closure is what makes the counting numbers solvable at all.
    closed: set[frozenset[str]] = set(clusters)
    frontier = list(closed)
    while frontier:
        fresh: set[frozenset[str]] = set()
        for left, right in itertools.combinations(frontier, 2):
            common = left & right
            if common and common not in closed:
                fresh.add(common)
        closed |= fresh
        frontier = list(fresh)

    ordered = sorted(closed, key=lambda s: (-len(s), sorted(s)))

    # Every factor sits in every region containing its scope, and the counting
    # numbers make the total one. This is not a partition and must not be
    # made one: a unary factor assigned to the variable's own region alone
    # would be counted `1 - d` times.
    owner: dict[int, list[str]] = {index: [] for index in range(len(ordered))}
    for factor in graph.factors:
        scope = frozenset(factor.variables)
        holders = [index for index, names in enumerate(ordered) if scope <= names]
        if not holders:
            msg = (
                f"factor {factor.name!r} over {sorted(scope)} lies inside no "
                "cluster, so its energy would be dropped rather than "
                "approximated; widen a cluster to contain it"
            )
            raise ValueError(msg)
        for index in holders:
            owner[index].append(factor.name)

    counting: list[int] = [0] * len(ordered)
    for index in range(len(ordered)):
        counting[index] = 1 - sum(counting[a] for a in _ancestors(ordered, index))

    parents: list[tuple[int, ...]] = []
    for index in range(len(ordered)):
        names = ordered[index]
        above = _ancestors(ordered, index)
        direct = [
            other
            for other in above
            if not any(ordered[other] > ordered[mid] > names for mid in above)
        ]
        parents.append(tuple(sorted(direct)))

    regions = tuple(
        Region(
            variables=tuple(sorted(names)),
            factors=tuple(sorted(owner[index])),
            counting=counting[index],
        )
        for index, names in enumerate(ordered)
    )
    built = RegionGraph(graph=graph, regions=regions, parents=tuple(parents))
    built.check()
    return built


def bethe_region_graph(graph: FactorGraph) -> RegionGraph:
    """The two-layer region graph whose free energy **is** the Bethe one.

    One region per factor, holding that factor's variables, and one per
    variable. Mobius inversion then gives the variable regions ``1 - d_v``,
    the degree correction of ``eq:bethe-factor`` --- which is the point of
    building it this way rather than writing those numbers down: the general
    construction reproduces the special case, so the special case referees the
    general one.
    """
    clusters = [frozenset(factor.variables) for factor in graph.factors]
    clusters += [frozenset({variable.name}) for variable in graph.variables]
    return region_graph(graph, clusters)


def kikuchi_free_energy(
    regions: RegionGraph, beliefs: dict[tuple[str, ...], np.ndarray]
) -> float:
    """``F_K``: the region-based free energy of a set of beliefs.

    ``sum_R c_R (sum_x b_R(x) log b_R(x) - sum_x b_R(x) log psi_R(x))``, the
    entropy term and the energy term of Yedidia, Freeman and Weiss. At the
    Bethe region graph this is ``eq:bethe-factor`` exactly, and ``-F_K`` is
    ``log Z`` wherever message passing is exact.

    Parameters
    ----------
    regions : RegionGraph
        Checked on construction.
    beliefs : dict[tuple[str, ...], np.ndarray]
        One array per region, keyed by the region's variables in its own
        order, shaped by their domains. Each must be non-negative and sum to
        one; a belief that does not is refused rather than renormalized,
        because a caller who has not normalized has not converged.

    Returns
    -------
    float
        ``F_K``. Negate it for the ``log Z`` estimate.

    Raises
    ------
    KeyError
        If a region has no belief.
    ValueError
        If a belief has the wrong shape, is negative, or does not sum to one
        within ``1e-9``.
    """
    cardinality = {v.name: v.cardinality for v in regions.graph.variables}
    tables = {factor.name: factor.log_table for factor in regions.graph.factors}
    scopes = {factor.name: tuple(factor.variables) for factor in regions.graph.factors}
    total = 0.0
    for region in regions.regions:
        if region.counting == 0:
            # Counted zero times: it contributes nothing, and demanding a
            # belief for it would make a caller compute a table for a region
            # the energy never reads.
            continue
        key = region.variables
        if key not in beliefs:
            msg = f"no belief for region {key}"
            raise KeyError(msg)
        belief = np.asarray(beliefs[key], dtype=np.float64)
        expected = tuple(cardinality[name] for name in key)
        if belief.shape != expected:
            msg = f"belief for region {key} is {belief.shape}, expected {expected}"
            raise ValueError(msg)
        if belief.min() < 0.0:
            msg = f"belief for region {key} carries a negative entry"
            raise ValueError(msg)
        mass = float(belief.sum())
        if abs(mass - 1.0) > 1e-9:
            msg = f"belief for region {key} sums to {mass}, not one"
            raise ValueError(msg)

        support = belief > 0.0
        entropy_term = float(np.sum(belief[support] * np.log(belief[support])))

        energy = np.zeros(expected, dtype=np.float64)
        for name in region.factors:
            energy = energy + _broadcast(tables[name], scopes[name], key)
        energy_term = float(np.sum(belief * energy))
        total += region.counting * (entropy_term - energy_term)
    return total


def _broadcast(
    table: np.ndarray, over: tuple[str, ...], onto: tuple[str, ...]
) -> np.ndarray:
    """A factor's log table, laid out over a region's axes.

    The factor names its variables in its own order and the region in its
    own; a transpose and a reshape put the table on the region's axes, and
    broadcasting fills the axes the factor does not name. Doing this with
    ``np.einsum`` would be shorter and would build the dense product, which is
    the array this avoids.
    """
    order = [over.index(name) for name in onto if name in over]
    moved = np.transpose(table, order)
    shape = [
        moved.shape[order.index(over.index(name))] if name in over else 1
        for name in onto
    ]
    return moved.reshape(shape)


def lattice_plaquettes(shape: tuple[int, int]) -> list[frozenset[str]]:
    """The unit cells of a 2-D square lattice, as clusters of variable names.

    The regions Kikuchi's approximation is worth taking on this problem: a
    plaquette is a 4-cycle, and a 4-cycle is exactly what the Bethe
    approximation cannot see (`sim/graph.py`, issue #172). On a ``(3, 3)``
    lattice the closure gives 4 plaquettes at ``c = 1``, their 4 shared edges
    at ``c = -1`` and the centre site at ``c = 1``.

    **This function knows one naming convention**, the row-major ``s{index}``
    that :func:`snakes_and_ladders.sim.factor_graph.from_potts` writes, and
    that coupling is why it takes a shape rather than a graph: a caller who
    built variables by another name passes its own clusters to
    :func:`region_graph`, which is the general door. A name this lattice does
    not carry is refused there.

    Parameters
    ----------
    shape : tuple[int, int]
        ``(rows, columns)``, each at least 2.

    Returns
    -------
    list[frozenset[str]]
        One cluster per unit cell, row-major.

    Raises
    ------
    ValueError
        If either extent is below 2, where a lattice has no cell at all.
    """
    rows, columns = shape
    if rows < 2 or columns < 2:
        msg = f"a plaquette needs two rows and two columns, got {shape}"
        raise ValueError(msg)
    return [
        frozenset(
            {
                f"s{row * columns + column}",
                f"s{row * columns + column + 1}",
                f"s{(row + 1) * columns + column}",
                f"s{(row + 1) * columns + column + 1}",
            }
        )
        for row in range(rows - 1)
        for column in range(columns - 1)
    ]
