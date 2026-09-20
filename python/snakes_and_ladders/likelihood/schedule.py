"""Which messages are sent, in what order, and when to stop.

The arithmetic of a message is one thing and the order the messages go in is
another, and until issue #592 this package had only the second hard-coded:
two orders, a two-member enum, and an ``if`` in
:func:`snakes_and_ladders.likelihood.message_passing.sum_product` choosing
between them. A third order could not be asked for, and the one that was
hard-coded built groups it did not need --- 20 steps carrying 20 messages on
a six-variable chain, each step's grouping machinery run to group one thing.

**A schedule that is not optimal is still valid.** The two-pass tree schedule
is exact and the others are not, but being unable to *ask* for another hides
what the exact one buys: the leaf-to-root half alone is Felsenstein pruning
and gives ``log Z`` for half the messages, and the root-to-leaf half alone is
what the second pass contributes. So a schedule declares a
:class:`~snakes_and_ladders.likelihood.schedule.Guarantee` rather than a boolean, and a caller that wants a cheaper
or a partial answer asks for it instead of reading a field that quietly means
something narrower than it says.

**One interface reaches every graph.** :mod:`snakes_and_ladders.sim.factor_graph`
carries six adapters --- ``from_potts``, ``from_hmm``, ``from_tree``,
``from_coupled``, ``from_parity_check``, ``from_trellis`` --- so the Potts
lattice, the hidden Markov chain, the phylogenetic tree, the coupled model,
the Tanner graph and the trellis already land on one representation. The
schedule was the only part that did not generalise; over :class:`~snakes_and_ladders.likelihood.schedule.Layout` it
now does, and `tests/regression/likelihood/test_schedule.py` runs every
schedule against every adapter rather than asserting that it would work.

**Two families of schedule, named apart.** This package schedules two
unrelated things: *messages*, here, and *temperatures*, in
:mod:`snakes_and_ladders.sample.schedule`. They were both called ``Schedule``,
which read as one concept and was not --- and which a documented package
cannot carry twice, since every cross-reference to either then resolves two
ways and the ``-W`` docs build refuses it. So each family carries its half in
its name: :class:`MessageSchedule` and :class:`TreeMessageSchedule` here,
``TempSchedule`` and ``LinearTempSchedule`` there. The name says which thing
is being ordered, which is the question a reader of either actually has.

**Why a base class and not a `Protocol`.** Five modules call through this ---
``message_passing``, ``message_passing_reference``, ``belief_propagation``,
``ldpc`` and ``search.ground_state`` --- which is past root ``CLAUDE.md``'s
three-consumer rule, and the five schedules share the group builders and the
guarantee. Sharing those by inheritance beats restating them five times, and
root ``CLAUDE.md`` admits a contract becoming a base class.
"""

from __future__ import annotations

import heapq
import math
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from snakes_and_ladders.sim.factor_graph import FactorGraph


class Guarantee(StrEnum):
    """What a schedule's answer is worth, which is not always a yes or a no."""

    EXACT = "exact"
    """``log Z`` and every marginal, exactly. Only on a tree."""

    PARTIAL = "partial"
    """Exact, but not everywhere: some marginals are undefined rather than
    approximate, and :meth:`MessageSchedule.defined` says which are not. Reading an
    undefined one raises rather than returning a number that looks like a
    posterior."""

    APPROXIMATE = "approximate"
    """The Bethe fixed point, reached by iterating to a tolerance. Exact on a
    tree at convergence, an approximation elsewhere."""


class MessageScheduleName(StrEnum):
    """The schedules by name, for a caller that does not want to import one.

    ``sum_product(graph, schedule="upward")`` and
    ``sum_product(graph, schedule=UpwardMessageSchedule())`` are the same call;
    :func:`resolve` takes either. A sixth schedule is addable without touching
    this enum, which is the point of the seam --- the enum is a convenience,
    not the registry's authority.
    """

    TREE = "tree"
    """Leaves to root then root to leaves: exact, and refused off a tree."""

    UPWARD = "upward"
    """The leaf-to-root half alone: Felsenstein pruning. Exact ``log Z`` and an
    exact root marginal, for about half the messages; no other marginal."""

    DOWNWARD = "downward"
    """The root-to-leaf half alone, from the priors. Not a posterior --- the
    beliefs are conditioned on what lies above each node and on nothing
    below --- so it is an ablation and the other half of a composition test."""

    FLOODING = "flooding"
    """Every message from its neighbours' previous values, damped, until the
    largest change falls below the tolerance: the Bethe approximation."""

    SEQUENTIAL = "sequential"
    """One message at a time in layout order, each read from the newest values
    rather than the previous sweep's: Gauss-Seidel where flooding is Jacobi.
    The same fixed points, reached by a different path, so the answer depends
    on the order in a way flooding's does not."""
    RESIDUAL = "residual"
    """Largest change first: a heap over the factors keyed on the last residual
    their inputs saw (Elidan, McGraw & Koller 2006). Converges where flooding
    oscillates and spends its sends where the graph is still moving; the same
    fixed points as flooding and sequential, reached in a data-dependent
    order (issue #825)."""


def _runs(indices: Iterable[int]) -> Iterator[tuple[int, int]]:
    """``(value, run length)`` over a sequence grouped by equal neighbours.

    The entries of one shape were appended level by level, so a level's rows
    are contiguous and a slice recovers them.
    """
    current: int | None = None
    length = 0
    for index in indices:
        if index == current:
            length += 1
            continue
        if current is not None:
            yield current, length
        current, length = index, 1
    if current is not None:
        yield current, length


@dataclass(frozen=True)
class VariableSends:
    """Variable-to-factor messages of one (degree, cardinality), as edge rows.

    ``targets[r]`` is the edge written; ``sources[r]`` the variable's other
    edges in factor order, whose factor-to-variable messages are added in
    that order onto zeros -- the reference's arithmetic, term for term.
    """

    cardinality: int
    targets: np.ndarray
    sources: np.ndarray


@dataclass(frozen=True)
class VariableBeliefs:
    """Variables of one (degree, cardinality) with all their edges, for the beliefs."""

    cardinality: int
    variables: np.ndarray
    edges: np.ndarray


@dataclass(frozen=True)
class FactorSends:
    """Factor-to-variable messages of one table shape along one axis.

    ``tables`` stacks the group's log tables as ``(n, *shape)``; ``incoming``
    holds each factor's edges per axis, and ``targets`` the edge row written
    -- or, when ``axis`` is ``None`` (the beliefs), the factor's own index.
    """

    shape: tuple[int, ...]
    axis: int | None
    targets: np.ndarray
    tables: np.ndarray
    incoming: np.ndarray


Step = tuple[list[VariableSends], list[FactorSends]]


class Layout:
    """The graph as contiguous edge arrays, and the groups a sweep runs over."""

    def __init__(self, graph: FactorGraph) -> None:
        self.graph = graph
        index = {variable.name: i for i, variable in enumerate(graph.variables)}
        self.cardinality = [variable.cardinality for variable in graph.variables]
        self.width = max(self.cardinality)
        self.edge_variable: list[int] = []
        self.edge_factor: list[int] = []
        self.variable_edges: list[list[int]] = [[] for _ in graph.variables]
        self.factor_edges: list[list[int]] = []
        for position, factor in enumerate(graph.factors):
            edges = []
            for name in factor.variables:
                edge = len(self.edge_variable)
                self.edge_variable.append(index[name])
                self.edge_factor.append(position)
                self.variable_edges[index[name]].append(edge)
                edges.append(edge)
            self.factor_edges.append(edges)
        self.n_edges = len(self.edge_variable)

    def variable_sends(self, pairs: Iterable[tuple[int, int]]) -> list[VariableSends]:
        """Group ``(variable, target edge)`` sends by (degree, cardinality)."""
        buckets: dict[tuple[int, int], tuple[list[int], list[list[int]]]] = {}
        for variable, target in pairs:
            sources = [e for e in self.variable_edges[variable] if e != target]
            key = (len(sources), self.cardinality[variable])
            targets, rows = buckets.setdefault(key, ([], []))
            targets.append(target)
            rows.append(sources)
        return [
            VariableSends(
                cardinality,
                np.asarray(targets, dtype=np.int64),
                np.asarray(rows, dtype=np.int64).reshape(len(targets), degree),
            )
            for (degree, cardinality), (targets, rows) in buckets.items()
        ]

    def factor_sends(
        self, pairs: Iterable[tuple[int, int | None]]
    ) -> list[FactorSends]:
        """Group ``(factor, axis)`` sends by (table shape, axis); ``None`` keeps every axis."""
        buckets: dict[
            tuple[tuple[int, ...], int | None],
            tuple[list[int], list[np.ndarray], list[list[int]]],
        ] = {}
        for position, axis in pairs:
            table = self.graph.factors[position].log_table
            targets, tables, incoming = buckets.setdefault(
                (table.shape, axis), ([], [], [])
            )
            edges = self.factor_edges[position]
            targets.append(position if axis is None else edges[axis])
            tables.append(table)
            incoming.append(edges)
        return [
            FactorSends(
                shape,
                axis,
                np.asarray(targets, dtype=np.int64),
                tables[0][np.newaxis] if len(tables) == 1 else np.stack(tables),
                np.asarray(incoming, dtype=np.int64),
            )
            for (shape, axis), (targets, tables, incoming) in buckets.items()
        ]

    def flooding_step(self) -> Step:
        """Every variable-to-factor send, then every factor-to-variable send."""
        variables = self.variable_sends(
            (v, e) for v, edges in enumerate(self.variable_edges) for e in edges
        )
        factors = self.factor_sends(
            (f, axis)
            for f, edges in enumerate(self.factor_edges)
            for axis in range(len(edges))
        )
        return variables, factors

    @property
    def root(self) -> int:
        """The variable the tree passes are rooted at: the first, as the reference is."""
        return 0

    def tree_passes(self) -> tuple[list[Step], list[Step]]:
        """The leaf-to-root sends grouped by height, and root-to-leaf by depth.

        Rooted at the first variable, as
        :mod:`snakes_and_ladders.likelihood.message_passing_reference` is. A
        node's up message needs only its children's, so every node of one
        height sends in one call; the down pass mirrors it by depth.
        Breadth-first rather than recursive, so a 20,000-step chain is not a
        recursion-depth error.

        Returned as two lists rather than one, because the halves are
        separately askable (issue #592):
        :class:`UpwardMessageSchedule` is the first and :class:`DownwardMessageSchedule` the
        second, and concatenating them is :class:`TreeMessageSchedule`.
        """
        n_variables = len(self.cardinality)
        n_nodes = n_variables + len(self.factor_edges)
        parent_edge = [-1] * n_nodes
        parent = [-1] * n_nodes
        depth = [0] * n_nodes
        order = [self.root]
        for node in order:
            for edge in self._edges_of(node):
                if edge == parent_edge[node]:
                    continue
                child = self._other_end(node, edge)
                parent_edge[child], parent[child], depth[child] = (
                    edge,
                    node,
                    depth[node] + 1,
                )
                order.append(child)
        height = [0] * n_nodes
        for node in reversed(order[1:]):
            height[parent[node]] = max(height[parent[node]], height[node] + 1)

        up: dict[int, tuple[list[tuple[int, int]], list[tuple[int, int | None]]]] = {}
        down: dict[int, tuple[list[tuple[int, int]], list[tuple[int, int | None]]]] = {}
        for node in order:
            # `enumerate` rather than `factor_edges[...].index(edge)`: the walk
            # already knows which axis it is on, and the scan it replaces was a
            # linear search per factor edge (issue #592).
            for axis, edge in enumerate(self._edges_of(node)):
                upward = edge == parent_edge[node]
                level = up if upward else down
                variables, factors = level.setdefault(
                    height[node] if upward else depth[node], ([], [])
                )
                if node < n_variables:
                    variables.append((node, edge))
                else:
                    factors.append((node - n_variables, axis))
        return tuple(  # type: ignore[return-value]
            list(
                zip(
                    self._batched_variable_sends([v for _, (v, _) in levels]),
                    self._batched_factor_sends([f for _, (_, f) in levels]),
                    strict=True,
                )
            )
            for levels in (sorted(up.items()), sorted(down.items()))
        )

    def _batched_variable_sends(
        self, levels: list[list[tuple[int, int]]]
    ) -> list[list[VariableSends]]:
        """:meth:`variable_sends` for many levels, over one array per shape.

        Called per level, that method builds two small arrays per group, and a
        chain has one message per level --- 8,000 levels at 2,000 variables,
        so 16,000 ``np.asarray`` calls over one-element lists, which measured
        as the whole of the schedule build (issue #592). Here every level's
        sends of one ``(degree, cardinality)`` go into one array and each
        level takes a **slice** of it: same rows, same order, no copy, and the
        allocation count stops growing with the graph.
        """
        order: dict[tuple[int, int], list[tuple[int, int, list[int]]]] = {}
        for index, pairs in enumerate(levels):
            for variable, target in pairs:
                sources = [e for e in self.variable_edges[variable] if e != target]
                key = (len(sources), self.cardinality[variable])
                order.setdefault(key, []).append((index, target, sources))
        out: list[list[VariableSends]] = [[] for _ in levels]
        for (degree, cardinality), entries in order.items():
            targets = np.asarray([t for _, t, _ in entries], dtype=np.int64)
            rows = np.asarray([s for _, _, s in entries], dtype=np.int64).reshape(
                len(entries), degree
            )
            start = 0
            for index, group in _runs(entry[0] for entry in entries):
                stop = start + group
                out[index].append(
                    VariableSends(cardinality, targets[start:stop], rows[start:stop])
                )
                start = stop
        return out

    def _batched_factor_sends(
        self, levels: list[list[tuple[int, int | None]]]
    ) -> list[list[FactorSends]]:
        """:meth:`factor_sends` for many levels, over one array per shape."""
        order: dict[
            tuple[tuple[int, ...], int | None],
            list[tuple[int, int, int, list[int]]],
        ] = {}
        for index, pairs in enumerate(levels):
            for position, axis in pairs:
                table = self.graph.factors[position].log_table
                edges = self.factor_edges[position]
                order.setdefault((table.shape, axis), []).append(
                    (index, position, position if axis is None else edges[axis], edges)
                )
        out: list[list[FactorSends]] = [[] for _ in levels]
        for (shape, axis), entries in order.items():
            targets = np.asarray([t for _, _, t, _ in entries], dtype=np.int64)
            incoming = np.asarray([e for _, _, _, e in entries], dtype=np.int64)
            tables = np.stack(
                [
                    self.graph.factors[position].log_table
                    for _, position, _, _ in entries
                ]
            )
            start = 0
            for index, group in _runs(entry[0] for entry in entries):
                stop = start + group
                out[index].append(
                    FactorSends(
                        shape,
                        axis,
                        targets[start:stop],
                        tables[start:stop],
                        incoming[start:stop],
                    )
                )
                start = stop
        return out

    def factor_neighbours(self) -> list[list[int]]:
        """The factors sharing a variable with each factor, excluding itself.

        What a residual schedule bumps when a factor's outgoing messages
        change: the factors whose incoming messages those are (issue #825).
        """
        by_variable: list[list[int]] = [[] for _ in self.variable_edges]
        for factor, edges in enumerate(self.factor_edges):
            for edge in edges:
                by_variable[self.edge_variable[edge]].append(factor)
        neighbours: list[list[int]] = []
        for factor, edges in enumerate(self.factor_edges):
            seen: dict[int, None] = {}
            for edge in edges:
                for other in by_variable[self.edge_variable[edge]]:
                    if other != factor:
                        seen.setdefault(other, None)
            neighbours.append(list(seen))
        return neighbours

    def sequential_steps(self) -> list[Step]:
        """One step per factor: its incoming messages, then its outgoing ones.

        Applied in order, so each step reads whatever the previous ones wrote
        --- Gauss-Seidel, where :meth:`flooding_step` is Jacobi. One step per
        factor rather than one per message is the layered schedule the LDPC
        literature uses, and it is the shape that would let
        :mod:`snakes_and_ladders.likelihood.ldpc` call through this module.
        """
        return [
            (
                self.variable_sends((self.edge_variable[edge], edge) for edge in edges),
                self.factor_sends((position, axis) for axis in range(len(edges))),
            )
            for position, edges in enumerate(self.factor_edges)
        ]

    def variable_beliefs(self) -> list[VariableBeliefs]:
        buckets: dict[tuple[int, int], tuple[list[int], list[list[int]]]] = {}
        for variable, edges in enumerate(self.variable_edges):
            variables, rows = buckets.setdefault(
                (len(edges), self.cardinality[variable]), ([], [])
            )
            variables.append(variable)
            rows.append(edges)
        return [
            VariableBeliefs(
                cardinality,
                np.asarray(variables, dtype=np.int64),
                np.asarray(rows, dtype=np.int64),
            )
            for (_, cardinality), (variables, rows) in buckets.items()
        ]

    def _edges_of(self, node: int) -> list[int]:
        n_variables = len(self.cardinality)
        if node < n_variables:
            return self.variable_edges[node]
        return self.factor_edges[node - n_variables]

    def _other_end(self, node: int, edge: int) -> int:
        n_variables = len(self.cardinality)
        if node < n_variables:
            return n_variables + self.edge_factor[edge]
        return self.edge_variable[edge]


class MessageSchedule(ABC):
    """An order over the messages, and what that order guarantees.

    Under the seam rule: one module names this base, and five call through it
    by :class:`MessageScheduleName` --- which is the convenience the enum
    exists to be, and the reason the consumer count `infra/gate_new_seams.py`
    reads is 1 while `tests/regression/test_duplication_guards.py` asserts
    that every schedule is written under this base and no consumer branches
    on its name. The rule counts a module naming the base; a seam reached
    through a registry is counted by neither that nor the import graph, and
    #813's review records the gap rather than the base pretending to a
    consumer it does not have.
    """

    #: The name :func:`resolve` registers it under.
    name: str

    @property
    @abstractmethod
    def guarantee(self) -> Guarantee:
        """What the answer is worth."""

    @property
    def bounded(self) -> bool:
        """Whether the plan is finite, as against iterating to a tolerance."""
        return True

    @property
    def requires_tree(self) -> bool:
        """Whether the schedule is refused on a graph with a cycle."""
        return False

    @property
    def compiled(self) -> bool:
        """Whether :mod:`snakes_and_ladders.likelihood.message_passing_rust` runs this order.

        False here, and true only on the two-pass tree schedule, which is the
        order issue #754 measured and ported. It is a property of the schedule
        rather than a lookup on its name, so a consumer asks the seam what it
        offers and no `if` on the name returns (issue #755).

        The two halves answer false although the kernel sends them: their
        ``log_partition`` reads the scale normalizing removed, which is a
        running float sum over the levels, and a kernel walking the order
        rather than the levels would reassociate it. What they would buy is
        the same passes; what it would cost is a ``log Z`` that is no longer
        bitwise against :func:`snakes_and_ladders.likelihood.pruning.log_likelihood`,
        which is the comparison they exist for.
        """
        return False

    @property
    def damped(self) -> bool:
        """Whether a send is mixed with the value it replaces.

        False for the bounded schedules, which write each message once and so
        have nothing to damp; true for the iterative ones. It is also what
        keeps the refactor bitwise: a bounded send normalizes once, and
        normalizing an already-normalized row a second time is not the
        identity in floating point.
        """
        return not self.bounded

    @property
    def adaptive(self) -> bool:
        """Whether :meth:`steps` chooses its next send from the last send's residual.

        False for a fixed order. True means the runner ``send``s each applied
        step's residual back into the generator, which is how an adaptive
        schedule sees the graph move without the runner naming it (issue
        #825).
        """
        return False

    def sweep_length(self, layout: Layout) -> int:
        """How many steps make one sweep, so a residual is compared over a full pass.

        One step per factor by default, which is the sequential order's
        sweep and the residual order's accounting unit; flooding sends
        everything in one step and says so. Comparing a residual against the
        tolerance mid-sweep would stop on a message that had simply not moved
        yet.
        """
        return len(layout.factor_edges)

    @abstractmethod
    def steps(self, layout: Layout) -> Iterator[Step]:
        """The sends, in order. May be infinite when :attr:`bounded` is false.

        An :attr:`adaptive` schedule is a generator that receives each yielded
        step's residual through ``send`` and chooses the next from it.
        """

    def log_partition(
        self, layout: Layout, to_variable: np.ndarray, bethe: float, scale: float
    ) -> float:
        """``log Z``, from whichever of the two routes this schedule has.

        The default is ``bethe`` --- the negated Bethe free energy over the
        beliefs, which is exact on a tree and the approximation elsewhere. A
        schedule that does not leave every belief correct cannot use it, and
        says here what it can use instead.
        """
        del layout, to_variable, scale
        return bethe

    def defined(self, layout: Layout) -> frozenset[int] | None:
        """The variables whose marginal this schedule leaves defined.

        ``None`` --- the default --- means every one. A
        :attr:`Guarantee.PARTIAL` schedule narrows it, and the caller gets a
        :class:`KeyError` for the rest rather than a number.
        """
        del layout
        return None


@dataclass(frozen=True)
class TreeMessageSchedule(MessageSchedule):
    """Leaves to root, then root to leaves. Exact, and refused off a tree."""

    name: str = "tree"

    @property
    def guarantee(self) -> Guarantee:
        return Guarantee.EXACT

    @property
    def requires_tree(self) -> bool:
        return True

    @property
    def compiled(self) -> bool:
        return True

    def steps(self, layout: Layout) -> Iterator[Step]:
        up, down = layout.tree_passes()
        yield from up
        yield from down


@dataclass(frozen=True)
class UpwardMessageSchedule(MessageSchedule):
    """The leaf-to-root pass alone: Felsenstein pruning.

    ``log Z`` is exact after it --- that is what pruning computes --- and so
    is the root's marginal, because every message into the root has arrived.
    No other variable has heard from above, so no other marginal is defined,
    and :meth:`defined` says so rather than letting a caller read one.
    """

    name: str = "upward"

    @property
    def guarantee(self) -> Guarantee:
        return Guarantee.PARTIAL

    @property
    def requires_tree(self) -> bool:
        return True

    def steps(self, layout: Layout) -> Iterator[Step]:
        yield from layout.tree_passes()[0]

    def log_partition(
        self, layout: Layout, to_variable: np.ndarray, bethe: float, scale: float
    ) -> float:
        """``scale`` plus the root's own normalizer: exactly what pruning returns.

        The Bethe route is unavailable --- it reads every belief, and after
        this pass only the root's is right. The scaled route is: every message
        was normalized on the way up, ``scale`` is the sum of what that
        removed, and the root's remaining normalizer is the last factor of
        ``Z``. This is Felsenstein's algorithm with the usual scaling, so it
        agrees with :func:`snakes_and_ladders.likelihood.pruning.log_likelihood`
        and with the two-pass schedule, and the tests assert both.
        """
        del bethe
        c = layout.cardinality[layout.root]
        total = np.zeros(c)
        for edge in layout.variable_edges[layout.root]:
            total = total + to_variable[edge, :c]
        peak = float(total.max())
        return scale + peak + float(np.log(np.exp(total - peak).sum()))

    def defined(self, layout: Layout) -> frozenset[int]:
        return frozenset({layout.root})


@dataclass(frozen=True)
class DownwardMessageSchedule(MessageSchedule):
    """The root-to-leaf pass alone, from the priors.

    Valid message passing over a well-defined order, and *not* a posterior:
    each belief is conditioned on what lies above its node and on nothing
    below it. Kept because it is the other half of :class:`TreeMessageSchedule` ---
    upward then downward must reproduce it exactly --- and because an
    ablation that says what the second pass contributes is worth being able
    to run.
    """

    name: str = "downward"

    @property
    def guarantee(self) -> Guarantee:
        return Guarantee.PARTIAL

    @property
    def requires_tree(self) -> bool:
        return True

    def steps(self, layout: Layout) -> Iterator[Step]:
        yield from layout.tree_passes()[1]

    def log_partition(
        self, layout: Layout, to_variable: np.ndarray, bethe: float, scale: float
    ) -> float:
        """Not available, and reported as ``nan`` rather than guessed.

        The down pass alone has seen no evidence from below, so neither route
        to ``log Z`` is open: the Bethe expression needs beliefs this schedule
        does not compute, and the scaled route needs the root's normalizer,
        which the up pass holds. A number here would be a number for
        something this schedule did not calculate.
        """
        del layout, to_variable, bethe, scale
        return math.nan

    def defined(self, layout: Layout) -> frozenset[int]:
        # Every variable has heard from above; the leaves have heard from
        # nothing else, so their belief is the prior propagated down.
        return frozenset(range(len(layout.cardinality)))


@dataclass(frozen=True)
class FloodingMessageSchedule(MessageSchedule):
    """Every message at once from the previous sweep's values: Jacobi."""

    name: str = "flooding"

    @property
    def guarantee(self) -> Guarantee:
        return Guarantee.APPROXIMATE

    @property
    def bounded(self) -> bool:
        return False

    def sweep_length(self, layout: Layout) -> int:
        del layout
        return 1

    def steps(self, layout: Layout) -> Iterator[Step]:
        step = layout.flooding_step()
        while True:
            yield step


@dataclass(frozen=True)
class SequentialMessageSchedule(MessageSchedule):
    """One message at a time, each from the newest values: Gauss-Seidel.

    Flooding reads every message from the previous sweep; this reads each from
    whatever is current, so information crosses the graph within a sweep
    rather than one edge per sweep. The fixed points are the same --- they are
    the Bethe stationary points either way --- and the path is not, so the
    sweep count is a measurement rather than a guarantee.
    """

    name: str = "sequential"

    @property
    def guarantee(self) -> Guarantee:
        return Guarantee.APPROXIMATE

    @property
    def bounded(self) -> bool:
        return False

    def steps(self, layout: Layout) -> Iterator[Step]:
        sweep = layout.sequential_steps()
        while True:
            yield from sweep


@dataclass(frozen=True)
class ResidualMessageSchedule(MessageSchedule):
    """The factor whose inputs moved most sends next: residual belief propagation.

    Elidan, McGraw & Koller (2006) order messages by the size of their last
    change. Here the unit is a factor's step --- its incoming sends, then its
    outgoing ones, :meth:`Layout.sequential_steps`'s shape --- and a factor's
    priority is the largest residual any neighbouring factor's outgoing
    messages last had, since those are its inputs. Every factor starts at
    infinity, so the first sweep visits each once; after that the heap decides,
    and a factor whose inputs have not moved is not sent again. The fixed
    points are flooding's and sequential's --- the Bethe stationary points ---
    and the path is not, so the sweep count is a measurement (issue #825).

    The residual reaches the schedule through ``send``: the runner applies
    the yielded step and sends back the largest change it made, which is the
    number this schedule keys on. A stale heap entry is skipped by its stamp
    rather than removed, the standard lazy deletion.
    """

    name: str = "residual"

    @property
    def guarantee(self) -> Guarantee:
        return Guarantee.APPROXIMATE

    @property
    def bounded(self) -> bool:
        return False

    @property
    def adaptive(self) -> bool:
        return True

    def steps(self, layout: Layout) -> Iterator[Step]:
        per_factor = layout.sequential_steps()
        neighbours = layout.factor_neighbours()
        n_factors = len(per_factor)
        priority = [math.inf] * n_factors
        stamp = [0] * n_factors
        # (-priority, factor) so the heap pops the largest; ties by index, so
        # the first sweep is the sequential order and the run is reproducible.
        heap = [(-math.inf, factor, 0) for factor in range(n_factors)]
        heapq.heapify(heap)
        while True:
            negated, factor, seen = heapq.heappop(heap)
            if seen != stamp[factor]:
                continue
            residual = yield per_factor[factor]
            moved = float(residual) if residual is not None else 0.0
            priority[factor] = 0.0
            stamp[factor] += 1
            heapq.heappush(heap, (0.0, factor, stamp[factor]))
            for other in neighbours[factor]:
                if moved > priority[other]:
                    priority[other] = moved
                    stamp[other] += 1
                    heapq.heappush(heap, (-moved, other, stamp[other]))


#: Every schedule this module offers, by the name :class:`MessageScheduleName`
#: uses. A seventh is registered by adding it here; nothing else changes.
SCHEDULES: dict[str, MessageSchedule] = {
    schedule.name: schedule
    for schedule in (
        TreeMessageSchedule(),
        UpwardMessageSchedule(),
        DownwardMessageSchedule(),
        FloodingMessageSchedule(),
        SequentialMessageSchedule(),
        ResidualMessageSchedule(),
    )
}


def resolve(
    schedule: MessageSchedule | MessageScheduleName | str,
) -> MessageSchedule:
    """A :class:`~snakes_and_ladders.likelihood.schedule.Schedule` from itself, its enum member, or its name.

    Raises
    ------
    ValueError
        If the name is not registered, listing the ones that are.
    """
    if isinstance(schedule, MessageSchedule):
        return schedule
    key = str(schedule)
    if key not in SCHEDULES:
        msg = f"unknown schedule {key!r}; the registered ones are {sorted(SCHEDULES)}"
        raise ValueError(msg)
    return SCHEDULES[key]


__all__ = [
    "SCHEDULES",
    "DownwardMessageSchedule",
    "FactorSends",
    "FloodingMessageSchedule",
    "Guarantee",
    "Layout",
    "MessageSchedule",
    "MessageScheduleName",
    "ResidualMessageSchedule",
    "SequentialMessageSchedule",
    "Step",
    "TreeMessageSchedule",
    "UpwardMessageSchedule",
    "VariableBeliefs",
    "VariableSends",
    "resolve",
]
