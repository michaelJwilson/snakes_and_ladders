"""Sum-product and max-product on a factor graph, exact on a tree.

One algorithm for the three evaluators this package already has:
``eq:sum-product`` of ``docs/tex/textbook.tex``. On a tree it
is exact in one leaf-to-root and one root-to-leaf pass (``app:sum-product``
derives both passes and their exactness): pruning is this on the
tree's factor graph (``eq:pruning``), the forward recursion is this on a chain
(``eq:forward``), and both are pinned here against the implementations that
predate it. On a loopy graph it is the Bethe approximation
:mod:`snakes_and_ladders.likelihood.belief_propagation` computes for the pairwise Potts
case, and it is pinned against that too: same messages, same free energy
(``eq:bethe-factor``, whose stationary point ``app:bethe`` identifies with the
fixed point).
Which regime a caller is in is a property of the graph, so :func:`sum_product`
refuses a tree schedule on a loopy graph rather than returning a number that
looks exact.

Messages live in the log domain and are normalized every sweep, for the
reason `belief_propagation.py` gives; a factor of any degree sums out its
other variables by ``logsumexp`` over the axes it does not send along, so
the code has no notion of "pairwise". Max-product replaces the sum by a max
and reads the MAP assignment off the max-marginals, which is exact on a tree
when the maximum is unique -- Viterbi, on a chain, and the test says so.

**Layout.** Every (factor, variable) incidence is an edge, numbered in factor
order; the two messages along it are rows of two preallocated ``(n_edges,
width)`` arrays, ``width`` the largest cardinality, a row's unused columns
held at zero and never read. A sweep is then one vectorized pass per group
of edges that share a shape -- variables by (degree, cardinality), factors by
(table shape, axis sent along) -- with the tables of a group stacked
contiguously once. The tree schedule groups the same kernels by height and
depth, so a wide level is one call and a chain is one call per position.
:mod:`snakes_and_ladders.likelihood.message_passing_reference` is the
dictionary-per-message implementation this replaced (issue #341); it is the
oracle, the arithmetic per message is the same in the same order, and the
regression suite pins the two bitwise on every schedule. Rooting at the first
variable is shared with the reference, which is what makes the tree messages
comparable edge for edge.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from snakes_and_ladders.sim.factor_graph import FactorGraph

DEFAULT_DAMPING = 0.5
DEFAULT_TOLERANCE = 1e-10
DEFAULT_MAX_ITERATIONS = 500


class MessageSchedule(StrEnum):
    """How messages are ordered."""

    TREE = "tree"
    """Leaves to root then root to leaves: exact, and refused off a tree."""

    FLOODING = "flooding"
    """Every message from its neighbours' previous values, damped, until the
    largest change falls below the tolerance: the Bethe approximation."""


class ConvergenceError(RuntimeError):
    """Flooding did not settle within the iteration cap."""


@dataclass(frozen=True)
class Marginals:
    """What message passing returns.

    Parameters
    ----------
    variable : Mapping[str, np.ndarray]
        Per variable, its (max-)marginal, normalized, in the linear domain.
    factor : Mapping[str, np.ndarray]
        Per factor, the belief over its variables, normalized, in the linear
        domain, shaped as its table.
    log_partition : float
        ``log Z`` exactly on a tree; the negated Bethe free energy otherwise.
        For max-product, the unnormalized log-density at the returned
        assignment: the maximum, on a tree with a unique maximum.
    iterations : int
        Sweeps run; ``2`` for the tree schedule.
    exact : bool
        Whether the graph is a tree, so the numbers above are exact.
    """

    variable: Mapping[str, np.ndarray]
    factor: Mapping[str, np.ndarray]
    log_partition: float
    iterations: int
    exact: bool


@dataclass(frozen=True)
class _VariableSends:
    """Variable-to-factor messages of one (degree, cardinality), as edge rows.

    ``targets[r]`` is the edge written; ``sources[r]`` the variable's other
    edges in factor order, whose factor-to-variable messages are added in
    that order onto zeros -- the reference's arithmetic, term for term.
    """

    cardinality: int
    targets: np.ndarray
    sources: np.ndarray


@dataclass(frozen=True)
class _VariableBeliefs:
    """Variables of one (degree, cardinality) with all their edges, for the beliefs."""

    cardinality: int
    variables: np.ndarray
    edges: np.ndarray


@dataclass(frozen=True)
class _FactorSends:
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


_Step = tuple[list[_VariableSends], list[_FactorSends]]


class _Layout:
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

    def variable_sends(self, pairs: Iterable[tuple[int, int]]) -> list[_VariableSends]:
        """Group ``(variable, target edge)`` sends by (degree, cardinality)."""
        buckets: dict[tuple[int, int], tuple[list[int], list[list[int]]]] = {}
        for variable, target in pairs:
            sources = [e for e in self.variable_edges[variable] if e != target]
            key = (len(sources), self.cardinality[variable])
            targets, rows = buckets.setdefault(key, ([], []))
            targets.append(target)
            rows.append(sources)
        return [
            _VariableSends(
                cardinality,
                np.asarray(targets, dtype=np.int64),
                np.asarray(rows, dtype=np.int64).reshape(len(targets), degree),
            )
            for (degree, cardinality), (targets, rows) in buckets.items()
        ]

    def factor_sends(
        self, pairs: Iterable[tuple[int, int | None]]
    ) -> list[_FactorSends]:
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
            _FactorSends(
                shape,
                axis,
                np.asarray(targets, dtype=np.int64),
                tables[0][np.newaxis] if len(tables) == 1 else np.stack(tables),
                np.asarray(incoming, dtype=np.int64),
            )
            for (shape, axis), (targets, tables, incoming) in buckets.items()
        ]

    def flooding_step(self) -> _Step:
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

    def tree_steps(self) -> list[_Step]:
        """Leaf-to-root sends grouped by height, then root-to-leaf by depth.

        Rooted at the first variable, as the reference is. A node's up message
        needs only its children's, so every node of one height sends in one
        call; the down pass mirrors it by depth. Breadth-first rather than
        recursive, so a 20,000-step chain is not a recursion-depth error.
        """
        n_variables = len(self.cardinality)
        n_nodes = n_variables + len(self.factor_edges)
        parent_edge = [-1] * n_nodes
        parent = [-1] * n_nodes
        depth = [0] * n_nodes
        order = [0]
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
            for edge in self._edges_of(node):
                level = up if edge == parent_edge[node] else down
                key = height[node] if edge == parent_edge[node] else depth[node]
                variables, factors = level.setdefault(key, ([], []))
                if node < n_variables:
                    variables.append((node, edge))
                else:
                    factors.append(
                        (
                            node - n_variables,
                            self.factor_edges[node - n_variables].index(edge),
                        )
                    )
        return [
            (self.variable_sends(variables), self.factor_sends(factors))
            for level in (up, down)
            for _, (variables, factors) in sorted(level.items())
        ]

    def variable_beliefs(self) -> list[_VariableBeliefs]:
        buckets: dict[tuple[int, int], tuple[list[int], list[list[int]]]] = {}
        for variable, edges in enumerate(self.variable_edges):
            variables, rows = buckets.setdefault(
                (len(edges), self.cardinality[variable]), ([], [])
            )
            variables.append(variable)
            rows.append(edges)
        return [
            _VariableBeliefs(
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


def _logsumexp_last(values: np.ndarray) -> np.ndarray:
    """:func:`snakes_and_ladders.numerics.logsumexp` over the last axis, keeping it.

    The same five operations in the same order -- peak, shift, ``exp``,
    ``sum``, ``log`` -- so a row agrees bitwise with the reference; inlined
    because on a chain the tree schedule pays this once per position and
    the general function's axis handling was a third of the call.
    """
    peak = values.max(axis=-1, keepdims=True)
    return np.asarray(peak + np.log(np.exp(values - peak).sum(axis=-1, keepdims=True)))


def _normalize(rows: np.ndarray) -> np.ndarray:
    """Each row minus its ``logsumexp``: the reference's ``_normalize``, batched."""
    return np.asarray(rows - _logsumexp_last(rows))


def _send_from_variables(
    group: _VariableSends, to_variable: np.ndarray, to_factor: np.ndarray
) -> None:
    """Write the group's variable-to-factor messages into ``to_factor``."""
    c = group.cardinality
    total = np.zeros((group.targets.shape[0], c))
    for i in range(group.sources.shape[1]):
        total = total + to_variable[group.sources[:, i], :c]
    to_factor[group.targets, :c] = _normalize(total)


def _factor_terms(group: _FactorSends, to_factor: np.ndarray) -> np.ndarray:
    """Each table plus its incoming messages on every axis but ``group.axis``, in axis order."""
    acc = group.tables
    n = group.targets.shape[0]
    for i, c in enumerate(group.shape):
        if i == group.axis:
            continue
        shape = [n] + [1] * len(group.shape)
        shape[1 + i] = c
        acc = acc + to_factor[group.incoming[:, i], :c].reshape(shape)
    return acc


def _send_from_factors(
    group: _FactorSends, to_factor: np.ndarray, maximum: bool
) -> np.ndarray:
    """The group's factor-to-variable messages, ``(n, cardinality of the axis)``, unnormalized."""
    acc = _factor_terms(group, to_factor)
    if acc.ndim == 2:
        return acc
    assert group.axis is not None
    n, c = acc.shape[0], group.shape[group.axis]
    others = tuple(i for i in range(1, acc.ndim) if i != 1 + group.axis)
    flat = acc.transpose(0, 1 + group.axis, *others).reshape(n, c, -1)
    if maximum:
        return np.asarray(flat.max(axis=2))
    return _logsumexp_last(flat)[:, :, 0]


def _run(
    graph: FactorGraph,
    schedule: MessageSchedule,
    maximum: bool,
    damping: float,
    tolerance: float,
    max_iterations: int,
) -> tuple[_Layout, np.ndarray, np.ndarray, int, bool]:
    """Messages in both directions as edge rows, the iteration count, and whether exact."""
    layout = _Layout(graph)
    tree = graph.is_tree()
    to_variable = np.zeros((layout.n_edges, layout.width))
    to_factor = np.zeros((layout.n_edges, layout.width))

    if schedule is MessageSchedule.TREE:
        if not tree:
            msg = "the tree schedule is exact only on a tree; this graph has a cycle or is disconnected"
            raise ValueError(msg)
        for variables, factors in layout.tree_steps():
            for group in variables:
                _send_from_variables(group, to_variable, to_factor)
            for sends in factors:
                c = sends.shape[sends.axis]  # type: ignore[index]
                to_variable[sends.targets, :c] = _normalize(
                    _send_from_factors(sends, to_factor, maximum)
                )
        return layout, to_variable, to_factor, 2, tree

    if not 0.0 <= damping < 1.0:
        msg = f"damping must be in [0, 1), got {damping}"
        raise ValueError(msg)
    variables, factors = layout.flooding_step()
    for iteration in range(1, max_iterations + 1):
        residual = 0.0
        for group in variables:
            _send_from_variables(group, to_variable, to_factor)
        for sends in factors:
            c = sends.shape[sends.axis]  # type: ignore[index]
            old = to_variable[sends.targets, :c]
            proposal = _normalize(_send_from_factors(sends, to_factor, maximum))
            updated = _normalize(damping * old + (1.0 - damping) * proposal)
            residual = max(residual, float(np.abs(updated - old).max()))
            to_variable[sends.targets, :c] = updated
        if residual <= tolerance:
            return layout, to_variable, to_factor, iteration, tree
    msg = f"flooding did not converge in {max_iterations} sweeps; residual {residual:.2e} above {tolerance:.0e}"
    raise ConvergenceError(msg)


def _beliefs(
    layout: _Layout, to_variable: np.ndarray, to_factor: np.ndarray, maximum: bool
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], float]:
    graph = layout.graph
    variable_rows: list[np.ndarray | None] = [None] * len(graph.variables)
    factor_rows: list[np.ndarray | None] = [None] * len(graph.factors)
    free_energy = 0.0
    for beliefs in layout.variable_beliefs():
        c = beliefs.cardinality
        total = np.zeros((beliefs.variables.shape[0], c))
        for i in range(beliefs.edges.shape[1]):
            total = total + to_variable[beliefs.edges[:, i], :c]
        if maximum:
            # Messages are normalized every sweep, so the max-marginals carry
            # the argmax and not the value; the caller reads the value off the
            # assignment with ``log_density``.
            b = np.exp(total - total.max(axis=1, keepdims=True))
        else:
            # Bethe free energy: sum_f b_f (log b_f - log psi_f) - sum_v (d_v - 1) b_v log b_v.
            log_b = _normalize(total)
            b = np.exp(log_b)
            entropy = (b * np.where(b > 0, log_b, 0.0)).sum(axis=1)
            free_energy -= float(((beliefs.edges.shape[1] - 1) * entropy).sum())
        for variable, row in zip(beliefs.variables, b, strict=True):
            variable_rows[variable] = row
    for sends in layout.factor_sends((f, None) for f in range(len(graph.factors))):
        acc = _factor_terms(sends, to_factor)
        flat = acc.reshape(acc.shape[0], -1)
        keep = (-1,) + (1,) * len(sends.shape)
        if maximum:
            b = np.exp(acc - flat.max(axis=1).reshape(keep))
        else:
            log_b = acc - _logsumexp_last(flat).reshape(keep)
            b = np.exp(log_b)
            with np.errstate(invalid="ignore"):
                term = np.where(b > 0, b * (log_b - sends.tables), 0.0)
            free_energy += float(term.sum())
        for position, table in zip(sends.targets, b, strict=True):
            factor_rows[position] = table
    variable = {
        v.name: row
        for v, row in zip(graph.variables, variable_rows, strict=True)
        if row is not None
    }
    factor = {
        f.name: row
        for f, row in zip(graph.factors, factor_rows, strict=True)
        if row is not None
    }
    return variable, factor, math.nan if maximum else -free_energy


def sum_product(
    graph: FactorGraph,
    *,
    schedule: MessageSchedule = MessageSchedule.TREE,
    damping: float = DEFAULT_DAMPING,
    tolerance: float = DEFAULT_TOLERANCE,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
) -> Marginals:
    """Marginals and ``log Z``: exact under the tree schedule, Bethe under flooding.

    Raises
    ------
    ValueError
        If the tree schedule is asked of a loopy graph, or ``damping`` is
        outside ``[0, 1)``.
    ConvergenceError
        If flooding does not settle in ``max_iterations`` sweeps.
    """
    layout, to_variable, to_factor, iterations, exact = _run(
        graph, schedule, False, damping, tolerance, max_iterations
    )
    variable, factor, log_partition = _beliefs(layout, to_variable, to_factor, False)
    return Marginals(variable, factor, log_partition, iterations, exact)


def max_product(
    graph: FactorGraph,
    *,
    schedule: MessageSchedule = MessageSchedule.TREE,
    damping: float = DEFAULT_DAMPING,
    tolerance: float = DEFAULT_TOLERANCE,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
) -> tuple[dict[str, int], Marginals]:
    """The MAP assignment by max-marginals, and the max-marginals themselves.

    On a tree with a unique maximum this is the exact MAP -- Viterbi on a
    chain. Ties are broken by the smallest state, which is the tie rule
    :func:`snakes_and_ladders.likelihood.hmm_paths.enumerate_hidden_paths` uses.
    """
    layout, to_variable, to_factor, iterations, exact = _run(
        graph, schedule, True, damping, tolerance, max_iterations
    )
    variable, factor, _ = _beliefs(layout, to_variable, to_factor, True)
    assignment = {name: int(np.argmax(values)) for name, values in variable.items()}
    return assignment, Marginals(
        variable, factor, graph.log_density(assignment), iterations, exact
    )
