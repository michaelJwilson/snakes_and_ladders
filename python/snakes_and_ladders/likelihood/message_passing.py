"""Sum-product and max-product on a factor graph, exact on a tree.

One algorithm for the three evaluators this package already has. On a tree it
is exact in one leaf-to-root and one root-to-leaf pass: pruning is this on the
tree's factor graph (``eq:pruning``), the forward recursion is this on a chain
(``eq:forward``), and both are pinned here against the implementations that
predate it. On a loopy graph it is the Bethe approximation
:mod:`snakes_and_ladders.likelihood.belief_propagation` computes for the pairwise Potts
case, and it is pinned against that too: same messages, same free energy.
Which regime a caller is in is a property of the graph, so :func:`sum_product`
refuses a tree schedule on a loopy graph rather than returning a number that
looks exact.

Messages live in the log domain and are normalized every sweep, for the
reason `belief_propagation.py` gives; a factor of any degree sums out its
other variables by ``logsumexp`` over the axes it does not send along, so
the code has no notion of "pairwise". Max-product replaces the sum by a max
and reads the MAP assignment off the max-marginals, which is exact on a tree
when the maximum is unique -- Viterbi, on a chain, and the test says so.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from snakes_and_ladders.numerics import logsumexp
from snakes_and_ladders.sim.factor_graph import Factor, FactorGraph

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


def _reduce(table: np.ndarray, keep: int, maximum: bool) -> np.ndarray:
    """Sum (or max) out every axis of ``table`` except ``keep``."""
    axes = tuple(axis for axis in range(table.ndim) if axis != keep)
    if not axes:
        return table
    flat = np.moveaxis(table, keep, 0).reshape(table.shape[keep], -1)
    if maximum:
        return np.asarray(flat.max(axis=1))
    return np.asarray(logsumexp(flat, axis=1))


def _factor_to_variable(
    factor: Factor, incoming: Mapping[str, np.ndarray], target: str, maximum: bool
) -> np.ndarray:
    """The message from ``factor`` to ``target``, given the other variables' messages."""
    table = factor.log_table
    for axis, name in enumerate(factor.variables):
        if name == target:
            continue
        shape = [1] * table.ndim
        shape[axis] = -1
        table = table + incoming[name].reshape(shape)
    return _reduce(table, factor.variables.index(target), maximum)


def _normalize(message: np.ndarray) -> np.ndarray:
    return np.asarray(message - logsumexp(message[np.newaxis, :], axis=1)[0])


def _tree_order(graph: FactorGraph) -> list[tuple[str, str, bool]]:
    """Edges ``(factor, variable, factor_sends)`` in a leaf-to-root order.

    Rooted at the first variable; the reverse order is the root-to-leaf pass.
    Each entry says which message along the edge is computable at that point.
    """
    root = graph.variables[0].name
    order: list[tuple[str, str, bool]] = []
    seen_factors: set[str] = set()
    seen_variables = {root}

    def visit_variable(name: str) -> None:
        for factor in graph.neighbours(name):
            if factor.name in seen_factors:
                continue
            seen_factors.add(factor.name)
            for other in factor.variables:
                if other in seen_variables:
                    continue
                seen_variables.add(other)
                visit_variable(other)
                order.append((factor.name, other, False))  # other -> factor
            order.append((factor.name, name, True))  # factor -> name

    visit_variable(root)
    return order


def _run(
    graph: FactorGraph,
    schedule: MessageSchedule,
    maximum: bool,
    damping: float,
    tolerance: float,
    max_iterations: int,
) -> tuple[dict[tuple[str, str], np.ndarray], dict[tuple[str, str], np.ndarray], int]:
    """Messages in both directions, keyed ``(factor, variable)``."""
    by_name = {factor.name: factor for factor in graph.factors}
    to_variable: dict[tuple[str, str], np.ndarray] = {}
    to_factor: dict[tuple[str, str], np.ndarray] = {}
    for factor in graph.factors:
        for name in factor.variables:
            zeros = np.zeros(graph.variable(name).cardinality)
            to_variable[(factor.name, name)] = zeros
            to_factor[(factor.name, name)] = zeros.copy()

    def variable_message(name: str, exclude: str) -> np.ndarray:
        total = np.zeros(graph.variable(name).cardinality)
        for factor in graph.neighbours(name):
            if factor.name != exclude:
                total = total + to_variable[(factor.name, name)]
        return total

    if schedule is MessageSchedule.TREE:
        if not graph.is_tree():
            msg = "the tree schedule is exact only on a tree; this graph has a cycle or is disconnected"
            raise ValueError(msg)
        order = _tree_order(graph)
        for pass_order in (order, [(f, v, not up) for f, v, up in reversed(order)]):
            for factor_name, variable_name, factor_sends in pass_order:
                factor = by_name[factor_name]
                if factor_sends:
                    incoming = {
                        n: to_factor[(factor_name, n)] for n in factor.variables
                    }
                    to_variable[(factor_name, variable_name)] = _normalize(
                        _factor_to_variable(factor, incoming, variable_name, maximum)
                    )
                else:
                    to_factor[(factor_name, variable_name)] = _normalize(
                        variable_message(variable_name, factor_name)
                    )
        return to_variable, to_factor, 2

    if not 0.0 <= damping < 1.0:
        msg = f"damping must be in [0, 1), got {damping}"
        raise ValueError(msg)
    for iteration in range(1, max_iterations + 1):
        residual = 0.0
        new_to_factor = {
            key: _normalize(variable_message(name, factor_name))
            for key in to_factor
            for factor_name, name in (key,)
        }
        to_factor = new_to_factor
        for factor in graph.factors:
            incoming = {n: to_factor[(factor.name, n)] for n in factor.variables}
            for name in factor.variables:
                proposal = _normalize(
                    _factor_to_variable(factor, incoming, name, maximum)
                )
                updated = _normalize(
                    damping * to_variable[(factor.name, name)]
                    + (1.0 - damping) * proposal
                )
                residual = max(
                    residual,
                    float(np.abs(updated - to_variable[(factor.name, name)]).max()),
                )
                to_variable[(factor.name, name)] = updated
        if residual <= tolerance:
            return to_variable, to_factor, iteration
    msg = f"flooding did not converge in {max_iterations} sweeps; residual {residual:.2e} above {tolerance:.0e}"
    raise ConvergenceError(msg)


def _beliefs(
    graph: FactorGraph,
    to_variable: Mapping[tuple[str, str], np.ndarray],
    to_factor: Mapping[tuple[str, str], np.ndarray],
    maximum: bool,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], float]:
    variable_log: dict[str, np.ndarray] = {}
    for variable in graph.variables:
        total = np.zeros(variable.cardinality)
        for factor in graph.neighbours(variable.name):
            total = total + to_variable[(factor.name, variable.name)]
        variable_log[variable.name] = total
    factor_log: dict[str, np.ndarray] = {}
    for factor in graph.factors:
        table = factor.log_table.copy()
        for axis, name in enumerate(factor.variables):
            shape = [1] * table.ndim
            shape[axis] = -1
            table = table + to_factor[(factor.name, name)].reshape(shape)
        factor_log[factor.name] = table
    if maximum:
        # Messages are normalized every sweep, so the max-marginals carry the
        # argmax and not the value; the caller reads the value off the
        # assignment with ``log_density``.
        norm = {k: v - np.max(v) for k, v in variable_log.items()}
        return (
            {k: np.exp(v) for k, v in norm.items()},
            {k: np.exp(v - np.max(v)) for k, v in factor_log.items()},
            math.nan,
        )
    # Bethe free energy: sum_f b_f (log b_f - log psi_f) - sum_v (d_v - 1) b_v log b_v.
    free_energy = 0.0
    beliefs_v: dict[str, np.ndarray] = {}
    beliefs_f: dict[str, np.ndarray] = {}
    for name, unnormalized in variable_log.items():
        log_b = _normalize(unnormalized)
        b = np.exp(log_b)
        beliefs_v[name] = b
        degree = graph.degree(name)
        free_energy -= (degree - 1) * float(np.sum(b * np.where(b > 0, log_b, 0.0)))
    for factor in graph.factors:
        log_b = factor_log[factor.name]
        log_b = log_b - logsumexp(log_b.reshape(1, -1), axis=1)[0]
        b = np.exp(log_b)
        beliefs_f[factor.name] = b
        with np.errstate(invalid="ignore"):
            term = np.where(b > 0, b * (log_b - factor.log_table), 0.0)
        free_energy += float(np.sum(term))
    return beliefs_v, beliefs_f, -free_energy


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
    to_variable, to_factor, iterations = _run(
        graph, schedule, False, damping, tolerance, max_iterations
    )
    variable, factor, log_partition = _beliefs(graph, to_variable, to_factor, False)
    return Marginals(variable, factor, log_partition, iterations, exact=graph.is_tree())


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
    to_variable, to_factor, iterations = _run(
        graph, schedule, True, damping, tolerance, max_iterations
    )
    variable, factor, _ = _beliefs(graph, to_variable, to_factor, True)
    assignment = {name: int(np.argmax(values)) for name, values in variable.items()}
    return assignment, Marginals(
        variable,
        factor,
        graph.log_density(assignment),
        iterations,
        exact=graph.is_tree(),
    )
