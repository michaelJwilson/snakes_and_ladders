"""Sum-product and max-product on a factor graph, exact on a tree.

One algorithm for the three evaluators this package already has:
``eq:sum-product`` of ``docs/tex/textbook.tex``. On a tree it is exact in one
leaf-to-root and one root-to-leaf pass (``app:sum-product`` derives both):
pruning is this on the tree's factor graph (``eq:pruning``), the forward
recursion is this on a chain (``eq:forward``), and both are pinned here
against the implementations that predate it. On a loopy graph it is the Bethe
approximation :mod:`snakes_and_ladders.likelihood.belief_propagation` computes
for the pairwise Potts case, and is pinned against that too: same messages,
same free energy (``eq:bethe-factor``, whose stationary point ``app:bethe``
identifies with the fixed point). The regime is a property of the graph, so
:func:`sum_product` refuses a tree schedule on a loopy graph rather than
returning a number that looks exact.

Messages live in the log domain and are normalized every sweep, for the reason
`belief_propagation.py` gives; a factor of any degree sums out its other
variables by ``logsumexp`` over the axes it does not send along, so the code
has no notion of "pairwise". Max-product replaces the sum by a max and reads
the MAP assignment off the max-marginals, exact on a tree when the maximum is
unique -- Viterbi, on a chain, and the test says so.

**Layout.** Every (factor, variable) incidence is an edge, numbered in factor
order; the two messages along it are rows of two preallocated ``(n_edges,
width)`` arrays, ``width`` the largest cardinality, a row's unused columns held
at zero and never read. A sweep is one vectorized pass per group of edges
sharing a shape -- variables by (degree, cardinality), factors by (table
shape, axis sent along) -- with the group's tables stacked contiguously once.
The tree schedule groups the same kernels by height and depth, so a wide level
is one call and a chain one call per position.
:mod:`snakes_and_ladders.likelihood.message_passing_reference` is the
dictionary-per-message implementation this replaced (issue #341): the oracle,
with the same arithmetic per message in the same order, pinned bitwise on
every schedule. Rooting at the first variable is shared with the reference, so
the tree messages compare edge for edge.

``Layout``'s ``variable_edges`` and ``factor_edges`` stay lists of lists, and
issue #586 measured rather than assumed that: building the whole layout is
**0.6 per cent** of a ``sum_product`` over a 2,000-variable chain, 0.24 on a
2,000-leaf star and 0.22 on a loopy 16x16 lattice, so moving it to
:class:`~snakes_and_ladders.incidence.SparseIncidence` would pay nothing
(root ``CLAUDE.md``, Profile first). What the profile did rank is the tree
*schedule*: :meth:`Layout.tree_steps` is 18.2 per cent of the run at 500
variables and 24.4 at 2,000, because a chain has one message per level and
the grouping machinery is paid per level to group one thing. That is issue
#592, and it is a different change from this one.

**A backend, for the one order a kernel implements.** The per-level dispatch
that grouping leaves is the cost on a chain --- one message per level, so the
NumPy calls are over single rows --- and issue #754's stress profile ranked
it. :mod:`snakes_and_ladders.likelihood.message_passing_rust` runs the two
tree passes over the same layout in Rust and is the default of
:func:`sum_product` and :func:`max_product`: **9.05x / 9.28x** at the chain
of 200, saving **21.6 ms** of that call's 24.3, with the NumPy route below
kept as its oracle and reachable by naming
:data:`~snakes_and_ladders.backend.Backend.PYTHON`. Which schedules it runs
is the schedule's own answer ---
:attr:`~snakes_and_ladders.likelihood.schedule.MessageSchedule.compiled` ---
so the seam is a method and not an ``if`` on a name (issue #755), and a
schedule without a kernel takes the NumPy route whichever backend is asked
for.
"""

from __future__ import annotations

import math
from collections.abc import Generator, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from snakes_and_ladders.backend import Backend
from snakes_and_ladders.likelihood.schedule import (
    FactorSends,
    Guarantee,
    Layout,
    MessageSchedule,
    MessageScheduleName,
    Step,
    VariableSends,
    resolve,
)
from snakes_and_ladders.sim.factor_graph import FactorGraph

DEFAULT_DAMPING = 0.5
DEFAULT_TOLERANCE = 1e-10
DEFAULT_MAX_ITERATIONS = 500


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
        Sweeps run; ``2`` for the tree schedule, ``1`` for each of its halves.
    guarantee : Guarantee
        What the schedule that produced these makes of them: ``EXACT``,
        ``PARTIAL`` --- exact but not everywhere, with the undefined marginals
        absent from ``variable`` rather than approximate --- or
        ``APPROXIMATE``. A field with three values rather than a boolean,
        because a half-pass answer is neither exact nor approximate and
        calling it either loses what it is (issue #592).
    tree : bool
        Whether the graph is a tree. Distinct from ``guarantee``: a loopy
        graph under flooding and a tree under flooding are both
        ``APPROXIMATE``, and only the second is exact at its fixed point.
    """

    variable: Mapping[str, np.ndarray]
    factor: Mapping[str, np.ndarray]
    log_partition: float
    iterations: int
    guarantee: Guarantee
    tree: bool

    @property
    def exact(self) -> bool:
        """Whether the numbers above are exact: the pre-#592 spelling, kept.

        ``guarantee is EXACT``, or an approximate schedule that converged on a
        tree --- which is what the boolean meant when it was the graph's
        property alone.
        """
        return self.guarantee is Guarantee.EXACT or (
            self.tree and self.guarantee is Guarantee.APPROXIMATE
        )


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
    group: VariableSends, to_variable: np.ndarray, to_factor: np.ndarray
) -> np.ndarray:
    """Write the group's variable-to-factor messages; return what normalizing removed.

    Every message is normalized as it is sent, which is what keeps a long
    chain off the floating-point floor --- and which throws away the scale.
    The scales are ``log Z``: their sum, plus the root's, is it exactly. Only
    a bounded schedule asks for them (issue #592); flooding reaches ``log Z``
    through the Bethe free energy instead and does not pay for this.
    """
    c = group.cardinality
    total = np.zeros((group.targets.shape[0], c))
    for i in range(group.sources.shape[1]):
        total = total + to_variable[group.sources[:, i], :c]
    scale = _logsumexp_last(total)
    to_factor[group.targets, :c] = total - scale
    return np.asarray(scale)


def _factor_terms(group: FactorSends, to_factor: np.ndarray) -> np.ndarray:
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
    group: FactorSends, to_factor: np.ndarray, maximum: bool
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


def _apply(
    step: Step,
    to_variable: np.ndarray,
    to_factor: np.ndarray,
    maximum: bool,
    damping: float | None,
    track: bool = False,
) -> tuple[float, float]:
    """Run one step's sends; return the largest change, and the scale removed.

    ``damping`` is ``None`` for a bounded schedule, which writes each message
    once and has nothing to mix it with. That is not only a shortcut: a
    bounded send normalizes once, and normalizing an already-normalized row a
    second time is not the identity in floating point, so routing the tree
    schedule through the damped path would move its output in the last bits.
    The two branches below are the two this function replaced, unchanged.
    """
    variables, factors = step
    scale = 0.0
    for group in variables:
        removed = _send_from_variables(group, to_variable, to_factor)
        if track:
            scale += float(removed.sum())
    residual = 0.0
    for sends in factors:
        c = sends.shape[sends.axis]  # type: ignore[index]
        raw = _send_from_factors(sends, to_factor, maximum)
        proposal = _normalize(raw)
        if track:
            scale += float(_logsumexp_last(raw).sum())
        if damping is None:
            to_variable[sends.targets, :c] = proposal
            continue
        old = to_variable[sends.targets, :c]
        updated = _normalize(damping * old + (1.0 - damping) * proposal)
        residual = max(residual, float(np.abs(updated - old).max()))
        to_variable[sends.targets, :c] = updated
    return residual, scale


@dataclass(frozen=True)
class _Run:
    """What one run of a schedule leaves behind.

    Parameters
    ----------
    layout : Layout
        The graph's edge and incidence arrays, built once for the run.
    to_variable : np.ndarray
        ``(n_edges, width)``, factor to variable.
    to_factor : np.ndarray
        ``(n_edges, width)``, variable to factor.
    iterations : int
        Sweeps run, as :attr:`Marginals.iterations` reports them.
    plan : MessageSchedule
        The schedule that ran, resolved from whatever named it.
    scale : float | None
        What normalization removed on the way up, which
        :meth:`~snakes_and_ladders.likelihood.schedule.MessageSchedule.log_partition`
        reads where the Bethe route is unavailable. ``nan`` on the unbounded
        route, which keeps no total, and ``None`` on the compiled one, which
        accumulates nothing back in Python. A schedule that reads the scale
        answers ``False`` to
        :attr:`~snakes_and_ladders.likelihood.schedule.MessageSchedule.compiled`,
        so no caller of that branch reads this field (issue #865).
    """

    layout: Layout
    to_variable: np.ndarray
    to_factor: np.ndarray
    iterations: int
    plan: MessageSchedule
    scale: float | None

    def __iter__(self) -> Iterator[Any]:
        """The six in the order callers unpack.

        ``Any`` because an unpacking gives all six the element type.
        """
        yield from (
            self.layout,
            self.to_variable,
            self.to_factor,
            self.iterations,
            self.plan,
            self.scale,
        )


def _run(
    graph: FactorGraph,
    schedule: MessageSchedule | MessageScheduleName | str,
    maximum: bool,
    damping: float,
    tolerance: float,
    max_iterations: int,
    backend: Backend,
) -> _Run:
    """Messages in both directions as edge rows, the sweeps run, and the schedule.

    One loop for every schedule (issue #592). A bounded schedule's plan is
    finite and runs once; an unbounded one repeats its sweep until the largest
    change falls below the tolerance. The compiled route replaces the first
    of those and nothing else: it is asked for by
    :attr:`~snakes_and_ladders.likelihood.schedule.MessageSchedule.compiled`,
    which is the schedule's own answer and not a branch on its name.
    """
    if backend not in (Backend.PYTHON, Backend.RUST):
        msg = (
            f"message passing runs on {Backend.PYTHON} or {Backend.RUST}, not {backend}"
        )
        raise ValueError(msg)
    plan = resolve(schedule)
    layout = Layout(graph)
    if plan.requires_tree and not graph.is_tree():
        msg = (
            f"the {plan.name} schedule is exact only on a tree; "
            "this graph has a cycle or is disconnected"
        )
        raise ValueError(msg)

    if backend is Backend.RUST and plan.compiled:
        # Local, because the twin imports this module's layout through
        # `schedule`: the seam is `pruning`/`pruning_rust`'s and
        # `convolutional`/`convolutional_rust`'s.
        from snakes_and_ladders.likelihood import message_passing_rust

        messages = message_passing_rust.tree_messages(layout, maximum=maximum)
        # `None` and not a number: a schedule reading the scale answers false
        # to `compiled`, so no caller of this branch reads the last field, and
        # a number here would be one this route did not compute (issue #865).
        return _Run(layout, messages.to_variable, messages.to_factor, 2, plan, None)

    to_variable = np.zeros((layout.n_edges, layout.width))
    to_factor = np.zeros((layout.n_edges, layout.width))

    if plan.bounded:
        steps = 0.0
        partial = plan.guarantee is Guarantee.PARTIAL
        for step in plan.steps(layout):
            _, removed = _apply(step, to_variable, to_factor, maximum, None, partial)
            steps += removed
        # Reported as passes, not steps: two for the tree schedule, one for
        # each of its halves, which is what the count meant before #592.
        passes = 2 if plan.guarantee is Guarantee.EXACT else 1
        return _Run(layout, to_variable, to_factor, passes, plan, steps)

    if not 0.0 <= damping < 1.0:
        msg = f"damping must be in [0, 1), got {damping}"
        raise ValueError(msg)
    length = plan.sweep_length(layout)
    residual, sweep, position = math.inf, 0, 0
    order = plan.steps(layout)
    step = next(order)
    while True:
        if position == 0:
            residual = 0.0
        moved = _apply(step, to_variable, to_factor, maximum, damping)[0]
        residual = max(residual, moved)
        # An adaptive schedule chooses its next step from what this one moved;
        # a fixed order ignores the value, so `send` is `next` to it.
        if plan.adaptive and isinstance(order, Generator):
            step = order.send(moved)
        else:
            step = next(order)
        position += 1
        if position < length:
            continue
        position, sweep = 0, sweep + 1
        if residual <= tolerance:
            return _Run(layout, to_variable, to_factor, sweep, plan, math.nan)
        if sweep >= max_iterations:
            break
    msg = (
        f"{plan.name} did not converge in {max_iterations} sweeps; "
        f"residual {residual:.2e} above {tolerance:.0e}"
    )
    raise ConvergenceError(msg)


def _beliefs(
    layout: Layout,
    to_variable: np.ndarray,
    to_factor: np.ndarray,
    maximum: bool,
    defined: frozenset[int] | None = None,
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
    # A partial schedule leaves most marginals undefined rather than
    # approximate, so they are absent from the mapping and a caller reading
    # one gets a `KeyError` instead of a number that looks like a posterior
    # (issue #592).
    variable = {
        v.name: row
        for index, (v, row) in enumerate(
            zip(graph.variables, variable_rows, strict=True)
        )
        if row is not None and (defined is None or index in defined)
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
    schedule: MessageSchedule | MessageScheduleName | str = MessageScheduleName.TREE,
    damping: float = DEFAULT_DAMPING,
    tolerance: float = DEFAULT_TOLERANCE,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    backend: Backend = Backend.RUST,
) -> Marginals:
    """Marginals and ``log Z``: exact under the tree schedule, Bethe under flooding.

    Parameters
    ----------
    graph : FactorGraph
    schedule : MessageSchedule | MessageScheduleName | str
    damping, tolerance, max_iterations
        What an unbounded schedule mixes, stops at, and gives up after.
    backend : Backend
        Which implementation sends the messages.
        :data:`~snakes_and_ladders.backend.Backend.PYTHON` is the NumPy
        route below, which stays as the oracle;
        :data:`~snakes_and_ladders.backend.Backend.RUST` is
        :func:`snakes_and_ladders.likelihood.message_passing_rust.tree_messages`
        and the default, because it is **9.05x / 9.28x** a tree-schedule
        ``sum_product`` at the chain of 200 the stress profile ranks, which
        saves **21.6 ms** of that call's 24.3 (``docs/experiments/027``). It
        runs the schedules that answer
        :attr:`~snakes_and_ladders.likelihood.schedule.MessageSchedule.compiled`
        --- the two-pass tree schedule --- and the others take the NumPy
        route whichever backend is named, since no kernel implements them.
        The two agree to **3.3e-15** absolute on the marginals and
        **1.4e-14** on ``log Z``, inside
        :data:`~snakes_and_ladders.likelihood.device.CROSS_DEVICE_RTOL_FLOAT64`:
        the arithmetic is the same in the same order and NumPy's vectorized
        ``exp`` and ``log`` differ from ``libm``'s in the last place.

    Raises
    ------
    ValueError
        If the tree schedule is asked of a loopy graph, ``damping`` is
        outside ``[0, 1)``, or ``backend`` names an implementation this
        function does not have.
    ConvergenceError
        If flooding does not settle in ``max_iterations`` sweeps.
    """
    run = _run(graph, schedule, False, damping, tolerance, max_iterations, backend)
    variable, factor, log_partition = _beliefs(
        run.layout, run.to_variable, run.to_factor, False, run.plan.defined(run.layout)
    )
    return Marginals(
        variable,
        factor,
        run.plan.log_partition(run.layout, run.to_variable, log_partition, run.scale),
        run.iterations,
        run.plan.guarantee,
        graph.is_tree(),
    )


def max_product(
    graph: FactorGraph,
    *,
    schedule: MessageSchedule | MessageScheduleName | str = MessageScheduleName.TREE,
    damping: float = DEFAULT_DAMPING,
    tolerance: float = DEFAULT_TOLERANCE,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    backend: Backend = Backend.RUST,
) -> tuple[dict[str, int], Marginals]:
    """The MAP assignment by max-marginals, and the max-marginals themselves.

    On a tree with a unique maximum this is the exact MAP -- Viterbi on a
    chain. Ties are broken by the smallest state, which is the tie rule
    :func:`snakes_and_ladders.likelihood.hmm_paths.enumerate_hidden_paths` uses.

    ``backend`` is :func:`sum_product`'s, on the same terms and with the same
    default: the kernel reduces by ``max`` where sum-product reduces by
    ``logsumexp``, and the assignment and its max-marginals are bitwise the
    NumPy route's.
    """
    run = _run(graph, schedule, True, damping, tolerance, max_iterations, backend)
    variable, factor, _ = _beliefs(
        run.layout, run.to_variable, run.to_factor, True, run.plan.defined(run.layout)
    )
    assignment = {name: int(np.argmax(values)) for name, values in variable.items()}
    density = (
        graph.log_density(assignment)
        if len(assignment) == len(graph.variables)
        else math.nan
    )
    return assignment, Marginals(
        variable, factor, density, run.iterations, run.plan.guarantee, graph.is_tree()
    )


#: What this module offers. Declared rather than left implicit: a bare
#: re-export is a private name under ``mypy --strict``, so `MessageScheduleName`
#: --- which every caller of `sum_product` names, and which moved to
#: `schedule` in issue #592 --- would stop type-checking at the call site
#: while the functions beside it passed.
#:
#: `MessageSchedule`, `Layout` and `Step` are *not* here though this module imports
#: them: they are `schedule`'s interface, and listing them gave Sphinx a
#: second target for each, which made every `MessageSchedule` reference ambiguous
#: against `sample.schedule.Schedule` and failed the `-W` docs build.
__all__ = [
    "DEFAULT_DAMPING",
    "DEFAULT_MAX_ITERATIONS",
    "DEFAULT_TOLERANCE",
    "ConvergenceError",
    "Guarantee",
    "Marginals",
    "MessageScheduleName",
    "max_product",
    "sum_product",
]
