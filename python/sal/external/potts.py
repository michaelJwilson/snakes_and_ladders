"""The Potts model's external solvers, in ``search``'s terms (issue #1282, steps 3 and 4).

``sal.external`` is namespaced by problem family, as the package is: this
module holds the Potts calls, and the root only the infrastructure they run
on. :func:`ground_state` (gco, PyMaxflow) mirrors
:func:`sal.search.ground_state.ground_state`; :func:`lower_bound` (HiGHS)
mirrors :func:`sal.search.trws.trws`.

**Ground states.** :func:`ground_state` takes :func:`sal.search.ground_state.ground_state`'s
arguments in its order, with a :class:`~sal.external.solvers.Solver` where
the sibling takes a method name, and returns its
:class:`~sal.search.ground_state.MethodRun`, as :class:`ExternalRun`: the
same fields, the answer's :class:`~sal.external.solvers.Provenance` beside
them.

**The model.** The package's energy is ``-sum_i h_i[s_i] - sum J_ij
[s_i == s_j]``. gco reads ``sum_i D_i(s_i) + sum J_ij [s_i != s_j]`` with
``D_i(a) = -h_i[a]`` less its row minimum (:func:`~sal.external.potts_inputs.expansion_inputs`);
PyMaxflow reads the two-state ferromagnet's cut, built with the capacities
:func:`sal.search.maxflow.ising_ground_state` builds (:func:`~sal.external.potts_inputs.ising_inputs`).
Both live in :mod:`sal.external.potts_inputs`; the adapters
:mod:`sal.validation.gco` and :mod:`sal.validation.pymaxflow` pose their
problems through the same two functions, so the bytes each
framework receives are one construction, and
``tests/regression/test_external_ground_state.py`` pins the labellings equal.

**Forbidden labels (#1139).** A forbidden pair is ``-inf`` in the field
(:func:`sal.sim.potts.forbid`); gco's integer costs carry no ``-inf``, so it
receives :func:`~sal.external.potts_inputs.stand_in`, #1274's large finite unary. Every labelling is
checked to hold no forbidden pair
(:func:`~sal.external.potts_inputs.allowed_by`), and its energy is
read under the field as given. PyMaxflow does not declare
:attr:`~sal.external.solvers.Capability.FORBIDDEN_LABELS` and refuses one.

**Cost.** Neither framework reports site visits, the sibling's unit, nor a
count of its own loop: gco runs its moves to convergence inside one call
and PyMaxflow cuts once. Each solver spends one call in the unit
:data:`UNITS` declares, and a budget in another unit is refused.

**Bounds.** :func:`lower_bound` takes :func:`sal.search.trws.trws`'s ``graph`` and
``field``, in its order, with a :class:`~sal.external.solvers.Solver` after
them, and returns its :class:`~sal.search.alpha_expansion.BoundedLabelling`,
as :class:`ExternalBound`: the same fields, with the cost spent, the
LP's integrality and the answer's :class:`~sal.external.solvers.Provenance`
beside them. TRW-S's keywords set its own loop --- ``max_iterations``,
``tolerance``, ``backend`` --- and HiGHS has no counterpart to any of them,
so none is taken.

**The LP.** HiGHS solves the primal of the relaxation TRW-S ascends the
dual of (#1063), the bytes posed by
:func:`~sal.external.potts_inputs.polytope_inputs`, which the adapter
:func:`sal.validation.highs.local_polytope` sends too, so the bound is the
adapter's value bitwise. The labelling is each site's largest node marginal,
its energy :func:`sal.sim.potts.energy` on ``field``; where every marginal is
integral the LP is tight and that labelling attains the bound.
``integral=True`` solves the ILP with HiGHS's ``milp`` to a zero gap
(#1274): its bound is the minimum and its labelling a minimizer.

**Forbidden labels in a bound.** HiGHS receives
:func:`~sal.external.potts_inputs.stand_in`, #1274's large finite unary, as
gco does. Every allowed labelling keeps its energy and a forbidden one drops
from ``+inf`` to a finite value, so the stand-in's minimum, and every bound on
it, is at most the problem's; the penalty exceeds what a site's couplings can
return, so no LP or ILP optimum puts weight on a forbidden label. The
labelling is checked to hold none (:func:`~sal.external.potts_inputs.allowed_by`).

**A bound's cost.** HiGHS runs its simplex or its branch and bound to its own criterion
inside one call, so a call spends one in :data:`BOUND_UNIT`, gco's unit; its own
count, simplex iterations or branch-and-bound nodes, is the
:class:`~sal.opt.termination.Termination`'s. A run HiGHS stops at its time
limit establishes nothing: its bound is ``-inf`` and its termination the
budget.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field as dataclass_field

import numpy as np

from sal.cost import Cost
from sal.external.potts_inputs import (
    OPTIMAL,
    allowed_by,
    expansion_inputs,
    ising_inputs,
    polytope_inputs,
    stand_in,
)
from sal.external.potts_inputs import integral as is_integral
from sal.external.runner import Run, ScriptError
from sal.external.sessions import Session
from sal.external.solvers import (
    Capability,
    ExternalUnavailable,
    Provenance,
    Solver,
    available,
    invoke,
    provenance,
    require,
)
from sal.opt.budget import Budget
from sal.opt.termination import Termination
from sal.search.alpha_expansion import BoundedLabelling
from sal.search.ground_state import MethodRun
from sal.sim.graph import PottsGraph
from sal.sim.potts import (
    SiteField,
    check_labelling,
    energy,
    log_weight_of,
    site_field,
)

#: The unit each solver's one call is charged in. PyMaxflow's cut is one
#: pass; gco's moves to convergence are one indivisible run of its optimizer.
UNITS: Mapping[Solver, Cost] = {
    Solver.GCO_EXPANSION: Cost.FITS,
    Solver.GCO_SWAP: Cost.FITS,
    Solver.PYMAXFLOW_EXACT: Cost.PASS,
}


@dataclass(frozen=True)
class ExternalRun(MethodRun):
    """A :class:`~sal.search.ground_state.MethodRun` from an external solver, with its provenance."""

    #: The framework, installed version and licence the labelling came from.
    provenance: Provenance = dataclass_field(kw_only=True)


def _labelling(solver: Solver, result: Run) -> np.ndarray:
    """The labelling a script returned, as ``int64`` states."""
    if solver is Solver.PYMAXFLOW_EXACT:
        return result.outputs["sink_side"].astype(bool).astype(np.int64)
    return result.outputs["labels"]


def _posed(
    graph: PottsGraph, values: np.ndarray, task: Capability
) -> tuple[np.ndarray, np.ndarray, set[Capability]]:
    """``values``, the labels it allows, and the capabilities ``task`` on it needs.

    ``values`` is one row per node, finite or ``-inf``, each site allowing a
    label; a problem with more than two states needs
    :attr:`~sal.external.solvers.Capability.MULTI_LABEL`, and one with a
    forbidden label :attr:`~sal.external.solvers.Capability.FORBIDDEN_LABELS`.
    Raises :class:`ValueError` otherwise, before any subprocess starts.
    """
    if values.ndim != 2 or values.shape[0] != graph.n_nodes:
        msg = (
            f"the field is one row per node, ({graph.n_nodes}, n_states); got "
            f"{values.shape}"
        )
        raise ValueError(msg)
    if np.isnan(values).any() or np.isposinf(values).any():
        msg = "the field is finite or -inf, a forbidden label"
        raise ValueError(msg)
    allowed = np.isfinite(values)
    if not allowed.any(axis=1).all():
        site = int(np.argmin(allowed.any(axis=1)))
        msg = f"site {site} allows no label"
        raise ValueError(msg)
    needs = {task}
    if values.shape[1] > 2:
        needs.add(Capability.MULTI_LABEL)
    if not allowed.all():
        needs.add(Capability.FORBIDDEN_LABELS)
    return values, allowed, needs


def _served_by(session: Session | None, solver: Solver) -> None:
    """Refuse a ``session`` opened on another solver than ``solver``."""
    if session is not None and session.solver is not solver:
        msg = f"the session serves {session.solver}, not {solver}"
        raise ValueError(msg)


def ground_state(
    graph: PottsGraph,
    field: SiteField | np.ndarray,
    solver: Solver,
    budget: Budget,
    rng: np.random.Generator,
    *,
    start: np.ndarray | None = None,
    session: Session | None = None,
) -> ExternalRun:
    """``solver``'s ground state of the Potts model ``(graph, field)``, as :func:`sal.search.ground_state.ground_state` returns one.

    The checks run in this process, before any subprocess starts: the
    capabilities the problem needs (:func:`~sal.external.solvers.require`),
    then the framework's presence. Both solvers are deterministic, so
    ``rng`` is taken for the sibling's signature and not drawn from.

    Parameters
    ----------
    graph : PottsGraph
        The lattice; every coupling non-negative.
    field : SiteField | np.ndarray
        ``h``, shape ``(n_nodes, n_states)``; ``-inf`` marks a forbidden
        label (:func:`sal.sim.potts.forbid`).
    solver : Solver
        One that declares :attr:`~sal.external.solvers.Capability.GROUND_STATE`.
    budget : Budget
        In ``UNITS[solver]``; the call spends one.
    rng : np.random.Generator
        Not drawn from.
    start : np.ndarray | None
        gco's initial labelling, shape ``(n_nodes,)``; gco's own default is
        label 0 everywhere. PyMaxflow's cut has none and refuses one.
    session : Session | None
        A worker :func:`sal.external.session` opened on ``solver``, which
        serves the call in place of a fresh subprocess.

    Returns
    -------
    ExternalRun
        The labelling, its energy under :func:`sal.sim.potts.energy` on
        ``field``, ``spent = 1``, the seconds the script measured around the
        framework's own call, a converged :class:`~sal.opt.termination.Termination`
        after one call, and the :class:`~sal.external.solvers.Provenance`.

    Raises
    ------
    CapabilityRefused
        If the problem needs what ``solver`` does not declare: more than two
        states, or a forbidden label.
    ExternalUnavailable
        If the framework is not installed.
    ValueError
        If ``field`` is not one row per node, holds ``nan`` or ``+inf``, or
        a site allows no label; the budget is in another unit; ``start`` is
        out of range, or given to PyMaxflow; ``session`` serves another
        solver.
    ScriptError
        If the framework fails, or returns a forbidden label.
    """
    # Both frameworks are deterministic: nothing is drawn.
    del rng
    values, allowed, needs = _posed(
        graph,
        np.asarray(log_weight_of(field), dtype=np.float64),
        Capability.GROUND_STATE,
    )
    n_states = int(values.shape[1])
    require(solver, needs)
    if budget.unit is not UNITS[solver]:
        msg = f"{solver} is charged in {UNITS[solver]}, not {budget.unit}"
        raise ValueError(msg)
    _served_by(session, solver)
    if start is not None:
        if solver is Solver.PYMAXFLOW_EXACT:
            msg = f"{solver} cuts once from no labelling, so takes no start"
            raise ValueError(msg)
        start = check_labelling(start, graph.n_nodes, n_states)
    if session is None and not available(solver):
        raise ExternalUnavailable(solver)

    if solver is Solver.PYMAXFLOW_EXACT:
        inputs = ising_inputs(graph, values)
    else:
        sent = values if allowed.all() else stand_in(graph, values, allowed)
        inputs = expansion_inputs(
            graph, sent, start=start, swap=solver is Solver.GCO_SWAP
        )
    result = (
        invoke(solver, needs, inputs)
        if session is None
        else session.invoke(needs, inputs)
    )
    labelling = _labelling(solver, result)
    if not allowed_by(allowed, labelling):
        msg = f"{solver} returned a forbidden label"
        raise ScriptError(msg)
    return ExternalRun(
        labelling=labelling,
        energy=energy(graph, values, labelling),
        spent=1,
        seconds=result.seconds,
        # One call, run by the framework to its own criterion: gco's cycle
        # that lowers nothing, or the cut's maximum flow.
        termination=Termination.after(1, converged=True),
        provenance=provenance(solver),
    )


#: The unit :func:`lower_bound` is charged in: one run of HiGHS to its own criterion.
BOUND_UNIT = Cost.FITS

#: HiGHS's status where an iteration or time limit stopped it.
LIMIT = 1


@dataclass(frozen=True, kw_only=True)
class ExternalBound(BoundedLabelling):
    """A :class:`~sal.search.alpha_expansion.BoundedLabelling` from an external solver, with what it spent and its provenance."""

    #: Calls spent, in :data:`BOUND_UNIT`.
    spent: int
    #: Whether every node marginal of the optimum is within
    #: :data:`~sal.external.potts_inputs.INTEGRALITY` of 0 or 1: the LP is tight and the labelling
    #: attains the bound. Always so for the ILP.
    integral: bool
    #: Wall seconds of HiGHS's own call, as the script measured them.
    seconds: float
    #: The framework, installed version and licence the bound came from.
    provenance: Provenance


def lower_bound(
    graph: PottsGraph,
    field: SiteField | np.ndarray,
    solver: Solver,
    *,
    integral: bool = False,
    timeout: float = 600.0,
    session: Session | None = None,
) -> ExternalBound:
    """``solver``'s lower bound on the minimum of :func:`sal.sim.potts.energy`, as :func:`sal.search.trws.trws` returns one.

    The checks run in this process, before any subprocess starts: the
    capabilities the problem needs (:func:`~sal.external.solvers.require`),
    then the framework's presence.

    Parameters
    ----------
    graph : PottsGraph
        The graph and its per-edge couplings, of either sign.
    field : SiteField | np.ndarray
        External field as a log-weight, ``(n_states,)`` or
        ``(n_nodes, n_states)``, or a :class:`~sal.sim.potts.SiteField`, as
        ``trws`` reads it; ``-inf`` marks a forbidden label
        (:func:`sal.sim.potts.forbid`).
    solver : Solver
        One that declares :attr:`~sal.external.solvers.Capability.LOWER_BOUND`.
    integral : bool
        Solve the ILP in place of the LP; needs
        :attr:`~sal.external.solvers.Capability.EXACT`.
    timeout : float
        Seconds the subprocess may take; HiGHS stops itself at 0.9 of it.
    session : Session | None
        A worker :func:`sal.external.session` opened on ``solver``, which
        serves the call in place of a fresh subprocess.

    Returns
    -------
    ExternalBound
        The optimum's value as ``bound``, the labelling of each site's
        largest marginal and its energy on ``field``, ``spent = 1``, the
        :class:`~sal.opt.termination.Termination` after HiGHS's own count,
        the integrality flag, and the :class:`~sal.external.solvers.Provenance`.

    Raises
    ------
    CapabilityRefused
        If ``solver`` gives no bound, or lacks what the problem needs: more
        than two states, a forbidden label, or the exact ILP.
    ExternalUnavailable
        If the framework is not installed.
    ValueError
        If ``field`` has neither shape, holds ``nan`` or ``+inf``, or a site
        allows no label; ``session`` serves another solver.
    ScriptError
        If HiGHS reports neither an optimum nor a limit, or returns a
        forbidden label.
    """
    values, allowed, needs = _posed(
        graph,
        site_field(np.asarray(log_weight_of(field), dtype=np.float64), graph.n_nodes),
        Capability.LOWER_BOUND,
    )
    if integral:
        needs.add(Capability.EXACT)
    require(solver, needs)
    _served_by(session, solver)
    if session is None and not available(solver):
        raise ExternalUnavailable(solver)

    inputs = polytope_inputs(
        graph,
        values if allowed.all() else stand_in(graph, values, allowed),
        integral=integral,
        time_limit=0.9 * timeout,
    )
    result = (
        invoke(solver, needs, inputs, timeout=timeout)
        if session is None
        else session.invoke(needs, inputs, timeout=timeout)
    )
    outputs = result.outputs
    status = int(outputs["status"])
    if status not in (OPTIMAL, LIMIT):
        msg = f"{solver} returned status {status}: {outputs['message']}"
        raise ScriptError(msg)
    marginals = outputs["node_marginals"]
    solved = status == OPTIMAL and bool(np.isfinite(marginals).all())
    # Without an optimum the marginals may be absent: each site's best
    # allowed label stands in, so the labelling is still one the field allows.
    labelling = np.argmax(marginals if solved else values, axis=1).astype(np.int64)
    if not allowed_by(allowed, labelling):
        msg = f"{solver} returned a forbidden label"
        raise ScriptError(msg)
    return ExternalBound(
        labelling=labelling,
        energy=energy(graph, values, labelling),
        bound=float(outputs["value"]) if solved else -np.inf,
        termination=Termination.after(int(outputs["iterations"]), converged=solved),
        spent=1,
        integral=solved and is_integral(marginals),
        seconds=result.seconds,
        provenance=provenance(solver),
    )
