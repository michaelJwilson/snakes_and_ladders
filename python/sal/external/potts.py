"""Potts ground states from gco and PyMaxflow, in ``search.ground_state``'s terms (issue #1282, step 3).

:func:`ground_state` takes :func:`sal.search.ground_state.ground_state`'s
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
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field as dataclass_field

import numpy as np

from sal.cost import Cost
from sal.external.potts_inputs import (
    allowed_by,
    expansion_inputs,
    ising_inputs,
    stand_in,
)
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
from sal.search.ground_state import MethodRun
from sal.sim.graph import PottsGraph
from sal.sim.potts import SiteField, check_labelling, energy, log_weight_of

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
    values = np.asarray(log_weight_of(field), dtype=np.float64)
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
    n_states = int(values.shape[1])
    needs = {Capability.GROUND_STATE}
    if n_states > 2:
        needs.add(Capability.MULTI_LABEL)
    if not allowed.all():
        needs.add(Capability.FORBIDDEN_LABELS)
    require(solver, needs)
    if budget.unit is not UNITS[solver]:
        msg = f"{solver} is charged in {UNITS[solver]}, not {budget.unit}"
        raise ValueError(msg)
    if session is not None and session.solver is not solver:
        msg = f"the session serves {session.solver}, not {solver}"
        raise ValueError(msg)
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
        posed = values if allowed.all() else stand_in(graph, values, allowed)
        inputs = expansion_inputs(
            graph, posed, start=start, swap=solver is Solver.GCO_SWAP
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
