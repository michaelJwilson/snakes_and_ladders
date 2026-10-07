"""The Potts model's external solvers, in ``search``'s terms (issue #1282, steps 3 and 4).

``sal.external`` is namespaced by problem family, as the package is: this
module holds the Potts calls, and the root only the infrastructure they run
on. :data:`ground_state` (gco, PyMaxflow, OpenGM's ICM, loopy BP, A*,
expansion and swap) mirrors :func:`sal.search.ground_state.ground_state`;
:data:`lower_bound` (HiGHS, OpenGM's TRW-S and dual decomposition) mirrors
:func:`sal.search.trws.trws`.

**Ground states.** :data:`ground_state` takes :func:`sal.search.ground_state.ground_state`'s
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

**OpenGM (#1279).** The script loads ``libsal_opengm.so``, which
``infra/build_opengm.sh`` compiles from OpenGM's headers; the bytes are
posed by :func:`~sal.external.potts_inputs.opengm_inputs`: the unary
``-field``, each edge ``-J [a == b]``, so OpenGM's value is the package's
energy with no constant restored. A forbidden label is
:func:`~sal.external.potts_inputs.stand_in`, as for gco. Each algorithm runs
its own loop to :data:`OPENGM_STEPS`, OpenGM's own defaults: one call, one
unit. Expansion and swap need a metric pairwise term, so a negative
coupling is refused before the subprocess; loopy BP and A* take no start.

**Cost.** No framework reports site visits, the sibling's unit, nor a
count of its own loop: gco runs its moves to convergence inside one call
and PyMaxflow cuts once. Each solver spends one call in the unit
:data:`UNITS` declares, and a budget in another unit is refused.

**Bounds.** :data:`lower_bound` takes :func:`sal.search.trws.trws`'s ``graph`` and
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

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from dataclasses import field as dataclass_field

import numpy as np

from sal.cost import Cost
from sal.external.frameworks import Framework
from sal.external.potts_inputs import (
    INTEGRALITY,
    OPTIMAL,
    allowed_by,
    expansion_inputs,
    ising_inputs,
    opengm_inputs,
    polytope_inputs,
    stand_in,
)
from sal.external.potts_inputs import integral as is_integral
from sal.external.runner import Run, ScriptError
from sal.external.sessions import Session, served_by
from sal.external.solvers import (
    Capability,
    ExternalUnavailable,
    Provenance,
    Solver,
    available,
    invoke,
    passed,
    provenance,
    refuse_framework,
    require,
    unmatched,
)
from sal.opt.budget import Budget
from sal.opt.termination import Termination, check_cap
from sal.search.alpha_expansion import BoundedLabelling
from sal.search.ground_state import MethodRun
from sal.search.trws import MAX_ITERATIONS as TRWS_ITERATIONS
from sal.search.trws import TOLERANCE as TRWS_TOLERANCE
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
    Solver.OPENGM_ICM: Cost.FITS,
    Solver.OPENGM_LBP: Cost.FITS,
    Solver.OPENGM_ASTAR: Cost.FITS,
    Solver.OPENGM_EXPANSION: Cost.FITS,
    Solver.OPENGM_SWAP: Cost.FITS,
}

#: The iteration cap each OpenGM ground-state algorithm runs to: OpenGM's
#: own default for loopy BP (100), expansion and swap (1000 each); ICM and
#: A* run to their own end and read none.
OPENGM_STEPS: Mapping[Solver, int] = {
    Solver.OPENGM_ICM: 1,
    Solver.OPENGM_LBP: 100,
    Solver.OPENGM_ASTAR: 1,
    Solver.OPENGM_EXPANSION: 1000,
    Solver.OPENGM_SWAP: 1000,
}

#: The solvers OpenGM's metric moves run, which refuse a negative coupling.
_METRIC = frozenset({Solver.OPENGM_EXPANSION, Solver.OPENGM_SWAP})


def _algorithm(solver: Solver) -> str:
    """The key of :data:`~sal.external.potts_inputs.OPENGM_ALGORITHMS` ``solver`` runs."""
    return str(solver).removeprefix("opengm_")


@dataclass(frozen=True)
class ExternalRun(MethodRun):
    """A :class:`~sal.search.ground_state.MethodRun` from an external solver, with its provenance."""

    #: The framework, installed version and licence the labelling came from.
    provenance: Provenance = dataclass_field(kw_only=True)


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


@dataclass(frozen=True)
class _Problem:
    """A Potts problem once checked: its field, the labels it allows, what it needs, its start."""

    values: np.ndarray
    allowed: np.ndarray
    needs: set[Capability]
    start: np.ndarray | None

    def sent(self, graph: PottsGraph) -> np.ndarray:
        """The field a framework receives: :attr:`values`, or its stand-in where a label is forbidden."""
        if self.allowed.all():
            return self.values
        return stand_in(graph, self.values, self.allowed)


def _ground_problem(
    graph: PottsGraph,
    field: SiteField | np.ndarray,
    solver: Solver,
    budget: Budget,
    start: np.ndarray | None,
    session: Session | None,
) -> _Problem:
    """Every ground-state call's checks, in this process, before any subprocess starts.

    The field's form, the capabilities it needs (:func:`~sal.external.solvers.require`),
    the budget's unit, the session's solver, the start's range, a metric
    pairwise term for OpenGM's moves, then the framework's presence.
    """
    values, allowed, needs = _posed(
        graph,
        np.asarray(log_weight_of(field), dtype=np.float64),
        Capability.GROUND_STATE,
    )
    require(solver, needs)
    if budget.unit is not UNITS[solver]:
        msg = f"{solver} is charged in {UNITS[solver]}, not {budget.unit}"
        raise ValueError(msg)
    served_by(session, solver)
    if start is not None:
        start = check_labelling(start, graph.n_nodes, int(values.shape[1]))
    if solver in _METRIC and (graph.edge_coupling < 0).any():
        msg = f"{solver} needs a metric pairwise term: every coupling >= 0"
        raise ValueError(msg)
    if session is None and not available(solver):
        raise ExternalUnavailable(solver)
    return _Problem(values, allowed, needs, start)


def _labels(result: Run) -> np.ndarray:
    """The labelling a script returned under ``labels``."""
    return result.outputs["labels"]


def _settled(result: Run) -> bool:
    """One call run by the framework to its own criterion."""
    del result
    return True


def _ground_run(
    graph: PottsGraph,
    problem: _Problem,
    solver: Solver,
    inputs: Mapping[str, np.ndarray],
    session: Session | None,
    *,
    read: Callable[[Run], np.ndarray] = _labels,
    converged: Callable[[Run], bool] = _settled,
) -> ExternalRun:
    """``solver``'s script on ``inputs``, its labelling checked and scored as an :class:`ExternalRun`."""
    result = (
        invoke(solver, problem.needs, inputs)
        if session is None
        else session.invoke(problem.needs, inputs)
    )
    labelling = read(result)
    if not allowed_by(problem.allowed, labelling):
        msg = f"{solver} returned a forbidden label"
        raise ScriptError(msg)
    return ExternalRun(
        labelling=labelling,
        energy=energy(graph, problem.values, labelling),
        spent=1,
        seconds=result.seconds,
        termination=Termination.after(1, converged=converged(result)),
        provenance=provenance(solver),
    )


def _gco(
    graph: PottsGraph,
    field: SiteField | np.ndarray,
    budget: Budget,
    solver: Solver,
    start: np.ndarray | None,
    session: Session | None,
) -> ExternalRun:
    """gco's expansion or swap, run to convergence inside one call."""
    problem = _ground_problem(graph, field, solver, budget, start, session)
    inputs = expansion_inputs(
        graph,
        problem.sent(graph),
        start=problem.start,
        swap=solver is Solver.GCO_SWAP,
    )
    return _ground_run(graph, problem, solver, inputs, session)


def _opengm_ground_state(
    graph: PottsGraph,
    field: SiteField | np.ndarray,
    budget: Budget,
    solver: Solver,
    start: np.ndarray | None,
    session: Session | None,
) -> ExternalRun:
    """One OpenGM algorithm's labelling, run to :data:`OPENGM_STEPS` iterations at most."""
    problem = _ground_problem(graph, field, solver, budget, start, session)
    steps = OPENGM_STEPS[solver]
    inputs = opengm_inputs(
        graph,
        problem.sent(graph),
        _algorithm(solver),
        max_iterations=steps,
        tolerance=0.0,
        start=problem.start,
    )

    def converged(result: Run) -> bool:
        # Loopy BP alone may reach its cap first, which its trace's length says.
        return (
            solver is not Solver.OPENGM_LBP or result.outputs["trace"].shape[0] < steps
        )

    return _ground_run(graph, problem, solver, inputs, session, converged=converged)


def _moves(
    graph: PottsGraph,
    field: SiteField | np.ndarray,
    budget: Budget,
    solver: Solver,
    start: np.ndarray | None,
    session: Session | None,
) -> ExternalRun:
    """Expansion or swap, by the framework ``solver`` runs: gco's or OpenGM's."""
    if solver.framework.name == Framework.GCO:
        return _gco(graph, field, budget, solver, start, session)
    return _opengm_ground_state(graph, field, budget, solver, start, session)


class GroundState:
    """:data:`ground_state`: the solver-keyed call, and one explicit call per algorithm (#1304).

    ``ground_state(graph, field, solver, budget, rng, ...)`` is a ``match``
    on ``solver`` onto one of the explicit calls below, which hold the
    algorithms; each takes the sibling's ``graph``, ``field``, ``budget``
    and ``rng`` and the keywords its algorithm reads. Where several
    frameworks run one algorithm, ``by`` names one.

    Every call checks in this process, before any subprocess starts: the
    capabilities the problem needs (:func:`~sal.external.solvers.require`),
    the budget's unit, then the framework's presence. Every framework is
    deterministic, so ``rng`` is taken for the sibling's signature and not
    drawn from. Each returns :class:`ExternalRun`: the labelling, its energy
    under :func:`sal.sim.potts.energy` on ``field``, ``spent = 1``, the
    seconds the script measured around the framework's own call, the
    :class:`~sal.opt.termination.Termination` after one call, and the
    :class:`~sal.external.solvers.Provenance`. Each raises
    :class:`~sal.external.solvers.CapabilityRefused` where the problem needs
    what the solver does not declare (more than two states, a forbidden
    label), :class:`~sal.external.solvers.ExternalUnavailable` where the
    framework is not installed or built, :class:`ValueError` where ``field``
    is not one row per node, holds ``nan`` or ``+inf``, or a site allows no
    label, the budget is in another unit than :data:`UNITS` states,
    ``start`` is out of range, or ``session`` serves another solver, and
    :class:`~sal.external.runner.ScriptError` where the framework fails or
    returns a forbidden label.
    """

    @staticmethod
    def expansion(
        graph: PottsGraph,
        field: SiteField | np.ndarray,
        budget: Budget,
        rng: np.random.Generator,
        *,
        by: Framework = Framework.GCO,
        start: np.ndarray | None = None,
        session: Session | None = None,
    ) -> ExternalRun:
        """Alpha expansion, by gco (the default) or OpenGM.

        ``budget`` is in :data:`UNITS`' unit for the framework's solver;
        ``start`` the initial labelling, label 0 everywhere by default.
        OpenGM's needs a metric pairwise term and refuses a negative
        coupling; any other ``by`` is refused.
        """
        del rng
        match by:
            case Framework.GCO:
                solver = Solver.GCO_EXPANSION
            case Framework.OPENGM:
                solver = Solver.OPENGM_EXPANSION
            case _:
                refuse_framework("expansion", by, _MOVERS)
        return _moves(graph, field, budget, solver, start, session)

    @staticmethod
    def swap(
        graph: PottsGraph,
        field: SiteField | np.ndarray,
        budget: Budget,
        rng: np.random.Generator,
        *,
        by: Framework = Framework.GCO,
        start: np.ndarray | None = None,
        session: Session | None = None,
    ) -> ExternalRun:
        """Alpha-beta swap, by gco (the default) or OpenGM, read as :meth:`expansion` reads its arguments."""
        del rng
        match by:
            case Framework.GCO:
                solver = Solver.GCO_SWAP
            case Framework.OPENGM:
                solver = Solver.OPENGM_SWAP
            case _:
                refuse_framework("swap", by, _MOVERS)
        return _moves(graph, field, budget, solver, start, session)

    @staticmethod
    def min_cut(
        graph: PottsGraph,
        field: SiteField | np.ndarray,
        budget: Budget,
        rng: np.random.Generator,
        *,
        session: Session | None = None,
    ) -> ExternalRun:
        """PyMaxflow's minimum cut: the exact ground state at q = 2, in one pass from no start.

        The capacities are :func:`sal.search.maxflow.ising_ground_state`'s
        (:func:`~sal.external.potts_inputs.ising_inputs`); more than two
        states or a forbidden label is refused.
        """
        del rng
        solver = Solver.PYMAXFLOW_EXACT
        problem = _ground_problem(graph, field, solver, budget, None, session)
        return _ground_run(
            graph,
            problem,
            solver,
            ising_inputs(graph, problem.values),
            session,
            read=_sink_side,
        )

    @staticmethod
    def icm(
        graph: PottsGraph,
        field: SiteField | np.ndarray,
        budget: Budget,
        rng: np.random.Generator,
        *,
        start: np.ndarray | None = None,
        session: Session | None = None,
    ) -> ExternalRun:
        """OpenGM's iterated conditional modes, from ``start``, label 0 everywhere by default."""
        del rng
        return _opengm_ground_state(
            graph, field, budget, Solver.OPENGM_ICM, start, session
        )

    @staticmethod
    def loopy_bp(
        graph: PottsGraph,
        field: SiteField | np.ndarray,
        budget: Budget,
        rng: np.random.Generator,
        *,
        session: Session | None = None,
    ) -> ExternalRun:
        """OpenGM's loopy belief propagation, from no start; unconverged where it reaches its cap."""
        del rng
        return _opengm_ground_state(
            graph, field, budget, Solver.OPENGM_LBP, None, session
        )

    @staticmethod
    def astar(
        graph: PottsGraph,
        field: SiteField | np.ndarray,
        budget: Budget,
        rng: np.random.Generator,
        *,
        session: Session | None = None,
    ) -> ExternalRun:
        """OpenGM's A*, the exact ground state, from no start."""
        del rng
        return _opengm_ground_state(
            graph, field, budget, Solver.OPENGM_ASTAR, None, session
        )

    def __call__(
        self,
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

        The sibling's arguments in its order, with a
        :class:`~sal.external.solvers.Solver` for its method; a ``match`` on
        ``solver`` onto its explicit call, with ``by`` its framework. A
        keyword the explicit call does not take --- ``start`` for
        PyMaxflow's cut, OpenGM's loopy BP or A* --- is refused with
        :class:`ValueError`, and a solver with no ground state with
        :class:`~sal.external.solvers.CapabilityRefused`, each before any
        subprocess starts.

        Parameters
        ----------
        graph : PottsGraph
            The lattice.
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
            The initial labelling, shape ``(n_nodes,)``, of the calls that take one.
        session : Session | None
            A worker :func:`sal.external.session` opened on ``solver``.

        Returns
        -------
        ExternalRun
            The explicit call's result, unchanged.
        """
        by: Framework | None = None
        call: Callable[..., ExternalRun]
        match solver:
            case Solver.GCO_EXPANSION:
                call, by = self.expansion, Framework.GCO
            case Solver.OPENGM_EXPANSION:
                call, by = self.expansion, Framework.OPENGM
            case Solver.GCO_SWAP:
                call, by = self.swap, Framework.GCO
            case Solver.OPENGM_SWAP:
                call, by = self.swap, Framework.OPENGM
            case Solver.PYMAXFLOW_EXACT:
                call = self.min_cut
            case Solver.OPENGM_ICM:
                call = self.icm
            case Solver.OPENGM_LBP:
                call = self.loopy_bp
            case Solver.OPENGM_ASTAR:
                call = self.astar
            case _:
                unmatched(solver, Capability.GROUND_STATE)
        keywords = passed(
            solver, call, {"start": (start, None), "session": (session, None)}
        )
        if by is not None:
            keywords["by"] = by
        return call(graph, field, budget, rng, **keywords)


#: The frameworks that run alpha expansion and alpha-beta swap.
_MOVERS = (Framework.GCO, Framework.OPENGM)


def _sink_side(result: Run) -> np.ndarray:
    """PyMaxflow's sink side as ``int64`` states."""
    return result.outputs["sink_side"].astype(bool).astype(np.int64)


#: ``ground_state(graph, field, solver, budget, rng, *, start, session)``,
#: the solver-keyed call, and its explicit calls ``ground_state.expansion``,
#: ``.swap``, ``.min_cut``, ``.icm``, ``.loopy_bp`` and ``.astar``
#: (:class:`GroundState`).
ground_state = GroundState()


#: The unit :data:`lower_bound` is charged in: one run of a solver to its own criterion.
BOUND_UNIT = Cost.FITS

#: The iterations OpenGM's dual decomposition runs at most by default,
#: OpenGM's own; TRW-S's is :func:`sal.search.trws.trws`'s.
DD_ITERATIONS = 100

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


def _bound_problem(
    graph: PottsGraph,
    field: SiteField | np.ndarray,
    solver: Solver,
    session: Session | None,
    *,
    exact: bool = False,
) -> _Problem:
    """Every bound's checks, in this process, before any subprocess starts.

    The field's form, the capabilities it needs (with
    :attr:`~sal.external.solvers.Capability.EXACT` where ``exact``), the
    session's solver, then the framework's presence.
    """
    values, allowed, needs = _posed(
        graph,
        site_field(np.asarray(log_weight_of(field), dtype=np.float64), graph.n_nodes),
        Capability.LOWER_BOUND,
    )
    if exact:
        needs.add(Capability.EXACT)
    require(solver, needs)
    served_by(session, solver)
    if session is None and not available(solver):
        raise ExternalUnavailable(solver)
    return _Problem(values, allowed, needs, None)


def _bound_run(
    problem: _Problem,
    solver: Solver,
    inputs: Mapping[str, np.ndarray],
    timeout: float,
    session: Session | None,
) -> Run:
    """``solver``'s script on ``inputs``, through ``session`` where one is open."""
    if session is None:
        return invoke(solver, problem.needs, inputs, timeout=timeout)
    return session.invoke(problem.needs, inputs, timeout=timeout)


def _opengm_bound(
    graph: PottsGraph,
    field: SiteField | np.ndarray,
    solver: Solver,
    *,
    max_iterations: int,
    tolerance: float,
    timeout: float,
    session: Session | None,
) -> ExternalBound:
    """OpenGM's TRW-S or dual decomposition bound, checked as :meth:`LowerBound.lp` checks HiGHS's."""
    problem = _bound_problem(graph, field, solver, session)
    steps = check_cap("max_iterations", max_iterations)
    inputs = opengm_inputs(
        graph,
        problem.sent(graph),
        _algorithm(solver),
        max_iterations=steps,
        tolerance=tolerance,
    )
    result = _bound_run(problem, solver, inputs, timeout, session)
    labelling = result.outputs["labels"]
    if not allowed_by(problem.allowed, labelling):
        msg = f"{solver} returned a forbidden label"
        raise ScriptError(msg)
    bound = float(result.outputs["bound"])
    found = energy(graph, problem.values, labelling)
    taken = int(result.outputs["trace"].shape[0])
    return ExternalBound(
        labelling=labelling,
        energy=found,
        bound=bound,
        termination=Termination.after(taken, converged=taken < steps),
        spent=1,
        # The labelling attains the bound: the relaxation is tight on it.
        integral=bool(found - bound <= INTEGRALITY * max(1.0, abs(bound))),
        seconds=result.seconds,
        provenance=provenance(solver),
    )


class LowerBound:
    """:data:`lower_bound`: the solver-keyed call, and one explicit call per algorithm (#1304).

    ``lower_bound(graph, field, solver, ...)`` is a ``match`` on ``solver``
    onto one of the explicit calls below, which hold the algorithms; each
    takes :func:`sal.search.trws.trws`'s ``graph`` and ``field`` and the
    keywords its algorithm reads, and returns :class:`ExternalBound`: the
    bound, the labelling and its energy on ``field``, ``spent = 1``, the
    :class:`~sal.opt.termination.Termination` after the framework's own
    count, the integrality flag, and the
    :class:`~sal.external.solvers.Provenance`.

    ``field`` is a log-weight, ``(n_states,)`` or ``(n_nodes, n_states)``,
    or a :class:`~sal.sim.potts.SiteField`, as ``trws`` reads it; ``-inf``
    marks a forbidden label (:func:`sal.sim.potts.forbid`). ``timeout`` is
    the seconds the subprocess may take, and ``session`` a worker
    :func:`sal.external.session` opened on the call's solver. Every call
    checks in this process, before any subprocess starts, and raises
    :class:`~sal.external.solvers.CapabilityRefused` where the problem needs
    what the solver lacks, :class:`~sal.external.solvers.ExternalUnavailable`
    where the framework is absent, :class:`ValueError` where ``field`` has
    neither shape, holds ``nan`` or ``+inf``, or a site allows no label, or
    ``session`` serves another solver, and
    :class:`~sal.external.runner.ScriptError` where the framework fails or
    returns a forbidden label.
    """

    @staticmethod
    def lp(
        graph: PottsGraph,
        field: SiteField | np.ndarray,
        *,
        integral: bool = False,
        timeout: float = 600.0,
        session: Session | None = None,
    ) -> ExternalBound:
        """HiGHS's local-polytope LP, or with ``integral`` its ILP to a zero gap.

        HiGHS runs to its own criterion and stops itself at 0.9 of
        ``timeout``; a run it stops at that limit establishes nothing, its
        bound ``-inf``. The labelling is each site's largest node marginal.
        ``integral`` needs :attr:`~sal.external.solvers.Capability.EXACT`.
        """
        solver = Solver.HIGHS_LP
        problem = _bound_problem(graph, field, solver, session, exact=integral)
        values = problem.values
        inputs = polytope_inputs(
            graph,
            problem.sent(graph),
            integral=integral,
            time_limit=0.9 * timeout,
        )
        result = _bound_run(problem, solver, inputs, timeout, session)
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
        if not allowed_by(problem.allowed, labelling):
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

    @staticmethod
    def trws(
        graph: PottsGraph,
        field: SiteField | np.ndarray,
        *,
        max_iterations: int = TRWS_ITERATIONS,
        tolerance: float = TRWS_TOLERANCE,
        timeout: float = 600.0,
        session: Session | None = None,
    ) -> ExternalBound:
        """OpenGM's TRW-S, with :func:`sal.search.trws.trws`'s loop keywords and defaults.

        At most ``max_iterations`` iterations, at least 1; it stops once the
        bound rises by at most ``tolerance |bound|`` in an iteration or the
        gap closes to that, as :func:`sal.search.trws.trws` reads its own
        with ``max(1, |bound|)``.
        """
        return _opengm_bound(
            graph,
            field,
            Solver.OPENGM_TRWS,
            max_iterations=max_iterations,
            tolerance=tolerance,
            timeout=timeout,
            session=session,
        )

    @staticmethod
    def dual_decomposition(
        graph: PottsGraph,
        field: SiteField | np.ndarray,
        *,
        max_iterations: int = DD_ITERATIONS,
        tolerance: float = TRWS_TOLERANCE,
        timeout: float = 600.0,
        session: Session | None = None,
    ) -> ExternalBound:
        """OpenGM's dual decomposition: at most ``max_iterations``, OpenGM's default, stopped once the relative gap closes to ``tolerance``."""
        return _opengm_bound(
            graph,
            field,
            Solver.OPENGM_DD,
            max_iterations=max_iterations,
            tolerance=tolerance,
            timeout=timeout,
            session=session,
        )

    def __call__(
        self,
        graph: PottsGraph,
        field: SiteField | np.ndarray,
        solver: Solver,
        *,
        integral: bool = False,
        max_iterations: int | None = None,
        tolerance: float = TRWS_TOLERANCE,
        timeout: float = 600.0,
        session: Session | None = None,
    ) -> ExternalBound:
        """``solver``'s lower bound on the minimum of :func:`sal.sim.potts.energy`, as :func:`sal.search.trws.trws` returns one.

        ``trws``'s ``graph`` and ``field`` in its order, with a
        :class:`~sal.external.solvers.Solver` after them; a ``match`` on
        ``solver`` onto its explicit call. A keyword set away from its
        default is passed where the explicit call takes it and refused with
        :class:`ValueError` where it does not --- ``max_iterations`` or
        ``tolerance`` for HiGHS, ``integral`` for OpenGM --- and a solver
        with no bound with :class:`~sal.external.solvers.CapabilityRefused`,
        each before any subprocess starts.

        Parameters
        ----------
        graph : PottsGraph
            The graph and its per-edge couplings, of either sign.
        field : SiteField | np.ndarray
            As :class:`LowerBound` reads it.
        solver : Solver
            One that declares :attr:`~sal.external.solvers.Capability.LOWER_BOUND`.
        integral : bool
            :meth:`LowerBound.lp`'s.
        max_iterations : int | None
            OpenGM's iterations at most; its explicit call's default where ``None``.
        tolerance : float
            OpenGM's stopping tolerance.
        timeout : float
            Seconds the subprocess may take.
        session : Session | None
            A worker :func:`sal.external.session` opened on ``solver``.

        Returns
        -------
        ExternalBound
            The explicit call's result, unchanged.
        """
        call: Callable[..., ExternalBound]
        match solver:
            case Solver.HIGHS_LP:
                call = self.lp
            case Solver.OPENGM_TRWS:
                call = self.trws
            case Solver.OPENGM_DD:
                call = self.dual_decomposition
            case _:
                unmatched(solver, Capability.LOWER_BOUND)
        keywords = passed(
            solver,
            call,
            {
                "integral": (integral, False),
                "max_iterations": (max_iterations, None),
                "tolerance": (tolerance, TRWS_TOLERANCE),
                "timeout": (timeout, 600.0),
                "session": (session, None),
            },
        )
        return call(graph, field, **keywords)


#: ``lower_bound(graph, field, solver, *, integral, max_iterations,
#: tolerance, timeout, session)``, the solver-keyed call, and its explicit
#: calls ``lower_bound.lp``, ``.trws`` and ``.dual_decomposition``
#: (:class:`LowerBound`).
lower_bound = LowerBound()
