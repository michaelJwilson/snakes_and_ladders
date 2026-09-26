"""Ground-state recovery on a Potts lattice in a per-site field, refereed twice.

Issue #551 asks which sampler or minimizer recovers the ground state of
`spatio_only/release` --- open triangular 71 x 71, q = 10, J = 0.7 --- and
answers it on three rungs that differ in what can referee an energy:
enumeration at nine sites, a graph cut at 5,041 sites and two states, and a
bracket at 5,041 sites and ten. Nothing here implements a method; it
constructs the rungs, charges every entry in one unit, and scores a labelling
by the two referees the ticket names.

**The two referees.** An energy alone cannot separate methods at the third
rung, where the bracket may be wider than the differences being ranked. The
second referee is the simulated truth --- the parameters that built the field
--- and it is exact at every size: the size tilt, the null class that carries
no field anywhere, and the per-class occupancy. A labelling can pass one and
fail the other, and a low energy that fails the structure is a low-energy
state of a *different* model.

**A ground state's tilt is not a thermal draw's tilt, and the difference has a
sign.** `spatio_only/release.yaml` records a tilt of 0.4935 over eight chains
of 60 sweeps at J = 0.7. That is a finite-temperature statistic. As the
temperature falls a ferromagnet orders, domain walls stop being affordable,
and the majority class takes sites whose own field points elsewhere --- so the
tilt of a *ground* state is **below** the thermal number, not above it. At
q = 2 it is 0.2151 against 0.5905, measured, and the exact ground state that
scores 0.2151 is the one enumeration confirms at nine sites. The thermal
constant therefore referees samplers, and the exact ground state referees
minimizers; carrying one across to the other is the arithmetic trap this
paragraph exists to name.

**The budget is site visits, not sweeps.** A Wolff step flips one cluster
while a heat-bath sweep touches every site, so equal sweeps hand the cluster
moves a free lattice per move. One visit is one read or write of a site's
label by a move: a heat-bath sweep costs ``n_nodes + 2 * n_edges``, a
Swendsen-Wang bond-and-recolour pass the same, and a Wolff step its cluster's
size times the mean degree plus one.

**A second family, for sizing rather than for ranking.** #596 asks where a
baseline *first fails*, which needs one instance family indexed by size rather
than three rungs chosen for what can referee them. :func:`lattice_rung` builds
the square lattice at the exact critical coupling at any side, in zero field or
in the size tilt, and :func:`uniform_ground_energy` is the closed form its zero
-field optimum has. The methods and the budget unit are shared, so a sizing
sweep and the three-rung comparison are billed the same way.

**One entry, from a start.** Every solver reads a :class:`Problem` --- the
lattice, the field and ``q`` --- and a :class:`Rung` hands its own.
:func:`ground_state` reaches every :data:`METHODS` entry and every
:data:`ARMS` entry (issue #1038's tuned Swendsen-Wang, matched Wolff and warm
chain, #1041's two expansion hybrids) from a ``(graph, field)``, and takes a
``start``, a ``schedule`` and a step count; with all three left ``None`` a run
is the entry's own bitwise (issue #1052). The warm chain and the two hybrids
are built by one factory, :func:`chain` over :func:`part`, and
:func:`compose` reads any chain of entries from a name such as
``field_argmax>descent>alpha-expansion`` (issue #1077).

See Boykov, Veksler & Zabih (2001) for the expansion bound and Baxter ch. 12
for the ordering coupling the rungs sit either side of.
"""

from __future__ import annotations

import copy
import functools
import operator
import re
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, replace
from dataclasses import field as dataclass_field
from typing import Any, NoReturn, Protocol

import numpy as np

from sal.backend import Backend
from sal.cost import Cost
from sal.likelihood.message_passing import (
    ConvergenceError,
    MessageScheduleName,
    max_product,
)
from sal.opt.budget import Budget, Comparison, Outcome
from sal.opt.compose import BestOf, Step, Then
from sal.opt.termination import Stop, Termination
from sal.sample.potts_mcmc import (
    ClusterCounter,
    PottsMove,
    anneal_potts,
    parallel_tempering,
)
from sal.sample.schedule import ScheduleParams, ScheduleShape
from sal.search.alpha_expansion import alpha_beta_swap, alpha_expansion
from sal.search.bifurcation import simulated_bifurcation
from sal.search.icm import (
    SweepOrder,
    check_min_sites,
    iterated_conditional_modes,
)
from sal.sim.factor_graph import from_potts
from sal.sim.graph import BoundaryCondition, PottsGraph, lattice_graph
from sal.sim.potts import (
    SpatioOnlyParams,
    check_labelling,
    critical_coupling,
    energy,
    spatio_only_field,
)

#: The annealing schedule every annealed entry runs, so a difference between
#: them is the move set. It starts above the ordering coupling's temperature
#: and ends cold enough that the last sweeps are a descent.
ANNEAL_START, ANNEAL_END = 2.0, 0.05

#: That schedule as parameters: exponential from :data:`ANNEAL_START` to
#: :data:`ANNEAL_END`, no hold. It builds the same
#: :class:`~sal.sample.schedule.ExponentialTempSchedule` the
#: annealed entries ran on before a schedule could be passed (issue #1038).
ANNEAL_SCHEDULE = ScheduleParams(ScheduleShape.EXPONENTIAL, ANNEAL_START, ANNEAL_END)

#: Replicas in the tempering ladder, geometric over the same endpoints. The
#: budget is divided by this, so a replica gets one sixth of the sweeps the
#: single-site entry gets and the comparison is at equal cost rather than at
#: equal sweeps per chain.
N_REPLICAS = 6


#: The move sets :func:`run_annealed` runs on the compiled cluster route.
_COMPILED_CLUSTERS = frozenset(
    {PottsMove.SWENDSEN_WANG, PottsMove.GHOST_SPIN, PottsMove.LABEL_DIRECTED}
)


@dataclass(frozen=True)
class Problem:
    """What a solver reads: the lattice, the field and the class count (issue #1052).

    A :class:`Rung` also carries the ladder, the sizes and the optimum, which
    the structural referee and the bracket read and no solver does. Every
    solver here reads a ``Problem``; handed a :class:`Rung`, it reads
    :attr:`Rung.problem`, so a caller with a graph and a field needs no
    fixture.

    Parameters
    ----------
    graph : PottsGraph
        The lattice; every coupling non-negative.
    field : np.ndarray
        ``h``, shape ``(n_nodes, n_states)``.
    n_states : int
        ``q``.
    """

    graph: PottsGraph
    field: np.ndarray
    n_states: int

    @property
    def n_nodes(self) -> int:
        """Sites."""
        return int(self.field.shape[0])

    @property
    def visits_per_sweep(self) -> int:
        """Site visits one heat-bath sweep costs: a write per site, a read per edge end."""
        return self.n_nodes + 2 * len(self.graph.edges)


@dataclass(frozen=True)
class Rung:
    """One instance of the comparison, and what can referee its energy.

    Parameters
    ----------
    name : str
        ``"ci-q3"``, ``"release-q2"``, ``"release-q10"``.
    graph : PottsGraph
        The lattice.
    field : np.ndarray
        ``h``, shape ``(n_nodes, n_states)``.
    alpha : np.ndarray
        The class ladder the field was built from, shape ``(n_states,)``.
        The structural checks are read against *this* ladder and never
        against another rung's, since a tilt is a mean of its own alphas.
    sizes : np.ndarray
        The per-site covariate, shape ``(n_nodes,)``.
    n_states : int
        ``q``.
    optimum : float | None
        The exact minimum energy where one is available --- by enumeration at
        nine sites, by graph cut at two states --- and ``None`` at q = 10,
        where the bracket is what there is.
    """

    name: str
    graph: PottsGraph
    field: np.ndarray
    alpha: np.ndarray
    sizes: np.ndarray
    n_states: int
    optimum: float | None

    @property
    def n_nodes(self) -> int:
        """Sites."""
        return int(self.field.shape[0])

    @property
    def visits_per_sweep(self) -> int:
        """Site visits one heat-bath sweep costs: a write per site, a read per edge end."""
        return self.n_nodes + 2 * len(self.graph.edges)

    @property
    def problem(self) -> Problem:
        """What a solver reads of this rung: the lattice, the field and ``q``."""
        return Problem(self.graph, self.field, self.n_states)


def _problem(instance: Problem | Rung) -> Problem:
    """``instance`` as a :class:`Problem`: a rung's own, or the problem itself."""
    return instance.problem if isinstance(instance, Rung) else instance


def _refuse_start(method: str, start: np.ndarray | None, reason: str) -> None:
    """Raise, naming ``method``, where a start is given to a method that has none.

    Raises
    ------
    ValueError
        If ``start`` is not ``None``.
    """
    if start is not None:
        msg = f"{method!r} takes no start: {reason}"
        raise ValueError(msg)


#: The class ladder the sizing family tilts by, when it carries a field at all.
#: Three equally spaced values, so the null class sits in the middle and the
#: tilt is symmetric: an asymmetric ladder would make the majority label a
#: property of the ladder rather than of the coupling.
SWEEP_ALPHA = (-1.0, 0.0, 1.0)
#: Spread of the lognormal per-site covariate. 0.6 puts the field's scale
#: within a factor of the coupling at `q = 3`, which is what makes the instance
#: a contest between the two terms rather than a field-only or coupling-only
#: problem.
SWEEP_SPREAD = 0.6


def uniform_ground_energy(rung: Rung) -> float:
    """``-J * n_edges``: the exact optimum of a zero-field ferromagnet.

    In zero field every site's unary term is zero, so the energy is the
    coupling term alone and is minimized by any labelling that agrees across
    every edge --- the ``q`` uniform ones. There is nothing to search for and
    that is the point of measuring here: a baseline that cannot reach *this*
    has failed on an instance whose answer is a closed form, and one that
    reaches it has not thereby solved anything.

    Raises
    ------
    ValueError
        If the field is not identically zero, where the closed form does not
        hold and returning it anyway would put a wrong target under a curve.
    """
    if bool(np.any(rung.field != 0.0)):
        msg = (
            "the closed form holds in zero field only; this rung's field has "
            f"a largest magnitude of {np.abs(rung.field).max():.6g}"
        )
        raise ValueError(msg)
    return -float(rung.graph.edge_coupling.sum())


def lattice_rung(
    side: int,
    n_states: int,
    *,
    seed: int | None = None,
    boundary: BoundaryCondition = BoundaryCondition.OPEN,
) -> Rung:
    """One member of the sizing family: a square lattice at its critical coupling.

    The coupling is :func:`~sal.sim.potts.critical_coupling`'s
    closed form at ``n_states`` rather than a stored float, for the reason
    `tests/regression/fixtures/potts_lattice/stress.yaml` gives: an instance
    declared *at* the transition cannot drift off it, and a family at a
    different label count is at its own transition rather than at this one.

    Parameters
    ----------
    side : int
        Extent of both dimensions, so the instance has ``side ** 2`` sites.
    n_states : int
        ``q``.
    seed : int | None
        ``None`` leaves the field identically zero, where
        :func:`uniform_ground_energy` is the exact optimum. An integer draws
        the per-site covariate lognormally and tilts the field by
        :data:`SWEEP_ALPHA`, which removes the closed form and is the instance
        a gate would be argued on.
    boundary : BoundaryCondition
        Open by default, matching the lattice fixture.

    Returns
    -------
    Rung
        With ``optimum`` set only in zero field, since that is the only case
        here where an optimum is known without running anything.
    """
    coupling = critical_coupling(n_states)
    graph = lattice_graph((side, side), boundary, coupling)
    n_nodes = side * side
    if seed is None:
        field = np.zeros((n_nodes, n_states))
        alpha = np.zeros(n_states)
        sizes = np.ones(n_nodes)
    else:
        ladder = np.asarray(SWEEP_ALPHA, dtype=float)
        alpha = ladder if n_states == ladder.size else np.array([ladder[0], ladder[-1]])
        if alpha.size != n_states:
            msg = (
                f"the sizing family tilts by {ladder.size} classes or 2, got "
                f"{n_states}: any other truncation keeps a lopsided ladder"
            )
            raise ValueError(msg)
        sizes = np.random.default_rng(seed).lognormal(0.0, SWEEP_SPREAD, size=n_nodes)
        field = spatio_only_field(alpha, sizes)
    rung = Rung(
        name=f"sweep-{side}x{side}-q{n_states}",
        graph=graph,
        field=field,
        alpha=alpha,
        sizes=sizes,
        n_states=n_states,
        optimum=None,
    )
    if seed is not None:
        return rung
    return Rung(
        name=rung.name,
        graph=graph,
        field=field,
        alpha=alpha,
        sizes=sizes,
        n_states=n_states,
        optimum=uniform_ground_energy(rung),
    )


@dataclass(frozen=True)
class RungField:
    """A rung's per-site field, and the class ladder it was built from.

    Parameters
    ----------
    field : np.ndarray
        Shape ``(n_nodes, n_states)``, as
        :func:`~sal.sim.potts.spatio_only_field` builds it.
    alpha : np.ndarray
        The ladder, length ``n_states``, ascending. The field's columns are
        read against it, so the two travel together.
    """

    field: np.ndarray
    alpha: np.ndarray

    def __iter__(self) -> Iterator[Any]:
        """``(field, alpha)``: the order callers unpack.

        ``Any`` and not ``np.ndarray``: the two are one type here, and a
        narrower annotation would still be widened by the next field added.
        """
        yield from (self.field, self.alpha)


def rung_field(params: SpatioOnlyParams, n_states: int) -> RungField:
    """The field and class ladder of a rung at ``n_states``, from one fixture.

    The ladder is taken from the fixture's own ``alpha`` --- its two extremes
    at q = 2, the whole of it at the declared q --- so every rung is built by
    :func:`sal.sim.potts.spatio_only_field` from the same sizes and
    the rungs differ in the label count alone.

    Raises
    ------
    ValueError
        If ``n_states`` is neither 2 nor the fixture's own class count. Any
        other truncation would drop the null class or keep a lopsided ladder,
        and the structural checks would then be read against a ladder no
        fixture declares.
    """
    if n_states == params.n_classes:
        return RungField(params.field, params.alpha)
    if n_states == 2:
        alpha = np.array([params.alpha[0], params.alpha[-1]])
        return RungField(spatio_only_field(alpha, params.sizes), alpha)
    msg = (
        f"a rung is built at 2 states or the fixture's {params.n_classes}, "
        f"got {n_states}: any other truncation reads the structural checks "
        "against a ladder no fixture declares"
    )
    raise ValueError(msg)


@dataclass(frozen=True)
class Structure:
    """What the generating parameters say about a labelling.

    Exact at every size, because it is read off the parameters rather than
    estimated: this is the referee that survives where the energy's does not.

    Parameters
    ----------
    tilt : float
        Mean alpha of a site's label in the largest size quartile less the
        mean in the smallest. Zero for a field that reached the wrong site.
    null_tilt : float
        The same difference restricted to the null class' occupancy --- the
        fraction of largest-quartile sites carrying the class whose alpha is
        exactly zero, less the fraction in the smallest quartile. That class
        carries no field at any site, so a non-zero value is the method's
        doing and not the instance's. ``nan`` where the ladder has no exact
        zero, which is q = 2's case.
    monotone : bool
        Whether the mean alpha rises across all four size quartiles. A
        stronger claim than the tilt, which reads only the ends.
    occupancy : tuple[int, ...]
        Sites per class.
    """

    tilt: float
    null_tilt: float
    monotone: bool
    occupancy: tuple[int, ...]


def quartiles(sizes: np.ndarray) -> np.ndarray:
    """Which size quartile each site is in, ``0`` smallest to ``3`` largest."""
    return np.asarray(np.digitize(sizes, np.quantile(sizes, [0.25, 0.5, 0.75])))


def structure(alpha: np.ndarray, sizes: np.ndarray, labelling: np.ndarray) -> Structure:
    """Score a labelling against the parameters that built the field.

    The tilt is `tests/regression/sim/test_spatio_only.py`'s, restated for a
    single labelling rather than an ensemble of chains: that file measures a
    sampler and this one measures a minimizer, and both must read the same
    statistic or the two results cannot be compared.
    """
    which = quartiles(sizes)
    means = [float(alpha[labelling[which == group]].mean()) for group in range(4)]
    nulls = np.flatnonzero(alpha == 0.0)
    if nulls.size:
        null = int(nulls[0])
        null_tilt = float(
            (labelling[which == 3] == null).mean()
            - (labelling[which == 0] == null).mean()
        )
    else:
        null_tilt = float("nan")
    return Structure(
        tilt=means[3] - means[0],
        null_tilt=null_tilt,
        monotone=means == sorted(means),
        occupancy=tuple(
            int(count) for count in np.bincount(labelling, minlength=alpha.shape[0])
        ),
    )


@dataclass(frozen=True)
class Bracket:
    """The interval an optimum lies in.

    Parameters
    ----------
    lower : float
        The largest energy the optimum cannot be above.
    upper : float
        The smallest it cannot be below, which is the method's own energy.
    """

    lower: float
    upper: float

    def __iter__(self) -> Iterator[Any]:
        """``(lower, upper)``: the order callers unpack."""
        yield from (self.lower, self.upper)


def expansion_bracket(rung: Rung, expansion_energy: float) -> Bracket:
    """The interval the optimum lies in, from the expansion's factor-2 bound.

    **The bound is on the non-negative form of the energy, and saying so is
    the point.** Boykov, Veksler & Zabih prove ``E(local) <= 2 E(global)`` for
    a metric pairwise term with ``D_p >= 0`` and ``V >= 0``. The energy this
    repository scores is ``-sum h - J sum [agree]``, which is negative, and
    the factor-2 statement is false as written on a negative quantity. Shifted
    by ``sum_n max_m h[n, m] + J |edges|`` both terms are non-negative, the
    bound applies, and the lower end is carried back through the same shift.

    Returns
    -------
    Bracket
        ``upper`` is the expansion's own energy, which the optimum is at most.
    """
    offset = float(rung.field.max(axis=1).sum()) + float(
        np.asarray(rung.graph.coupling).sum()
    )
    return Bracket(expansion_energy / 2.0 - offset / 2.0, expansion_energy)


def ends_labelling(rung: Rung, labelling: np.ndarray) -> np.ndarray:
    """``labelling`` with every inner class moved to the ladder end its sites favour, at no higher energy (issue #1041).

    **Why the optimum of a `spatio_only` rung at any class count is its
    two-state one.** The field is ``h[i, m] = alpha[m] c_i``
    (:func:`~sal.sim.potts.spatio_only_field`), linear in the
    class value. Take every site carrying one class ``k`` other than the
    ladder's lowest and highest and give them all the lowest or the highest,
    whichever ``sum_i c_i`` over them favours: the field term,
    ``alpha[k] sum c_i``, is linear in ``alpha[k]`` and so no smaller at one
    end of the ladder; a bond inside the set keeps its agreement, and a bond
    leaving it agreed with nothing before and may agree now, which with
    ``J >= 0`` lowers the energy or leaves it. Class by class, any labelling
    maps to one on the two ends at no higher energy, so the minimum over all
    classes is the minimum over two --- which a graph cut solves exactly.

    Returns
    -------
    np.ndarray
        The relabelled copy, carrying only the two end classes.

    Raises
    ------
    ValueError
        If the field is not ``alpha`` times one number per site, or a
        coupling is negative, where the argument fails.
    """
    alpha = np.asarray(rung.alpha, dtype=float)
    low, high = int(np.argmin(alpha)), int(np.argmax(alpha))
    per_site = (rung.field[:, high] - rung.field[:, low]) / (alpha[high] - alpha[low])
    if not np.allclose(rung.field, np.outer(per_site, alpha), rtol=0.0, atol=1e-12):
        msg = "the field is not alpha times one number per site"
        raise ValueError(msg)
    if float(np.min(rung.graph.edge_coupling, initial=0.0)) < 0.0:
        msg = "a negative coupling breaks the argument"
        raise ValueError(msg)
    moved = np.array(labelling, dtype=np.int64)
    for inner in sorted(set(range(rung.n_states)) - {low, high}):
        members = moved == inner
        if members.any():
            moved[members] = high if per_site[members].sum() >= 0.0 else low
    return moved


@dataclass(frozen=True)
class MethodRun:
    """One method on one rung under one generator.

    Parameters
    ----------
    labelling : np.ndarray
        What it returned, so the structural referee can score it.
    energy : float
        Its energy.
    spent : int
        Site visits, the budget's unit.
    seconds : float
        Wall clock. Meaningful only from a host verified quiet; the runner
        records it either way and the report says which host it came from.
    trace : tuple[ClusterCounter, ...]
        The cluster instrumentation, empty for a move set that builds none.
    converged : bool
        ``False`` where the method reached its budget without producing an
        answer, which only message passing can do here. Such a run carries an
        infinite energy rather than a fallback labelling's, so it ranks last
        and a reader is told it failed instead of being shown another
        method's numbers under its name.
    termination : Termination | None
        Why the method's own loop ended, where the method says (issue #860).
        Not a restatement of ``converged``: that field answers *did this
        return an answer*, and a sweep loop that answers it with ``True``
        still ran to its budget and met no criterion. So the rows carrying
        one are the two that know --- the cut-based moves, which return on a
        cycle that lowers nothing, and max-product, which converges or
        refuses. The Monte Carlo and descent rows leave it ``None``: their
        kernels report a labelling and an energy and not which branch ended
        the loop.
    """

    labelling: np.ndarray
    energy: float
    spent: int
    seconds: float
    trace: tuple[ClusterCounter, ...] = ()
    converged: bool = True
    termination: Termination | None = None


def step_cost(problem: Problem | Rung, move: PottsMove) -> int:
    """Site visits one step of ``move`` is budgeted at: a sweep's, and a ghost bond per site for the ghost-spin pass."""
    extra = problem.n_nodes if move is PottsMove.GHOST_SPIN else 0
    return problem.visits_per_sweep + extra


def run_annealed(
    problem: Problem | Rung,
    budget: Budget,
    rng: np.random.Generator,
    move: PottsMove,
    *,
    schedule: ScheduleParams = ANNEAL_SCHEDULE,
    steps: int | None = None,
    start: np.ndarray | None = None,
) -> MethodRun:
    """One annealed run, its step count fixed before the run starts.

    The step count is ``budget // visits_per_sweep`` for every move set,
    including Wolff, whose step is far cheaper. Wolff therefore *underspends*
    its budget rather than stopping when it is exhausted: a stop on
    accumulated cluster size is a stop on the state, which `search/CLAUDE.md`
    refuses, and the underspend is reported as the finding it is.

    The ghost-spin pass also reads one ghost bond per site, so its step costs
    :func:`step_cost` and it runs fewer steps on the same budget (issue
    #1041); every other move's step costs ``visits_per_sweep``.

    ``schedule``, ``steps`` and ``start`` are what issue #1038 varies, and
    their defaults are the run above bitwise. ``steps`` replaces the count
    with one a caller fixed beforehand, still not read from the run's state;
    ``start`` starts the chain from a labelling instead of a uniform draw
    (``initial`` until issue #1052 gave every solver the one name).
    """
    problem = _problem(problem)
    count = max(1, budget.size // step_cost(problem, move)) if steps is None else steps
    started = time.perf_counter()
    # Swendsen-Wang on the compiled pass: the same law on another order of
    # draws, and the comparison reads no cluster counter (issue #923). The
    # ghost-spin and label-directed passes merge their bonds on the compiled
    # union-find, which returns the Python roots, so the chain (issue #1041).
    run = anneal_potts(
        problem.graph,
        problem.field,
        schedule.build(count),
        rng,
        move=move,
        cluster_backend=Backend.RUST if move in _COMPILED_CLUSTERS else Backend.PYTHON,
        start=start,
    )
    return MethodRun(
        labelling=run.labelling,
        energy=run.energy,
        spent=run.site_visits,
        seconds=time.perf_counter() - started,
        trace=run.trace,
    )


def descend(
    problem: Problem | Rung,
    rng: np.random.Generator,
    max_iterations: int,
    *,
    start: np.ndarray | None = None,
    backend: Backend | None = None,
    min_sites: int = 0,
) -> tuple[np.ndarray, int]:
    """Index-order ICM from a uniform draw, or from ``start``, and the sweeps it ran.

    The descent :func:`run_icm` runs, on the same draws, with its sweep count
    read out of :attr:`~sal.search.alpha_expansion.Labelling.sweeps`: the
    count is what a warm chain is charged, where :func:`run_icm` charges its
    whole budget. The sweep that changes nothing is counted, as
    :func:`~sal.search.potts_starts.polish_by_icm` counts it. One call since
    #1059, which reran ICM one sweep at a time to count.

    ``backend`` and ``min_sites`` are
    :func:`~sal.search.icm.iterated_conditional_modes`'s;
    ``None`` is its default backend. A floored descent draws its
    ``max_iterations * n_nodes`` uniforms up front, as :func:`run_icm`'s one call
    does, so the labelling is the one sweep-at-a-time descents reached and
    the generator is left where one call leaves it.

    Returns
    -------
    tuple[np.ndarray, int]
        The labelling, and the sweeps run, at most ``max_iterations``.
    """
    problem = _problem(problem)
    check_min_sites(min_sites, problem.n_nodes)
    labelling = (
        rng.integers(0, problem.n_states, size=problem.n_nodes)
        if start is None
        else np.array(start, dtype=np.int64)
    )
    settled = iterated_conditional_modes(
        problem.graph,
        problem.field,
        rng,
        start=labelling,
        max_iterations=max_iterations,
        min_sites=min_sites,
        backend=Backend.NUMBA if backend is None else backend,
        n_states=problem.n_states,
    )
    return settled.labelling, settled.sweeps


def run_descent(
    problem: Problem | Rung,
    budget: Budget,
    rng: np.random.Generator,
    *,
    start: np.ndarray | None = None,
) -> MethodRun:
    """ICM from a uniform draw, or from ``start``, to its first clean sweep, charged the sweeps it ran.

    :func:`descend` as a stage: the warm chain's first part (issue #1038).
    Where :func:`run_icm` charges its whole budget, this charges
    ``visits_per_sweep`` per sweep run, the clean one included, so a stage
    after it gets the rest (issue #1077).
    """
    problem = _problem(problem)
    started = time.perf_counter()
    labelling, sweeps = descend(
        problem, rng, budget.size // problem.visits_per_sweep, start=start
    )
    return MethodRun(
        labelling=labelling,
        energy=energy(problem.graph, problem.field, labelling),
        spent=sweeps * problem.visits_per_sweep,
        seconds=time.perf_counter() - started,
    )


#: What a stage hands the next: its labelling (issue #1077).
handover = operator.attrgetter("labelling")

#: The options an annealed stage takes.
ANNEAL_OPTIONS = frozenset({"schedule", "steps"})

#: The options a single-site descent takes.
DESCENT_OPTIONS = frozenset({"backend", "min_sites"})


def run_field_argmax(
    problem: Problem | Rung,
    budget: Budget,
    rng: np.random.Generator,
    *,
    start: np.ndarray | None = None,
) -> MethodRun:
    """The field-only labelling: every site takes its own best class, coupling ignored.

    The trap baseline. It maximizes the tilt by construction, so it is the
    control for the structural referee as well as for the energy: a method it
    ties on both is a method the instance cannot distinguish.
    """
    _refuse_start(
        "field_argmax",
        start,
        "it is built from the field alone, with no labelling to start from",
    )
    problem = _problem(problem)
    del budget, rng
    started = time.perf_counter()
    labelling = problem.field.argmax(axis=1).astype(np.int64)
    return MethodRun(
        labelling=labelling,
        energy=energy(problem.graph, problem.field, labelling),
        spent=problem.n_nodes,
        seconds=time.perf_counter() - started,
    )


def run_icm(
    problem: Problem | Rung,
    budget: Budget,
    rng: np.random.Generator,
    *,
    start: np.ndarray | None = None,
    backend: Backend | None = None,
    min_sites: int = 0,
) -> MethodRun:
    """Iterated conditional modes: single-site descent in index order.

    From a uniform draw, or from ``start``. ``backend`` and ``min_sites`` are
    :func:`~sal.search.icm.iterated_conditional_modes`'s;
    ``None`` is the compiled sweep, the default before either was a
    parameter (issue #1055).
    """
    problem = _problem(problem)
    steps = max(1, budget.size // problem.visits_per_sweep)
    started = time.perf_counter()
    settled = iterated_conditional_modes(
        problem.graph,
        problem.field,
        rng,
        start=start,
        max_iterations=steps,
        min_sites=min_sites,
        backend=Backend.NUMBA if backend is None else backend,
        n_states=problem.n_states,
    )
    return MethodRun(
        labelling=settled.labelling,
        energy=settled.energy,
        spent=steps * problem.visits_per_sweep,
        seconds=time.perf_counter() - started,
    )


def run_icm_random(
    problem: Problem | Rung,
    budget: Budget,
    rng: np.random.Generator,
    *,
    start: np.ndarray | None = None,
    backend: Backend | None = None,
    min_sites: int = 0,
) -> MethodRun:
    """ICM in a random sweep order: the heat bath at T = 0, every sweep of the budget run.

    Reported on ICM's axis rather than beside it. The heat bath at ``T -> 0``
    is the argmin over each site's conditional, which is ICM's update, so the
    two differ in the order sites are visited and in nothing else. Running
    both and reporting one axis is what keeps the comparison from counting
    the same method twice; the pair's spread *is* the sweep order's effect.

    That is why this is :func:`run_icm`'s call under two parameters rather
    than a second sweep written beside it (issue #858): a random order, and
    every sweep of the budget run whether or not one changes nothing. Named
    for what it runs, ICM under a permutation, and `gibbs-T0` until #923,
    which also moved it onto the compiled sweep: the permutations are drawn
    in the Python sweep's order, and the labelling is that sweep's bitwise.
    ``backend`` and ``min_sites`` are as :func:`run_icm` takes them.
    """
    problem = _problem(problem)
    steps = max(1, budget.size // problem.visits_per_sweep)
    started = time.perf_counter()
    settled = iterated_conditional_modes(
        problem.graph,
        problem.field,
        rng,
        start=start,
        max_iterations=steps,
        sweep_order=SweepOrder.RANDOM,
        stop_when_clean=False,
        min_sites=min_sites,
        backend=Backend.NUMBA if backend is None else backend,
        n_states=problem.n_states,
    )
    return MethodRun(
        labelling=settled.labelling,
        energy=settled.energy,
        spent=steps * problem.visits_per_sweep,
        seconds=time.perf_counter() - started,
    )


class Method(Protocol):
    """One method of the comparison: a problem, a budget and a generator to a run.

    ``start`` is the labelling a method descends or anneals from in place of
    its own; a method with no single starting labelling raises on one
    (issue #1052).
    """

    def __call__(
        self,
        problem: Problem | Rung,
        budget: Budget,
        rng: np.random.Generator,
        /,
        *,
        start: np.ndarray | None = None,
    ) -> MethodRun:
        """Run on ``problem`` within ``budget``."""
        ...


#: The three annealed entries are one run with a move set (issue #717):
#: single-site is the fair annealed baseline, Swendsen-Wang recolours every
#: cluster with its own accept step, Wolff one cluster per step with the
#: accept step on its field.
run_anneal = functools.partial(run_annealed, move=PottsMove.SINGLE_SITE)
run_swendsen_wang = functools.partial(run_annealed, move=PottsMove.SWENDSEN_WANG)
run_wolff = functools.partial(run_annealed, move=PottsMove.WOLFF)


def run_tempering(
    problem: Problem | Rung,
    budget: Budget,
    rng: np.random.Generator,
    *,
    start: np.ndarray | None = None,
) -> MethodRun:
    """Parallel tempering over a geometric ladder, charged for every replica.

    The budget buys ``budget // (N_REPLICAS * visits_per_sweep)`` sweeps per
    replica rather than that many per chain, which is the whole difference
    between a comparison at equal cost and one at equal sweeps.
    """
    _refuse_start("tempering", start, "its ladder draws one labelling per replica")
    problem = _problem(problem)
    per_replica = max(1, budget.size // (N_REPLICAS * problem.visits_per_sweep))
    ladder = tuple(
        float(value) for value in np.geomspace(ANNEAL_START, ANNEAL_END, N_REPLICAS)
    )
    started = time.perf_counter()
    run = parallel_tempering(
        problem.graph,
        problem.field,
        ladder,
        rng,
        per_replica,
    )
    return MethodRun(
        labelling=run.best,
        energy=run.best_energy,
        spent=N_REPLICAS * per_replica * problem.visits_per_sweep,
        seconds=time.perf_counter() - started,
    )


def run_alpha_expansion(
    problem: Problem | Rung,
    budget: Budget,
    rng: np.random.Generator,
    *,
    start: np.ndarray | None = None,
) -> MethodRun:
    """Alpha expansion: the only entry carrying a bound, and the bracket's lower end.

    It starts from the field's argmax, the field_argmax labelling, or from
    ``start``, and draws nothing from ``rng``.
    """
    problem = _problem(problem)
    del rng
    cycles = max(1, budget.size // (problem.n_states * problem.visits_per_sweep))
    started = time.perf_counter()
    run = alpha_expansion(
        problem.graph,
        problem.field,
        start=start,
        max_iterations=cycles,
        backend=Backend.RUST,
        n_states=problem.n_states,
    )
    return MethodRun(
        labelling=run.labelling,
        energy=run.energy,
        spent=run.cycles * problem.n_states * problem.visits_per_sweep,
        seconds=time.perf_counter() - started,
        termination=run.termination,
    )


def run_alpha_beta_swap(
    problem: Problem | Rung,
    budget: Budget,
    rng: np.random.Generator,
    *,
    start: np.ndarray | None = None,
) -> MethodRun:
    """Alpha-beta swap: the cheaper move, and the one with no bound.

    A cycle is ``q (q - 1) / 2`` cuts against the expansion's ``q``, and each
    cut covers only the sites carrying its two labels. Summed over a cycle
    every site is therefore visited ``q - 1`` times, not once per pair, so a
    cycle costs ``(q - 1)`` sweeps against the expansion's ``q`` --- the swap
    is much cheaper per *cut* and barely cheaper per *cycle*, which is the
    trade, and charging it by the pair count would have hidden it. It starts
    where :func:`run_alpha_expansion` starts.
    """
    problem = _problem(problem)
    del rng
    per_cycle = (problem.n_states - 1) * problem.visits_per_sweep
    cycles = max(1, budget.size // per_cycle)
    started = time.perf_counter()
    run = alpha_beta_swap(
        problem.graph,
        problem.field,
        start=start,
        max_iterations=cycles,
        backend=Backend.RUST,
        n_states=problem.n_states,
    )
    return MethodRun(
        labelling=run.labelling,
        energy=run.energy,
        spent=run.cycles * per_cycle,
        termination=run.termination,
        seconds=time.perf_counter() - started,
    )


def run_max_product(
    problem: Problem | Rung,
    budget: Budget,
    rng: np.random.Generator,
    *,
    start: np.ndarray | None = None,
) -> MethodRun:
    """Max-product on the factor graph: MAP-exact on a tree, and this is not a tree.

    Included because it carries **no** bound on a loopy graph, which is what
    it is reported as.

    Flooding is *refused* rather than truncated when it does not settle, and
    that refusal is carried through rather than papered over: a run that does
    not converge within its budget returns an infinite energy and
    ``converged=False``. Substituting the field-only labelling there would
    enter the field_argmax baseline's numbers in max-product's row, which is the
    quiet failure this branch exists to prevent.
    """
    _refuse_start("max-product", start, "it iterates messages, not a labelling")
    problem = _problem(problem)
    del rng
    iterations = max(1, budget.size // problem.visits_per_sweep)
    graph = from_potts(problem.graph, problem.field)
    started = time.perf_counter()
    try:
        assignment, marginals = max_product(
            graph, schedule=MessageScheduleName.FLOODING, max_iterations=iterations
        )
    except ConvergenceError:
        return MethodRun(
            labelling=problem.field.argmax(axis=1).astype(np.int64),
            energy=float("inf"),
            spent=iterations * problem.visits_per_sweep,
            seconds=time.perf_counter() - started,
            converged=False,
            termination=Termination(
                converged=False, iterations=iterations, reason=Stop.REFUSED
            ),
        )
    labelling = np.array(
        [assignment[f"s{node}"] for node in range(problem.n_nodes)], dtype=np.int64
    )
    return MethodRun(
        labelling=labelling,
        energy=energy(problem.graph, problem.field, labelling),
        spent=iterations * problem.visits_per_sweep,
        seconds=time.perf_counter() - started,
        # Flooding refuses rather than truncating, so a return is a settled
        # fixed point and the sweeps it took are the marginals' own count.
        termination=Termination.after(marginals.iterations, converged=True),
    )


def run_bifurcation(
    problem: Problem | Rung,
    budget: Budget,
    rng: np.random.Generator,
    *,
    start: np.ndarray | None = None,
) -> MethodRun:
    """Simulated bifurcation, one replica, the budget spent in integration steps.

    A step reads every edge twice and writes every site once, the heat-bath
    sweep's unit, so ``steps = budget // visits_per_sweep`` matches the rows
    beside it in what they spend (issue #823).
    """
    _refuse_start(
        "bifurcation", start, "it integrates continuous amplitudes, not labels"
    )
    problem = _problem(problem)
    steps = max(1, budget.size // problem.visits_per_sweep)
    started = time.perf_counter()
    result = simulated_bifurcation(
        problem.graph, problem.field, rng, steps=steps, n_states=problem.n_states
    )
    return MethodRun(
        labelling=result.labelling,
        energy=result.energy,
        spent=steps * problem.visits_per_sweep,
        seconds=time.perf_counter() - started,
        termination=result.termination,
    )


#: Every entry, in report order. ICM and Gibbs at T = 0 are two rows of one
#: axis, named so the report cannot present them as independent methods.
METHODS: dict[str, Method] = {
    "field_argmax": run_field_argmax,
    "icm": run_icm,
    "icm-random": run_icm_random,
    "anneal": run_anneal,
    "swendsen-wang": run_swendsen_wang,
    "wolff": run_wolff,
    "tempering": run_tempering,
    "alpha-expansion": run_alpha_expansion,
    "alpha-beta-swap": run_alpha_beta_swap,
    "max-product": run_max_product,
    "bifurcation": run_bifurcation,
}

#: The two rows that are one axis, so a reader of the table is told rather
#: than left to notice.
ONE_AXIS = ("icm", "icm-random")


#: Cycles of the expansion the ``swendsen-wang>expansion`` arm holds
#: back from Swendsen-Wang's share. From a uniform start the expansion ends in
#: 3 cycles on `spatio_only/release` at ten states; 10 leaves it room from a
#: labelling that is not its own.
EXPANSION_RESERVE_CYCLES = 10


@dataclass(frozen=True)
class ExpansionReserve:
    """Site visits of ``cycles`` expansion cycles on a problem: what a stage before the expansion holds back."""

    cycles: int

    def __call__(self, problem: Problem | Rung) -> int:
        """``cycles`` times one cycle's ``n_states`` sweeps."""
        return self.cycles * problem.n_states * problem.visits_per_sweep


@dataclass(frozen=True)
class SweepReserve:
    """Site visits of ``sweeps`` heat-bath sweeps on a problem: a reserve in sweeps."""

    sweeps: int

    def __call__(self, problem: Problem | Rung) -> int:
        """``sweeps`` times ``visits_per_sweep``."""
        return self.sweeps * problem.visits_per_sweep


#: The names that are a single-site descent, and so take its ``backend`` and
#: ``min_sites`` floor (issue #1055). The annealers, the cuts and the hybrids
#: have no floor of their own and refuse one.
FLOORED = frozenset({"icm", "icm-random"})

#: The :data:`METHODS` names that run an anneal.
_ANNEALED_METHODS = frozenset({"anneal", "swendsen-wang", "wolff"})

#: Parts a chain can name beside :data:`METHODS` and :data:`ARMS`: the warm
#: chain's descent, which charges only the sweeps it ran (issue #1077).
STAGES: dict[str, Method] = {"descent": run_descent}

#: Separates the parts of a chain in a method name.
CHAIN = ">"


def _options(name: str) -> frozenset[str]:
    """The keyword options the part ``name`` takes: an arm's are an anneal's."""
    if name in FLOORED:
        return DESCENT_OPTIONS
    if name in _ANNEALED_METHODS:
        return ANNEAL_OPTIONS
    if name in METHODS or name in STAGES:
        return frozenset()
    return ANNEAL_OPTIONS


def _solver(name: str) -> Method:
    """The solver a part names, from :data:`METHODS`, :data:`STAGES` or :data:`ARMS`."""
    for table in (METHODS, STAGES):
        if name in table:
            return table[name]
    # :data:`ARMS` is read last and only here: it is built from parts named
    # in the other two, so it is defined by the time a name reaches it.
    if name in ARMS:
        return ARMS[name]
    msg = (
        f"no ground-state part {name!r}; the parts are {sorted(METHODS)}, "
        f"{sorted(ARMS)} and {sorted(STAGES)}"
    )
    raise ValueError(msg)


def part(
    solver: str | Callable[..., MethodRun],
    *,
    reserve: Callable[[Any], int] | None = None,
    takes: frozenset[str] | None = None,
    **bound: Any,
) -> Step:
    """One part of a chain: a solver by name or itself, with keywords bound (issue #1077).

    A solver given itself may take more than a :class:`Method` does, such as
    :func:`run_annealed`'s ``move``, which ``bound`` then supplies. ``bound`` is bound to the solver, as ``schedule=`` or ``move=``, and a
    keyword of the same name passed to the chain at call time replaces it.
    ``reserve`` is what the part holds back for the parts after it. ``takes``
    is read from the name where one is given, and is otherwise the options
    the caller routes to the solver, none by default.
    """
    stage: Callable[..., MethodRun]
    if isinstance(solver, str):
        stage = _solver(solver)
        routed = _options(solver) if takes is None else takes
    else:
        stage = solver
        routed = frozenset() if takes is None else takes
    if bound:
        stage = functools.partial(stage, **bound)
    return Step(stage, takes=routed, reserve=reserve)


def chain(*parts: str | Step) -> Then:
    """A chain of parts, each a name or a :func:`part`: the one factory for every hybrid (issue #1077).

    Each part starts from the labelling of the one before, gets the budget the
    parts before it left less its reserve, and draws from the one generator.
    The run is the last part's, ``spent`` summed. ``schedule`` and ``steps``
    reach every annealed part, ``backend`` and ``min_sites`` every
    single-site descent. A part that refuses a start, such as
    ``field_argmax``, can only come first; and since :func:`run_icm` charges
    its whole budget, a descent before another part is ``descent``.
    """
    return Then(
        tuple(item if isinstance(item, Step) else part(item) for item in parts),
        handover,
    )


#: One stage of a chain's text: a solver's name and, in parentheses, its
#: arguments as ``key=value`` pairs separated by commas.
_STAGE = re.compile(r"^\s*([A-Za-z0-9_\-]+)\s*(?:\((.*)\))?\s*$")

#: The fields of an annealed stage's schedule an argument sets; the others
#: are :data:`ANNEAL_SCHEDULE`'s.
SCHEDULE_FIELDS = ("shape", "t_start", "t_end", "hold")

#: Every argument a stage can take, and how a value, typed or text, is read.
ARGUMENTS: dict[str, Callable[[Any], Any]] = {
    "shape": ScheduleShape,
    "t_start": float,
    "t_end": float,
    "hold": float,
    "steps": int,
    "backend": Backend,
    "min_sites": int,
    "reserve_cycles": int,
    "reserve_sweeps": int,
}


def _refuse_argument(name: str, keys: Iterable[str], reason: str) -> None:
    """Raise for arguments a stage cannot take."""
    msg = f"{name!r} takes no {sorted(keys)!r}: {reason}"
    raise ValueError(msg)


@dataclass
class SolverStage:
    """One stage of a :class:`SolverChain`: a solver's name and its arguments (issue #1077).

    ``arguments`` are :data:`ARGUMENTS`' keys: the schedule's fields and
    ``steps`` on an annealed stage, ``backend`` and ``min_sites`` on a
    single-site descent, and ``reserve_cycles`` or ``reserve_sweeps`` on any
    stage, held back for the stages after it. Set them with :meth:`update`,
    which checks each against the stage. An :data:`ARMS` name is a chain
    already and takes none.
    """

    name: str
    arguments: dict[str, Any] = dataclass_field(default_factory=dict)

    def __post_init__(self) -> None:
        _solver(self.name)
        given, self.arguments = self.arguments, {}
        self.update(**given)

    @classmethod
    def parse(cls, text: str) -> SolverStage:
        """A stage from ``name`` or ``name(key=value,...)``."""
        match = _STAGE.match(text)
        if match is None:
            msg = f"a stage is a name and optional (key=value, ...), got {text!r}"
            raise ValueError(msg)
        name, body = match.groups()
        arguments: dict[str, str] = {}
        for item in (body or "").split(","):
            if not item.strip():
                continue
            key, equals, value = item.partition("=")
            if not equals or not key.strip() or not value.strip():
                msg = f"an argument is key=value, got {item.strip()!r} in {text!r}"
                raise ValueError(msg)
            arguments[key.strip()] = value.strip()
        return cls(name, arguments)

    def update(self, **arguments: Any) -> SolverStage:
        """Set arguments, checked against the stage; ``schedule=`` sets all four fields.

        Returns the stage, so calls chain.

        Raises
        ------
        ValueError
            If an argument is unknown, unreadable or not the stage's, or
            both reserves would be set.
        """
        if "schedule" in arguments:
            schedule = arguments.pop("schedule")
            arguments = {
                **{key: getattr(schedule, key) for key in SCHEDULE_FIELDS},
                **arguments,
            }
        if not arguments:
            return self
        unknown = set(arguments) - set(ARGUMENTS)
        if unknown:
            _refuse_argument(self.name, unknown, "unknown arguments")
        if self.name in ARMS:
            _refuse_argument(self.name, arguments, "an arm is a chain already")
        takes = _options(self.name)
        annealing = set(arguments) & {*SCHEDULE_FIELDS, "steps"}
        if annealing and not takes & ANNEAL_OPTIONS:
            _refuse_argument(self.name, annealing, "it runs no anneal")
        descending = set(arguments) & {"backend", "min_sites"}
        if descending and not takes & DESCENT_OPTIONS:
            _refuse_argument(self.name, descending, "it is no single-site descent")
        read = {key: ARGUMENTS[key](value) for key, value in arguments.items()}
        merged = {**self.arguments, **read}
        if "reserve_cycles" in merged and "reserve_sweeps" in merged:
            _refuse_argument(self.name, read, "one reserve, in cycles or in sweeps")
        self.arguments = merged
        return self

    def step(self) -> Step:
        """The stage as a :class:`~sal.opt.compose.Step`."""
        arguments = dict(self.arguments)
        bound: dict[str, Any] = {}
        fields = {
            key: arguments.pop(key) for key in SCHEDULE_FIELDS if key in arguments
        }
        if fields:
            bound["schedule"] = replace(ANNEAL_SCHEDULE, **fields)
        for key in ("steps", "backend", "min_sites"):
            if key in arguments:
                bound[key] = arguments.pop(key)
        reserve: Callable[[Any], int] | None = None
        if "reserve_cycles" in arguments:
            reserve = ExpansionReserve(arguments.pop("reserve_cycles"))
        if "reserve_sweeps" in arguments:
            reserve = SweepReserve(arguments.pop("reserve_sweeps"))
        return part(self.name, reserve=reserve, **bound)

    def __str__(self) -> str:
        """The stage's text, which :meth:`parse` reads back."""
        if not self.arguments:
            return self.name
        shown = ",".join(
            f"{key}={getattr(value, 'value', value)!s}"
            if not isinstance(value, float)
            else f"{key}={value!r}"
            for key, value in self.arguments.items()
        )
        return f"{self.name}({shown})"


#: The separators of a chain's text: stages in order, realizations, and a
#: realization count.
REALIZATIONS = "|"
REPEAT = "**"

#: A stage's name in a chain's text.
_NAME = re.compile(r"[A-Za-z0-9_\-]+")

#: What a realization's run is judged on.
_ENERGY = operator.attrgetter("energy")


def _as_step(node: SolverStage | SolverChain | SolverRealizations) -> Step:
    """A node as one step of the chain around it."""
    if isinstance(node, SolverStage):
        return node.step()
    built = node.then() if isinstance(node, SolverChain) else node.best_of()
    return Step(built, takes=built.takes)


@dataclass
class SolverChain:
    """Stages run in order under one budget, built when called (issue #1077).

    ``SolverChain.parse("swendsen-wang>alpha-expansion")`` reads one from
    its text, ``chain.stages[0].update(t_start=1.0, reserve_cycles=10)``
    sets a stage's arguments, and :func:`ground_state` runs it. Each stage
    starts from the labelling of the one before and gets the budget the
    stages before it left less its reserve; the run is the last stage's,
    ``spent`` summed. A stage is a :class:`SolverStage` or a
    :class:`SolverRealizations`. ``schedule`` and ``steps`` passed at call
    time reach every annealed stage and replace its arguments; ``backend``
    and ``min_sites`` every single-site descent. A stage that refuses a
    start, such as ``field_argmax``, can only come first; and since
    :func:`run_icm` charges its whole budget, a descent before another stage
    is ``descent``.

    The text is stages joined by ``>``; ``a|b`` is realizations of ``a`` and
    ``b`` and ``x**n`` is ``n`` realizations of ``x``, :class:`SolverRealizations`;
    parentheses group. ``**`` binds tightest, then ``>``, then ``|``, so
    ``descent>alpha-expansion**2`` repeats the expansion alone and
    ``(descent>alpha-expansion)**2`` the whole chain. ``str(chain)`` is text
    :meth:`parse` reads back.
    """

    stages: list[SolverStage | SolverRealizations]

    def __post_init__(self) -> None:
        if not self.stages:
            msg = "a chain has at least one stage"
            raise ValueError(msg)

    @classmethod
    def parse(cls, text: str) -> SolverChain:
        """A chain from its text; realizations at the top are a chain of one stage.

        Raises
        ------
        ValueError
            If the text does not read, or a stage or argument is refused.
        """
        node = _Reader(text).read()
        return node if isinstance(node, SolverChain) else cls([node])

    def then(self) -> Then:
        """The chain as it runs, built from the stages' current arguments."""
        return Then(tuple(_as_step(stage) for stage in self.stages), handover)

    @property
    def takes(self) -> frozenset[str]:
        """Every option some stage takes at call time."""
        return self.then().takes

    def __call__(
        self,
        problem: Problem | Rung,
        budget: Budget,
        rng: np.random.Generator,
        /,
        *,
        start: np.ndarray | None = None,
        **options: Any,
    ) -> MethodRun:
        """Run the stages on ``problem`` within ``budget``, the first from ``start``."""
        run: MethodRun = self.then()(problem, budget, rng, start=start, **options)
        return run

    def __str__(self) -> str:
        """The chain's text, which :meth:`parse` reads back."""
        return CHAIN.join(
            f"({stage})"
            if isinstance(stage, SolverRealizations) and len(self.stages) > 1
            else str(stage)
            for stage in self.stages
        )


@dataclass
class SolverRealizations:
    """Independent realizations from one start, the lowest energy kept (issue #1077).

    :class:`~sal.opt.compose.BestOf`: each branch, a :class:`SolverStage` or
    a :class:`SolverChain`, runs from the same start on an equal share of
    the budget and its own generator spawned in branch order; the run is
    the lowest-energy branch's, ``spent`` summed. ``x**n`` in a chain's text
    is ``n`` copies of ``x``, each updated on its own.
    """

    branches: list[SolverStage | SolverChain]

    def __post_init__(self) -> None:
        if len(self.branches) < 2:
            msg = f"realizations are at least two, got {len(self.branches)}"
            raise ValueError(msg)

    def update(self, **arguments: Any) -> SolverRealizations:
        """:meth:`SolverStage.update` on every branch; update one through :attr:`branches`.

        Raises
        ------
        ValueError
            If a branch is a chain, whose stages are updated one by one, or
            an argument is refused; no branch is changed then.
        """
        updated = [
            branch
            for branch in copy.deepcopy(self.branches)
            if isinstance(branch, SolverStage)
        ]
        if len(updated) != len(self.branches):
            msg = "a branch is a chain: update its stages through branches"
            raise ValueError(msg)
        for branch in updated:
            branch.update(**arguments)
        self.branches = list(updated)
        return self

    def best_of(self) -> BestOf:
        """The realizations as they run."""
        return BestOf(
            tuple(
                branch.then()
                if isinstance(branch, SolverChain)
                else Then((_as_step(branch),), handover)
                for branch in self.branches
            ),
            _ENERGY,
        )

    @property
    def takes(self) -> frozenset[str]:
        """Every option some branch takes at call time."""
        return self.best_of().takes

    def __call__(
        self,
        problem: Problem | Rung,
        budget: Budget,
        rng: np.random.Generator,
        /,
        *,
        start: np.ndarray | None = None,
        **options: Any,
    ) -> MethodRun:
        """Run every realization on its share and keep the lowest."""
        run: MethodRun = self.best_of()(problem, budget, rng, start=start, **options)
        return run

    def __str__(self) -> str:
        """``x**n`` where every branch is the same, else branches joined by ``|``."""
        first = self.branches[0]
        if all(branch == first for branch in self.branches):
            shown = f"({first})" if isinstance(first, SolverChain) else str(first)
            return f"{shown}{REPEAT}{len(self.branches)}"
        return REALIZATIONS.join(str(branch) for branch in self.branches)


class _Reader:
    """Recursive descent over a chain's text: ``|`` below ``>`` below ``**``."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.at = 0

    def _fail(self, what: str) -> NoReturn:
        msg = f"expected {what} at {self.at} in {self.text!r}"
        raise ValueError(msg)

    def _peek(self, token: str) -> bool:
        while self.at < len(self.text) and self.text[self.at].isspace():
            self.at += 1
        return self.text.startswith(token, self.at)

    def read(self) -> SolverStage | SolverChain | SolverRealizations:
        node = self._branches()
        if self._peek("") and self.at != len(self.text):
            self._fail("the end")
        return node

    def _branches(self) -> SolverStage | SolverChain | SolverRealizations:
        branches = [self._sequence()]
        while self._peek(REALIZATIONS):
            self.at += len(REALIZATIONS)
            branches.append(self._sequence())
        if len(branches) == 1:
            return branches[0]
        return SolverRealizations(
            [
                SolverChain([branch])
                if isinstance(branch, SolverRealizations)
                else branch
                for branch in branches
            ]
        )

    def _sequence(self) -> SolverStage | SolverChain | SolverRealizations:
        stages: list[SolverStage | SolverRealizations] = []
        while True:
            node = self._repeated()
            # A group that is a chain is its stages here: the same run.
            stages.extend(node.stages if isinstance(node, SolverChain) else [node])
            if not self._peek(CHAIN):
                break
            self.at += len(CHAIN)
        return stages[0] if len(stages) == 1 else SolverChain(stages)

    def _repeated(self) -> SolverStage | SolverChain | SolverRealizations:
        node = self._atom()
        if not self._peek(REPEAT):
            return node
        self.at += len(REPEAT)
        self._peek("")
        count = re.match(r"\d+", self.text[self.at :])
        if count is None:
            self._fail("a count after **")
        self.at += count.end()
        n = int(count.group())
        if n < 1:
            self._fail("a count of at least one")
        if n == 1:
            return node
        branch = SolverChain([node]) if isinstance(node, SolverRealizations) else node
        return SolverRealizations([copy.deepcopy(branch) for _ in range(n)])

    def _atom(self) -> SolverStage | SolverChain | SolverRealizations:
        if self._peek("("):
            self.at += 1
            node = self._branches()
            if not self._peek(")"):
                self._fail("a closing parenthesis")
            self.at += 1
            return node
        name = _NAME.match(self.text, self.at)
        if name is None:
            self._fail("a stage name")
        self.at = name.end()
        body = ""
        if self.text.startswith("(", self.at):
            close = self.text.find(")", self.at)
            if close < 0:
                self._fail("a closing parenthesis")
            body = self.text[self.at : close + 1]
            self.at = close + 1
        return SolverStage.parse(name.group() + body)


def compose(text: str) -> SolverChain:
    """:meth:`SolverChain.parse`: any chain or realizations of :data:`METHODS`, :data:`ARMS` and :data:`STAGES` stages, from its text."""
    return SolverChain.parse(text)


# The arms' constants are issue #1038's results, copied from
# `docs/nb/data/potts_schedule.json` (written by `qa.potts_schedule`) and
# frozen here so importing this module reads no file, and issue #1041's
# hand-over schedule, which `qa.potts_clusters` reads from here.

#: Swendsen-Wang's tuned schedule: the file's ``moves.swendsen-wang.chosen``,
#: the lowest mean energy on the tuning seeds at `spatio_only/release`, q = 10.
SWENDSEN_WANG_SCHEDULE = ScheduleParams(
    ScheduleShape.LINEAR, 0.7835758686849045, 0.3235944681535294, 0.0
)
#: Wolff's step count at matched spend: the file's ``matched_wolff.steps``,
#: at which Wolff's mean spend on the tuning seeds meets the 1,000-sweep
#: budget of `spatio_only/release` at q = 10, on :data:`ANNEAL_SCHEDULE`.
#: Fixed, so on another budget or instance it is matched to nothing.
WOLFF_MATCHED_STEPS = 41250
#: The warm chain's schedule: Swendsen-Wang's tuned shape, hold and end from
#: ``t_start = 1.0``, the notebook's ``swendsen-wang warm 1.0`` arm.
WARM_SCHEDULE = replace(SWENDSEN_WANG_SCHEDULE, t_start=1.0)
#: Where the ``expansion>swendsen-wang`` arm's anneal starts and ends:
#: Swendsen-Wang's tuned end, cooled to :data:`ANNEAL_END` (issue #1041).
EXPANSION_SW_SCHEDULE = ScheduleParams(ScheduleShape.LINEAR, 0.3236, ANNEAL_END)

#: The tuned, warm and hybrid solvers of issues #1038 and #1041, by name.
#: Not in :data:`METHODS`, which is issue #906's table and whose rows
#: `docs/nb/potts_starts.ipynb` reproduces; :func:`ground_state` reaches both.
ARMS: dict[str, Method] = {
    "tuned-swendsen-wang": functools.partial(
        run_annealed, move=PottsMove.SWENDSEN_WANG, schedule=SWENDSEN_WANG_SCHEDULE
    ),
    "matched-wolff": functools.partial(
        run_annealed, move=PottsMove.WOLFF, steps=WOLFF_MATCHED_STEPS
    ),
    # ICM to its first clean sweep, then Swendsen-Wang from 1.0 (#1038).
    "warm-anneal": chain("descent", part("swendsen-wang", schedule=WARM_SCHEDULE)),
    # Swendsen-Wang, holding ten expansion cycles back, then the expansion
    # from its labelling on what is left (#1041).
    "swendsen-wang>expansion": chain(
        part(
            "swendsen-wang",
            schedule=SWENDSEN_WANG_SCHEDULE,
            reserve=ExpansionReserve(EXPANSION_RESERVE_CYCLES),
        ),
        "alpha-expansion",
    ),
    # The expansion, then Swendsen-Wang from its labelling (#1041).
    "expansion>swendsen-wang": chain(
        "alpha-expansion", part("swendsen-wang", schedule=EXPANSION_SW_SCHEDULE)
    ),
}

#: The names that run an anneal, and so take a ``schedule`` and ``steps``.
ANNEALED = _ANNEALED_METHODS | frozenset(ARMS)


def ground_state(
    graph: PottsGraph,
    field: np.ndarray,
    method: str | SolverChain | SolverRealizations | Then | BestOf,
    budget: Budget,
    rng: np.random.Generator,
    *,
    start: np.ndarray | None = None,
    schedule: ScheduleParams | None = None,
    steps: int | None = None,
    backend: Backend | None = None,
    min_sites: int = 0,
) -> MethodRun:
    """One :data:`METHODS` or :data:`ARMS` entry on any Potts problem, with no fixture behind it (issues #933, #1052).

    Every solver reads a :class:`Problem` --- the lattice, the field and the
    class count --- and nothing else: the ladder, the sizes and the optimum
    a :class:`Rung` also carries are what the structural referee and the
    bracket read. So a caller with a graph and a field --- a label step's
    energy, say --- reaches every solver here without constructing a
    fixture's rung.

    With ``start``, ``schedule`` and ``steps`` left ``None`` the run is the
    entry's own, bitwise. ``start`` replaces the labelling the solver would
    draw or build: ICM and ICM in random order descend from it, the annealers
    and the warm chain's descent start from it, the expansion and the swap
    cut from it, and each hybrid hands it to its first part. ``field_argmax``,
    ``tempering``, ``max-product`` and ``bifurcation`` have no single
    starting labelling and refuse one.

    Parameters
    ----------
    graph : PottsGraph
        The lattice; every coupling non-negative.
    field : np.ndarray
        ``h``, shape ``(n_nodes, n_states)``.
    method : str | SolverChain | Then
        A key of :data:`METHODS` or :data:`ARMS`; the text of a
        :class:`SolverChain`, stages joined by ``>`` with their arguments,
        e.g. ``swendsen-wang(t_start=1.0,reserve_cycles=10)>alpha-expansion``;
        a :class:`SolverChain`; or a chain built by :func:`chain`. A key is
        read as itself before it is read as a chain.
    budget : Budget
        In :attr:`~sal.cost.Cost.SITE_VISITS`, the unit every
        entry is charged in.
    rng : np.random.Generator
        The method's generator.
    start : np.ndarray | None
        A labelling, shape ``(n_nodes,)``, states in ``[0, n_states)``.
    schedule : ScheduleParams | None
        The anneal's temperatures, in place of the entry's own; a name in
        :data:`ANNEALED` only.
    steps : int | None
        The anneal's step count, fixed beforehand, in place of the budget's
        rule; a name in :data:`ANNEALED` only.
    backend : Backend | None
        The descent's sweep, ``None`` for its default; a name in
        :data:`FLOORED` only.
    min_sites : int
        The descent's floor (issue #1055): after each sweep a state holding
        fewer sites is dissolved into those at or above it. A name in
        :data:`FLOORED` only, where ``0``, the default, is the run before
        the floor existed, bitwise.

    Returns
    -------
    MethodRun

    Raises
    ------
    ValueError
        If ``method`` is in neither table, ``field`` is not one row per node,
        the budget is in another unit, ``start`` is not one state in range
        per node, the method refuses a start, ``schedule`` or ``steps``
        is given to a method that runs no anneal, or ``backend`` or
        ``min_sites > 0`` to one outside :data:`FLOORED`.
    """
    solvers = METHODS | ARMS
    solver: Callable[..., MethodRun]
    if isinstance(method, Then | BestOf | SolverChain | SolverRealizations):
        solver, takes = method, method.takes
    elif method in solvers:
        solver, takes = solvers[method], _options(method)
    elif any(token in method for token in (CHAIN, "(", REALIZATIONS, REPEAT)):
        composed = SolverChain.parse(method)
        solver, takes = composed, composed.takes
    else:
        msg = (
            f"no ground-state method {method!r}; the methods are {sorted(METHODS)} "
            f"and the arms {sorted(ARMS)}, or a chain of those and "
            f"{sorted(STAGES)} joined by {CHAIN!r}, each with its arguments"
        )
        raise ValueError(msg)
    values = np.asarray(field, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != graph.n_nodes:
        msg = (
            f"the field is one row per node, ({graph.n_nodes}, n_states); got "
            f"{values.shape}"
        )
        raise ValueError(msg)
    if budget.unit is not Cost.SITE_VISITS:
        msg = f"every entry is charged in site visits, not {budget.unit}"
        raise ValueError(msg)
    if (schedule is not None or steps is not None) and not takes & ANNEAL_OPTIONS:
        msg = (
            f"{method!r} runs no anneal, so takes no schedule or steps; those "
            f"apply to {sorted(ANNEALED)}"
        )
        raise ValueError(msg)
    if (backend is not None or min_sites != 0) and not takes & DESCENT_OPTIONS:
        msg = (
            f"{method!r} runs no single-site descent, so takes no backend or "
            f"min_sites; those apply to {sorted(FLOORED)}"
        )
        raise ValueError(msg)
    problem = Problem(graph, values, int(values.shape[1]))
    if start is not None:
        start = check_labelling(start, problem.n_nodes, problem.n_states)
    keywords: dict[str, Any] = {}
    if schedule is not None:
        keywords["schedule"] = schedule
    if steps is not None:
        keywords["steps"] = steps
    if takes & DESCENT_OPTIONS:
        keywords["backend"] = backend
        keywords["min_sites"] = min_sites
    return solver(problem, budget, rng, start=start, **keywords)


def outcome(run: MethodRun) -> Outcome:
    """A :class:`~sal.opt.budget.Outcome` from a run, for :func:`~sal.opt.budget.compare`."""
    return Outcome(value=run.energy, spent=run.spent)


@dataclass(frozen=True)
class MethodRecord:
    """One entry's run, carried out of the comparison on its outcome.

    :func:`~sal.opt.budget.compare` scores an energy and a
    spend, and the structural referee needs the *labelling*. Rather than run
    every method twice --- once for the energy and once for the structure ---
    an entry returns the run it already made, on
    :attr:`~sal.opt.budget.Outcome.detail`. A module-level list
    held the same memo and was correct only at ``workers=1``, since a cell
    past that runs in another process (issue #856).

    Parameters
    ----------
    method : str
        The entry's name, a key of :data:`METHODS`.
    rung : str
        The instance's name, :attr:`Rung.name`.
    run : MethodRun
        What the method returned, labelling included.
    """

    method: str
    rung: str
    run: MethodRun


def recorded(comparison: Comparison) -> tuple[MethodRecord, ...]:
    """Every run ``comparison``'s cells made, in cell order.

    A cell run by anything other than an :class:`Entry` carries no record and
    is left out, so this is the entries' runs and not the comparison's cells.
    """
    return tuple(
        cell.detail
        for cell in comparison.outcomes
        if isinstance(cell.detail, MethodRecord)
    )


@dataclass(frozen=True)
class Entry:
    """One named method, in the shape :func:`~sal.opt.budget.compare` calls.

    A frozen dataclass rather than a closure for the reason
    :class:`sal.opt.budget.Restarts` is one: a closure cannot
    cross a process boundary and the comparison's cells may.
    """

    name: str

    def __call__(self, rung: Rung, budget: Budget, rng: np.random.Generator) -> Outcome:
        run = METHODS[self.name](rung, budget, rng)
        return replace(outcome(run), detail=MethodRecord(self.name, rung.name, run))


def entries() -> dict[str, Entry]:
    """Every entry by name, for :func:`~sal.opt.budget.compare`."""
    return {name: Entry(name) for name in METHODS}
