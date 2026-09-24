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
is the entry's own bitwise (issue #1052).

See Boykov, Veksler & Zabih (2001) for the expansion bound and Baxter ch. 12
for the ordering coupling the rungs sit either side of.
"""

from __future__ import annotations

import functools
import time
from collections.abc import Iterator
from dataclasses import dataclass, replace
from typing import Any, Protocol

import numpy as np

from snakes_and_ladders.backend import Backend
from snakes_and_ladders.cost import Cost
from snakes_and_ladders.likelihood.message_passing import (
    ConvergenceError,
    MessageScheduleName,
    max_product,
)
from snakes_and_ladders.opt.budget import Budget, Comparison, Outcome
from snakes_and_ladders.opt.termination import Stop, Termination
from snakes_and_ladders.sample.potts_mcmc import (
    ClusterCounter,
    PottsMove,
    anneal_potts,
    parallel_tempering,
)
from snakes_and_ladders.sample.schedule import ScheduleParams, ScheduleShape
from snakes_and_ladders.search.alpha_expansion import (
    SweepOrder,
    alpha_beta_swap,
    alpha_expansion,
    iterated_conditional_modes,
)
from snakes_and_ladders.search.bifurcation import simulated_bifurcation
from snakes_and_ladders.sim.factor_graph import from_potts
from snakes_and_ladders.sim.graph import BoundaryCondition, PottsGraph, lattice_graph
from snakes_and_ladders.sim.potts import (
    SpatioOnlyParams,
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
#: :class:`~snakes_and_ladders.sample.schedule.ExponentialTempSchedule` the
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

    The coupling is :func:`~snakes_and_ladders.sim.potts.critical_coupling`'s
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
        :func:`~snakes_and_ladders.sim.potts.spatio_only_field` builds it.
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
    :func:`snakes_and_ladders.sim.potts.spatio_only_field` from the same sizes and
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
    (:func:`~snakes_and_ladders.sim.potts.spatio_only_field`), linear in the
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
    threshold: float | None = None,
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
    ``threshold`` is Niedermayer's ``E_0``, passed to
    :func:`~snakes_and_ladders.sample.potts_mcmc.anneal_potts`; ``None`` is
    the graph's own threshold, the run above bitwise (issue #1046).
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
        initial=start,
        threshold=threshold,
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
    max_sweeps: int,
    *,
    start: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    """Index-order ICM from a uniform draw, or from ``start``, one sweep at a time, and the sweeps it ran.

    The descent :func:`run_icm` runs, on the same draws, with its sweep count
    read out: the count is what a warm chain is charged, where
    :func:`run_icm` charges its whole budget. The sweep that changes nothing
    is counted, as :func:`~snakes_and_ladders.search.potts_starts.polish_by_icm`
    counts it.

    Returns
    -------
    tuple[np.ndarray, int]
        The labelling, and the sweeps run, at most ``max_sweeps``.
    """
    problem = _problem(problem)
    labelling = (
        rng.integers(0, problem.n_states, size=problem.n_nodes)
        if start is None
        else np.array(start, dtype=np.int64)
    )
    sweeps = 0
    while sweeps < max_sweeps:
        settled = iterated_conditional_modes(
            problem.graph,
            problem.field,
            problem.n_states,
            rng,
            start=labelling,
            max_sweeps=1,
        )
        sweeps += 1
        if np.array_equal(settled.labelling, labelling):
            break
        labelling = settled.labelling
    return labelling, sweeps


def warm_anneal(
    problem: Problem | Rung,
    budget: Budget,
    rng: np.random.Generator,
    move: PottsMove,
    schedule: ScheduleParams,
    *,
    steps: int | None = None,
    start: np.ndarray | None = None,
) -> MethodRun:
    """ICM from a uniform draw, or from ``start``, to its first clean sweep, then an anneal from its labelling.

    A warm chain (issue #1038): the descent's sweeps are charged at
    ``visits_per_sweep`` each, clean sweep included, and the anneal gets what
    is left of ``budget`` by :func:`run_annealed`'s rule, or ``steps`` where the
    caller fixed a count. Both draw from ``rng`` in that order, so the
    descent is the one :func:`run_icm` runs on the same generator.

    Returns
    -------
    MethodRun
        The anneal's labelling and energy, ``spent`` the two charges summed.
    """
    problem = _problem(problem)
    started = time.perf_counter()
    labelling, sweeps = descend(
        problem, rng, budget.size // problem.visits_per_sweep, start=start
    )
    descent = sweeps * problem.visits_per_sweep
    run = run_annealed(
        problem,
        Budget(budget.unit, budget.size - descent),
        rng,
        move,
        schedule=schedule,
        steps=steps,
        start=labelling,
    )
    return replace(
        run, spent=descent + run.spent, seconds=time.perf_counter() - started
    )


def run_greedy(
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
        "greedy",
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
) -> MethodRun:
    """Iterated conditional modes: single-site descent in index order.

    From a uniform draw, or from ``start``.
    """
    problem = _problem(problem)
    steps = max(1, budget.size // problem.visits_per_sweep)
    started = time.perf_counter()
    settled = iterated_conditional_modes(
        problem.graph,
        problem.field,
        problem.n_states,
        rng,
        start=start,
        max_sweeps=steps,
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
    """
    problem = _problem(problem)
    steps = max(1, budget.size // problem.visits_per_sweep)
    started = time.perf_counter()
    settled = iterated_conditional_modes(
        problem.graph,
        problem.field,
        problem.n_states,
        rng,
        start=start,
        max_sweeps=steps,
        sweep_order=SweepOrder.RANDOM,
        stop_when_clean=False,
        backend=Backend.NUMBA,
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

    It starts from the field's argmax, the greedy labelling, or from
    ``start``, and draws nothing from ``rng``.
    """
    problem = _problem(problem)
    del rng
    cycles = max(1, budget.size // (problem.n_states * problem.visits_per_sweep))
    started = time.perf_counter()
    run = alpha_expansion(
        problem.graph,
        problem.field,
        problem.n_states,
        start=start,
        max_cycles=cycles,
        backend=Backend.RUST,
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
        problem.n_states,
        start=start,
        max_cycles=cycles,
        backend=Backend.RUST,
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
    enter the greedy baseline's numbers in max-product's row, which is the
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
        problem.graph, problem.field, problem.n_states, rng, steps=steps
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
    "greedy": run_greedy,
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


#: Cycles of the expansion :func:`run_swendsen_wang_then_expansion` holds
#: back from Swendsen-Wang's share. From a uniform start the expansion ends in
#: 3 cycles on `spatio_only/release` at ten states; 10 leaves it room from a
#: labelling that is not its own.
EXPANSION_RESERVE_CYCLES = 10


def run_swendsen_wang_then_expansion(
    problem: Problem | Rung,
    budget: Budget,
    rng: np.random.Generator,
    *,
    schedule: ScheduleParams,
    reserve_cycles: int = EXPANSION_RESERVE_CYCLES,
    steps: int | None = None,
    start: np.ndarray | None = None,
) -> MethodRun:
    """Swendsen-Wang on ``schedule``, then alpha-expansion from its labelling (issue #1041).

    The anneal runs from a uniform draw or ``start`` on ``budget`` less
    ``reserve_cycles`` expansion cycles, by :func:`run_annealed`'s rule or at
    ``steps``; the expansion starts from the anneal's best labelling with
    what is left as its cap, and is charged the cycles it ran.

    Returns
    -------
    MethodRun
        The expansion's labelling and energy, ``spent`` both parts summed.
    """
    problem = _problem(problem)
    started = time.perf_counter()
    per_cycle = problem.n_states * problem.visits_per_sweep
    anneal = run_annealed(
        problem,
        Budget(budget.unit, budget.size - reserve_cycles * per_cycle),
        rng,
        PottsMove.SWENDSEN_WANG,
        schedule=schedule,
        steps=steps,
        start=start,
    )
    left = budget.size - anneal.spent
    expansion = alpha_expansion(
        problem.graph,
        problem.field,
        problem.n_states,
        start=anneal.labelling,
        max_cycles=max(1, left // per_cycle),
        backend=Backend.RUST,
    )
    return MethodRun(
        labelling=expansion.labelling,
        energy=expansion.energy,
        spent=anneal.spent + expansion.cycles * per_cycle,
        seconds=time.perf_counter() - started,
        termination=expansion.termination,
    )


def run_expansion_then_swendsen_wang(
    problem: Problem | Rung,
    budget: Budget,
    rng: np.random.Generator,
    *,
    schedule: ScheduleParams,
    steps: int | None = None,
    start: np.ndarray | None = None,
) -> MethodRun:
    """Alpha-expansion, then Swendsen-Wang on ``schedule`` from its labelling (issue #1041).

    The expansion runs as :func:`run_alpha_expansion` does, from ``start``
    where one is given, and is charged its cycles; the anneal gets the rest
    of ``budget`` by :func:`run_annealed`'s rule or ``steps``. The anneal
    returns the lowest energy it visited, its start included, so the arm
    hands over the expansion's energy or lower.

    Returns
    -------
    MethodRun
        The anneal's labelling and energy, ``spent`` both parts summed.
    """
    problem = _problem(problem)
    started = time.perf_counter()
    expansion = run_alpha_expansion(problem, budget, rng, start=start)
    anneal = run_annealed(
        problem,
        Budget(budget.unit, budget.size - expansion.spent),
        rng,
        PottsMove.SWENDSEN_WANG,
        schedule=schedule,
        steps=steps,
        start=expansion.labelling,
    )
    return replace(
        anneal,
        spent=expansion.spent + anneal.spent,
        seconds=time.perf_counter() - started,
    )


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
#: Where :func:`run_expansion_then_swendsen_wang`'s chain starts and ends:
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
    "warm-anneal": functools.partial(
        warm_anneal, move=PottsMove.SWENDSEN_WANG, schedule=WARM_SCHEDULE
    ),
    "swendsen-wang>expansion": functools.partial(
        run_swendsen_wang_then_expansion, schedule=SWENDSEN_WANG_SCHEDULE
    ),
    "expansion>swendsen-wang": functools.partial(
        run_expansion_then_swendsen_wang, schedule=EXPANSION_SW_SCHEDULE
    ),
}

#: The names that run an anneal, and so take a ``schedule`` and ``steps``.
ANNEALED = frozenset({"anneal", "swendsen-wang", "wolff", *ARMS})


def ground_state(
    graph: PottsGraph,
    field: np.ndarray,
    method: str,
    budget: Budget,
    rng: np.random.Generator,
    *,
    start: np.ndarray | None = None,
    schedule: ScheduleParams | None = None,
    steps: int | None = None,
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
    cut from it, and each hybrid hands it to its first part. ``greedy``,
    ``tempering``, ``max-product`` and ``bifurcation`` have no single
    starting labelling and refuse one.

    Parameters
    ----------
    graph : PottsGraph
        The lattice; every coupling non-negative.
    field : np.ndarray
        ``h``, shape ``(n_nodes, n_states)``.
    method : str
        A key of :data:`METHODS` or :data:`ARMS`.
    budget : Budget
        In :attr:`~snakes_and_ladders.cost.Cost.SITE_VISITS`, the unit every
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

    Returns
    -------
    MethodRun

    Raises
    ------
    ValueError
        If ``method`` is in neither table, ``field`` is not one row per node,
        the budget is in another unit, ``start`` is not one state in range
        per node, the method refuses a start, or ``schedule`` or ``steps``
        is given to a method that runs no anneal.
    """
    solvers = METHODS | ARMS
    if method not in solvers:
        msg = (
            f"no ground-state method {method!r}; the methods are {sorted(METHODS)} "
            f"and the arms {sorted(ARMS)}"
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
    if (schedule is not None or steps is not None) and method not in ANNEALED:
        msg = (
            f"{method!r} runs no anneal, so takes no schedule or steps; those "
            f"apply to {sorted(ANNEALED)}"
        )
        raise ValueError(msg)
    problem = Problem(graph, values, int(values.shape[1]))
    if start is not None:
        start = _checked_start(problem, start)
    keywords: dict[str, Any] = {}
    if schedule is not None:
        keywords["schedule"] = schedule
    if steps is not None:
        keywords["steps"] = steps
    return solvers[method](problem, budget, rng, start=start, **keywords)


def _checked_start(problem: Problem, start: np.ndarray) -> np.ndarray:
    """``start`` as an ``int64`` copy, checked to be one state in range per node.

    Raises
    ------
    ValueError
        If it is not integer, not shape ``(n_nodes,)``, or holds a state
        outside ``[0, n_states)``.
    """
    labelling = np.asarray(start)
    if (
        labelling.shape != (problem.n_nodes,)
        or not np.issubdtype(labelling.dtype, np.integer)
        or not ((labelling >= 0).all() and (labelling < problem.n_states).all())
    ):
        msg = (
            f"start must hold one integer state in [0, {problem.n_states}) per "
            f"node of {problem.n_nodes}; got shape {labelling.shape}, dtype "
            f"{labelling.dtype}"
        )
        raise ValueError(msg)
    return np.array(labelling, dtype=np.int64)


def outcome(run: MethodRun) -> Outcome:
    """A :class:`~snakes_and_ladders.opt.budget.Outcome` from a run, for :func:`~snakes_and_ladders.opt.budget.compare`."""
    return Outcome(value=run.energy, spent=run.spent)


@dataclass(frozen=True)
class MethodRecord:
    """One entry's run, carried out of the comparison on its outcome.

    :func:`~snakes_and_ladders.opt.budget.compare` scores an energy and a
    spend, and the structural referee needs the *labelling*. Rather than run
    every method twice --- once for the energy and once for the structure ---
    an entry returns the run it already made, on
    :attr:`~snakes_and_ladders.opt.budget.Outcome.detail`. A module-level list
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
    """One named method, in the shape :func:`~snakes_and_ladders.opt.budget.compare` calls.

    A frozen dataclass rather than a closure for the reason
    :class:`snakes_and_ladders.opt.budget.Restarts` is one: a closure cannot
    cross a process boundary and the comparison's cells may.
    """

    name: str

    def __call__(self, rung: Rung, budget: Budget, rng: np.random.Generator) -> Outcome:
        run = METHODS[self.name](rung, budget, rng)
        return replace(outcome(run), detail=MethodRecord(self.name, rung.name, run))


def entries() -> dict[str, Entry]:
    """Every entry by name, for :func:`~snakes_and_ladders.opt.budget.compare`."""
    return {name: Entry(name) for name in METHODS}
