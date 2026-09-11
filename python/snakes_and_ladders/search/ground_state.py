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

See Boykov, Veksler & Zabih (2001) for the expansion bound and Baxter ch. 12
for the ordering coupling the rungs sit either side of.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from snakes_and_ladders.likelihood.message_passing import (
    ConvergenceError,
    MessageSchedule,
    max_product,
)
from snakes_and_ladders.opt.budget import Budget, Outcome
from snakes_and_ladders.opt.schedule import Exponential
from snakes_and_ladders.search.alpha_expansion import (
    alpha_beta_swap,
    alpha_expansion,
    energy,
    iterated_conditional_modes,
)
from snakes_and_ladders.search.backend import Backend
from snakes_and_ladders.search.potts_mcmc import (
    ClusterCounter,
    PottsMove,
    anneal_potts,
    parallel_tempering,
)
from snakes_and_ladders.sim.factor_graph import from_potts
from snakes_and_ladders.sim.graph import PottsGraph
from snakes_and_ladders.sim.potts import SpatioOnlyParams, spatio_only_field

#: The annealing schedule every annealed entry runs, so a difference between
#: them is the move set. It starts above the ordering coupling's temperature
#: and ends cold enough that the last sweeps are a descent.
ANNEAL_START, ANNEAL_END = 2.0, 0.05

#: Replicas in the tempering ladder, geometric over the same endpoints. The
#: budget is divided by this, so a replica gets one sixth of the sweeps the
#: single-site entry gets and the comparison is at equal cost rather than at
#: equal sweeps per chain.
N_REPLICAS = 6


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


def rung_field(
    params: SpatioOnlyParams, n_states: int
) -> tuple[np.ndarray, np.ndarray]:
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
        return params.field, params.alpha
    if n_states == 2:
        alpha = np.array([params.alpha[0], params.alpha[-1]])
        return spatio_only_field(alpha, params.sizes), alpha
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


def expansion_bracket(rung: Rung, expansion_energy: float) -> tuple[float, float]:
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
    tuple[float, float]
        ``(lower, upper)``: the largest energy the optimum cannot be above,
        and the smallest it cannot be below. ``upper`` is the expansion's own
        energy, which the optimum is at most.
    """
    offset = float(rung.field.max(axis=1).sum()) + float(
        np.asarray(rung.graph.coupling).sum()
    )
    return expansion_energy / 2.0 - offset / 2.0, expansion_energy


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
    """

    labelling: np.ndarray
    energy: float
    spent: int
    seconds: float
    trace: tuple[ClusterCounter, ...] = ()
    converged: bool = True


def _anneal(
    rung: Rung, budget: Budget, rng: np.random.Generator, move: PottsMove
) -> MethodRun:
    """One annealed run, its step count fixed before the run starts.

    The step count is ``budget // visits_per_sweep`` for every move set,
    including Wolff, whose step is far cheaper. Wolff therefore *underspends*
    its budget rather than stopping when it is exhausted: a stop on
    accumulated cluster size is a stop on the state, which `search/CLAUDE.md`
    refuses, and the underspend is reported as the finding it is.
    """
    steps = max(1, budget.size // rung.visits_per_sweep)
    start = time.perf_counter()
    run = anneal_potts(
        rung.graph,
        rung.field,
        Exponential(ANNEAL_START, ANNEAL_END, steps),
        rng,
        move=move,
    )
    return MethodRun(
        labelling=run.labelling,
        energy=run.energy,
        spent=run.site_visits,
        seconds=time.perf_counter() - start,
        trace=run.trace,
    )


def run_greedy(rung: Rung, budget: Budget, rng: np.random.Generator) -> MethodRun:
    """The field-only labelling: every site takes its own best class, coupling ignored.

    The trap baseline. It maximizes the tilt by construction, so it is the
    control for the structural referee as well as for the energy: a method it
    ties on both is a method the instance cannot distinguish.
    """
    del budget, rng
    start = time.perf_counter()
    labelling = rung.field.argmax(axis=1).astype(np.int64)
    return MethodRun(
        labelling=labelling,
        energy=energy(rung.graph, rung.field, labelling),
        spent=rung.n_nodes,
        seconds=time.perf_counter() - start,
    )


def run_icm(rung: Rung, budget: Budget, rng: np.random.Generator) -> MethodRun:
    """Iterated conditional modes: single-site descent in index order."""
    steps = max(1, budget.size // rung.visits_per_sweep)
    start = time.perf_counter()
    labelling, value = iterated_conditional_modes(
        rung.graph,
        rung.field,
        rung.n_states,
        rng,
        max_sweeps=steps,
    )
    return MethodRun(
        labelling=labelling,
        energy=value,
        spent=steps * rung.visits_per_sweep,
        seconds=time.perf_counter() - start,
    )


def run_gibbs_zero(rung: Rung, budget: Budget, rng: np.random.Generator) -> MethodRun:
    """Gibbs at T = 0 in a random sweep order, which is ICM under a permutation.

    Reported on ICM's axis rather than beside it. The heat bath at ``T -> 0``
    is the argmin over each site's conditional, which is ICM's update, so the
    two differ in the order sites are visited and in nothing else. Running
    both and reporting one axis is what keeps the comparison from counting
    the same method twice; the pair's spread *is* the sweep order's effect.
    """
    steps = max(1, budget.size // rung.visits_per_sweep)
    start = time.perf_counter()
    labelling = rng.integers(0, rung.n_states, size=rung.n_nodes)
    neighbours: list[list[tuple[int, float]]] = [[] for _ in range(rung.n_nodes)]
    for (first, second), coupling in rung.graph.weighted_edges():
        neighbours[first].append((second, coupling))
        neighbours[second].append((first, coupling))
    for _ in range(steps):
        for node in rng.permutation(rung.n_nodes):
            local = -rung.field[node].copy()
            for neighbour, coupling in neighbours[node]:
                local[labelling[neighbour]] -= coupling
            labelling[node] = int(np.argmin(local))
    return MethodRun(
        labelling=labelling,
        energy=energy(rung.graph, rung.field, labelling),
        spent=steps * rung.visits_per_sweep,
        seconds=time.perf_counter() - start,
    )


def run_anneal(rung: Rung, budget: Budget, rng: np.random.Generator) -> MethodRun:
    """Simulated annealing by heat-bath sweeps: the fair annealed single-site entry."""
    return _anneal(rung, budget, rng, PottsMove.SINGLE_SITE)


def run_swendsen_wang(
    rung: Rung, budget: Budget, rng: np.random.Generator
) -> MethodRun:
    """Annealed Swendsen-Wang: every cluster recoloured, each with its own accept step."""
    return _anneal(rung, budget, rng, PottsMove.SWENDSEN_WANG)


def run_wolff(rung: Rung, budget: Budget, rng: np.random.Generator) -> MethodRun:
    """Annealed Wolff: one cluster per step, with the accept step on its field."""
    return _anneal(rung, budget, rng, PottsMove.WOLFF)


def run_tempering(rung: Rung, budget: Budget, rng: np.random.Generator) -> MethodRun:
    """Parallel tempering over a geometric ladder, charged for every replica.

    The budget buys ``budget // (N_REPLICAS * visits_per_sweep)`` sweeps per
    replica rather than that many per chain, which is the whole difference
    between a comparison at equal cost and one at equal sweeps.
    """
    per_replica = max(1, budget.size // (N_REPLICAS * rung.visits_per_sweep))
    ladder = tuple(
        float(value) for value in np.geomspace(ANNEAL_START, ANNEAL_END, N_REPLICAS)
    )
    start = time.perf_counter()
    run = parallel_tempering(
        rung.graph,
        rung.field,
        ladder,
        rng,
        per_replica,
    )
    return MethodRun(
        labelling=run.best,
        energy=run.best_energy,
        spent=N_REPLICAS * per_replica * rung.visits_per_sweep,
        seconds=time.perf_counter() - start,
    )


def run_alpha_expansion(
    rung: Rung, budget: Budget, rng: np.random.Generator
) -> MethodRun:
    """Alpha expansion: the only entry carrying a bound, and the bracket's lower end."""
    del rng
    cycles = max(1, budget.size // (rung.n_states * rung.visits_per_sweep))
    start = time.perf_counter()
    run = alpha_expansion(
        rung.graph,
        rung.field,
        rung.n_states,
        max_cycles=cycles,
        backend=Backend.RUST,
    )
    return MethodRun(
        labelling=run.labelling,
        energy=run.energy,
        spent=run.cycles * rung.n_states * rung.visits_per_sweep,
        seconds=time.perf_counter() - start,
    )


def run_alpha_beta_swap(
    rung: Rung, budget: Budget, rng: np.random.Generator
) -> MethodRun:
    """Alpha-beta swap: the cheaper move, and the one with no bound.

    A cycle is ``q (q - 1) / 2`` cuts against the expansion's ``q``, and each
    cut covers only the sites carrying its two labels. Summed over a cycle
    every site is therefore visited ``q - 1`` times, not once per pair, so a
    cycle costs ``(q - 1)`` sweeps against the expansion's ``q`` --- the swap
    is much cheaper per *cut* and barely cheaper per *cycle*, which is the
    trade, and charging it by the pair count would have hidden it.
    """
    del rng
    per_cycle = (rung.n_states - 1) * rung.visits_per_sweep
    cycles = max(1, budget.size // per_cycle)
    start = time.perf_counter()
    run = alpha_beta_swap(
        rung.graph,
        rung.field,
        rung.n_states,
        max_cycles=cycles,
        backend=Backend.RUST,
    )
    return MethodRun(
        labelling=run.labelling,
        energy=run.energy,
        spent=run.cycles * per_cycle,
        seconds=time.perf_counter() - start,
    )


def run_max_product(rung: Rung, budget: Budget, rng: np.random.Generator) -> MethodRun:
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
    del rng
    iterations = max(1, budget.size // rung.visits_per_sweep)
    graph = from_potts(rung.graph, rung.field)
    start = time.perf_counter()
    try:
        assignment, _ = max_product(
            graph, schedule=MessageSchedule.FLOODING, max_iterations=iterations
        )
    except ConvergenceError:
        return MethodRun(
            labelling=rung.field.argmax(axis=1).astype(np.int64),
            energy=float("inf"),
            spent=iterations * rung.visits_per_sweep,
            seconds=time.perf_counter() - start,
            converged=False,
        )
    labelling = np.array(
        [assignment[f"s{node}"] for node in range(rung.n_nodes)], dtype=np.int64
    )
    return MethodRun(
        labelling=labelling,
        energy=energy(rung.graph, rung.field, labelling),
        spent=iterations * rung.visits_per_sweep,
        seconds=time.perf_counter() - start,
    )


#: Every entry, in report order. ICM and Gibbs at T = 0 are two rows of one
#: axis, named so the report cannot present them as independent methods.
METHODS = {
    "greedy": run_greedy,
    "icm": run_icm,
    "gibbs-T0": run_gibbs_zero,
    "anneal": run_anneal,
    "swendsen-wang": run_swendsen_wang,
    "wolff": run_wolff,
    "tempering": run_tempering,
    "alpha-expansion": run_alpha_expansion,
    "alpha-beta-swap": run_alpha_beta_swap,
    "max-product": run_max_product,
}

#: The two rows that are one axis, so a reader of the table is told rather
#: than left to notice.
ONE_AXIS = ("icm", "gibbs-T0")


def outcome(run: MethodRun) -> Outcome:
    """A :class:`~snakes_and_ladders.opt.budget.Outcome` from a run, for :func:`~snakes_and_ladders.opt.budget.compare`."""
    return Outcome(value=run.energy, spent=run.spent)


#: Every run this process has made, in call order. :func:`~snakes_and_ladders.opt.budget.compare`
#: returns an energy and a spend and nothing else, and the structural referee
#: needs the *labelling*. Rather than run every method twice --- once for the
#: energy and once for the structure --- an entry records its run here as it
#: goes. It is a memo of work already done, not a second code path, and it is
#: correct only at ``workers=1``, which is where the comparison is run and
#: which `opt/budget.py` states is the default until a measurement says
#: otherwise.
_RUNS: list[tuple[str, str, MethodRun]] = []


def recorded() -> tuple[tuple[str, str, MethodRun], ...]:
    """Every run since :func:`forget`, as ``(method, rung, run)``."""
    return tuple(_RUNS)


def forget() -> None:
    """Drop the recorded runs, so one process can run two comparisons."""
    _RUNS.clear()


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
        _RUNS.append((self.name, rung.name, run))
        return outcome(run)


def entries() -> dict[str, Entry]:
    """Every entry by name, for :func:`~snakes_and_ladders.opt.budget.compare`."""
    return {name: Entry(name) for name in METHODS}
