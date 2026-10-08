"""The drivers: a recorded chain, annealing, parallel tempering, the two-replica Houdayer chain, and the sweep a move set names.

Each runs the kernels of :mod:`~sal.sample.potts_mcmc.sweeps`
over a schedule and returns what the run recorded; :func:`sweep_for` is where a
:class:`~sal.sample.potts_mcmc.moves.PottsMove` becomes a
kernel.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.random import Generator

from sal.backend import Backend
from sal.cost import Cost
from sal.opt.budget import Budget
from sal.opt.termination import Stop, Termination
from sal.parallel import Pool, map_tasks
from sal.sample.accept import accept
from sal.sample.loop import (
    Exchanging,
    Moved,
    Step,
    anneal,
    anneal_spent,
    swap_log_ratio,
    temper,
)
from sal.sample.potts_mcmc.moves import (
    PottsMove,
    PottsMoves,
    Recolour,
    move_set,
    refuse_negative_coupling,
)
from sal.sample.potts_mcmc.sweeps import (
    ClusterCounter,
    adjacency_lists,
    balanced_sweep_at,
    houdayer_move,
    sweep_at,
    swendsen_wang_heat_bath_sweep,
    swendsen_wang_sweep,
    wolff_heat_bath_sweep,
    wolff_sweep,
)
from sal.sample.schedule import (
    Annealed,
    Monotone,
    Tempered,
    TempSchedule,
    check_ladder,
    ladder,
)
from sal.sample.statistics import integrated_autocorrelation_time, split_rhat
from sal.sample.tune import (
    Ladder,
    LadderTuning,
    Schedule,
    ScheduleTuning,
    TunedLadder,
    TunedSchedule,
    resolve_ladder,
    resolve_schedule,
)
from sal.sim.graph import PottsGraph
from sal.sim.potts import (
    SiteField,
    check_labelling,
    energies,
    log_weight_of,
    site_field,
)

#: The move sets :func:`balanced_sweep_at` serves.
_BALANCED_MOVES = frozenset(
    {PottsMove.LOCALLY_BALANCED, PottsMove.GIBBS_WITH_GRADIENTS}
)

#: The move sets that grow one cluster a step, charged by its size.
_SINGLE_CLUSTER_MOVES = frozenset({PottsMove.WOLFF, PottsMove.WOLFF_HEAT_BATH})


@dataclass(frozen=True)
class PottsChain:
    """One chain's recorded configurations, and what a sweep cost.

    Parameters
    ----------
    states : np.ndarray
        Integer states, shape ``(n_sweeps, n_nodes)``.
    mean_cluster_size : float
        Sites per cluster flip, averaged over the run. A Wolff sweep flips one
        cluster while the other two touch every site, so an autocorrelation
        time in sweeps is not comparable across the three without it. For the
        move sets that build no clusters it is ``n_nodes``.
    acceptance : float
        Steps that changed the labelling over steps run, burn-in included.
        For a cluster move it is accepted over proposed; a step that redraws
        every label it touches to its own value counts as refused, since it
        moved the chain nowhere. Read from the state before and after each
        step, so it draws no random number (issue #1316).
    largest_cluster_share : float
        The largest cluster a step built, over ``n_nodes``: ``1.0`` is a
        spanning cluster, whose move is a global relabelling. ``1.0`` for the
        move sets that build no clusters, as ``mean_cluster_size`` is
        ``n_nodes`` for them.
    ess : np.ndarray
        Effective draws per observable, ``n_records / tau`` with ``tau`` from
        :func:`~sal.sample.statistics.integrated_autocorrelation_time`: entry
        0 the energy, entry ``1 + k`` the occupancy of label ``k``
        (:func:`observables`). A constant energy series is ``0``: a chain
        whose energy never moved has no sample to count. A constant
        occupancy is ``inf`` and decides nothing, since a label no draw holds
        is constant in a mixed chain too.
    termination : Termination
        :attr:`~sal.opt.termination.Stop.CONVERGED` where every entry of
        ``ess`` is at least :data:`ESS_FLOOR`, else
        :attr:`~sal.opt.termination.Stop.NOT_MIXING`; ``iterations`` counts
        every step run. The draws are kept either way.
    """

    states: np.ndarray
    mean_cluster_size: float
    acceptance: float
    largest_cluster_share: float
    ess: np.ndarray
    termination: Termination


#: Effective draws under which a chain is reported as not mixing: the floor
#: `tests/regression/sample/test_exact_lattice_referees.py` declares (#1276),
#: with the same estimator ``n_records / tau``.
ESS_FLOOR = 50.0

#: Split R-hat above which chains from distinct starts are reported as not
#: mixing: Vehtari et al. (2021), *Rank-normalization, folding, and
#: localization*, Bayesian Analysis 16(2), recommend 1.01, against 1.1 in
#: Gelman et al., *Bayesian Data Analysis*, 3rd ed.; the stricter is declared.
RHAT_THRESHOLD = 1.01


def observables(graph: PottsGraph, field: np.ndarray, states: np.ndarray) -> np.ndarray:
    """The energy and each label's occupancy per recorded state, shape ``(1 + q, n_records)``.

    The scalar series :attr:`PottsChain.ess` and :attr:`PottsStarts.rhat`
    are read on, in that order.
    """
    rows = site_field(field, graph.n_nodes)
    n_states = int(rows.shape[1])
    energy = energies(graph, rows, states)
    occupancy = np.stack([(states == label).mean(axis=1) for label in range(n_states)])
    return np.concatenate([energy[None, :], occupancy], axis=0)


def effective_draws(series: np.ndarray) -> np.ndarray:
    """:attr:`PottsChain.ess` from :func:`observables`' rows."""
    n_records = series.shape[1]
    ess = np.empty(series.shape[0])
    for index, row in enumerate(series):
        if n_records < 2:
            ess[index] = float(n_records)
        elif np.ptp(row) == 0.0:
            ess[index] = 0.0 if index == 0 else np.inf
        else:
            ess[index] = n_records / integrated_autocorrelation_time(row)
    return ess


def _mixing(ess: np.ndarray, steps: int) -> Termination:
    """``CONVERGED`` where every observable clears :data:`ESS_FLOOR`, else ``NOT_MIXING``."""
    mixed = bool(np.all(ess >= ESS_FLOOR))
    return Termination(
        converged=mixed,
        iterations=steps,
        reason=Stop.CONVERGED if mixed else Stop.NOT_MIXING,
    )


@dataclass(frozen=True)
class TemperedModel:
    """A model rescaled to the temperature it is to be sampled at.

    Parameters
    ----------
    graph : PottsGraph
        The couplings divided by the temperature.
    field : np.ndarray
        The field divided by the same, so the pair is one model and not two
        scalings a caller has to keep together.
    """

    graph: PottsGraph
    field: np.ndarray

    def __iter__(self) -> Iterator[Any]:
        """``(graph, field)``: the order callers unpack.

        ``Any`` and not a union: an unpacking gives every name the element
        type, so a union would mistype each of them.
        """
        yield from (self.graph, self.field)


def tempered(graph: PottsGraph, field: np.ndarray, temperature: float) -> TemperedModel:
    """The model whose Boltzmann weight at temperature 1 is this one's at ``temperature``.

    ``(J, h) / T``. At ``T = 1`` the division is the identity bitwise, which
    lets every untempered chain be the tempered one at the default rather than
    a separate path.

    Raises
    ------
    ValueError
        If ``temperature`` is not positive: at zero the sweep is a descent
        and the chain samples nothing.
    """
    if not temperature > 0.0:
        msg = f"temperature must be positive, got {temperature}"
        raise ValueError(msg)
    scaled = PottsGraph(
        n_nodes=graph.n_nodes,
        edges=graph.edges,
        coupling=tuple(coupling / temperature for coupling in graph.coupling),
        shape=graph.shape,
    )
    return TemperedModel(
        graph=scaled, field=np.asarray(field, dtype=float) / temperature
    )


def sample_potts(
    graph: PottsGraph,
    field: SiteField | np.ndarray,
    move: PottsMoves,
    rng: np.random.Generator,
    n_sweeps: int | Budget,
    burn_in: int = 0,
    thin: int = 1,
    *,
    temperature: float = 1.0,
    recolour: Recolour = Recolour.PER_MOVE,
    backend: Backend = Backend.RUST,
    cluster_backend: Backend = Backend.RUST,
    start: np.ndarray | None = None,
) -> PottsChain:
    """Run one chain and return the configuration after every sweep.

    Parameters
    ----------
    graph : PottsGraph
        The lattice. Couplings may vary per edge.
    field : SiteField | np.ndarray
        External field ``h``, shape ``(n_states,)``.
    move : PottsMoves
        The move set, or a sequence of them applied in order as one step
        (issue #1317); a single move is its one-element set, bitwise. Every
        one leaves the same Boltzmann distribution invariant, which is what
        `tests/regression/search/test_potts_mcmc.py` asserts.
    rng : np.random.Generator
        Passed in rather than seeded here: seeding inside a call makes every
        draw of an ensemble identical (`sim/CLAUDE.md`, issue #240).
    n_sweeps : int | Budget
        Recorded sweeps. A sweep is ``n_nodes`` heat-bath updates, ``n_nodes``
        gradient-informed proposals, one Swendsen-Wang bond-and-recolour pass
        over the whole lattice, or *one* Wolff cluster step ---
        see :func:`wolff_sweep` for why a single-cluster sweep cannot be
        sized to match the others. A :class:`~sal.opt.budget.Budget` in
        :attr:`~sal.cost.Cost.SITE_VISITS` is that size instead (issue
        #1344): after burn-in the chain takes steps, each charged as
        :func:`step_visits` charges it, until they have spent
        ``budget.size``, recording every ``thin``-th, so a Wolff chain spends
        its budget whatever its clusters' sizes. ``states`` then holds as many
        records as the steps bought.
    burn_in : int
        Sweeps run and discarded before recording starts.
    thin : int
        Record one sweep in every ``thin``. Successive sweeps are correlated,
        so a goodness-of-fit test run on every sweep rejects a *correct*
        sampler: the chi-square assumes independent draws and the correlation
        inflates it. Thinning by several autocorrelation times makes the test
        measure the sampler rather than the correlation.
    temperature : float
        The chain targets ``exp(-E / temperature)``; 1 is the model as
        declared. Implemented as :func:`tempered` model scaling, so the
        cluster moves' bond probabilities and the field accept step are
        tempered by the same division as the heat bath.
    backend : Backend
        Which implementation runs the **heat-bath** sweep; the cluster and
        gradient-informed moves have one and ignore it. The chain is the same
        either way, state for state, which is what makes
        :data:`~sal.backend.Backend.RUST` the default
        (:func:`sweep_at`, issue #599).
    cluster_backend : Backend
        Which implementation runs the **Swendsen-Wang** pass and the two
        **Wolff** steps (#1362); Niedermayer has one and ignores it. A
        separate argument rather than the one above because the two are not the same decision: the Rust pass draws
        the same uniforms in a different order and so returns a chain of the
        same law rather than the same chain (:func:`_cluster_pass_rust`,
        issue #754). :data:`~sal.backend.Backend.RUST` is the default
        nonetheless, 35.6x the oracle on ten sweeps of a 64x64 lattice;
        :data:`~sal.backend.Backend.PYTHON` replays the oracle's stream, which
        was the default before #1283.
    recolour : Recolour
        How each cluster move in ``move`` draws a cluster's label
        (:func:`~sal.sample.potts_mcmc.moves.move_set`). ``UNIFORM``, the
        default, is each move's behaviour before #1317.
    start : np.ndarray | None
        The starting labelling. ``None``, the default, draws one uniformly
        from ``rng`` as before; a given start draws nothing, so
        :func:`sample_potts_starts` can run the ordered start (issue #1316).

    Returns
    -------
    PottsChain
        The recorded configurations, the mean cluster size where the move
        set builds clusters, and the mixing diagnostics of issue #1316.

    Raises
    ------
    ValueError
        If a cluster move is asked for on a graph with a negative coupling. The
        bond probability ``1 - exp(-J)`` is not a probability there, and an
        antiferromagnet has no like-spin clusters to flip.
    """
    field = log_weight_of(field)
    site_visits(n_sweeps if isinstance(n_sweeps, Budget) else None)
    move = move_set(move, recolour)
    refuse_negative_coupling(move, graph)

    model = tempered(graph, field, temperature)
    graph, field = model.graph, model.field
    rows = site_field(field, graph.n_nodes)
    n_states = int(rows.shape[1])
    # Contiguous `int64` because the kernel borrows this buffer rather than
    # copying it; `integers` already returns one here, so this asserts the
    # layout rather than paying for it.
    if start is None:
        state = np.ascontiguousarray(
            rng.integers(0, n_states, size=graph.n_nodes), dtype=np.int64
        )
    else:
        state = np.ascontiguousarray(
            check_labelling(start, graph.n_nodes, n_states), dtype=np.int64
        )
    lattice = _Lattice(
        graph, rows, graph.compressed_adjacency(), backend, cluster_backend
    )
    return _record_chain(lattice, move, state, rng, n_sweeps, burn_in, thin)


def _advance(
    lattice: _Lattice, moves: Sequence[PottsMove]
) -> Callable[[np.ndarray, np.random.Generator, float], int]:
    """``moves`` in order as one sweep on ``lattice``: :func:`sweep_for`'s closure (issue #1365)."""
    if len(moves) == 1:
        return lattice.sweep(moves[0])
    parts = [lattice.sweep(each) for each in moves]

    def composed(state: np.ndarray, rng: np.random.Generator, beta: float = 1.0) -> int:
        return sum(part(state, rng, beta) for part in parts)

    return composed


def _record_chain(
    lattice: _Lattice,
    move: Sequence[PottsMove],
    state: np.ndarray,
    rng: np.random.Generator,
    n_sweeps: int | Budget,
    burn_in: int,
    thin: int,
) -> PottsChain:
    """:func:`sample_potts`' recording loop on a built lattice and start (issue #1365).

    Shared with :mod:`sal.sandbox.potts_moves`, whose lattice adds the moves
    that left :class:`PottsMove`, so the two drivers record alike.
    """
    graph, rows = lattice.graph, lattice.rows
    advance = _advance(lattice, move)

    if isinstance(n_sweeps, Budget):
        return _spend_chain(
            graph,
            rows,
            state,
            rng,
            move,
            n_sweeps.size,
            burn_in,
            thin,
            lattice,
        )
    recorded = np.empty((n_sweeps, graph.n_nodes), dtype=np.int64)
    before = np.empty_like(state)
    cluster_total, cluster_count, largest, accepted = 0, 0, 0, 0
    for step in range(-burn_in * thin, n_sweeps * thin):
        before[:] = state
        size = advance(state, rng, 1.0)
        accepted += not np.array_equal(before, state)
        if size:
            cluster_total += size
            cluster_count += 1
            largest = max(largest, size)
        if step >= 0 and (step + 1) % thin == 0:
            recorded[step // thin] = state
    steps = (burn_in + n_sweeps) * thin
    mean_cluster = (
        cluster_total / cluster_count if cluster_count else float(graph.n_nodes)
    )
    ess = effective_draws(observables(graph, rows, recorded))
    return PottsChain(
        states=recorded,
        mean_cluster_size=mean_cluster,
        acceptance=accepted / steps if steps else 0.0,
        largest_cluster_share=largest / graph.n_nodes if cluster_count else 1.0,
        ess=ess,
        termination=_mixing(ess, steps),
    )


def _spend_chain(
    graph: PottsGraph,
    rows: np.ndarray,
    state: np.ndarray,
    rng: np.random.Generator,
    move: Sequence[PottsMove],
    budget: int,
    burn_in: int,
    thin: int,
    lattice: _Lattice,
) -> PottsChain:
    """:func:`sample_potts`' loop under a site-visit budget (issue #1344).

    The moves run in :func:`sweep_for`'s order on the same closures, so a
    step draws what it draws there; each move is charged by
    :func:`step_visits`, and the steps after burn-in run until they have
    spent ``budget``, overspending by less than the last.
    """
    parts = [(each, lattice.sweep(each)) for each in move]
    kept: list[np.ndarray] = []
    before = np.empty_like(state)
    cluster_total, cluster_count, largest, accepted = 0, 0, 0, 0
    steps, spent = -burn_in * thin, 0
    while steps < 0 or spent < budget:
        before[:] = state
        size, visits = 0, 0
        for each, sweep in parts:
            built = sweep(state, rng, 1.0)
            size += built
            visits += lattice.visits(each, built)
        if visits < 1:
            msg = "a step charged nothing, so a budget in its unit is never spent"
            raise ValueError(msg)
        accepted += not np.array_equal(before, state)
        if size:
            cluster_total += size
            cluster_count += 1
            largest = max(largest, size)
        if steps >= 0:
            spent += visits
            if (steps + 1) % thin == 0:
                kept.append(state.copy())
        steps += 1
    ran = steps + burn_in * thin
    recorded = np.stack(kept) if kept else np.empty((0, graph.n_nodes), dtype=np.int64)
    mean_cluster = (
        cluster_total / cluster_count if cluster_count else float(graph.n_nodes)
    )
    ess = effective_draws(observables(graph, rows, recorded))
    return PottsChain(
        states=recorded,
        mean_cluster_size=mean_cluster,
        acceptance=accepted / ran if ran else 0.0,
        largest_cluster_share=largest / graph.n_nodes if cluster_count else 1.0,
        ess=ess,
        termination=_mixing(ess, ran),
    )


def site_visits(budget: Budget | None) -> None:
    """Refuse a ``budget`` in any unit but :attr:`~sal.cost.Cost.SITE_VISITS`, the unit a Potts step charges (issue #1344).

    Raises
    ------
    ValueError
        If ``budget`` is given in another unit.
    """
    if budget is not None and budget.unit is not Cost.SITE_VISITS:
        msg = f"a Potts run spends {Cost.SITE_VISITS}, got a budget in {budget.unit}"
        raise ValueError(msg)


def step_visits(move: PottsMove, graph: PottsGraph, cluster_sites: int = 0) -> int:
    """Site visits one step of ``move`` costs, the unit :func:`anneal_potts` charges.

    A heat-bath sweep reads every site's label once as a neighbour of each
    incident edge and writes it once; the bond pass of Swendsen-Wang reads
    the same two labels per edge. Counting both in one unit is what makes
    the budget comparable across move sets (issue #551). The gradient-informed
    sets pay ``n_nodes`` sweeps' reads a step, and a single-cluster step reads each of
    its ``cluster_sites`` members' neighbours and writes the members.

    Parameters
    ----------
    move : PottsMove
        The move set.
    graph : PottsGraph
        The instance.
    cluster_sites : int
        Sites the step's clusters held, read only for the single-cluster
        moves (Wolff, heat-bath Wolff).

    Returns
    -------
    int
    """
    per_sweep = graph.n_nodes + 2 * len(graph.edges)
    if move in _BALANCED_MOVES:
        return graph.n_nodes * per_sweep
    if move in _SINGLE_CLUSTER_MOVES:
        return cluster_sites * (1 + 2 * len(graph.edges) // graph.n_nodes)
    return per_sweep


@dataclass(frozen=True, kw_only=True)
class AnnealedPotts(Annealed[np.ndarray]):
    """What one annealing run found, and what it cost (issue #1090).

    An :class:`~sal.sample.schedule.Annealed` over labellings: ``best`` is the
    lowest-energy configuration visited, shape ``(n_nodes,)``, ``final``
    where the chain ended, and ``spent`` the site visits, the unit a budget
    is matched on rather than the sweep count: a Wolff sweep flips one
    cluster while a heat-bath sweep touches every site, so equal sweeps
    hand the cluster moves a free lattice per move (issue #551).

    Parameters
    ----------
    energy : float
        ``best``'s energy, in :func:`energies`' convention.
    n_sweeps : int
        Sweeps run: one per schedule step, or under a ``budget`` the steps
        that spent it (issue #1344).
    trace : tuple[ClusterCounter, ...]
        One counter per schedule step for a cluster move set, empty for
        single-site. Kept per step because the quantity issue #551 predicts
        is a function of temperature and the schedule is what varies it.
    tuned : TunedSchedule | None
        The pilots that chose the schedule under ``schedule="auto"`` (issue
        #1317), ``None`` for a given schedule; their spend is not in ``spent``.
    """

    energy: float
    n_sweeps: int
    trace: tuple[ClusterCounter, ...] = ()
    tuned: TunedSchedule | None = None


def anneal_potts(
    graph: PottsGraph,
    field: SiteField | np.ndarray,
    schedule: Schedule,
    rng: np.random.Generator,
    *,
    move: PottsMoves = PottsMove.SINGLE_SITE,
    recolour: Recolour = Recolour.PER_MOVE,
    backend: Backend = Backend.RUST,
    cluster_backend: Backend = Backend.RUST,
    start: np.ndarray | None = None,
    tuning: ScheduleTuning | None = None,
    budget: Budget | None = None,
) -> AnnealedPotts:
    """Simulated annealing by heat-bath sweeps on a temperature schedule.

    One :func:`single_site_sweep` per schedule step at that step's
    temperature, tracking the lowest energy seen (Kirkpatrick, Gelatt & Vecchi,
    1983). It is :func:`~sal.search.icm.iterated_conditional_modes`
    at finite temperature: at ``T -> 0`` the heat bath is the argmin over each
    site's conditional, ICM's update, so the two are one search separated by the schedule and a
    difference between them is a statement about the schedule.

    ``move`` names the move set. Single-site is the default and the fair
    annealed baseline. The two gradient-informed move sets anneal a coupling
    of either sign, at ``n_nodes`` times the site visits per sweep: each of
    their ``n_nodes`` proposals reads every site's conditional, where a
    heat-bath sweep reads each site's once. The two cluster move sets refuse a
    negative coupling,
    so they anneal only a ferromagnet --- which `spatio_only` is, and which
    issue #551 anneals them on. They are **not** a faster route to the same
    answer there: the Fortuin-Kasteleyn bond construction is exact at zero
    field, so in a field every recolouring carries the accept step
    :func:`_recolour` applies, and its acceptance falls as the cluster grows.
    That compounding is the measurement, not an implementation detail, so the
    run returns the counters that show it.
    The heat-bath cluster move sets (issue #1142) draw each cluster's label
    from its field weight instead, with no accept step; heat-bath
    Swendsen-Wang keeps no counter.

    Parameters
    ----------
    graph : PottsGraph
        The instance. Couplings of either sign.
    field : SiteField | np.ndarray
        External field, shape ``(n_states,)``.
    schedule : TempSchedule | Literal["auto"]
        Temperature per sweep. Its length is the budget. ``"auto"`` chooses
        it by :func:`~sal.sample.tune.tune_schedule`'s pilots, which
        ``tuning`` describes and which draw from generators spawned from
        ``rng`` (issue #1317); their site visits are ``tuned.spent``, not
        the run's.
    rng : np.random.Generator
        Source of every draw, the start included. Passed in rather than
        seeded here, for the reason :func:`sample_potts` gives.

    backend : Backend
        :data:`~sal.backend.Backend.RUST` runs the
        extension's sweep and is the default;
        :data:`~sal.backend.Backend.PYTHON` runs the
        oracle that pins it. The two produce the same chain state for state,
        on the same uniforms in the same order (:func:`sweep_at`).
    cluster_backend : Backend
        Which implementation runs the Swendsen-Wang pass.
        :data:`~sal.backend.Backend.RUST`, the default since #1283, is a chain
        of the same law as the oracle on another order of draws
        (:func:`_cluster_pass_rust`) and keeps no counter, so its steps leave
        ``trace`` empty (issue #923); :data:`~sal.backend.Backend.PYTHON`
        records every cluster's accept step in ``trace``.
        For both Wolff moves it runs the step (:func:`_wolff_rust`, #1362),
        a chain of the same law on a ChaCha8 stream seeded from ``rng``,
        whose clusters ``trace`` still records.
        For the heat-bath Swendsen-Wang pass it merges the bonds
        (:func:`bond_roots`), on the same roots either way, and keeps no
        counter (issue #1142).
    start : np.ndarray | None
        The labelling the chain starts from, copied, shape ``(n_nodes,)``,
        checked by :func:`~sal.sim.potts.check_labelling`;
        ``None`` draws it uniformly from ``rng``, as before the parameter
        existed. A given start draws nothing, so the chain's first draw is
        the generator's next (issue #1038).
    tuning : ScheduleTuning | None
        The pilots that choose ``schedule="auto"``, required with it and
        refused without it (issue #1317).
    budget : Budget | None
        Site visits to spend (issue #1344). ``None``, the default, runs one
        step per schedule entry, bitwise the run before the parameter
        existed. Given, the run takes steps until ``spent`` reaches
        ``budget.size`` and reads the schedule at the spent fraction
        (:func:`~sal.sample.loop.anneal_spent`), so a Wolff run, charged by
        its clusters, spends the budget and ends its ramp at it; ``n_sweeps``
        is then the steps run. A move set of fixed cost per step ``c`` given
        ``budget.size = schedule.n_steps * c`` is the default run, bitwise.

    Returns
    -------
    AnnealedPotts

    Raises
    ------
    ValueError
        If ``start`` is not one integer state in range per node, as
        :func:`~sal.sample.tune.resolve_schedule` refuses, or if ``budget``
        is not in :attr:`~sal.cost.Cost.SITE_VISITS`.
    """
    field = log_weight_of(field)
    site_visits(budget)
    given = move
    move = move_set(move, recolour)
    refuse_negative_coupling(move, graph)
    schedule, tuned = resolve_schedule(
        schedule,
        tuning,
        graph=graph,
        field=field,
        move=given,
        recolour=recolour,
        rng=rng,
    )

    rows = site_field(np.asarray(field, dtype=float), graph.n_nodes)
    lattice = _Lattice(
        graph, rows, graph.compressed_adjacency(), backend, cluster_backend
    )
    return _anneal_lattice(lattice, move, schedule, tuned, rng, start, budget)


def _anneal_lattice(
    lattice: _Lattice,
    move: Sequence[PottsMove],
    schedule: TempSchedule,
    tuned: TunedSchedule | None,
    rng: np.random.Generator,
    start: np.ndarray | None,
    budget: Budget | None,
) -> AnnealedPotts:
    """:func:`anneal_potts` on a built lattice, shared with :mod:`sal.sandbox.potts_moves` (issue #1365)."""
    graph, rows = lattice.graph, lattice.rows
    drawn = (
        rng.integers(0, int(rows.shape[1]), size=graph.n_nodes)
        if start is None
        else check_labelling(start, graph.n_nodes, int(rows.shape[1]))
    )
    state = np.ascontiguousarray(drawn, dtype=np.int64)
    trace: list[ClusterCounter] = []
    origin = Moved(state, lattice.energy(state), None, 0)
    step = lattice.rung(move, trace)
    walked = (
        anneal(step, schedule, origin, rng, np.copy)
        if budget is None
        else anneal_spent(step, schedule, origin, rng, np.copy, budget=budget.size)
    )
    return AnnealedPotts(
        best=walked.best,
        energy=walked.energy,
        final=walked.final,
        n_sweeps=walked.termination.iterations,
        spent=walked.spent,
        unit=Cost.SITE_VISITS,
        termination=walked.termination,
        trace=tuple(trace),
        tuned=tuned,
    )


def _rung_starts(
    start: np.ndarray | None,
    temperatures: Sequence[float],
    ladder_tuning: LadderTuning | None,
    children: Sequence[np.random.Generator],
    n_nodes: int,
    n_states: int,
) -> np.ndarray:
    """One labelling per rung, as both Potts temperings read ``start`` (issue #1343).

    ``None`` draws each rung's from its replica's child generator, as before
    the parameter existed. ``(n_nodes,)`` is every rung's. ``(n_rungs,
    n_nodes)`` is one per rung of the ladder given, coldest first; under
    ``temperatures="auto"`` that ladder is ``ladder_tuning.start``, and each
    tuned rung takes the row of the starting rung nearest it in temperature,
    a tie going to the colder. The tuned ladder keeps the starting
    endpoints, so the two ends take their own rows.

    Returns
    -------
    np.ndarray
        One contiguous ``int64`` row per replica: the kernel borrows a row of
        this block rather than copying it.

    Raises
    ------
    ValueError
        If ``start`` is neither shape, or a row is not a labelling.
    """
    if start is None:
        drawn = [child.integers(0, n_states, size=n_nodes) for child in children]
        return np.ascontiguousarray(np.stack(drawn), dtype=np.int64)
    given = temperatures if ladder_tuning is None else tuple(ladder_tuning.start)
    shape = np.shape(start)
    if len(shape) == 1:
        row = check_labelling(start, n_nodes, n_states)
        return np.ascontiguousarray(
            np.repeat(np.asarray(row)[None], len(temperatures), axis=0),
            dtype=np.int64,
        )
    if len(shape) != 2 or shape[0] != len(given):
        msg = (
            f"start is one labelling, ({n_nodes},), or one per rung of the "
            f"ladder given, ({len(given)}, {n_nodes}); got shape {shape}"
        )
        raise ValueError(msg)
    rows = [check_labelling(row, n_nodes, n_states) for row in start]
    if ladder_tuning is not None:
        anchors = np.asarray(given, dtype=float)
        rows = [
            rows[int(np.argmin(np.abs(anchors - temperature)))]
            for temperature in temperatures
        ]
    return np.ascontiguousarray(np.stack(rows), dtype=np.int64)


@dataclass(frozen=True)
class _Lattice:
    """A Potts instance as its moves and loops read it: the field rows and the two backends."""

    graph: PottsGraph
    rows: np.ndarray
    adjacency: Iterable[np.ndarray]
    backend: Backend
    cluster_backend: Backend

    def energy(self, state: np.ndarray) -> float:
        """``state``'s energy, in :func:`energies`' convention."""
        return float(energies(self.graph, self.rows, state[None])[0])

    def visits(self, move: PottsMove, cluster_sites: int) -> int:
        """:func:`step_visits` on this lattice, where a sandbox lattice charges its own moves (issue #1365)."""
        return step_visits(move, self.graph, cluster_sites)

    def sweep(
        self, move: PottsMove, trace: list[ClusterCounter] | None = None
    ) -> Callable[[np.ndarray, np.random.Generator, float], int]:
        """:func:`sweep_for`, appending a cluster move's :class:`ClusterCounter` to ``trace`` per call.

        Only the moves whose pass reads a cluster's members keep a counter: the
        compiled Swendsen-Wang pass and the heat-bath Swendsen-Wang pass build
        their clusters as roots (issues #923, #1142).
        """
        graph, rows, backend, cluster_backend = (
            self.graph,
            self.rows,
            self.backend,
            self.cluster_backend,
        )
        offsets, neighbours, couplings = self.adjacency
        if move is PottsMove.SINGLE_SITE or move in _BALANCED_MOVES:
            single = (
                sweep_at(rows, offsets, neighbours, couplings, backend)
                if move is PottsMove.SINGLE_SITE
                else balanced_sweep_at(rows, offsets, neighbours, couplings, move)
            )

            def sweep(
                state: np.ndarray, rng: np.random.Generator, beta: float = 1.0
            ) -> int:
                single(state, rng, beta)
                return 0

            return sweep

        # The adjacency as lists, once for every cluster the closure grows (#919).
        one = move in _SINGLE_CLUSTER_MOVES
        # The two Wolff moves take `cluster_backend` (#1362); the Rust route
        # reads the arrays and not the lists.
        wolff = move in (PottsMove.WOLFF, PottsMove.WOLFF_HEAT_BATH)
        compiled = wolff and cluster_backend is Backend.RUST
        lists = (
            adjacency_lists(offsets, neighbours, couplings)
            if one and not compiled
            else None
        )
        # A single cluster's growth, one signature for the two.
        grow: Callable[..., int] = (
            functools.partial(wolff_heat_bath_sweep, backend=cluster_backend)
            if move is PottsMove.WOLFF_HEAT_BATH
            else functools.partial(wolff_sweep, backend=cluster_backend)
        )
        arrays = (rows, offsets, neighbours, couplings)
        keeps = trace is not None and (
            one
            or (move is PottsMove.SWENDSEN_WANG and cluster_backend is Backend.PYTHON)
        )

        def cluster_pass(
            state: np.ndarray, rng: np.random.Generator, beta: float = 1.0
        ) -> int:
            counter = ClusterCounter() if keeps else None
            size = 0
            if move is PottsMove.SWENDSEN_WANG:
                swendsen_wang_sweep(
                    state, graph, rows, rng, counter, beta, backend=cluster_backend
                )
            elif move is PottsMove.SWENDSEN_WANG_HEAT_BATH:
                swendsen_wang_heat_bath_sweep(
                    state, graph, rows, rng, beta, backend=cluster_backend
                )
            else:
                size = grow(state, *arrays, rng, counter, graph, beta=beta, lists=lists)
            if counter is not None and trace is not None:
                trace.append(counter)
            return size

        return cluster_pass

    def rung(
        self,
        moves: Sequence[PottsMove],
        trace: list[ClusterCounter] | None = None,
        *,
        scored: bool = True,
    ) -> Step[np.ndarray, None, np.random.Generator]:
        """``moves`` in order as one step, charged in site visits as :func:`step_visits` states.

        One :func:`sweep_for` closure per move, built here and drawing
        nothing. Unscored, the step returns ``nan`` for a loop that scores
        every rung in one block.
        """
        sweeps = [(each, self.sweep(each, trace)) for each in moves]

        def step(
            state: np.ndarray, _: float, __: None, temperature: float, rng: Generator
        ) -> Moved[np.ndarray, None]:
            beta = 1.0 / temperature
            visits = sum(
                self.visits(each, sweep(state, rng, beta)) for each, sweep in sweeps
            )
            energy = self.energy(state) if scored else math.nan
            return Moved(state, energy, None, visits)

        return step

    def temper(
        self,
        steps: Sequence[Step[np.ndarray, None, np.random.Generator]],
        temperatures: Sequence[float],
        states: np.ndarray,
        children: Sequence[np.random.Generator],
        rng: np.random.Generator,
        budget: tuple[int, int, int],
        *,
        kept: int,
        observe: Callable[[int, Exchanging[np.ndarray]], None] | None = None,
        between: Callable[[Exchanging[np.ndarray]], None] | None = None,
    ) -> tuple[Exchanging[np.ndarray], np.ndarray]:
        """:func:`~sal.sample.loop.temper` on labellings, and the first ``kept`` recorded rounds.

        ``budget`` is ``(n_sweeps, burn_in, thin)``, the first two counting
        blocks of ``thin`` steps, each recorded at its end. Every rung is
        scored in one block per round, and ``rng`` draws the exchange
        uniforms after any draws ``between`` makes.
        """
        n_sweeps, burn_in, thin = budget
        recorded = np.empty((kept, len(steps), self.graph.n_nodes), dtype=np.int64)
        written = [0]

        def record(rungs: Sequence[np.ndarray], _: Sequence[float]) -> None:
            if written[0] < kept:
                recorded[written[0]] = rungs
            written[0] += 1

        burned = burn_in * thin + thin - 1
        current = energies(self.graph, self.rows, states).tolist()
        run = temper(
            steps,
            temperatures,
            [Moved(row, e, None, 0) for row, e in zip(states, current, strict=True)],
            children,
            lambda log_ratio: accept(log_ratio, rng),
            (burn_in + n_sweeps) * thin - burned,
            burned,
            thin,
            keep=np.copy,
            record=record,
            observe=observe,
            between=between,
            # Read from this module, where a test replaces it.
            ratio=swap_log_ratio,
            score=lambda rungs: energies(
                self.graph, self.rows, np.stack(rungs)
            ).tolist(),
        )
        return run, recorded


@dataclass(frozen=True, kw_only=True)
class ClusterTempered(Tempered[np.ndarray]):
    """What :func:`cluster_tempering` produced (issue #1090).

    A :class:`~sal.sample.schedule.Tempered` over labellings: ``best`` is the
    lowest-energy configuration seen at any temperature, and ``spent`` the
    site visits: every replica's passes plus every Houdayer move, each
    charged one sweep's ``n_nodes + 2 n_edges``, since the move reads every
    site and the defect sites' edges.

    Parameters
    ----------
    states : np.ndarray
        Recorded configurations, ``(n_recorded, n_replicas, n_nodes)``;
        empty along the first axis unless the run was asked to record.
    houdayer_acceptance : np.ndarray
        Houdayer moves accepted over proposed, per pair that runs one.
    houdayer_sizes : tuple[int, ...]
        Every proposed Houdayer cluster's size, in order; a pair that agrees
        everywhere proposes none.
    houdayer_accepts : int
        Of those, the moves accepted.
    energy : float
        ``best``'s energy.
    n_sweeps : int
        Steps run, each one Swendsen-Wang pass per replica.
    tuned_ladder : TunedLadder | None
        The pilot that chose the ladder under ``temperatures="auto"`` (issue
        #1337), ``None`` for a given ladder; its spend is not in ``spent``.
    """

    states: np.ndarray
    houdayer_acceptance: np.ndarray
    houdayer_sizes: tuple[int, ...]
    houdayer_accepts: int
    energy: float
    n_sweeps: int
    tuned_ladder: TunedLadder | None = None


def cluster_tempering(
    graph: PottsGraph,
    field: SiteField | np.ndarray,
    temperatures: Ladder,
    rng: np.random.Generator,
    n_sweeps: int,
    *,
    houdayer_pairs: int = 1,
    burn_in: int = 0,
    thin: int = 1,
    record: bool = False,
    cluster_backend: Backend = Backend.RUST,
    start: np.ndarray | None = None,
    ladder_tuning: LadderTuning | None = None,
) -> ClusterTempered:
    """Parallel tempering on Swendsen-Wang passes, with Houdayer moves at the cold end (issue #1041).

    Per step: one :func:`swendsen_wang_sweep` per replica at its own
    temperature, then one :func:`houdayer_move` on each of the
    ``houdayer_pairs`` coldest adjacent pairs, then an exchange proposal on
    every adjacent pair, accepted on :func:`swap_log_ratio`.

    **Houdayer across two temperatures needs an accept step.** The move
    exchanges the two replicas' labels on one component of the sites where
    they disagree; ``E(s) + E(s')`` is invariant, for Potts labels as for
    Ising spins, because a boundary bond's outside end is a site where the
    two agree. The proposal is symmetric --- the disagreement set is what the
    exchange leaves alone. At one temperature that makes the acceptance one;
    at ``beta`` and ``beta'`` the product law changes by
    ``exp(-(beta - beta') (E(s_new) - E(s)))``, and that is the Metropolis
    ratio applied here.

    Parameters
    ----------
    graph : PottsGraph
        Every coupling non-negative.
    field : SiteField | np.ndarray
        ``(n_states,)`` or ``(n_nodes, n_states)``.
    temperatures : TempSchedule | Sequence[float] | Literal["auto"]
        The ladder, coldest first and strictly increasing, the one order of
        both Potts temperings (issue #1343); at least two, all positive. ``"auto"``
        chooses it as :func:`~sal.sandbox.potts_tempering.parallel_tempering` does, on pilot runs of this
        function (issue #1337).
    rng : np.random.Generator
        Spawns one child per replica, then draws the Houdayer seed sites and
        every accept uniform; under ``"auto"`` it first spawns the pilots'.
    n_sweeps, burn_in, thin : int
        As :func:`~sal.sandbox.potts_tempering.parallel_tempering`.
    houdayer_pairs : int
        How many of the coldest adjacent pairs run a Houdayer move a step.
    record : bool
        Whether to keep the thinned configurations, which the enumeration
        test reads and a ground-state search does not.
    cluster_backend : Backend
        Runs the Swendsen-Wang pass, as :func:`anneal_potts` states.
    start : np.ndarray | None
        As :func:`~sal.sandbox.potts_tempering.parallel_tempering`'s: ``(n_nodes,)`` for every rung or
        ``(n_rungs, n_nodes)`` per rung, mapped onto a tuned ladder by
        nearest temperature (issue #1343); ``None`` draws as before.
    ladder_tuning : LadderTuning | None
        As :func:`~sal.sandbox.potts_tempering.parallel_tempering`'s.

    Returns
    -------
    ClusterTempered

    Raises
    ------
    ValueError
        If the ladder is not strictly increasing, coldest first, or
        ``houdayer_pairs`` is outside ``[0, n_replicas - 1]``, or ``start``
        is neither shape.
    """
    refuse_negative_coupling(PottsMove.SWENDSEN_WANG, graph)

    def pilot(candidate: tuple[float, ...]) -> tuple[list[float], int]:
        run = cluster_tempering(
            graph,
            field,
            candidate,
            pilot_rng,
            ladder_tuning.n_sweeps if ladder_tuning else 0,
            houdayer_pairs=min(houdayer_pairs, len(candidate) - 1),
            cluster_backend=cluster_backend,
        )
        return [float(value) for value in run.swap_acceptance], run.spent

    pilot_rng = rng.spawn(1)[0] if isinstance(temperatures, str) else rng
    given, tuned_ladder = resolve_ladder(temperatures, ladder_tuning, pilot)
    temperatures = check_ladder(
        ladder(given),
        needed_by="cluster tempering",
        monotone=Monotone.INCREASING,
    )
    n_replicas = len(temperatures)
    if not 0 <= houdayer_pairs <= n_replicas - 1:
        msg = f"houdayer_pairs is in [0, {n_replicas - 1}], got {houdayer_pairs}"
        raise ValueError(msg)
    rows = site_field(np.asarray(log_weight_of(field), dtype=float), graph.n_nodes)
    betas = [1.0 / temperature for temperature in temperatures]
    children = rng.spawn(n_replicas)
    n_states = int(rows.shape[1])
    states = _rung_starts(
        start, temperatures, ladder_tuning, children, graph.n_nodes, n_states
    )
    adjacency = graph.compressed_adjacency()
    per_sweep = graph.n_nodes + 2 * len(graph.edges)
    houdayer = np.zeros((2, houdayer_pairs))
    sizes: list[int] = []

    def houdayer_moves(run: Exchanging[np.ndarray]) -> None:
        """Houdayer's move on each of the coldest pairs, accepted across their two temperatures."""
        labels, current = run.states, run.energies
        for pair in range(houdayer_pairs):
            first, second = labels[pair].copy(), labels[pair + 1].copy()
            size = houdayer_move(
                first, second, adjacency.offsets, adjacency.neighbours, rng
            )
            run.spent += per_sweep
            if size == 0:
                continue
            sizes.append(size)
            houdayer[0, pair] += 1
            moved = energies(graph, rows, np.stack([first, second]))
            log_ratio = -(betas[pair] - betas[pair + 1]) * (moved[0] - current[pair])
            if accept(float(log_ratio), rng):
                houdayer[1, pair] += 1
                labels[pair], labels[pair + 1] = first, second
                current[pair], current[pair + 1] = float(moved[0]), float(moved[1])

    lattice = _Lattice(graph, rows, adjacency, Backend.RUST, cluster_backend)
    step = lattice.rung((PottsMove.SWENDSEN_WANG,), scored=False)
    run, recorded = lattice.temper(
        [step] * n_replicas,
        temperatures,
        states,
        children,
        rng,
        (n_sweeps, burn_in, thin),
        kept=n_sweeps if record else 0,
        between=houdayer_moves,
    )
    return ClusterTempered(
        states=recorded,
        temperatures=tuple(temperatures),
        swap_acceptance=run.accepted / np.maximum(run.proposed, 1),
        houdayer_acceptance=houdayer[1] / np.maximum(houdayer[0], 1),
        houdayer_sizes=tuple(sizes),
        houdayer_accepts=int(houdayer[1].sum()),
        best=run.best,
        energy=run.energy,
        n_sweeps=n_sweeps,
        spent=run.spent,
        unit=Cost.SITE_VISITS,
        termination=Termination.after((burn_in + n_sweeps) * thin, converged=False),
        tuned_ladder=tuned_ladder,
    )


def sweep_for(
    move: PottsMoves,
    graph: PottsGraph,
    rows: np.ndarray,
    offsets: np.ndarray,
    neighbours: np.ndarray,
    couplings: np.ndarray,
    backend: Backend,
    cluster_backend: Backend = Backend.RUST,
    *,
    recolour: Recolour = Recolour.PER_MOVE,
) -> Callable[[np.ndarray, np.random.Generator, float], int]:
    """One sweep of ``move``, as a call taking a state, a generator and ``beta``.

    The one place a move set is turned into a sweep, so
    :func:`sample_potts` and :func:`sample_potts_pair` run one dispatch rather
    than two that can drift. Each branch calls the sweep it already called,
    with the same arguments in the same order, so every existing chain is
    bitwise what it was. ``backend`` runs the heat-bath sweep and
    ``cluster_backend`` the Swendsen-Wang pass: two arguments because they are
    two decisions, one keeping the oracle's stream and one not
    (:func:`sample_potts`, issues #754, #1283).

    Returns
    -------
    Callable[[np.ndarray, np.random.Generator, float], int]
        The sweep, returning the size of the cluster it built and ``0`` where
        the move set builds none --- which is what
        :attr:`PottsChain.mean_cluster_size` averages.
    """
    adjacency = (offsets, neighbours, couplings)
    lattice = _Lattice(graph, rows, adjacency, backend, cluster_backend)
    return _advance(lattice, move_set(move, recolour))


@dataclass(frozen=True)
class PottsPair:
    """The two replicas :func:`sample_potts_pair` ran, recorded alike.

    Parameters
    ----------
    first, second : PottsChain
        One replica each, on the same recording schedule and at the same
        temperature. Neither is the other's reference: Houdayer's move is
        symmetric in the pair, and the order is the order the generator
        spawned the children in.
    """

    first: PottsChain
    second: PottsChain

    def __iter__(self) -> Iterator[Any]:
        """``(first, second)``: the order callers unpack.

        ``Any`` for :meth:`TemperedModel.__iter__`'s reason.
        """
        yield from (self.first, self.second)


def sample_potts_pair(
    graph: PottsGraph,
    field: SiteField | np.ndarray,
    move: PottsMoves,
    rng: np.random.Generator,
    n_sweeps: int,
    burn_in: int = 0,
    thin: int = 1,
    *,
    temperature: float = 1.0,
    houdayer: bool = True,
    recolour: Recolour = Recolour.PER_MOVE,
    backend: Backend = Backend.RUST,
    cluster_backend: Backend = Backend.RUST,
) -> PottsPair:
    """Two replicas at one temperature, joined by Houdayer's isoenergetic move.

    Each recorded step is one sweep of ``move`` on each replica, then --- where
    ``houdayer`` --- one :func:`houdayer_move` on the pair. The two replicas
    are the *pair* Houdayer (2001) defines the move on, so this is where the
    move lives rather than in :func:`sample_potts`, which has one chain and
    nothing to exchange with.

    **Each replica's marginal is the same Boltzmann law one chain targets.**
    The joint target is the product of the two, and the move is an involution
    on it with a symmetric proposal, so it leaves the product invariant; the
    marginal follows. `tests/regression/search/test_potts_mcmc.py` asserts it
    against the enumerated law rather than against this paragraph.

    **Houdayer's move alone is not a sampler**, and nothing here pretends
    otherwise: it exchanges labels between replicas and so leaves the pair
    ``{s_i, s'_i}`` at every site exactly as it found it. Run without a move
    that changes those pairs it explores an orbit of the initial draw, which is
    a property this file pins rather than a limitation it works around. So
    ``move`` is the replicas' own sweep and Houdayer's move sits beside it.

    Parameters
    ----------
    graph, field, move, rng, n_sweeps, burn_in, thin, temperature, backend, cluster_backend
        As :func:`sample_potts`, applied to each replica. ``rng`` spawns one
        child per replica and keeps the pair's own draws, so the two replicas
        do not share a stream --- :func:`~sal.sandbox.potts_tempering.parallel_tempering`'s rule, for its
        reason.
    houdayer : bool
        Whether the pair takes the isoenergetic move after each pair of
        sweeps. ``False`` is the control: two independent chains, which is
        what a round-trip time or an autocorrelation is read against.

    Returns
    -------
    PottsPair
        One per replica, recorded on the same schedule. ``mean_cluster_size``
        is the replica's own move's, so Houdayer's clusters are not counted
        into a number that means the within-replica move's cost.

    Raises
    ------
    ValueError
        If the field carries other than two states while ``houdayer`` ---
        Houdayer's overlap ``q_i = s_i s'_i`` is the Ising one and issue #756
        validates the move at ``q = 2`` alone --- or if ``move`` is a
        Fortuin-Kasteleyn cluster move on a graph with a negative coupling, as
        :func:`sample_potts` refuses it.
    """
    field = log_weight_of(field)
    move = move_set(move, recolour)
    refuse_negative_coupling(move, graph)

    model = tempered(graph, field, temperature)
    graph, field = model.graph, model.field
    rows = site_field(field, graph.n_nodes)
    n_states = int(rows.shape[1])
    if houdayer and n_states != 2:
        msg = (
            f"Houdayer's move is defined on the Ising overlap q_i = s_i s'_i "
            f"and this model has {n_states} states: pass houdayer=False, or "
            "use two states (issue #756)"
        )
        raise ValueError(msg)
    lattice = _Lattice(
        graph, rows, graph.compressed_adjacency(), backend, cluster_backend
    )
    return _record_pair(lattice, move, rng, n_sweeps, burn_in, thin, houdayer=houdayer)


def _record_pair(
    lattice: _Lattice,
    move: Sequence[PottsMove],
    rng: np.random.Generator,
    n_sweeps: int,
    burn_in: int,
    thin: int,
    *,
    houdayer: bool,
) -> PottsPair:
    """:func:`sample_potts_pair`' loop on a built lattice, shared with :mod:`sal.sandbox.potts_moves` (issue #1365)."""
    graph, rows = lattice.graph, lattice.rows
    n_states = int(rows.shape[1])
    children = rng.spawn(2)
    states = [
        np.ascontiguousarray(
            child.integers(0, n_states, size=graph.n_nodes), dtype=np.int64
        )
        for child in children
    ]
    offsets, neighbours, _ = lattice.adjacency
    advance = _advance(lattice, move)

    recorded = [np.empty((n_sweeps, graph.n_nodes), dtype=np.int64) for _ in range(2)]
    totals, counts, largest, accepted = [0, 0], [0, 0], [0, 0], [0, 0]
    before = np.empty_like(states[0])
    for step in range(-burn_in * thin, n_sweeps * thin):
        for replica in range(2):
            before[:] = states[replica]
            size = advance(states[replica], children[replica], 1.0)
            accepted[replica] += not np.array_equal(before, states[replica])
            if size:
                totals[replica] += size
                counts[replica] += 1
                largest[replica] = max(largest[replica], size)
        if houdayer:
            houdayer_move(states[0], states[1], offsets, neighbours, rng)
        if step >= 0 and (step + 1) % thin == 0:
            for replica in range(2):
                recorded[replica][step // thin] = states[replica]
    steps = (burn_in + n_sweeps) * thin
    ess = [effective_draws(observables(graph, rows, recorded[r])) for r in range(2)]
    first, second = (
        PottsChain(
            states=recorded[replica],
            mean_cluster_size=(
                totals[replica] / counts[replica]
                if counts[replica]
                else float(graph.n_nodes)
            ),
            acceptance=accepted[replica] / steps if steps else 0.0,
            largest_cluster_share=(
                largest[replica] / graph.n_nodes if counts[replica] else 1.0
            ),
            ess=ess[replica],
            termination=_mixing(ess[replica], steps),
        )
        for replica in range(2)
    )
    return PottsPair(first=first, second=second)


#: The starts :func:`sample_potts_starts` runs one chain from, in order: every
#: site on label 0, a uniform draw, and a uniform draw after single-site sweeps.
STARTS = ("ordered", "drawn", "equilibrated")


@dataclass(frozen=True)
class PottsStarts:
    """Chains of one move from :data:`STARTS`, and whether they agree (issue #1316).

    Parameters
    ----------
    chains : tuple[PottsChain, ...]
        One per entry of :data:`STARTS`, in that order.
    rhat : np.ndarray
        :func:`~sal.sample.statistics.split_rhat` over the chains, per
        observable in :func:`observables`' order: energy, then each label's
        occupancy.
    termination : Termination
        :attr:`~sal.opt.termination.Stop.NOT_MIXING` where any entry of
        ``rhat`` exceeds :data:`RHAT_THRESHOLD` or any chain ended
        ``NOT_MIXING``, else :attr:`~sal.opt.termination.Stop.CONVERGED`;
        ``iterations`` sums the chains' steps.
    """

    chains: tuple[PottsChain, ...]
    rhat: np.ndarray
    termination: Termination


def _chain_from(
    origin: str,
    generator: np.random.Generator,
    *,
    graph: PottsGraph,
    field: np.ndarray,
    move: PottsMoves,
    n_sweeps: int,
    burn_in: int,
    thin: int,
    temperature: float,
    equilibration_sweeps: int,
    backend: Backend,
    cluster_backend: Backend,
) -> PottsChain:
    """One :func:`sample_potts_starts` body: its start, then its chain, on its own generator.

    Thread-safe: it draws from ``generator`` alone and writes only arrays it
    allocates, so a thread pool returns the serial run bitwise.
    """
    start: np.ndarray | None
    if origin == "ordered":
        start = np.zeros(graph.n_nodes, dtype=np.int64)
    elif origin == "drawn":
        start = None
    else:
        start = sample_potts(
            graph,
            field,
            PottsMove.SINGLE_SITE,
            generator,
            n_sweeps=1,
            burn_in=equilibration_sweeps - 1,
            temperature=temperature,
            backend=backend,
        ).states[-1]
    return sample_potts(
        graph,
        field,
        move,
        generator,
        n_sweeps,
        burn_in,
        thin,
        temperature=temperature,
        backend=backend,
        cluster_backend=cluster_backend,
        start=start,
        # ``move`` is resolved by the caller; under ``UNIFORM`` a resolved
        # set is returned unchanged (issue #1323).
        recolour=Recolour.UNIFORM,
    )


def sample_potts_starts(
    graph: PottsGraph,
    field: SiteField | np.ndarray,
    move: PottsMoves,
    rng: np.random.Generator,
    n_sweeps: int,
    burn_in: int = 0,
    thin: int = 1,
    *,
    temperature: float = 1.0,
    recolour: Recolour = Recolour.PER_MOVE,
    backend: Backend = Backend.RUST,
    cluster_backend: Backend = Backend.RUST,
    equilibration_sweeps: int = 100,
    workers: int = 1,
    pool: Pool = "serial",
) -> PottsStarts:
    """:func:`sample_potts` from each of :data:`STARTS`, and the split R-hat across them.

    A chain stuck where it started reports a mean, and a single chain cannot
    say the mean is its start's: the ordered start on #1314's instance
    reported occupancy (0, 0, 1) against (0.31, 0.33, 0.36) from an
    equilibrated one. Chains from distinct starts that disagree can.

    Parameters
    ----------
    graph, field, move, n_sweeps, burn_in, thin, temperature, backend, cluster_backend
        As :func:`sample_potts`, for every chain.
    rng : np.random.Generator
        Spawns one child per start through :func:`sal.parallel.map_tasks`;
        each chain draws from its child alone.
    equilibration_sweeps : int
        Single-site sweeps run from a uniform draw to make the
        ``"equilibrated"`` start, at least 1.
    workers, pool
        As :func:`sal.parallel.map_tasks`. Every body is thread-safe, so
        ``pool="threads"`` returns the serial result bitwise.

    Returns
    -------
    PottsStarts
        The chains, ``rhat`` per observable, and the termination.

    Raises
    ------
    ValueError
        If ``n_sweeps`` is below 4, where a split half holds fewer than two
        draws, or ``equilibration_sweeps`` is below 1.
    """
    move = move_set(move, recolour)
    if n_sweeps < 4:
        msg = f"split R-hat needs at least 4 recorded sweeps, got {n_sweeps}"
        raise ValueError(msg)
    if equilibration_sweeps < 1:
        msg = f"equilibration_sweeps must be at least 1, got {equilibration_sweeps}"
        raise ValueError(msg)
    field = log_weight_of(field)
    body = functools.partial(
        _chain_from,
        graph=graph,
        field=field,
        move=move,
        n_sweeps=n_sweeps,
        burn_in=burn_in,
        thin=thin,
        temperature=temperature,
        equilibration_sweeps=equilibration_sweeps,
        backend=backend,
        cluster_backend=cluster_backend,
    )
    chains = tuple(map_tasks(body, STARTS, workers=workers, pool=pool, generator=rng))
    series = np.stack([observables(graph, field, chain.states) for chain in chains])
    rhat = np.array([split_rhat(series[:, k]) for k in range(series.shape[1])])
    mixed = bool(np.all(rhat <= RHAT_THRESHOLD)) and all(
        chain.termination.converged for chain in chains
    )
    return PottsStarts(
        chains=chains,
        rhat=rhat,
        termination=Termination(
            converged=mixed,
            iterations=sum(chain.termination.iterations for chain in chains),
            reason=Stop.CONVERGED if mixed else Stop.NOT_MIXING,
        ),
    )
