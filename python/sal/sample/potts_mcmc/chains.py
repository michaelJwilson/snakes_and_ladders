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
from sal.opt.termination import Termination
from sal.sample.accept import accept
from sal.sample.loop import Exchanging, Moved, Step, anneal, swap_log_ratio, temper
from sal.sample.potts_mcmc import sweeps
from sal.sample.potts_mcmc.moves import (
    PottsMove,
    refuse_negative_coupling,
)
from sal.sample.potts_mcmc.sweeps import (
    ClusterCounter,
    adjacency_lists,
    balanced_sweep_at,
    ghost_spin_sweep,
    houdayer_move,
    label_directed_sweep,
    niedermayer_sweep,
    niedermayer_threshold,
    sweep_at,
    swendsen_wang_heat_bath_sweep,
    swendsen_wang_sweep,
    wolff_heat_bath_sweep,
    wolff_sweep,
)
from sal.sample.schedule import (
    AdaptedLadder,
    Annealed,
    Monotone,
    Tempered,
    TempSchedule,
    adapt_ladder,
    check_ladder,
    ladder,
)
from sal.sim.graph import PottsGraph
from sal.sim.potts import (
    SiteField,
    check_labelling,
    critical_coupling,
    energies,
    log_weight_of,
    site_field,
)

# `current` is aliased: `parallel_tempering` already binds that name to the
# replicas' energies, and one of the two has to give.
from sal.track import TrackedOptimization
from sal.track import current as current_tracked

#: The move sets :func:`balanced_sweep_at` serves.
_BALANCED_MOVES = frozenset(
    {PottsMove.LOCALLY_BALANCED, PottsMove.GIBBS_WITH_GRADIENTS}
)

#: The move sets that grow one cluster a step, charged by its size.
_SINGLE_CLUSTER_MOVES = frozenset(
    {PottsMove.WOLFF, PottsMove.NIEDERMAYER, PottsMove.WOLFF_HEAT_BATH}
)


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
    """

    states: np.ndarray
    mean_cluster_size: float


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
    move: PottsMove,
    rng: np.random.Generator,
    n_sweeps: int,
    burn_in: int = 0,
    thin: int = 1,
    *,
    temperature: float = 1.0,
    backend: Backend = Backend.RUST,
    cluster_backend: Backend = Backend.PYTHON,
) -> PottsChain:
    """Run one chain and return the configuration after every sweep.

    Parameters
    ----------
    graph : PottsGraph
        The lattice. Couplings may vary per edge.
    field : SiteField | np.ndarray
        External field ``h``, shape ``(n_states,)``.
    move : PottsMove
        The move set. Every one leaves the same Boltzmann distribution
        invariant, which is what
        `tests/regression/search/test_potts_mcmc.py` asserts.
    rng : np.random.Generator
        Passed in rather than seeded here: seeding inside a call makes every
        draw of an ensemble identical (`sim/CLAUDE.md`, issue #240).
    n_sweeps : int
        Recorded sweeps. A sweep is ``n_nodes`` heat-bath updates, ``n_nodes``
        gradient-informed proposals, one Swendsen-Wang bond-and-recolour pass
        over the whole lattice, or *one* Wolff or Niedermayer cluster step ---
        see :func:`wolff_sweep` for why a single-cluster sweep cannot be
        sized to match the others.
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
        Which implementation runs the **Swendsen-Wang** pass; the Wolff move
        has one and ignores it. A separate argument rather than the one
        above because the two are not the same decision:
        :data:`~sal.backend.Backend.PYTHON` is the default
        here, where it is not there, because the Rust pass draws the same
        uniforms in a different order and so returns a chain of the same law
        rather than the same chain (:func:`_cluster_pass_rust`, issue #754).

    Returns
    -------
    PottsChain
        The recorded configurations, and the mean cluster size where the move
        set builds clusters.

    Raises
    ------
    ValueError
        If a cluster move is asked for on a graph with a negative coupling. The
        bond probability ``1 - exp(-J)`` is not a probability there, and an
        antiferromagnet has no like-spin clusters to flip.
    """
    field = log_weight_of(field)
    refuse_negative_coupling(move, graph)

    model = tempered(graph, field, temperature)
    graph, field = model.graph, model.field
    rows = site_field(field, graph.n_nodes)
    n_states = int(rows.shape[1])
    # Contiguous `int64` because the kernel borrows this buffer rather than
    # copying it; `integers` already returns one here, so this asserts the
    # layout rather than paying for it.
    state = np.ascontiguousarray(
        rng.integers(0, n_states, size=graph.n_nodes), dtype=np.int64
    )
    offsets, neighbours, couplings = graph.compressed_adjacency()
    advance = sweep_for(
        move, graph, rows, offsets, neighbours, couplings, backend, cluster_backend
    )

    recorded = np.empty((n_sweeps, graph.n_nodes), dtype=np.int64)
    cluster_total, cluster_count = 0, 0
    for step in range(-burn_in * thin, n_sweeps * thin):
        size = advance(state, rng, 1.0)
        if size:
            cluster_total += size
            cluster_count += 1
        if step >= 0 and (step + 1) % thin == 0:
            recorded[step // thin] = state
    mean_cluster = (
        cluster_total / cluster_count if cluster_count else float(graph.n_nodes)
    )
    return PottsChain(states=recorded, mean_cluster_size=mean_cluster)


def step_visits(move: PottsMove, graph: PottsGraph, cluster_sites: int = 0) -> int:
    """Site visits one step of ``move`` costs, the unit :func:`anneal_potts` charges.

    A heat-bath sweep reads every site's label once as a neighbour of each
    incident edge and writes it once; the bond pass of Swendsen-Wang reads
    the same two labels per edge. Counting both in one unit is what makes
    the budget comparable across move sets (issue #551). The gradient-informed
    sets pay ``n_nodes`` sweeps' reads a step, the ghost-spin pass one ghost
    bond per site beside the edges, and a single-cluster step reads each of
    its ``cluster_sites`` members' neighbours and writes the members.

    Parameters
    ----------
    move : PottsMove
        The move set.
    graph : PottsGraph
        The instance.
    cluster_sites : int
        Sites the step's clusters held, read only for the single-cluster
        moves (Wolff, Niedermayer, heat-bath Wolff).

    Returns
    -------
    int
    """
    per_sweep = graph.n_nodes + 2 * len(graph.edges)
    if move in _BALANCED_MOVES:
        return graph.n_nodes * per_sweep
    if move is PottsMove.GHOST_SPIN:
        return per_sweep + graph.n_nodes
    if move in _SINGLE_CLUSTER_MOVES:
        return cluster_sites * (1 + 2 * len(graph.edges) // graph.n_nodes)
    return per_sweep


#: What :func:`parallel_tempering` takes as ``move``: one move set for every
#: rung, or per rung a move set or a sequence of them run in order (#1158).
RungMoves = PottsMove | Sequence[PottsMove | Sequence[PottsMove]]


def moves_per_rung(move: RungMoves, n_rungs: int) -> tuple[tuple[PottsMove, ...], ...]:
    """``move`` as one non-empty tuple of move sets per rung (issue #1158).

    A single :class:`~sal.sample.potts_mcmc.moves.PottsMove` is checked
    first: it is a ``str``, and so a ``Sequence``, which read as one would be
    its characters.

    Raises
    ------
    ValueError
        If ``move`` is a sequence whose length is not ``n_rungs``, or a
        rung's sequence is empty.
    TypeError
        If an entry holds anything but a ``PottsMove``.
    """
    if isinstance(move, PottsMove):
        return ((move,),) * n_rungs
    entries = tuple(move)
    if len(entries) != n_rungs:
        msg = f"move holds one entry per rung, {n_rungs}, got {len(entries)}"
        raise ValueError(msg)
    per_rung = []
    for entry in entries:
        rung = (entry,) if isinstance(entry, PottsMove) else tuple(entry)
        if not rung:
            msg = "a rung's moves are a non-empty sequence"
            raise ValueError(msg)
        for each in rung:
            if not isinstance(each, PottsMove):
                msg = f"a rung's moves are PottsMove, got {each!r}"
                raise TypeError(msg)
        per_rung.append(rung)
    return tuple(per_rung)


#: :func:`rung_moves`' threshold on :func:`critical_ratio`: a rung at or
#: above it runs a single-site sweep after its cluster move. The transition
#: itself; the measurement brackets it between the ladder's rungs at 0.76
#: and 1.58 (``tests/regression/search/test_tempering_rung_moves.py``).
PAIR_FROM = 1.0


def critical_ratio(graph: PottsGraph, n_states: int, temperature: float) -> float:
    """``beta J / K_c``: a rung's coupling against the Potts transition (issue #1158).

    ``J`` is the mean coupling and ``K_c = ln(1 + sqrt(q)) * 4 / z``, with
    ``z = 2 |E| / n`` the mean degree: the square lattice's exact self-dual
    point (:func:`~sal.sim.potts.critical_coupling`) scaled by ``4 / z``, as
    the mean-field transition scales with ``1 / z``. Exact for the square
    lattice; a proxy elsewhere, ``0.951`` against the exact ``0.912`` (the
    root of ``v^3 + 3 v^2 = q``, ``K = ln(1 + v)``) on the triangular lattice
    at ``q = 10``, 4% high.

    Returns
    -------
    float
        ``0`` on a graph with no edges.
    """
    if not graph.edges:
        return 0.0
    degree = 2.0 * len(graph.edges) / graph.n_nodes
    k_c = critical_coupling(n_states) * 4.0 / degree
    return float(np.mean(graph.coupling)) / temperature / k_c


def field_ratio(graph: PottsGraph, rows: np.ndarray) -> float:
    """The median site's field spread against its coupling, ``median_i (max h_i - min h_i) / (J z)`` (issue #1158).

    The spread is over each site's allowed labels, those of finite field, so
    a forbidden label's ``-inf`` does not make it infinite; ``J`` is the mean
    coupling and ``z = 2 |E| / n`` the mean degree, so the ratio compares the
    field one site carries with the coupling it has to its neighbours. Both
    scale with ``beta`` alike, so the ratio is the rung's at every rung.

    Returns
    -------
    float
        ``inf`` on a graph with no edges or no coupling.
    """
    allowed = np.isfinite(rows)
    spread = np.where(allowed, rows, -np.inf).max(axis=1) - np.where(
        allowed, rows, np.inf
    ).min(axis=1)
    bond = (
        float(np.mean(graph.coupling)) * 2.0 * len(graph.edges) / graph.n_nodes
        if graph.edges
        else 0.0
    )
    return float(np.median(spread)) / bond if bond > 0.0 else float("inf")


def rung_moves(
    graph: PottsGraph,
    field: SiteField | np.ndarray,
    temperatures: TempSchedule | Sequence[float],
) -> tuple[tuple[PottsMove, ...], ...]:
    """The moves each rung of a ladder runs, read from its temperature (issue #1158).

    A deterministic rule on :func:`critical_ratio` ``k = beta J / K_c``:

    - ``k < 1`` (:data:`PAIR_FROM`), hot of the transition: heat-bath
      Swendsen-Wang alone. Clusters span correlated regions and relabel
      them in one step.
    - ``k >= 1``, at the transition and colder: heat-bath Swendsen-Wang, then
      a single-site sweep. The bonds close over whole domains, which the
      cluster move relabels and cannot reshape; the sweep moves their
      boundaries.

    **The field enters through the label draw, not the rule.** Every cluster
    move chosen draws its cluster's label ``~ exp(beta sum_C h)``, near
    uniform in a weak field and on the best label in a strong one, so no
    threshold switches between the uniform-proposal and heat-bath variants;
    the uniform ones are reached through an explicit ``move=``.
    :func:`field_ratio` does not change the composition either: measured at
    field x1, x10 and x30 (ratio 0.08 to 5.3), it moved no ladder's energy
    at x10 or x30.

    **Measured** through :func:`~sal.search.ground_state.run_tempering` at
    1,000 sweeps' site visits, five seeds, on ``spatio_only/release`` (71 x 71
    triangular, q = 10, J = 0.7, ``k`` = 0.36, 0.76, 1.58, 3.30, 6.91, 14.5),
    mean energy (standard error): single-site everywhere -9,900.1 (10.9);
    the cluster move alone on the three hottest rungs, single-site below,
    -9,740.4 (13.2); the pair on every rung -10,215.1 (19.1); the pair on
    the two hottest rungs and single-site below -9,820.3 (37.4); this rule
    -10,215.9 (19.0), the same as single-site on the two hottest rungs with
    the pair below. So the rung at 1.58 needs the pair and those at 0.36 and 0.76
    are indifferent to their move; the threshold sits at the transition
    between them. On ``spatio_tiling/release`` this rule -17,018.9 (0.6)
    against single-site's -17,001.2 (5.4).

    Parameters
    ----------
    graph : PottsGraph
        The instance. A negative coupling refuses every cluster move, so
        such a graph is single-site on every rung.
    field : SiteField | np.ndarray
        As :func:`parallel_tempering` takes it.
    temperatures : TempSchedule | Sequence[float]
        The ladder, in :func:`parallel_tempering`'s order.

    Returns
    -------
    tuple[tuple[PottsMove, ...], ...]
        One entry per rung, in the ladder's order, as ``move`` takes it.
    """
    rows = site_field(np.asarray(log_weight_of(field), dtype=float), graph.n_nodes)
    n_states = int(rows.shape[1])
    if min(graph.coupling, default=0.0) < 0.0:
        return tuple((PottsMove.SINGLE_SITE,) for _ in ladder(temperatures))
    return tuple(
        (PottsMove.SWENDSEN_WANG_HEAT_BATH,)
        if critical_ratio(graph, n_states, temperature) < PAIR_FROM
        else (PottsMove.SWENDSEN_WANG_HEAT_BATH, PottsMove.SINGLE_SITE)
        for temperature in ladder(temperatures)
    )


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
        Sweeps run, one per schedule step.
    trace : tuple[ClusterCounter, ...]
        One counter per schedule step for a cluster move set, empty for
        single-site. Kept per step because the quantity issue #551 predicts
        is a function of temperature and the schedule is what varies it.
    """

    energy: float
    n_sweeps: int
    trace: tuple[ClusterCounter, ...] = ()


def anneal_potts(
    graph: PottsGraph,
    field: SiteField | np.ndarray,
    schedule: TempSchedule,
    rng: np.random.Generator,
    *,
    move: PottsMove = PottsMove.SINGLE_SITE,
    backend: Backend = Backend.RUST,
    cluster_backend: Backend = Backend.PYTHON,
    start: np.ndarray | None = None,
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
    Swendsen-Wang keeps no counter, as the ghost-spin pass keeps none.

    Parameters
    ----------
    graph : PottsGraph
        The instance. Couplings of either sign.
    field : SiteField | np.ndarray
        External field, shape ``(n_states,)``.
    schedule : TempSchedule
        Temperature per sweep. Its length is the budget.
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
        :data:`~sal.backend.Backend.PYTHON`, the default,
        records every cluster's accept step in ``trace``;
        :data:`~sal.backend.Backend.RUST` is a chain of the
        same law on another order of draws (:func:`_cluster_pass_rust`) and
        keeps no counter, so its steps leave ``trace`` empty (issue #923).
        For the ghost-spin, label-directed and heat-bath Swendsen-Wang
        passes it merges the bonds (:func:`bond_roots`), on the same roots
        either way, and none keeps a counter (issues #1041, #1142).
    start : np.ndarray | None
        The labelling the chain starts from, copied, shape ``(n_nodes,)``,
        checked by :func:`~sal.sim.potts.check_labelling`;
        ``None`` draws it uniformly from ``rng``, as before the parameter
        existed. A given start draws nothing, so the chain's first draw is
        the generator's next (issue #1038).

    Returns
    -------
    AnnealedPotts

    Raises
    ------
    ValueError
        If ``start`` is not one integer state in range per node.
    """
    field = log_weight_of(field)
    refuse_negative_coupling(move, graph)

    rows = site_field(np.asarray(field, dtype=float), graph.n_nodes)
    drawn = (
        rng.integers(0, int(rows.shape[1]), size=graph.n_nodes)
        if start is None
        else check_labelling(start, graph.n_nodes, int(rows.shape[1]))
    )
    state = np.ascontiguousarray(drawn, dtype=np.int64)
    trace: list[ClusterCounter] = []
    lattice = _Lattice(
        graph, rows, graph.compressed_adjacency(), backend, cluster_backend
    )
    origin = Moved(state, lattice.energy(state), None, 0)
    walked = anneal(lattice.rung((move,), trace), schedule, origin, rng, np.copy)
    return AnnealedPotts(
        best=walked.best,
        energy=walked.energy,
        final=walked.final,
        n_sweeps=schedule.n_steps,
        spent=walked.spent,
        unit=Cost.SITE_VISITS,
        termination=walked.termination,
        trace=tuple(trace),
    )


@dataclass(frozen=True, kw_only=True)
class TemperedChains(Tempered[np.ndarray]):
    """What a parallel-tempering run produced (issue #1090).

    A :class:`~sal.sample.schedule.Tempered` over labellings: ``best`` is the
    lowest-energy configuration seen at any temperature, and ``spent`` the
    site visits of every replica's steps, burn-in included, each move charged
    as :func:`anneal_potts` charges it (issue #1156).

    Parameters
    ----------
    states : np.ndarray
        Recorded configurations, shape ``(n_sweeps, n_replicas, n_nodes)``;
        replica ``r`` sits at ``temperatures[r]`` throughout, because a swap
        exchanges *configurations* between temperatures rather than moving a
        chain along the ladder.
    energy : float
        ``best``'s energy, in :func:`energies`' convention.
    n_sweeps : int
        Sweeps run per replica after burn-in --- the budget per replica, so
        the whole run cost ``n_replicas`` times this.
    walkers : np.ndarray
        ``walkers[t, w]`` is the rung walker ``w`` sat at, at recorded sweep
        ``t``, shape ``(n_sweeps, n_replicas)``. The other reading of the
        same run: a *replica* is a temperature configurations pass through,
        where a *walker* is a configuration followed through the swaps, and
        a round trip is a statement about the second.
        :func:`sal.sample.tempered.round_trips` and
        :func:`sal.sample.tempered.up_fraction` read it, an
        exchange acceptance being a per-pair number a ladder can look
        healthy in while nothing crosses it (issue #756).
    """

    states: np.ndarray
    energy: float
    n_sweeps: int
    walkers: np.ndarray


def parallel_tempering(
    graph: PottsGraph,
    field: SiteField | np.ndarray,
    temperatures: TempSchedule | Sequence[float],
    rng: np.random.Generator,
    n_sweeps: int,
    burn_in: int = 0,
    thin: int = 1,
    *,
    move: RungMoves = PottsMove.SINGLE_SITE,
    backend: Backend = Backend.RUST,
    cluster_backend: Backend = Backend.PYTHON,
    start: np.ndarray | None = None,
) -> TemperedChains:
    """Replicas at fixed temperatures, exchanging configurations by Metropolis.

    Each replica runs its rung's moves once per step at its own temperature,
    in the order given, then every adjacent pair proposes to exchange configurations and accepts
    on :func:`swap_log_ratio`. The hot replicas cross barriers the cold one
    cannot, and an exchange carries what they find down the ladder (Swendsen &
    Wang, 1986; Geyer, 1991; Earl & Deem, 2005).

    **The exchange is exact for every move set** (issue #1156). The swap ratio
    reads energies alone, so the product law ``prod_r exp(-beta_r E)`` is
    invariant whenever each replica's move leaves its own rung's law
    invariant, which every :class:`~sal.sample.potts_mcmc.moves.PottsMove`
    does, and so does any sequence of them run in order on one rung. Each
    rung may therefore run its own moves (issue #1158); :func:`rung_moves`
    chooses them from the temperature. This is not :func:`cluster_tempering`,
    which adds Houdayer moves between replicas.

    **Moves belong to rungs, configurations to walkers.** An exchange swaps
    configurations between rungs and leaves each rung's moves where they
    are, so a configuration handed to a colder rung is next moved by that
    rung's moves at that rung's ``beta``. Every closure :func:`sweep_for`
    builds reads ``beta`` per call and stores nothing read from a
    configuration; the one state any keeps is the label-directed pass's call
    counter, which cycles the target label and is a rung's own. Each
    ``(rung, position)`` gets its own closure, so no two share it.

    **The replicas must not share a stream and must be reproducible from one
    seed.** The passed generator spawns a child per replica; the parent
    then draws only the exchange uniforms. Sharing one stream would correlate
    the replicas, which is the whole point lost while every diagnostic looks
    healthy.

    Parameters
    ----------
    graph : PottsGraph
        The instance. Couplings of either sign, except under a cluster move,
        which refuses a negative one as :func:`anneal_potts` does.
    field : SiteField | np.ndarray
        External field, shape ``(n_states,)``.
    temperatures : TempSchedule | Sequence[float]
        The ladder, in any order; at least two, all positive. The stationary
        distribution depends on which pairs are adjacent for exchange, not on
        the order.
    rng : np.random.Generator
        The parent generator: it spawns one child per replica and then draws
        only the exchange uniforms, so one seeded generator reproduces the run.
    n_sweeps, burn_in, thin : int
        As :func:`sample_potts`, applied per replica.
    move : PottsMove | Sequence[PottsMove | Sequence[PottsMove]]
        One move set every rung runs, or one entry per rung in the ladder's
        order, each a move set or a non-empty sequence of them run in order
        every step. Each is built by :func:`sweep_for` once per rung and
        position, so no rung shares another's mutable state. A single
        ``PottsMove`` is the chain of issue #1156 bitwise; single-site, the
        default, is the chain before the parameter existed, bitwise.
    backend : Backend
        As :func:`anneal_potts`: the Rust sweep by default, the oracle that
        pins it on request, each replica on its own child generator either
        way.
    cluster_backend : Backend
        Runs the cluster passes, as :func:`anneal_potts` states.
    start : np.ndarray | None
        One labelling per rung, shape ``(n_replicas, n_nodes)`` and in the
        ladder's order, each row checked by
        :func:`~sal.sim.potts.check_labelling`; ``None`` draws each from its
        replica's child generator, as before the parameter existed. A given
        start draws nothing, as :func:`anneal_potts`' does, so one step from a
        fixed pair is a draw from the product kernel's row (issue #1156).

    Returns
    -------
    TemperedChains

    Raises
    ------
    ValueError
        If fewer than two temperatures are given --- a ladder of one has
        nothing to exchange and is :func:`sample_potts` --- or any is not
        positive, or any move is a cluster move and a coupling is negative,
        or ``move`` is not one entry per rung, or ``start`` is not one
        labelling per rung.
    TypeError
        If a rung's entry holds anything but a ``PottsMove``.
    """
    field = log_weight_of(field)
    temperatures = check_ladder(ladder(temperatures), needed_by="parallel tempering")
    per_rung = moves_per_rung(move, len(temperatures))
    for each in dict.fromkeys(m for rung in per_rung for m in rung):
        refuse_negative_coupling(each, graph)

    rows = site_field(np.asarray(field, dtype=float), graph.n_nodes)
    n_replicas = len(temperatures)
    children = rng.spawn(n_replicas)
    n_states = int(rows.shape[1])
    # One contiguous `int64` row per replica: the kernel borrows a row of
    # this block rather than copying it.
    if start is None:
        drawn = [child.integers(0, n_states, size=graph.n_nodes) for child in children]
    else:
        if np.shape(start)[:1] != (n_replicas,):
            msg = (
                f"start holds one labelling per rung, {n_replicas}, "
                f"got shape {np.shape(start)}"
            )
            raise ValueError(msg)
        drawn = [check_labelling(row, graph.n_nodes, n_states) for row in start]
    states = np.ascontiguousarray(np.stack(drawn), dtype=np.int64)
    lattice = _Lattice(
        graph, rows, graph.compressed_adjacency(), backend, cluster_backend
    )
    # One sweep per rung and position: the label-directed pass keeps a call
    # counter, and a rung's counter is its own.
    steps = [lattice.rung(rung, scored=False) for rung in per_rung]
    # `swap_acceptance` is the mean over adjacent pairs of the fraction
    # accepted so far, recorded from the first step after burn-in. Round
    # trips and rung occupation are not: neither is a number this run
    # computes, and a hook does not define a metric (issue #778).
    tracked: TrackedOptimization = current_tracked()
    burned = burn_in * thin

    def observe(sweep: int, run: Exchanging[np.ndarray]) -> None:
        if sweep >= burned:
            tracked.record(
                sweep - burned,
                state=run.best,
                swap_acceptance=float(np.mean(run.swap_acceptance)),
            )

    run, recorded = lattice.temper(
        steps,
        temperatures,
        states,
        children,
        rng,
        (n_sweeps, burn_in, thin),
        kept=n_sweeps,
        observe=observe,
    )
    tracked.record_cost(max(n_sweeps * thin - 1, 0), states.nbytes)
    return TemperedChains(
        states=recorded,
        temperatures=tuple(temperatures),
        swap_acceptance=run.swap_acceptance,
        best=run.best,
        energy=run.energy,
        n_sweeps=n_sweeps,
        walkers=run.walkers,
        spent=run.spent,
        unit=Cost.SITE_VISITS,
        termination=Termination.after((burn_in + n_sweeps) * thin, converged=False),
    )


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

    def sweep(
        self, move: PottsMove, trace: list[ClusterCounter] | None = None
    ) -> Callable[[np.ndarray, np.random.Generator, float], int]:
        """:func:`sweep_for`, appending a cluster move's :class:`ClusterCounter` to ``trace`` per call.

        Only the moves whose pass reads a cluster's members keep a counter: the
        compiled Swendsen-Wang pass, the ghost-spin, label-directed and heat-bath
        Swendsen-Wang passes build their clusters as roots (issues #923, #1041,
        #1142).
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

        # The adjacency as lists, once for every cluster the closure grows (#919);
        # the ghost couplings, fixed by the field, once (#1041).
        one = move in _SINGLE_CLUSTER_MOVES
        lists = adjacency_lists(offsets, neighbours, couplings) if one else None
        # Read from `sweeps`, where a test replaces it.
        ghost = sweeps.ghost_couplings(rows) if move is PottsMove.GHOST_SPIN else None
        # A single cluster's growth, one signature for the three.
        grow: Callable[..., int] = (
            functools.partial(
                niedermayer_sweep, threshold=niedermayer_threshold(couplings)
            )
            if move is PottsMove.NIEDERMAYER
            else wolff_heat_bath_sweep
            if move is PottsMove.WOLFF_HEAT_BATH
            else wolff_sweep
        )
        arrays = (rows, offsets, neighbours, couplings)
        keeps = trace is not None and (
            one
            or (move is PottsMove.SWENDSEN_WANG and cluster_backend is Backend.PYTHON)
        )
        # The label-directed target cycles through the labels, one per call.
        calls = [0]

        def cluster_pass(
            state: np.ndarray, rng: np.random.Generator, beta: float = 1.0
        ) -> int:
            counter = ClusterCounter() if keeps else None
            size = 0
            if move is PottsMove.SWENDSEN_WANG:
                swendsen_wang_sweep(
                    state, graph, rows, rng, counter, beta, backend=cluster_backend
                )
            elif move is PottsMove.GHOST_SPIN:
                ghost_spin_sweep(
                    state, graph, rows, rng, beta, backend=cluster_backend, ghost=ghost
                )
            elif move is PottsMove.LABEL_DIRECTED:
                target = calls[0] % int(rows.shape[1])
                calls[0] += 1
                label_directed_sweep(
                    state, graph, rows, rng, target, beta, backend=cluster_backend
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
        graph = self.graph
        sweeps = [(each, self.sweep(each, trace)) for each in moves]

        def step(
            state: np.ndarray, _: float, __: None, temperature: float, rng: Generator
        ) -> Moved[np.ndarray, None]:
            beta = 1.0 / temperature
            visits = sum(
                step_visits(each, graph, sweep(state, rng, beta))
                for each, sweep in sweeps
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
    """

    states: np.ndarray
    houdayer_acceptance: np.ndarray
    houdayer_sizes: tuple[int, ...]
    houdayer_accepts: int
    energy: float
    n_sweeps: int


def cluster_tempering(
    graph: PottsGraph,
    field: SiteField | np.ndarray,
    temperatures: Sequence[float],
    rng: np.random.Generator,
    n_sweeps: int,
    *,
    houdayer_pairs: int = 1,
    burn_in: int = 0,
    thin: int = 1,
    record: bool = False,
    cluster_backend: Backend = Backend.RUST,
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
    temperatures : Sequence[float]
        The ladder, coldest first; at least two, all positive.
    rng : np.random.Generator
        Spawns one child per replica, then draws the Houdayer seed sites and
        every accept uniform.
    n_sweeps, burn_in, thin : int
        As :func:`parallel_tempering`.
    houdayer_pairs : int
        How many of the coldest adjacent pairs run a Houdayer move a step.
    record : bool
        Whether to keep the thinned configurations, which the enumeration
        test reads and a ground-state search does not.
    cluster_backend : Backend
        Runs the Swendsen-Wang pass, as :func:`anneal_potts` states.

    Returns
    -------
    ClusterTempered

    Raises
    ------
    ValueError
        If the ladder is not strictly increasing, coldest first, or
        ``houdayer_pairs`` is outside ``[0, n_replicas - 1]``.
    """
    refuse_negative_coupling(PottsMove.SWENDSEN_WANG, graph)
    temperatures = check_ladder(
        ladder(temperatures),
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
    states = np.ascontiguousarray(
        np.stack(
            [child.integers(0, n_states, size=graph.n_nodes) for child in children]
        ),
        dtype=np.int64,
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
    )


def adapt_ladder_potts(
    graph: PottsGraph,
    field: SiteField | np.ndarray,
    temperatures: TempSchedule | Sequence[float],
    rng: np.random.Generator,
    n_sweeps: int,
    band: tuple[float, float],
    max_iterations: int,
    max_replicas: int,
    *,
    backend: Backend = Backend.RUST,
) -> AdaptedLadder:
    """A ladder for :func:`parallel_tempering`, from its own exchange acceptances.

    :func:`sal.sample.schedule.adapt_ladder`, the measurement being
    a :func:`parallel_tempering` run of ``n_sweeps`` per replica on the
    candidate ladder, drawn from ``rng`` in sequence so one seed reproduces the
    warm-up. ``replicas_measured * n_sweeps`` is the warm-up's cost in sweeps,
    which a comparison against a hand ladder at equal budget charges (issue
    #333).

    Parameters
    ----------
    graph, field, rng, backend
        As :func:`parallel_tempering`.
    temperatures : TempSchedule | Sequence[float]
        The starting ladder, in either spelling and read by
        :func:`~sal.sample.schedule.ladder` into the same
        floats; its endpoints are kept.
    n_sweeps : int
        Sweeps per replica per measurement. Each acceptance is a fraction of
        ``n_sweeps`` proposals, so this sets what the band can resolve.
    band, max_iterations, max_replicas
        As :func:`sal.sample.schedule.adapt_ladder`.

    Returns
    -------
    AdaptedLadder
    """
    field = log_weight_of(field)

    def measure(candidate: tuple[float, ...]) -> list[float]:
        run = parallel_tempering(
            graph, field, candidate, rng, n_sweeps, backend=backend
        )
        return [float(value) for value in run.swap_acceptance]

    return adapt_ladder(
        measure, ladder(temperatures), band, max_iterations, max_replicas
    )


def sweep_for(
    move: PottsMove,
    graph: PottsGraph,
    rows: np.ndarray,
    offsets: np.ndarray,
    neighbours: np.ndarray,
    couplings: np.ndarray,
    backend: Backend,
    cluster_backend: Backend = Backend.PYTHON,
) -> Callable[[np.ndarray, np.random.Generator, float], int]:
    """One sweep of ``move``, as a call taking a state, a generator and ``beta``.

    The one place a move set is turned into a sweep, so
    :func:`sample_potts` and :func:`sample_potts_pair` run one dispatch rather
    than two that can drift. Each branch calls the sweep it already called,
    with the same arguments in the same order, so every existing chain is
    bitwise what it was. ``backend`` runs the heat-bath sweep and
    ``cluster_backend`` the Swendsen-Wang pass: two arguments because they are
    two decisions with two defaults (:func:`sample_potts`, issue #754).

    Returns
    -------
    Callable[[np.ndarray, np.random.Generator, float], int]
        The sweep, returning the size of the cluster it built and ``0`` where
        the move set builds none --- which is what
        :attr:`PottsChain.mean_cluster_size` averages.
    """
    adjacency = (offsets, neighbours, couplings)
    return _Lattice(graph, rows, adjacency, backend, cluster_backend).sweep(move)


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
    move: PottsMove,
    rng: np.random.Generator,
    n_sweeps: int,
    burn_in: int = 0,
    thin: int = 1,
    *,
    temperature: float = 1.0,
    houdayer: bool = True,
    backend: Backend = Backend.RUST,
    cluster_backend: Backend = Backend.PYTHON,
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
        do not share a stream --- :func:`parallel_tempering`'s rule, for its
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

    children = rng.spawn(2)
    states = [
        np.ascontiguousarray(
            child.integers(0, n_states, size=graph.n_nodes), dtype=np.int64
        )
        for child in children
    ]
    offsets, neighbours, couplings = graph.compressed_adjacency()
    advance = sweep_for(
        move, graph, rows, offsets, neighbours, couplings, backend, cluster_backend
    )

    recorded = [np.empty((n_sweeps, graph.n_nodes), dtype=np.int64) for _ in range(2)]
    totals, counts = [0, 0], [0, 0]
    for step in range(-burn_in * thin, n_sweeps * thin):
        for replica in range(2):
            size = advance(states[replica], children[replica], 1.0)
            if size:
                totals[replica] += size
                counts[replica] += 1
        if houdayer:
            houdayer_move(states[0], states[1], offsets, neighbours, rng)
        if step >= 0 and (step + 1) % thin == 0:
            for replica in range(2):
                recorded[replica][step // thin] = states[replica]
    first, second = (
        PottsChain(
            states=recorded[replica],
            mean_cluster_size=(
                totals[replica] / counts[replica]
                if counts[replica]
                else float(graph.n_nodes)
            ),
        )
        for replica in range(2)
    )
    return PottsPair(first=first, second=second)
