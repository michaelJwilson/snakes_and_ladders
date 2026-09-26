"""The drivers: a recorded chain, annealing, parallel tempering, the two-replica Houdayer chain, and the sweep a move set names.

Each runs the kernels of :mod:`~sal.sample.potts_mcmc.sweeps`
over a schedule and returns what the run recorded; :func:`sweep_for` is where a
:class:`~sal.sample.potts_mcmc.moves.PottsMove` becomes a
kernel.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from sal.backend import Backend
from sal.cost import Cost
from sal.opt.termination import Termination
from sal.sample.accept import accept
from sal.sample.potts_mcmc.moves import (
    PottsMove,
    refuse_negative_coupling,
)
from sal.sample.potts_mcmc.sweeps import (
    ClusterCounter,
    adjacency_lists,
    balanced_sweep_at,
    ghost_couplings,
    ghost_spin_sweep,
    houdayer_move,
    label_directed_sweep,
    niedermayer_sweep,
    niedermayer_threshold,
    sweep_at,
    swendsen_wang_sweep,
    wolff_sweep,
)
from sal.sample.runs import Annealed
from sal.sample.schedule import (
    AdaptedLadder,
    Monotone,
    TempSchedule,
    adapt_ladder,
    check_ladder,
    ladder,
)
from sal.sim.graph import PottsGraph
from sal.sim.potts import (
    SiteField,
    check_labelling,
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
        The move set. All six leave the same Boltzmann distribution
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


@dataclass(frozen=True, kw_only=True)
class AnnealedPotts(Annealed[np.ndarray]):
    """What one annealing run found, and what it cost (issue #1090).

    An :class:`~sal.sample.runs.Annealed` over labellings: ``best`` is the
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
        For the ghost-spin and label-directed passes it merges the bonds
        (:func:`bond_roots`), on the same roots either way, and neither
        keeps a counter (issue #1041).
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
    offsets, neighbours, couplings = graph.compressed_adjacency()
    lists = adjacency_lists(offsets, neighbours, couplings)

    best_state = state.copy()
    best_energy = float(energies(graph, rows, state[None])[0])
    sweep = (
        balanced_sweep_at(rows, offsets, neighbours, couplings, move)
        if move in _BALANCED_MOVES
        else sweep_at(rows, offsets, neighbours, couplings, backend)
    )
    # A heat-bath sweep reads every site's label once as a neighbour of each
    # incident edge and writes it once; the bond pass of Swendsen-Wang reads
    # the same two labels per edge. Counting both in one unit is what makes
    # the budget comparable across move sets (issue #551).
    per_sweep = graph.n_nodes + 2 * len(graph.edges)
    # The ghost couplings are fixed by the field, so stored once (#1041).
    ghost = ghost_couplings(rows) if move is PottsMove.GHOST_SPIN else None
    visits, trace = 0, []
    # One lookup for the run (`sal.track`) and one `record` a
    # sweep. `energy` is the best energy so far, which is what
    # `AnnealedPotts.energy` returns; the visited state's energy is computed
    # below either way, so the hook costs a call and no arithmetic. The best
    # labelling is the state passed, so a bound `PottsMetrics` scores what
    # the run returns.
    tracked: TrackedOptimization = current_tracked()
    for step in range(schedule.n_steps):
        temperature = schedule(step)
        if move is PottsMove.SINGLE_SITE:
            sweep(state, rng, 1.0 / temperature)
            visits += per_sweep
        elif move in _BALANCED_MOVES:
            sweep(state, rng, 1.0 / temperature)
            visits += graph.n_nodes * per_sweep
        else:
            counter = ClusterCounter()
            kept = True
            beta = 1.0 / temperature
            if move is PottsMove.SWENDSEN_WANG:
                # The compiled pass reads no cluster's members, so it keeps
                # no counter, and its step is not in the trace (issue #923).
                compiled = cluster_backend is Backend.RUST
                swendsen_wang_sweep(
                    state,
                    graph,
                    rows,
                    rng,
                    None if compiled else counter,
                    beta,
                    backend=cluster_backend,
                )
                visits += per_sweep
                kept = not compiled
            elif move is PottsMove.GHOST_SPIN:
                # No counter: the pass builds its clusters as roots, and a
                # ghost bond read per site is charged beside the edges.
                ghost_spin_sweep(
                    state,
                    graph,
                    rows,
                    rng,
                    beta,
                    backend=cluster_backend,
                    ghost=ghost,
                )
                visits += per_sweep + graph.n_nodes
                kept = False
            elif move is PottsMove.LABEL_DIRECTED:
                # The target label cycles with the step, so every label is
                # proposed once per `n_states` steps.
                label_directed_sweep(
                    state,
                    graph,
                    rows,
                    rng,
                    step % int(rows.shape[1]),
                    beta,
                    backend=cluster_backend,
                )
                visits += per_sweep
                kept = False
            else:
                if move is PottsMove.NIEDERMAYER:
                    niedermayer_sweep(
                        state,
                        rows,
                        offsets,
                        neighbours,
                        couplings,
                        rng,
                        counter,
                        graph,
                        beta,
                        niedermayer_threshold(couplings),
                        lists=lists,
                    )
                else:
                    wolff_sweep(
                        state,
                        rows,
                        offsets,
                        neighbours,
                        couplings,
                        rng,
                        counter,
                        graph,
                        beta,
                        lists=lists,
                    )
                # A single-cluster step reads each member's neighbours and
                # writes the members; a heat-bath sweep is charged the same
                # way, so one budget covers both.
                visits += sum(counter.sizes) * (
                    1 + 2 * len(graph.edges) // graph.n_nodes
                )
            if kept:
                trace.append(counter)
        energy = float(energies(graph, rows, state[None])[0])
        if energy < best_energy:
            best_state, best_energy = state.copy(), energy
        tracked.record(
            step, state=best_state, energy=best_energy, temperature=temperature
        )
    tracked.record_cost(max(schedule.n_steps - 1, 0), state.nbytes)
    return AnnealedPotts(
        best=best_state,
        energy=best_energy,
        final=state,
        n_sweeps=schedule.n_steps,
        spent=visits,
        unit=Cost.SITE_VISITS,
        termination=Termination.after(schedule.n_steps, converged=False),
        trace=tuple(trace),
    )


@dataclass(frozen=True)
class TemperedChains:
    """What a parallel-tempering run produced.

    Parameters
    ----------
    states : np.ndarray
        Recorded configurations, shape ``(n_sweeps, n_replicas, n_nodes)``;
        replica ``r`` sits at ``temperatures[r]`` throughout, because a swap
        exchanges *configurations* between temperatures rather than moving a
        chain along the ladder.
    temperatures : tuple[float, ...]
        The ladder, as given.
    swap_acceptance : np.ndarray
        Fraction of proposed exchanges accepted per adjacent pair, shape
        ``(n_replicas - 1,)``. Near zero means the ladder has a gap no
        configuration crosses and the replicas are independent chains; near
        one means two temperatures are close enough that one is redundant.
    best : np.ndarray
        The lowest-energy configuration seen at any temperature.
    best_energy : float
        Its energy, in :func:`energies`' convention.
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
    temperatures: tuple[float, ...]
    swap_acceptance: np.ndarray
    best: np.ndarray
    best_energy: float
    n_sweeps: int
    walkers: np.ndarray


def swap_log_ratio(
    beta_low: float, beta_high: float, energy_low: float, energy_high: float
) -> float:
    """Log acceptance of exchanging the configurations at two temperatures.

    The joint target is the product of the tempered marginals, so the ratio is
    ``(beta_i - beta_j)(E_i - E_j)``: an exchange handing the colder replica
    the lower energy is always accepted. A version omitting this term still
    runs, still mixes, and converges to the wrong distribution ---
    `tests/regression/search/test_potts_mcmc.py` replaces this function with it
    and asserts the chi-square catches it.
    """
    return (beta_low - beta_high) * (energy_low - energy_high)


def parallel_tempering(
    graph: PottsGraph,
    field: np.ndarray,
    temperatures: TempSchedule | Sequence[float],
    rng: np.random.Generator,
    n_sweeps: int,
    burn_in: int = 0,
    thin: int = 1,
    *,
    backend: Backend = Backend.RUST,
) -> TemperedChains:
    """Replicas at fixed temperatures, exchanging configurations by Metropolis.

    Each replica runs one heat-bath sweep per step at its own temperature,
    then every adjacent pair proposes to exchange configurations and accepts
    on :func:`swap_log_ratio`. The hot replicas cross barriers the cold one
    cannot, and an exchange carries what they find down the ladder (Swendsen &
    Wang, 1986; Geyer, 1991; Earl & Deem, 2005).

    **The replicas must not share a stream and must be reproducible from one
    seed.** The passed generator spawns a child per replica; the parent
    then draws only the exchange uniforms. Sharing one stream would correlate
    the replicas, which is the whole point lost while every diagnostic looks
    healthy.

    Parameters
    ----------
    graph : PottsGraph
        The instance. Couplings of either sign; single-site moves only, for
        the reason :func:`anneal_potts` gives.
    field : np.ndarray
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
    backend : Backend
        As :func:`anneal_potts`: the Rust sweep by default, the oracle that
        pins it on request, each replica on its own child generator either
        way.

    Returns
    -------
    TemperedChains

    Raises
    ------
    ValueError
        If fewer than two temperatures are given --- a ladder of one has
        nothing to exchange and is :func:`sample_potts` --- or any is not
        positive.
    """
    temperatures = check_ladder(ladder(temperatures), needed_by="parallel tempering")

    rows = site_field(np.asarray(field, dtype=float), graph.n_nodes)
    n_replicas = len(temperatures)
    betas = [1.0 / temperature for temperature in temperatures]
    children = rng.spawn(n_replicas)
    n_states = int(rows.shape[1])
    # One contiguous `int64` row per replica: the kernel borrows a row of
    # this block rather than copying it.
    states = np.ascontiguousarray(
        np.stack(
            [child.integers(0, n_states, size=graph.n_nodes) for child in children]
        ),
        dtype=np.int64,
    )
    offsets, neighbours, couplings = graph.compressed_adjacency()

    recorded = np.empty((n_sweeps, n_replicas, graph.n_nodes), dtype=np.int64)
    # Which walker sits at each rung, as `sample.tempered.exchange` tracks
    # it: a swap moves configurations between temperatures, so this is what
    # says a configuration crossed the ladder.
    at_rung = list(range(n_replicas))
    trace = np.empty((n_sweeps, n_replicas), dtype=np.int64)
    proposed = np.zeros(n_replicas - 1)
    accepted = np.zeros(n_replicas - 1)
    current = energies(graph, rows, states)
    best_index = int(np.argmin(current))
    best, best_energy = states[best_index].copy(), float(current[best_index])

    sweep = sweep_at(rows, offsets, neighbours, couplings, backend)
    # `swap_acceptance` is the mean over adjacent pairs of the fraction
    # accepted so far -- the mean of the vector `TemperedChains` returns, and
    # so equal to it at the last sweep. Round trips and rung occupation are
    # not recorded: neither is a number this run computes, and a hook does
    # not define a metric (issue #778).
    tracked: TrackedOptimization = current_tracked()
    for step in range(-burn_in * thin, n_sweeps * thin):
        for replica in range(n_replicas):
            sweep(states[replica], children[replica], betas[replica])
        current = energies(graph, rows, states)
        for pair in range(n_replicas - 1):
            log_ratio = swap_log_ratio(
                betas[pair], betas[pair + 1], current[pair], current[pair + 1]
            )
            proposed[pair] += 1
            if accept(log_ratio, rng):
                accepted[pair] += 1
                states[[pair, pair + 1]] = states[[pair + 1, pair]]
                current[[pair, pair + 1]] = current[[pair + 1, pair]]
                at_rung[pair], at_rung[pair + 1] = at_rung[pair + 1], at_rung[pair]
        lowest = int(np.argmin(current))
        if current[lowest] < best_energy:
            best, best_energy = states[lowest].copy(), float(current[lowest])
        if step >= 0 and (step + 1) % thin == 0:
            recorded[step // thin] = states
            for rung, walker in enumerate(at_rung):
                trace[step // thin, walker] = rung
        if step >= 0:
            tracked.record(
                step, state=best, swap_acceptance=float(np.mean(accepted / proposed))
            )
    tracked.record_cost(max(n_sweeps * thin - 1, 0), states.nbytes)
    return TemperedChains(
        states=recorded,
        temperatures=tuple(temperatures),
        swap_acceptance=accepted / proposed,
        best=best,
        best_energy=best_energy,
        n_sweeps=n_sweeps,
        walkers=trace,
    )


@dataclass(frozen=True)
class ClusterTempered:
    """What :func:`cluster_tempering` produced.

    Parameters
    ----------
    states : np.ndarray
        Recorded configurations, ``(n_recorded, n_replicas, n_nodes)``;
        empty along the first axis unless the run was asked to record.
    temperatures : tuple[float, ...]
        The ladder, as given.
    swap_acceptance : np.ndarray
        Exchanges accepted over proposed, per adjacent pair.
    houdayer_acceptance : np.ndarray
        Houdayer moves accepted over proposed, per pair that runs one.
    houdayer_sizes : tuple[int, ...]
        Every proposed Houdayer cluster's size, in order; a pair that agrees
        everywhere proposes none.
    houdayer_accepts : int
        Of those, the moves accepted.
    best : np.ndarray
        The lowest-energy configuration seen at any temperature.
    best_energy : float
        Its energy.
    n_sweeps : int
        Steps run, each one Swendsen-Wang pass per replica.
    site_visits : int
        Every replica's passes plus every Houdayer move, each charged one
        sweep's ``n_nodes + 2 n_edges``: the move reads every site and the
        defect sites' edges.
    """

    states: np.ndarray
    temperatures: tuple[float, ...]
    swap_acceptance: np.ndarray
    houdayer_acceptance: np.ndarray
    houdayer_sizes: tuple[int, ...]
    houdayer_accepts: int
    best: np.ndarray
    best_energy: float
    n_sweeps: int
    site_visits: int


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
    offsets, neighbours, _ = graph.compressed_adjacency()
    per_sweep = graph.n_nodes + 2 * len(graph.edges)
    kept = n_sweeps if record else 0
    recorded = np.empty((kept, n_replicas, graph.n_nodes), dtype=np.int64)
    swaps = np.zeros((2, n_replicas - 1))
    houdayer = np.zeros((2, houdayer_pairs))
    sizes: list[int] = []
    current = energies(graph, rows, states)
    lowest = int(np.argmin(current))
    best, best_energy = states[lowest].copy(), float(current[lowest])
    visits = 0
    for step in range(-burn_in * thin, n_sweeps * thin):
        for replica in range(n_replicas):
            swendsen_wang_sweep(
                states[replica],
                graph,
                rows,
                children[replica],
                None,
                betas[replica],
                backend=cluster_backend,
            )
        visits += n_replicas * per_sweep
        current = energies(graph, rows, states)
        for pair in range(houdayer_pairs):
            first, second = states[pair].copy(), states[pair + 1].copy()
            size = houdayer_move(first, second, offsets, neighbours, rng)
            visits += per_sweep
            if size == 0:
                continue
            sizes.append(size)
            houdayer[0, pair] += 1
            moved = energies(graph, rows, np.stack([first, second]))
            log_ratio = -(betas[pair] - betas[pair + 1]) * (moved[0] - current[pair])
            if accept(float(log_ratio), rng):
                houdayer[1, pair] += 1
                states[pair], states[pair + 1] = first, second
                current[pair], current[pair + 1] = moved[0], moved[1]
        for pair in range(n_replicas - 1):
            log_ratio = swap_log_ratio(
                betas[pair], betas[pair + 1], current[pair], current[pair + 1]
            )
            swaps[0, pair] += 1
            if accept(log_ratio, rng):
                swaps[1, pair] += 1
                states[[pair, pair + 1]] = states[[pair + 1, pair]]
                current[[pair, pair + 1]] = current[[pair + 1, pair]]
        lowest = int(np.argmin(current))
        if current[lowest] < best_energy:
            best, best_energy = states[lowest].copy(), float(current[lowest])
        if record and step >= 0 and (step + 1) % thin == 0:
            recorded[step // thin] = states
    return ClusterTempered(
        states=recorded[: n_sweeps if record else 0],
        temperatures=tuple(temperatures),
        swap_acceptance=swaps[1] / np.maximum(swaps[0], 1),
        houdayer_acceptance=houdayer[1] / np.maximum(houdayer[0], 1),
        houdayer_sizes=tuple(sizes),
        houdayer_accepts=int(houdayer[1].sum()),
        best=best,
        best_energy=best_energy,
        n_sweeps=n_sweeps,
        site_visits=visits,
    )


def adapt_ladder_potts(
    graph: PottsGraph,
    field: np.ndarray,
    start: TempSchedule | Sequence[float],
    rng: np.random.Generator,
    n_sweeps: int,
    band: tuple[float, float],
    max_rounds: int,
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
    start : TempSchedule | Sequence[float]
        The starting ladder, in either spelling and read by
        :func:`~sal.sample.schedule.ladder` into the same
        floats; its endpoints are kept.
    n_sweeps : int
        Sweeps per replica per measurement. Each acceptance is a fraction of
        ``n_sweeps`` proposals, so this sets what the band can resolve.
    band, max_rounds, max_replicas
        As :func:`sal.sample.schedule.adapt_ladder`.

    Returns
    -------
    AdaptedLadder
    """

    def measure(candidate: tuple[float, ...]) -> list[float]:
        run = parallel_tempering(
            graph, field, candidate, rng, n_sweeps, backend=backend
        )
        return [float(value) for value in run.swap_acceptance]

    return adapt_ladder(measure, ladder(start), band, max_rounds, max_replicas)


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

    if move is PottsMove.SWENDSEN_WANG:

        def bond_pass(
            state: np.ndarray, rng: np.random.Generator, beta: float = 1.0
        ) -> int:
            swendsen_wang_sweep(
                state, graph, rows, rng, None, beta, backend=cluster_backend
            )
            return 0

        return bond_pass

    if move is PottsMove.GHOST_SPIN:

        def ghost_pass(
            state: np.ndarray, rng: np.random.Generator, beta: float = 1.0
        ) -> int:
            ghost_spin_sweep(state, graph, rows, rng, beta, backend=cluster_backend)
            return 0

        return ghost_pass

    if move is PottsMove.LABEL_DIRECTED:
        # The target cycles through the labels, one per call, as
        # `anneal_potts` cycles it by step.
        calls = [0]

        def directed_pass(
            state: np.ndarray, rng: np.random.Generator, beta: float = 1.0
        ) -> int:
            target = calls[0] % int(rows.shape[1])
            calls[0] += 1
            label_directed_sweep(
                state, graph, rows, rng, target, beta, backend=cluster_backend
            )
            return 0

        return directed_pass

    # The adjacency as lists, once for every cluster the closure grows (#919).
    lists = adjacency_lists(offsets, neighbours, couplings)
    if move is PottsMove.NIEDERMAYER:
        threshold = niedermayer_threshold(couplings)

        def generalized(
            state: np.ndarray, rng: np.random.Generator, beta: float = 1.0
        ) -> int:
            return niedermayer_sweep(
                state,
                rows,
                offsets,
                neighbours,
                couplings,
                rng,
                beta=beta,
                threshold=threshold,
                lists=lists,
            )

        return generalized

    def one_cluster(
        state: np.ndarray, rng: np.random.Generator, beta: float = 1.0
    ) -> int:
        return wolff_sweep(
            state, rows, offsets, neighbours, couplings, rng, beta=beta, lists=lists
        )

    return one_cluster


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
    field: np.ndarray,
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
