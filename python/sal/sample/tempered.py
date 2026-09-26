"""A tempered ensemble over structures, composed from moves that already exist (issue #331).

Replica exchange over the factor graph and over tree topologies: replicas at
fixed temperatures, each stepped by a move :mod:`sal.sample.gibbs`
already holds -- the heat-bath sweep for a labelling or a decoding, the
Metropolis move for a topology -- and exchanged on the ratio
:func:`sal.sample.potts_mcmc.parallel_tempering` uses
(``docs/tex/textbook.tex``, ``eq:exchange``). Nothing here is a new sampler;
the ensemble is the moves of issue #309 on the ladder of issue #267.

The replica at temperature one is the one that matters. Its marginal is the
unscaled model -- the Boltzmann weight of a labelling, the path posterior of
a decoding, the flat-prior weight over fitted likelihoods of a topology --
so the fraction of its recorded sweeps spent at a structure estimates that
structure's weight over the *whole* space (``eq:tempered-weight``), where
the neighbourhood weight is conditional on a neighbourhood and enumeration
is refused past its cap. The hot replicas are there to move it: a chain at
temperature one alone stays where it starts on a rugged surface, and an
exchange carries what the hot replicas find down the ladder.

**The exchange loop below is not this module's alone.**
:func:`sal.sample.hmc.parallel_tempering` runs it too, supplying
one Hamiltonian transition per replica as its step and its own torch stream as
the swap's draw (issue #861). The two temperings stay two entry points with two
referees --- enumeration for the lattice, quadrature for the posterior --- and
what they share is the loop, not the rung.

**Every replica draws from its own generator**, spawned from the parent that
then draws only the exchange uniforms, as ``potts_mcmc.parallel_tempering``
does and for the reason it gives. And the estimate is a Monte Carlo one:
:mod:`sal.search.support` names it a posterior weight only
beside the exchange acceptance and the autocorrelation time that show the
ensemble mixed.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import TypeVar

import numpy as np

from sal.backend import Backend
from sal.sample.accept import accept
from sal.sample.gibbs import (
    Indexed,
    cached_topology_score,
    gibbs_sweep,
    topology_step,
)
from sal.sample.potts_mcmc import (
    PottsMove,
    energies,
    houdayer_move,
    parallel_tempering,
    refuse_negative_coupling,
    swap_log_ratio,
    sweep_for,
)
from sal.sample.schedule import (
    FeedbackLadder,
    TempSchedule,
    adapt_ladder_by_round_trips,
    check_ladder,
    ladder,
)
from sal.sim.factor_graph import FactorGraph
from sal.sim.graph import PottsGraph
from sal.sim.potts import SiteField, log_weight_of, site_field
from sal.sim.topology import Model, MoveSet, Topology, leaf_bipartitions
from sal.track import TrackedOptimization, current

S = TypeVar("S")
G = TypeVar("G")


@dataclass(frozen=True)
class TemperedEnsemble:
    """What a replica-exchange run over structures produced.

    Parameters
    ----------
    temperatures : tuple[float, ...]
        The ladder, as given; replica ``r`` sits at ``temperatures[r]``
        throughout, because an exchange swaps structures between
        temperatures rather than moving a replica along the ladder.
    keys : tuple[tuple[Hashable, ...], ...]
        Per replica, the canonical key of its structure at every recorded
        sweep: the labelling as a tuple of states, a topology as its leaf
        bipartitions.
    log_densities : np.ndarray
        The unnormalized log-density of every recorded structure at
        temperature one, shape ``(n_recorded, n_replicas)``.
    swap_acceptance : np.ndarray
        Fraction of proposed exchanges accepted per adjacent pair, shape
        ``(n_replicas - 1,)``: near zero, a gap in the ladder no structure
        crosses; near one, a temperature that is redundant.
    scores : Mapping[Hashable, float]
        Every structure any replica held at a recorded sweep, keyed, with
        its log-density at temperature one.
    walkers : np.ndarray
        ``walkers[t, w]`` is the rung walker ``w`` sat at, at recorded sweep
        ``t``, shape ``(n_recorded, n_replicas)``. A *walker* is a structure
        followed through the exchanges, where a *replica* is a temperature
        that structures pass through --- the two readings of the same run, and
        a round trip is a statement about the first. :func:`round_trips`
        counts them.
    """

    temperatures: tuple[float, ...]
    keys: tuple[tuple[Hashable, ...], ...]
    log_densities: np.ndarray
    swap_acceptance: np.ndarray
    scores: Mapping[Hashable, float]
    walkers: np.ndarray

    @property
    def round_trip_time(self) -> float:
        """Recorded sweeps a walker spends per round trip, averaged over walkers.

        :func:`round_trip_time` on :attr:`walkers`, which is where the
        arithmetic lives: a
        :class:`~sal.sample.potts_mcmc.TemperedChains` carries
        the same trace and is read by the same definition.
        """
        return round_trip_time(self.walkers)

    def replica_at(self, temperature: float) -> int:
        """The index of the replica at ``temperature``.

        Raises
        ------
        ValueError
            If no replica sits there. The marginal weight is read at
            temperature one, and a ladder without it has no unscaled
            replica to read.
        """
        for index, value in enumerate(self.temperatures):
            if value == temperature:
                return index
        msg = f"no replica at temperature {temperature}; the ladder is {self.temperatures}"
        raise ValueError(msg)


def round_trips(walkers: np.ndarray) -> np.ndarray:
    """Round trips completed per walker: the coldest rung to the hottest and back.

    The one definition of a round trip in this package. A walker is counted
    when it returns to rung ``0`` having reached the last rung since it was
    there, so a walker rattling at the cold end scores nothing however often
    it revisits rung ``0`` --- which is the whole reason the statistic is
    preferred to an exchange acceptance, that being a per-pair quantity a
    ladder can look healthy in while no structure crosses it
    (Katzgraber, Trebst, Huse & Troyer 2006).

    Parameters
    ----------
    walkers : np.ndarray
        :attr:`TemperedEnsemble.walkers`: the rung of each walker at each
        recorded sweep, shape ``(n_recorded, n_replicas)``.

    Returns
    -------
    np.ndarray
        ``(n_replicas,)`` of counts, one per walker.

    Raises
    ------
    ValueError
        If the trace is not two-dimensional.
    """
    trace = np.asarray(walkers, dtype=np.int64)
    if trace.ndim != 2:
        msg = f"a walker trace is (n_recorded, n_replicas), got {trace.shape}"
        raise ValueError(msg)
    top = int(trace.max(initial=0))
    counts = np.zeros(trace.shape[1], dtype=np.int64)
    for walker in range(trace.shape[1]):
        reached_top = False
        for rung in trace[:, walker].tolist():
            if rung == top:
                reached_top = True
            elif rung == 0 and reached_top:
                counts[walker] += 1
                reached_top = False
    return counts


def round_trip_time(walkers: np.ndarray) -> float:
    """Recorded sweeps a walker spends per round trip, averaged over walkers.

    ``inf`` where no walker completed one, which is the reading a ladder with
    a gap gives and is not a division to guard against: a ladder no structure
    crosses has an infinite round-trip time, and reporting it as such is what
    makes it comparable with one that does.

    Parameters
    ----------
    walkers : np.ndarray
        The rung of each walker at each recorded sweep, shape
        ``(n_recorded, n_replicas)``.

    Returns
    -------
    float
    """
    counts = round_trips(walkers)
    total = float(counts.sum())
    if total == 0.0:
        return float("inf")
    trace = np.asarray(walkers)
    return float(trace.shape[0] * trace.shape[1]) / total


def up_fraction(walkers: np.ndarray) -> np.ndarray:
    """Fraction of labelled visits at each rung made by a walker on its way up.

    The measurement a feedback-optimized ladder is placed from
    (Katzgraber, Trebst, Huse & Troyer 2006), and the same trace
    :func:`round_trips` counts trips in: a walker is labelled *up* from the
    moment it touches rung ``0`` and *down* from the moment it touches the
    last rung, and every recorded sweep between is a visit carrying that
    label. ``f`` is 1 at rung 0 and 0 at the last rung by construction, and
    falls steeply where walkers are held up --- which is what the placement
    reads.

    Visits before a walker has touched either end carry no label and are not
    counted; they are the walker's start, which says nothing about a
    direction it has not yet had.

    Parameters
    ----------
    walkers : np.ndarray
        :attr:`TemperedEnsemble.walkers`, shape
        ``(n_recorded, n_replicas)``.

    Returns
    -------
    np.ndarray
        ``(n_rungs,)``, and ``nan`` at a rung no labelled walker visited ---
        a ladder that carries no placement, reported rather than filled in.

    Raises
    ------
    ValueError
        If the trace is not two-dimensional.
    """
    trace = np.asarray(walkers, dtype=np.int64)
    if trace.ndim != 2:
        msg = f"a walker trace is (n_recorded, n_replicas), got {trace.shape}"
        raise ValueError(msg)
    top = int(trace.max(initial=0))
    up = np.zeros(top + 1)
    down = np.zeros(top + 1)
    for walker in range(trace.shape[1]):
        label = 0
        for rung in trace[:, walker].tolist():
            if rung == 0:
                label = 1
            elif rung == top:
                label = -1
            if label == 1:
                up[rung] += 1.0
            elif label == -1:
                down[rung] += 1.0
    total = up + down
    with np.errstate(invalid="ignore"):
        return np.where(total > 0.0, up / total, np.nan)


def _swap_drawn(rng: np.random.Generator) -> Callable[[float], bool]:
    """The exchange's accept step on a NumPy stream: one uniform, and only where the ratio is negative."""
    return lambda log_ratio: accept(log_ratio, rng)


def _check_budget(n_sweeps: int, thin: int, burn_in: int) -> None:
    """What a run is asked for, beside what its ladder is."""
    if n_sweeps < 1 or thin < 1 or burn_in < 0:
        msg = f"n_sweeps {n_sweeps} and thin {thin} must be >= 1 and burn_in {burn_in} >= 0"
        raise ValueError(msg)


def exchange(
    step: Callable[[S, float, float, G], tuple[S, float]],
    key: Callable[[S], Hashable] | None,
    states: list[S],
    values: list[float],
    temperatures: Sequence[float],
    children: Sequence[G],
    swap: Callable[[float], bool],
    n_sweeps: int,
    burn_in: int,
    thin: int,
    record: Callable[[Sequence[S], Sequence[float]], None] | None = None,
    stop: Callable[[], bool] | None = None,
) -> TemperedEnsemble:
    """The replica-exchange loop, over any structure with a step and a key.

    ``step(state, value, temperature, generator)`` advances one replica one
    sweep and returns its new state and log-density at temperature one; the
    exchange ratio takes the energy ``-value``.

    **The draw is the caller's and the test is here**, which is
    :mod:`sal.sample.accept`'s own division: ``swap`` is
    handed the log ratio and answers whether the pair exchanges, so a site
    that draws one NumPy uniform only where the ratio is negative and one
    that draws a torch uniform every time are the same loop on their own
    streams (issue #861).

    ``key`` is ``None`` where a state has no canonical key --- a continuous
    position is a point, not a structure --- and the ensemble then carries
    no keys and no scores. ``record`` is called at every recorded sweep with
    the states and their log-densities after the exchanges, for a caller
    whose result carries the states themselves; the ensemble carries the
    keys of them.

    ``stop`` is asked before every sweep after the first, and ``True`` ends
    the loop there, so ``burn_in + n_sweeps`` is a ceiling rather than a
    count. It is for a wall clock and must read no replica's state: a sweep
    that stops on a state-dependent condition biases what it records
    (`sample/CLAUDE.md`). ``None`` runs every sweep, as before it existed.
    """
    n_replicas = len(temperatures)
    betas = [1.0 / temperature for temperature in temperatures]
    proposed = np.zeros(n_replicas - 1)
    accepted = np.zeros(n_replicas - 1)
    recorded_keys: list[list[Hashable]] = [[] for _ in range(n_replicas)]
    densities: list[list[float]] = []
    scores: dict[Hashable, float] = {}
    # Which walker sits at each rung. An exchange swaps structures between
    # temperatures, so this is what says a *structure* crossed the ladder,
    # which the per-pair acceptance cannot.
    at_rung = list(range(n_replicas))
    trace: list[list[int]] = []
    # One lookup for both ensembles (`sal.track`), since both
    # run this loop. `swap_acceptance` is the mean over adjacent pairs of the
    # fraction accepted so far -- the mean of the vector `TemperedEnsemble`
    # returns -- and `log_density` is replica 0's, the first column of the
    # `log_densities` it returns, whose state is replica 0's, so a bound
    # `Metrics` reads the same replica the density is taken from.
    tracked: TrackedOptimization = current()
    started = time.perf_counter()
    swept = 0
    for sweep in range(burn_in + n_sweeps):
        if stop is not None and sweep > 0 and stop():
            break
        swept += 1
        for replica in range(n_replicas):
            states[replica], values[replica] = step(
                states[replica],
                values[replica],
                temperatures[replica],
                children[replica],
            )
        for pair in range(n_replicas - 1):
            log_ratio = swap_log_ratio(
                betas[pair], betas[pair + 1], -values[pair], -values[pair + 1]
            )
            proposed[pair] += 1
            if swap(log_ratio):
                accepted[pair] += 1
                states[pair], states[pair + 1] = states[pair + 1], states[pair]
                values[pair], values[pair + 1] = values[pair + 1], values[pair]
                at_rung[pair], at_rung[pair + 1] = at_rung[pair + 1], at_rung[pair]
        if sweep >= burn_in and (sweep - burn_in) % thin == 0:
            if record is not None:
                record(states, values)
            if key is not None:
                for replica in range(n_replicas):
                    name = key(states[replica])
                    scores[name] = values[replica]
                    recorded_keys[replica].append(name)
            densities.append(list(values))
            rungs = [0] * n_replicas
            for rung, walker in enumerate(at_rung):
                rungs[walker] = rung
            trace.append(rungs)
        if not tracked.is_null:
            # The round trips and the up fraction are read from the trace so
            # far by the two functions the result is read with, so the last
            # entry is what `round_trips(ensemble.walkers)` returns; the
            # re-read per sweep is paid by a listening run only (issue #799).
            wall = time.perf_counter() - started
            so_far = np.array(trace, dtype=np.int64).reshape(len(trace), n_replicas)
            tracked.record(
                sweep,
                state=states[0],
                swap_acceptance=float(np.mean(accepted / proposed)),
                log_density=values[0],
                round_trips=float(round_trips(so_far).sum()) if len(trace) else 0.0,
                up_fraction=float(np.nanmean(up_fraction(so_far)))
                if len(trace)
                else 0.0,
                sweeps_per_second=(sweep + 1) / wall if wall else 0.0,
                wall_s=wall,
            )
    walkers = np.array(trace, dtype=np.int64).reshape(len(trace), n_replicas)
    log_densities = np.array(densities)
    tracked.record_cost(max(swept - 1, 0), walkers.nbytes + log_densities.nbytes)
    return TemperedEnsemble(
        temperatures=tuple(temperatures),
        keys=tuple(tuple(names) for names in recorded_keys),
        log_densities=log_densities,
        swap_acceptance=accepted / proposed,
        scores=scores,
        walkers=walkers,
    )


def tempered_factor_graph(
    graph: FactorGraph,
    temperatures: TempSchedule | Sequence[float],
    rng: np.random.Generator,
    n_sweeps: int,
    burn_in: int = 0,
    thin: int = 1,
    *,
    start: np.ndarray | None = None,
) -> TemperedEnsemble:
    """Replicas of the heat-bath sweep over ``graph`` at fixed temperatures, exchanging by Metropolis.

    One :func:`sal.sample.gibbs.gibbs_sweep` per replica per
    step at its own ``beta``, then every adjacent pair proposes an exchange.
    Serves a Potts labelling and a hidden path alike, through the adapters
    of :mod:`sal.sim.factor_graph`.

    Parameters
    ----------
    graph : FactorGraph
        Any of the adapters' graphs.
    temperatures : TempSchedule | Sequence[float]
        The ladder, at least two, all positive; the order fixes which pairs
        are adjacent for exchange.
    rng : np.random.Generator
        The parent generator: one child per replica, then the exchange
        uniforms only.
    n_sweeps, burn_in, thin : int
        As :func:`sal.sample.gibbs.sample_factor_graph`, per
        replica.
    start : np.ndarray | None
        A starting state every replica takes; ``None`` draws one per replica
        uniformly from its own child.

    Raises
    ------
    ValueError
        As :func:`sal.sample.gibbs.sample_factor_graph`, and if
        the ladder has fewer than two temperatures or one that is not
        positive.
    """
    temperatures = check_ladder(ladder(temperatures), needed_by="a tempered ensemble")
    _check_budget(n_sweeps, thin, burn_in)
    indexed = Indexed(graph)
    children = rng.spawn(len(temperatures))
    states = [indexed.start(child, start) for child in children]
    values = [indexed.log_density(state) for state in states]

    def step(
        state: np.ndarray, _: float, temperature: float, child: np.random.Generator
    ) -> tuple[np.ndarray, float]:
        gibbs_sweep(indexed, state, child, beta=1.0 / temperature)
        return state, indexed.log_density(state)

    return exchange(
        step,
        lambda state: tuple(int(value) for value in state),
        states,
        values,
        temperatures,
        children,
        _swap_drawn(rng),
        n_sweeps,
        burn_in,
        thin,
    )


def tempered_potts_pair(
    graph: PottsGraph,
    field: SiteField | np.ndarray,
    temperatures: TempSchedule | Sequence[float],
    rng: np.random.Generator,
    n_sweeps: int,
    burn_in: int = 0,
    thin: int = 1,
    *,
    move: PottsMove = PottsMove.SINGLE_SITE,
    houdayer: bool = True,
    backend: Backend = Backend.RUST,
    cluster_backend: Backend = Backend.PYTHON,
) -> TemperedEnsemble:
    """A replica pair per rung, exchanging along the ladder, joined by Houdayer's move.

    The ensemble Houdayer (2001) defines his move on: two systems run side by
    side, and at each rung the pair takes the isoenergetic cluster move between
    the two sweeps and the exchange. The state carried along the ladder is the
    *pair*, so the joint target at a rung is the product of the two tempered
    marginals and the exchange ratio takes the pair's summed energy --- which
    is :func:`exchange`'s own ratio on that energy, not a second one.

    Nothing here is a new sampler. The within-replica sweep is
    :func:`~sal.sample.potts_mcmc.sample_potts`'s own, through
    the one dispatch
    :func:`~sal.sample.potts_mcmc.sweep_for` holds, and the
    exchange is the loop the factor graph and the topologies already run.

    Parameters
    ----------
    graph : PottsGraph
        The lattice. Couplings of either sign, subject to ``move``'s own
        refusal.
    field : SiteField | np.ndarray
        External field ``h``, shape ``(n_states,)`` or ``(n_nodes, n_states)``.
    temperatures : TempSchedule | Sequence[float]
        The ladder, at least two, all positive, in the order that fixes which
        pairs are adjacent for exchange.
    rng : np.random.Generator
        The parent: one child per rung, then the exchange uniforms only.
    n_sweeps, burn_in, thin : int
        As :func:`tempered_factor_graph`, per rung.
    move : PottsMove
        The within-replica move set, applied to each replica of every pair.
    houdayer : bool
        Whether each rung's pair takes the isoenergetic move between the
        sweeps and the exchange. ``False`` is the control the round-trip time
        is read against.
    backend : Backend
        As :func:`~sal.sample.potts_mcmc.sample_potts`.
    cluster_backend : Backend
        Which implementation runs a cluster move's pass, as
        :func:`~sal.sample.potts_mcmc.sample_potts` takes it;
        :data:`~sal.backend.Backend.PYTHON`, the default, is the chain before
        #1059 threaded it here, bitwise.

    Returns
    -------
    TemperedEnsemble
        Keyed by the pair, so :attr:`TemperedEnsemble.keys` names both
        replicas of a rung and :attr:`TemperedEnsemble.log_densities` is
        their summed log-density at temperature one.

    Raises
    ------
    ValueError
        If the ladder is unusable, as :func:`tempered_factor_graph` states, if
        ``houdayer`` is asked for at other than two states --- Houdayer's
        overlap is the Ising one (issue #756) --- or if ``move`` is a
        Fortuin-Kasteleyn cluster move on a graph with a negative coupling, as
        :func:`~sal.sample.potts_mcmc.sample_potts` refuses it.
    """
    field = log_weight_of(field)
    temperatures = check_ladder(ladder(temperatures), needed_by="a tempered ensemble")
    _check_budget(n_sweeps, thin, burn_in)
    refuse_negative_coupling(move, graph)
    rows = site_field(np.asarray(field, dtype=float), graph.n_nodes)
    n_states = int(rows.shape[1])
    if houdayer and n_states != 2:
        msg = (
            f"Houdayer's move is defined on the Ising overlap q_i = s_i s'_i "
            f"and this model has {n_states} states: pass houdayer=False, or "
            "use two states (issue #756)"
        )
        raise ValueError(msg)

    offsets, neighbours, couplings = graph.compressed_adjacency()
    advance = sweep_for(
        move, graph, rows, offsets, neighbours, couplings, backend, cluster_backend
    )
    children = rng.spawn(len(temperatures))

    def start(child: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.ascontiguousarray(
                child.integers(0, n_states, size=graph.n_nodes), dtype=np.int64
            ),
            np.ascontiguousarray(
                child.integers(0, n_states, size=graph.n_nodes), dtype=np.int64
            ),
        )

    def density(pair: tuple[np.ndarray, np.ndarray]) -> float:
        return -float(energies(graph, rows, np.stack(pair)).sum())

    states = [start(child) for child in children]
    values = [density(pair) for pair in states]

    def step(
        pair: tuple[np.ndarray, np.ndarray],
        _: float,
        temperature: float,
        child: np.random.Generator,
    ) -> tuple[tuple[np.ndarray, np.ndarray], float]:
        beta = 1.0 / temperature
        for replica in pair:
            advance(replica, child, beta)
        if houdayer:
            houdayer_move(pair[0], pair[1], offsets, neighbours, child)
        return pair, density(pair)

    return exchange(
        step,
        lambda pair: tuple(int(value) for value in np.concatenate(pair)),
        states,
        values,
        temperatures,
        children,
        _swap_drawn(rng),
        n_sweeps,
        burn_in,
        thin,
    )


def tempered_topologies(
    alignment: Mapping[str, np.ndarray],
    n_states: int,
    temperatures: TempSchedule | Sequence[float],
    rng: np.random.Generator,
    n_sweeps: int,
    burn_in: int = 0,
    thin: int = 1,
    *,
    start: Topology,
    moves: MoveSet = MoveSet.NNI,
    model: Model = Model.JC,
    scores: dict[frozenset[frozenset[str]], float] | None = None,
) -> TemperedEnsemble:
    """Replicas of the Metropolis topology move at fixed temperatures, exchanging by Metropolis.

    One :func:`sal.sample.gibbs.topology_step` per replica
    per step on the fitted log-likelihood, every replica from ``start``.
    The energy in the exchange ratio is the negative fitted log-likelihood,
    so the replica at temperature one targets the flat-prior weight over
    maximized likelihoods that
    :func:`sal.search.support.enumerated_support` computes --
    a weight over fitted likelihoods, not a marginal over branch lengths.

    ``scores`` caches fitted log-likelihoods by leaf bipartitions across
    calls, since the fit is the whole cost; every replica shares it.

    Raises
    ------
    ValueError
        If the ladder has fewer than two temperatures or one that is not
        positive, or ``n_sweeps``, ``thin`` or ``burn_in`` is unusable.
    """
    temperatures = check_ladder(ladder(temperatures), needed_by="a tempered ensemble")
    _check_budget(n_sweeps, thin, burn_in)
    cache = {} if scores is None else scores
    score = cached_topology_score(alignment, n_states, cache, model=model)
    children = rng.spawn(len(temperatures))
    value = score(start)

    def step(
        state: Topology, current: float, temperature: float, child: np.random.Generator
    ) -> tuple[Topology, float]:
        taken = topology_step(state, current, temperature, child, score, moves=moves)
        return taken.topology, taken.log_likelihood

    return exchange(
        step,
        leaf_bipartitions,
        [start] * len(temperatures),
        [value] * len(temperatures),
        temperatures,
        children,
        _swap_drawn(rng),
        n_sweeps,
        burn_in,
        thin,
    )


def adapt_ladder_round_trips(
    graph: PottsGraph,
    field: SiteField | np.ndarray,
    temperatures: TempSchedule | Sequence[float],
    rng: np.random.Generator,
    n_sweeps: int,
    tolerance: float,
    max_iterations: int,
    *,
    backend: Backend = Backend.RUST,
) -> FeedbackLadder:
    """A ladder for :func:`~sal.sample.potts_mcmc.parallel_tempering`, placed by its own round trips.

    :func:`sal.sample.schedule.adapt_ladder_by_round_trips`, the
    measurement being a :func:`~sal.sample.potts_mcmc.parallel_tempering`
    run of ``n_sweeps`` per replica on the candidate ladder, read through
    :func:`up_fraction`. The sibling of
    :func:`~sal.sample.potts_mcmc.adapt_ladder_potts`, which
    places the same ladder by its exchange acceptance; both draw from ``rng``
    in sequence, so one seed reproduces the warm-up, and
    ``replicas_measured * n_sweeps`` is its cost in sweeps, which a comparison
    at equal budget charges.

    Parameters
    ----------
    graph, field, rng, backend
        As :func:`~sal.sample.potts_mcmc.parallel_tempering`.
    temperatures : TempSchedule | Sequence[float]
        The starting ladder, in either spelling and read by
        :func:`~sal.sample.schedule.ladder` into the same
        floats; its endpoints and its length are the result's.
    n_sweeps : int
        Sweeps per replica per measurement. The up-fraction is a ratio of
        visit counts over these, so it sets what the placement can resolve.
    tolerance, max_iterations
        As :func:`sal.sample.schedule.adapt_ladder_by_round_trips`.

    Returns
    -------
    FeedbackLadder
    """
    field = log_weight_of(field)

    def measure(candidate: tuple[float, ...]) -> list[float]:
        run = parallel_tempering(
            graph, field, candidate, rng, n_sweeps, backend=backend
        )
        return [float(value) for value in up_fraction(run.walkers)]

    return adapt_ladder_by_round_trips(
        measure, ladder(temperatures), tolerance, max_iterations
    )
