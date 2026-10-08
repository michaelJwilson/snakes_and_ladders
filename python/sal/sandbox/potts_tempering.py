"""The Potts ``parallel_tempering`` and what runs through it, no longer supported (issue #1352).

Moved here on the sandbox's "unsupported" rule, by the owner's decision
alone: replicas at fixed temperatures each running their rung's
:class:`~sal.sample.potts_mcmc.PottsMove` set and exchanging configurations by
Metropolis, with ``temperatures="auto"``; :func:`adapt_ladder_potts` and
:func:`adapt_ladder_round_trips`, which place its ladder from its own
exchanges and round trips; and the ``tempering`` and ``tempering-mixed``
ground-state arms, :func:`run_tempering` and :func:`run_tempering_mixed`,
which left :data:`~sal.search.ground_state.METHODS`; and the per-rung move
rule :func:`rung_moves`, with :data:`RungMoves`, :func:`moves_per_rung`,
:func:`critical_ratio` and :func:`field_ratio`, which only this tempering
reads (issue #1365). The supported tempering
on the lattice is :func:`~sal.sample.potts_mcmc.cluster_tempering`;
:func:`sal.sample.hmc.parallel_tempering` is a different sampler and stays.
The tests moved with it, under ``tests/regression/sandbox/``.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from sal.backend import Backend
from sal.cost import Cost
from sal.opt.budget import Budget
from sal.opt.termination import Termination
from sal.sample.loop import Exchanging
from sal.sample.potts_mcmc import PottsMove, Recolour
from sal.sample.potts_mcmc.chains import _Lattice, _rung_starts
from sal.sample.potts_mcmc.moves import composed, move_set, refuse_negative_coupling
from sal.sample.schedule import (
    AdaptedLadder,
    FeedbackLadder,
    Monotone,
    Tempered,
    TempSchedule,
    adapt_ladder,
    adapt_ladder_by_round_trips,
    check_ladder,
    ladder,
)
from sal.sample.tempered import up_fraction
from sal.sample.tune import Ladder, LadderTuning, TunedLadder, resolve_ladder
from sal.search.ground_state import (
    _COMPILED_CLUSTERS,
    ANNEAL_END,
    ANNEAL_START,
    MethodRun,
    Problem,
    Rung,
    _problem,
    _refuse_start,
    step_cost,
)
from sal.sim.graph import PottsGraph
from sal.sim.potts import SiteField, critical_coupling, log_weight_of, site_field
from sal.track import TrackedOptimization
from sal.track import current as current_tracked

__all__ = [
    "N_REPLICAS",
    "TemperedChains",
    "adapt_ladder_potts",
    "adapt_ladder_round_trips",
    "parallel_tempering",
    "run_tempering",
    "run_tempering_mixed",
    "tempering_ladder",
]

#: Replicas in the tempering ladder, geometric over the same endpoints. The
#: budget is divided by this, so a replica gets one sixth of the sweeps the
#: single-site entry gets and the comparison is at equal cost rather than at
#: equal sweeps per chain.
N_REPLICAS = 6


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
        return (composed(move),) * n_rungs
    entries = tuple(move)
    if len(entries) != n_rungs:
        msg = f"move holds one entry per rung, {n_rungs}, got {len(entries)}"
        raise ValueError(msg)
    per_rung = []
    for entry in entries:
        rung = composed(entry) if isinstance(entry, PottsMove) else tuple(entry)
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
#: and 1.58 (``tests/regression/sandbox/test_potts_tempering_rung_threshold.py``).
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

    **Measured** through :func:`run_tempering` at
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
class TemperedChains(Tempered[np.ndarray]):
    """What a parallel-tempering run produced (issue #1090).

    A :class:`~sal.sample.schedule.Tempered` over labellings: ``best`` is the
    lowest-energy configuration seen at any temperature, and ``spent`` the
    site visits of every replica's steps, burn-in included, each move charged
    as :func:`~sal.sample.potts_mcmc.anneal_potts` charges it (issue #1156).

    Parameters
    ----------
    states : np.ndarray
        Recorded configurations, shape ``(n_sweeps, n_replicas, n_nodes)``;
        replica ``r`` sits at ``temperatures[r]`` throughout, because a swap
        exchanges *configurations* between temperatures rather than moving a
        chain along the ladder.
    energy : float
        ``best``'s energy, in :func:`~sal.sim.potts.energies`' convention.
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
    tuned_ladder : TunedLadder | None
        The pilot that chose the ladder under ``temperatures="auto"`` (issue
        #1337), ``None`` for a given ladder; its spend is not in ``spent``.
    """

    states: np.ndarray
    energy: float
    n_sweeps: int
    walkers: np.ndarray
    tuned_ladder: TunedLadder | None = None


def parallel_tempering(
    graph: PottsGraph,
    field: SiteField | np.ndarray,
    temperatures: Ladder,
    rng: np.random.Generator,
    n_sweeps: int,
    burn_in: int = 0,
    thin: int = 1,
    *,
    move: RungMoves = PottsMove.SINGLE_SITE,
    recolour: Recolour = Recolour.PER_MOVE,
    backend: Backend = Backend.RUST,
    cluster_backend: Backend = Backend.RUST,
    start: np.ndarray | None = None,
    ladder_tuning: LadderTuning | None = None,
) -> TemperedChains:
    """Replicas at fixed temperatures, exchanging configurations by Metropolis.

    Each replica runs its rung's moves once per step at its own temperature,
    in the order given, then every adjacent pair proposes to exchange configurations and accepts
    on :func:`~sal.sample.loop.swap_log_ratio`. The hot replicas cross barriers the cold one
    cannot, and an exchange carries what they find down the ladder (Swendsen &
    Wang, 1986; Geyer, 1991; Earl & Deem, 2005).

    **The exchange is exact for every move set** (issue #1156). The swap ratio
    reads energies alone, so the product law ``prod_r exp(-beta_r E)`` is
    invariant whenever each replica's move leaves its own rung's law
    invariant, which every :class:`~sal.sample.potts_mcmc.moves.PottsMove`
    does, and so does any sequence of them run in order on one rung. Each
    rung may therefore run its own moves (issue #1158); :func:`rung_moves`
    chooses them from the temperature. This is not :func:`~sal.sample.potts_mcmc.cluster_tempering`,
    which adds Houdayer moves between replicas.

    **Moves belong to rungs, configurations to walkers.** An exchange swaps
    configurations between rungs and leaves each rung's moves where they
    are, so a configuration handed to a colder rung is next moved by that
    rung's moves at that rung's ``beta``. Every closure :func:`~sal.sample.potts_mcmc.sweep_for`
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
        which refuses a negative one as :func:`~sal.sample.potts_mcmc.anneal_potts` does.
    field : SiteField | np.ndarray
        External field, shape ``(n_states,)``.
    temperatures : TempSchedule | Sequence[float] | Literal["auto"]
        The ladder, coldest first and strictly increasing; at least two, all
        positive. The order is :func:`~sal.sample.potts_mcmc.cluster_tempering`'s, which 20 of 32
        explicit-ladder call sites passed when the two were made one (issue
        #1343); another order is refused, not reversed. ``"auto"`` chooses it by
        :func:`~sal.sample.schedule.adapt_ladder` on pilot runs of this
        function from ``ladder_tuning.start``, drawn from one child spawned
        from ``rng`` first (issue #1337).
    rng : np.random.Generator
        The parent generator: it spawns one child per replica and then draws
        only the exchange uniforms, so one seeded generator reproduces the run.
    n_sweeps, burn_in, thin : int
        As :func:`~sal.sample.potts_mcmc.sample_potts`, applied per replica.
    move : PottsMove | Sequence[PottsMove | Sequence[PottsMove]]
        One move set every rung runs, or one entry per rung in the ladder's
        order, each a move set or a non-empty sequence of them run in order
        every step. Each is built by :func:`~sal.sample.potts_mcmc.sweep_for` once per rung and
        position, so no rung shares another's mutable state. A single
        ``PottsMove`` is the chain of issue #1156 bitwise; single-site, the
        default, is the chain before the parameter existed, bitwise.
    backend : Backend
        As :func:`~sal.sample.potts_mcmc.anneal_potts`: the Rust sweep by default, the oracle that
        pins it on request, each replica on its own child generator either
        way.
    cluster_backend : Backend
        Runs the cluster passes, as :func:`~sal.sample.potts_mcmc.anneal_potts` states.
    start : np.ndarray | None
        ``(n_nodes,)`` for every rung, or ``(n_rungs, n_nodes)`` one per rung
        of the ladder given, coldest first; under ``"auto"`` each tuned rung
        takes the row of ``ladder_tuning.start``'s rung nearest it in
        temperature, a tie to the colder (issue #1343). Each row is checked by
        :func:`~sal.sim.potts.check_labelling`; ``None`` draws each from its
        replica's child generator, as before the parameter existed. A given
        start draws nothing, as :func:`~sal.sample.potts_mcmc.anneal_potts`' does, so one step from a
        fixed pair is a draw from the product kernel's row (issue #1156).
    ladder_tuning : LadderTuning | None
        The pilot that chooses ``temperatures="auto"``, required with it and
        refused without it. A given ladder draws nothing for it, so its run
        is bitwise the run before #1337.

    Returns
    -------
    TemperedChains

    Raises
    ------
    ValueError
        If fewer than two temperatures are given --- a ladder of one has
        nothing to exchange and is :func:`~sal.sample.potts_mcmc.sample_potts` --- or any is not
        positive, or the ladder is not strictly increasing, coldest first,
        or any move is a cluster move and a coupling is negative, or
        ``move`` is not one entry per rung, or ``start`` is neither shape.
    TypeError
        If a rung's entry holds anything but a ``PottsMove``.
    """
    field = log_weight_of(field)

    def pilot(candidate: tuple[float, ...]) -> tuple[list[float], int]:
        run = parallel_tempering(
            graph,
            field,
            candidate,
            pilot_rng,
            ladder_tuning.n_sweeps if ladder_tuning else 0,
            move=move,
            recolour=recolour,
            backend=backend,
            cluster_backend=cluster_backend,
        )
        return [float(value) for value in run.swap_acceptance], run.spent

    pilot_rng = rng.spawn(1)[0] if isinstance(temperatures, str) else rng
    given, tuned_ladder = resolve_ladder(temperatures, ladder_tuning, pilot)
    temperatures = check_ladder(
        ladder(given),
        needed_by="parallel tempering",
        monotone=Monotone.INCREASING,
    )
    per_rung = tuple(
        move_set(rung, recolour) for rung in moves_per_rung(move, len(temperatures))
    )
    for each in dict.fromkeys(m for rung in per_rung for m in rung):
        refuse_negative_coupling(each, graph)

    rows = site_field(np.asarray(field, dtype=float), graph.n_nodes)
    n_replicas = len(temperatures)
    children = rng.spawn(n_replicas)
    n_states = int(rows.shape[1])
    states = _rung_starts(
        start, temperatures, ladder_tuning, children, graph.n_nodes, n_states
    )
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
        tuned_ladder=tuned_ladder,
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
    """A ladder for :func:`parallel_tempering`, placed by its own round trips.

    :func:`sal.sample.schedule.adapt_ladder_by_round_trips`, the
    measurement being a :func:`parallel_tempering`
    run of ``n_sweeps`` per replica on the candidate ladder, read through
    :func:`~sal.sample.tempered.up_fraction`. The sibling of
    :func:`adapt_ladder_potts`, which
    places the same ladder by its exchange acceptance; both draw from ``rng``
    in sequence, so one seed reproduces the warm-up, and
    ``replicas_measured * n_sweeps`` is its cost in sweeps, which a comparison
    at equal budget charges.

    Parameters
    ----------
    graph, field, rng, backend
        As :func:`parallel_tempering`.
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


def run_tempering(
    problem: Problem | Rung,
    budget: Budget,
    rng: np.random.Generator,
    *,
    move: RungMoves = PottsMove.SINGLE_SITE,
    start: np.ndarray | None = None,
    temperatures: Sequence[float] | Literal["auto"] | None = None,
    ladder_tuning: LadderTuning | None = None,
) -> MethodRun:
    """Parallel tempering over a geometric ladder, charged for every replica.

    The budget buys ``budget // sum_r step_cost(rung r)`` steps per replica,
    a rung's step costing the :func:`~sal.search.ground_state.step_cost` of each move it runs, rather
    than that many per chain, which is the whole difference between a
    comparison at equal cost and one at equal sweeps. ``move`` is the move set
    each replica runs (issue #1156), or one entry per rung of the hot-first
    ladder, each a move or a sequence of moves (issue #1158), as
    :func:`parallel_tempering` takes it. The cluster
    backend is the compiled one where any move is on :func:`~sal.search.ground_state.run_annealed`'s
    compiled route; the single-cluster moves read none. The step count is
    fixed as :func:`~sal.search.ground_state.run_annealed` fixes one, so a single-cluster move
    underspends, and ``spent`` is what the run charged.

    ``temperatures`` is :func:`tempering_ladder` unless given; ``"auto"``
    chooses it as :func:`parallel_tempering` does,
    from ``ladder_tuning`` (issue #1337), for one ``PottsMove`` on every
    rung. The pilots' site visits come out of ``budget`` and are in
    ``spent``, so a tuned ladder is compared at equal cost.
    """
    _refuse_start("tempering", start, "its ladder draws one labelling per replica")
    problem = _problem(problem)
    tuned_ladder = None
    if isinstance(temperatures, str) and not isinstance(move, PottsMove):
        msg = "temperatures='auto' tunes one move set for every rung, not one per rung"
        raise ValueError(msg)

    def pilot(candidate: tuple[float, ...]) -> tuple[list[float], int]:
        run = parallel_tempering(
            problem.graph,
            problem.field,
            candidate,
            pilot_rng,
            ladder_tuning.n_sweeps if ladder_tuning else 0,
            move=move,
            recolour=Recolour.UNIFORM,
            cluster_backend=Backend.RUST
            if move in _COMPILED_CLUSTERS
            else Backend.PYTHON,
        )
        return [float(value) for value in run.swap_acceptance], run.spent

    pilot_rng = rng.spawn(1)[0] if isinstance(temperatures, str) else rng
    given, tuned_ladder = resolve_ladder(
        tempering_ladder() if temperatures is None else temperatures,
        ladder_tuning,
        pilot,
    )
    temperatures = ladder(given)
    pilot_spent = 0 if tuned_ladder is None else tuned_ladder.spent
    n_rungs = len(temperatures)
    # Each bare move alone, as recorded: a sequence is taken as given (#1323).
    entries = [move] * n_rungs if isinstance(move, PottsMove) else list(move)
    per_rung = moves_per_rung(
        [(each,) if isinstance(each, PottsMove) else each for each in entries],
        n_rungs,
    )
    per_step = sum(step_cost(problem, each) for rung in per_rung for each in rung)
    per_replica = max(1, (budget.size - pilot_spent) // per_step)
    compiled = any(each in _COMPILED_CLUSTERS for rung in per_rung for each in rung)
    started = time.perf_counter()
    run = parallel_tempering(
        problem.graph,
        problem.field,
        temperatures,
        rng,
        per_replica,
        move=per_rung,
        # As the annealed arm: the recorded results' recolouring (#1323).
        recolour=Recolour.UNIFORM,
        cluster_backend=Backend.RUST if compiled else Backend.PYTHON,
    )
    return MethodRun(
        labelling=run.best,
        energy=run.energy,
        spent=run.spent + pilot_spent,
        seconds=time.perf_counter() - started,
        termination=Termination.after(per_replica, converged=False),
        tuned_ladder=tuned_ladder,
    )


def run_tempering_mixed(
    problem: Problem | Rung,
    budget: Budget,
    rng: np.random.Generator,
    *,
    start: np.ndarray | None = None,
) -> MethodRun:
    """:func:`run_tempering` on the moves :func:`rung_moves` picks per rung (issue #1158).

    Heat-bath Swendsen-Wang on the rungs hot of the transition, heat-bath
    Swendsen-Wang then a single-site sweep on the rest, at equal site visits
    to ``tempering``: each rung's step charges both its moves, so the ladder
    runs fewer steps on the same budget.
    """
    _refuse_start(
        "tempering-mixed", start, "its ladder draws one labelling per replica"
    )
    problem = _problem(problem)
    return run_tempering(
        problem,
        budget,
        rng,
        move=rung_moves(problem.graph, problem.field, tempering_ladder()),
    )


def tempering_ladder() -> tuple[float, ...]:
    """The temperatures :func:`run_tempering` exchanges across, coldest first.

    :data:`N_REPLICAS` rungs geometric from :data:`~sal.search.ground_state.ANNEAL_END` to
    :data:`~sal.search.ground_state.ANNEAL_START`, the annealing schedule's endpoints, in the one order
    the Potts temperings take (issue #1343).
    """
    return tuple(
        float(value) for value in np.geomspace(ANNEAL_END, ANNEAL_START, N_REPLICAS)
    )
