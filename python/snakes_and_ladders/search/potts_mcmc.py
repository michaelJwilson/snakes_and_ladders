"""Monte Carlo move sets on a Potts lattice: single-site, Swendsen-Wang, Wolff.

`ROADMAP.md` §1.4 names both cluster algorithms. Single-site flips slow
critically near the transition --- the autocorrelation time of the energy
diverges as the correlation length does --- so no Potts result at a useful
lattice size is reachable through them. Cluster updates flip whole correlated
regions at once and do not.

**The field is easy to get silently wrong.** The reference instance is a Potts
model *in an external field*, and the Fortuin-Kasteleyn construction both
cluster algorithms rest on is exact only at zero field: recolouring a cluster
changes the field term by ``|C| * (h[new] - h[old])``, which the bond
construction knows nothing about. Left there, the sampler runs, produces
plausible configurations, and converges to the wrong distribution. So a cluster
recolouring carries a Metropolis accept step on that difference, and the
chi-square tests in `tests/regression/search/test_potts_mcmc.py` are run with
and without a field because only the first catches its absence.

These are samplers, not optimizers: they are validated by the distribution they
converge to, and nothing here claims to find a ground state. The exception is
:func:`anneal_potts`, an optimizer built *from* the sampler: the same sweep on
a schedule of falling temperatures (issue #267).

**Temperature is model scaling.** The Potts coupling absorbs ``beta``:
``exp(-E / T)`` with ``E = -h[s] - J [s = s']`` is the Boltzmann weight of the
model with ``(J / T, h / T)`` at temperature 1. So a tempered chain runs the
untempered sweeps on the scaled model and there is no second code path:
:func:`tempered` is checked against the energies, and the chain it produces
against ``exp(-E / T)`` enumerated from the *unscaled* model. Tempering a
likelihood is a different object (`snakes_and_ladders.opt.schedule` says why);
here the objective is an energy and the temperature is physical.

See ``docs/tex/textbook.tex``, ``sec:potts`` (Newman &
Barkema chs. 4 and 6 for both algorithms and for Sokal's windowing; Mezard &
Montanari ch. 2).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from enum import StrEnum
from typing import NamedTuple

import numpy as np

from snakes_and_ladders.likelihood.potts import log_weights
from snakes_and_ladders.opt.schedule import AdaptedLadder, Schedule, adapt_ladder
from snakes_and_ladders.search.backend import Backend
from snakes_and_ladders.sim.graph import PottsGraph
from snakes_and_ladders.sim.potts import site_field


class PottsMove(StrEnum):
    """Which Monte Carlo move set a chain proposes from.

    A ``StrEnum`` for the reason `snakes_and_ladders.search.infer.MoveSet` is one: an
    unrecognized move is rejected by ``mypy --strict`` at the call site rather
    than by a branch that silently falls through to a default.
    """

    SINGLE_SITE = "single-site"
    SWENDSEN_WANG = "swendsen-wang"
    WOLFF = "wolff"


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


class Recolour(NamedTuple):
    """What one cluster recolouring proposed, and whether the field accepted it.

    Two booleans rather than one, because a proposal that drew the colour the
    cluster already has is not a rejection: counting it as one understates the
    accept rate by ``1 / q`` and would make the cluster moves look worse than
    they are at exactly the point issue #551 measures them.
    """

    proposed: bool
    accepted: bool


@dataclass
class ClusterCounter:
    """Cluster sizes and field acceptances, accumulated over a run.

    The instrumentation issue #551 exists to collect. Mutable and passed in
    rather than returned, because a sweep recolours a variable number of
    clusters and threading a growing tuple back through every call would cost
    more than the sweep.

    Attributes
    ----------
    sizes : list[int]
        One entry per cluster recoloured.
    proposals, accepts : int
        Colour changes proposed, and of those accepted. The accept rate is
        the ratio, and is undefined rather than 1.0 when nothing was proposed.
    spanning : int
        Clusters reaching from one edge of the lattice to the opposite one.
        Counted only where the graph declares a 2-D ``shape``; a graph without
        one has no sides to span and this stays zero.
    """

    sizes: list[int] = dataclass_field(default_factory=list)
    proposals: int = 0
    accepts: int = 0
    spanning: int = 0

    def record(
        self, members: np.ndarray, outcome: Recolour, graph: PottsGraph | None
    ) -> None:
        """Add one recoloured cluster."""
        self.sizes.append(int(members.shape[0]))
        self.proposals += int(outcome.proposed)
        self.accepts += int(outcome.accepted)
        self.spanning += int(_spans(members, graph))

    @property
    def accept_rate(self) -> float:
        """Accepted over proposed; ``nan`` where nothing was proposed."""
        return self.accepts / self.proposals if self.proposals else float("nan")

    @property
    def mean_size(self) -> float:
        """Sites per recoloured cluster; ``nan`` where none was built."""
        return float(np.mean(self.sizes)) if self.sizes else float("nan")

    @property
    def max_size(self) -> int:
        """The largest cluster recoloured; zero where none was built."""
        return max(self.sizes) if self.sizes else 0

    @property
    def spanning_fraction(self) -> float:
        """Clusters that spanned the lattice, as a fraction of those built."""
        return self.spanning / len(self.sizes) if self.sizes else float("nan")


def _spans(members: np.ndarray, graph: PottsGraph | None) -> bool:
    """Whether a cluster reaches both opposite sides of a 2-D lattice.

    The node index of a lattice built by
    :func:`snakes_and_ladders.sim.graph.triangular_lattice_graph` is
    ``row * columns + column``, so the coordinates are a ``divmod``. A
    spanning cluster is what makes a cluster move a global move rather than a
    large local one, which is the distinction issue #551 reports against
    temperature.
    """
    if graph is None or graph.shape is None or len(graph.shape) != 2:
        return False
    rows, columns = graph.shape
    row, column = np.divmod(members, columns)
    return bool(
        (row.min() == 0 and row.max() == rows - 1)
        or (column.min() == 0 and column.max() == columns - 1)
    )


def tempered(
    graph: PottsGraph, field: np.ndarray, temperature: float
) -> tuple[PottsGraph, np.ndarray]:
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
    return scaled, np.asarray(field, dtype=float) / temperature


def sample_potts(
    graph: PottsGraph,
    field: np.ndarray,
    move: PottsMove,
    rng: np.random.Generator,
    n_sweeps: int,
    burn_in: int = 0,
    thin: int = 1,
    *,
    temperature: float = 1.0,
) -> PottsChain:
    """Run one chain and return the configuration after every sweep.

    Parameters
    ----------
    graph : PottsGraph
        The lattice. Couplings may vary per edge.
    field : np.ndarray
        External field ``h``, shape ``(n_states,)``.
    move : PottsMove
        The move set. All three leave the same Boltzmann distribution
        invariant, which is what
        `tests/regression/search/test_potts_mcmc.py` asserts.
    rng : np.random.Generator
        Passed in rather than seeded here: seeding inside a call makes every
        draw of an ensemble identical (`sim/CLAUDE.md`, issue #240).
    n_sweeps : int
        Recorded sweeps. A sweep is ``n_nodes`` heat-bath updates, one
        Swendsen-Wang bond-and-recolour pass over the whole lattice, or *one*
        Wolff cluster flip --- see :func:`_wolff_sweep` for why the Wolff
        sweep cannot be sized to match the other two.
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
    if move is not PottsMove.SINGLE_SITE and min(graph.coupling, default=0.0) < 0.0:
        msg = (
            f"{move} needs every coupling >= 0: the bond probability "
            "1 - exp(-J) is not a probability for J < 0, and an "
            "antiferromagnet has no like-spin clusters to flip"
        )
        raise ValueError(msg)

    graph, field = tempered(graph, field, temperature)
    rows = site_field(field, graph.n_nodes)
    n_states = int(rows.shape[1])
    state = rng.integers(0, n_states, size=graph.n_nodes)
    neighbours = _adjacency(graph)

    recorded = np.empty((n_sweeps, graph.n_nodes), dtype=np.int64)
    cluster_total, cluster_count = 0, 0
    for step in range(-burn_in * thin, n_sweeps * thin):
        if move is PottsMove.SINGLE_SITE:
            _single_site_sweep(state, rows, neighbours, rng)
        elif move is PottsMove.SWENDSEN_WANG:
            _swendsen_wang_sweep(state, graph, rows, rng)
        else:
            cluster_total += _wolff_sweep(state, rows, neighbours, rng)
            cluster_count += 1
        if step >= 0 and (step + 1) % thin == 0:
            recorded[step // thin] = state
    mean_cluster = (
        cluster_total / cluster_count if cluster_count else float(graph.n_nodes)
    )
    return PottsChain(states=recorded, mean_cluster_size=mean_cluster)


@dataclass(frozen=True)
class AnnealedPotts:
    """What one annealing run found, and what it cost.

    Parameters
    ----------
    labelling : np.ndarray
        The lowest-energy configuration visited, shape ``(n_nodes,)``. The
        *best* rather than the last: the final sweeps run cold but not at
        zero, so the chain can leave the best state it found.
    energy : float
        Its energy, in :func:`energies`' convention.
    final : np.ndarray
        Where the chain ended, kept so a caller can see whether the best was
        the end or a state passed through.
    n_sweeps : int
        Sweeps run, one per schedule step.
    site_visits : int
        Site labels read or written by the move set. **This** is what a
        budget is matched on, not the sweep count: a Wolff sweep flips one
        cluster while a heat-bath sweep touches every site, so equal sweeps
        hand the cluster moves a free lattice per move (issue #551).
    trace : tuple[ClusterCounter, ...]
        One counter per schedule step for a cluster move set, empty for
        single-site. Kept per step because the quantity issue #551 predicts
        is a function of temperature and the schedule is what varies it.
    """

    labelling: np.ndarray
    energy: float
    final: np.ndarray
    n_sweeps: int
    site_visits: int = 0
    trace: tuple[ClusterCounter, ...] = ()


def anneal_potts(
    graph: PottsGraph,
    field: np.ndarray,
    schedule: Schedule,
    rng: np.random.Generator,
    *,
    move: PottsMove = PottsMove.SINGLE_SITE,
    backend: Backend = Backend.PYTHON,
) -> AnnealedPotts:
    """Simulated annealing by heat-bath sweeps on a temperature schedule.

    One :func:`_single_site_sweep` per schedule step at that step's
    temperature, tracking the lowest energy seen (Kirkpatrick, Gelatt & Vecchi,
    1983). It is :func:`iterated_conditional_modes` at finite temperature: at
    ``T -> 0`` the heat bath is the argmin over each site's conditional, ICM's
    update, so the two are one search separated by the schedule and a
    difference between them is a statement about the schedule.

    ``move`` names the move set. Single-site is the default and the fair
    annealed baseline. The two cluster move sets refuse a negative coupling,
    so they anneal only a ferromagnet --- which `potts_spots` is, and which
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
    field : np.ndarray
        External field, shape ``(n_states,)``.
    schedule : Schedule
        Temperature per sweep. Its length is the budget.
    rng : np.random.Generator
        Source of every draw, the start included. Passed in rather than
        seeded here, for the reason :func:`sample_potts` gives.

    backend : Backend
        :data:`~snakes_and_ladders.search.backend.Backend.PYTHON` runs the oracle
        sweep; :data:`~snakes_and_ladders.search.backend.Backend.RUST` runs the
        extension's, on the same uniforms in the same order. Opt-in rather
        than default for the reason :mod:`snakes_and_ladders.search.potts_mcmc_rust`
        gives: the two agree distributionally, not draw for draw, so the
        default path keeps every committed chain unchanged.

    Returns
    -------
    AnnealedPotts
    """
    if move is not PottsMove.SINGLE_SITE and min(graph.coupling, default=0.0) < 0.0:
        msg = (
            f"{move} needs every coupling >= 0: the bond probability "
            "1 - exp(-J) is not a probability for J < 0, and an "
            "antiferromagnet has no like-spin clusters to flip"
        )
        raise ValueError(msg)

    rows = site_field(np.asarray(field, dtype=float), graph.n_nodes)
    state = rng.integers(0, int(rows.shape[1]), size=graph.n_nodes)
    neighbours = _adjacency(graph)

    best_state = state.copy()
    best_energy = float(energies(graph, rows, state[None])[0])
    sweep = _sweep_at(graph, rows, neighbours, backend)
    # A heat-bath sweep reads every site's label once as a neighbour of each
    # incident edge and writes it once; the bond pass of Swendsen-Wang reads
    # the same two labels per edge. Counting both in one unit is what makes
    # the budget comparable across move sets (issue #551).
    per_sweep = graph.n_nodes + 2 * len(graph.edges)
    visits, trace = 0, []
    for step in range(schedule.n_steps):
        temperature = schedule(step)
        if move is PottsMove.SINGLE_SITE:
            sweep(state, rng, 1.0 / temperature)
            visits += per_sweep
        else:
            counter = ClusterCounter()
            beta = 1.0 / temperature
            if move is PottsMove.SWENDSEN_WANG:
                _swendsen_wang_sweep(state, graph, rows, rng, counter, beta)
                visits += per_sweep
            else:
                _wolff_sweep(state, rows, neighbours, rng, counter, graph, beta)
                # A Wolff step reads each cluster member's neighbours and
                # writes the members; a heat-bath sweep is charged the same
                # way, so one budget covers both.
                visits += sum(counter.sizes) * (
                    1 + 2 * len(graph.edges) // graph.n_nodes
                )
            trace.append(counter)
        energy = float(energies(graph, rows, state[None])[0])
        if energy < best_energy:
            best_state, best_energy = state.copy(), energy
    return AnnealedPotts(
        labelling=best_state,
        energy=best_energy,
        final=state,
        n_sweeps=schedule.n_steps,
        site_visits=visits,
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
    """

    states: np.ndarray
    temperatures: tuple[float, ...]
    swap_acceptance: np.ndarray
    best: np.ndarray
    best_energy: float
    n_sweeps: int


def _swap_log_ratio(
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
    temperatures: tuple[float, ...],
    rng: np.random.Generator,
    n_sweeps: int,
    burn_in: int = 0,
    thin: int = 1,
    *,
    backend: Backend = Backend.PYTHON,
) -> TemperedChains:
    """Replicas at fixed temperatures, exchanging configurations by Metropolis.

    Each replica runs one heat-bath sweep per step at its own temperature,
    then every adjacent pair proposes to exchange configurations and accepts
    on :func:`_swap_log_ratio`. The hot replicas cross barriers the cold one
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
    temperatures : tuple[float, ...]
        The ladder, in any order; at least two, all positive. The stationary
        distribution depends on which pairs are adjacent for exchange, not on
        the order.
    rng : np.random.Generator
        The parent generator: it spawns one child per replica and then draws
        only the exchange uniforms, so one seeded generator reproduces the run.
    n_sweeps, burn_in, thin : int
        As :func:`sample_potts`, applied per replica.
    backend : Backend
        As :func:`anneal_potts`: the oracle sweep by default, the Rust sweep
        on request, each replica on its own child generator either way.

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
    if len(temperatures) < 2:
        msg = (
            f"parallel tempering needs at least two temperatures, got "
            f"{len(temperatures)}: a ladder of one has nothing to exchange"
        )
        raise ValueError(msg)
    for temperature in temperatures:
        if not temperature > 0.0:
            msg = f"every temperature must be positive, got {temperature}"
            raise ValueError(msg)

    rows = site_field(np.asarray(field, dtype=float), graph.n_nodes)
    n_replicas = len(temperatures)
    betas = [1.0 / temperature for temperature in temperatures]
    children = rng.spawn(n_replicas)
    n_states = int(rows.shape[1])
    states = np.stack(
        [child.integers(0, n_states, size=graph.n_nodes) for child in children]
    )
    neighbours = _adjacency(graph)

    recorded = np.empty((n_sweeps, n_replicas, graph.n_nodes), dtype=np.int64)
    proposed = np.zeros(n_replicas - 1)
    accepted = np.zeros(n_replicas - 1)
    current = energies(graph, rows, states)
    best_index = int(np.argmin(current))
    best, best_energy = states[best_index].copy(), float(current[best_index])

    sweep = _sweep_at(graph, rows, neighbours, backend)
    for step in range(-burn_in * thin, n_sweeps * thin):
        for replica in range(n_replicas):
            sweep(states[replica], children[replica], betas[replica])
        current = energies(graph, rows, states)
        for pair in range(n_replicas - 1):
            log_ratio = _swap_log_ratio(
                betas[pair], betas[pair + 1], current[pair], current[pair + 1]
            )
            proposed[pair] += 1
            if log_ratio >= 0.0 or rng.random() < np.exp(log_ratio):
                accepted[pair] += 1
                states[[pair, pair + 1]] = states[[pair + 1, pair]]
                current[[pair, pair + 1]] = current[[pair + 1, pair]]
        lowest = int(np.argmin(current))
        if current[lowest] < best_energy:
            best, best_energy = states[lowest].copy(), float(current[lowest])
        if step >= 0 and (step + 1) % thin == 0:
            recorded[step // thin] = states
    return TemperedChains(
        states=recorded,
        temperatures=tuple(temperatures),
        swap_acceptance=accepted / proposed,
        best=best,
        best_energy=best_energy,
        n_sweeps=n_sweeps,
    )


def adapt_ladder_potts(
    graph: PottsGraph,
    field: np.ndarray,
    ladder: tuple[float, ...],
    rng: np.random.Generator,
    n_sweeps: int,
    band: tuple[float, float],
    max_rounds: int,
    max_replicas: int,
    *,
    backend: Backend = Backend.PYTHON,
) -> AdaptedLadder:
    """A ladder for :func:`parallel_tempering`, from its own exchange acceptances.

    :func:`snakes_and_ladders.opt.schedule.adapt_ladder`, the measurement being
    a :func:`parallel_tempering` run of ``n_sweeps`` per replica on the
    candidate ladder, drawn from ``rng`` in sequence so one seed reproduces the
    warm-up. ``replicas_measured * n_sweeps`` is the warm-up's cost in sweeps,
    which a comparison against a hand ladder at equal budget charges (issue
    #333).

    Parameters
    ----------
    graph, field, rng, backend
        As :func:`parallel_tempering`.
    ladder : tuple[float, ...]
        The starting ladder; its endpoints are kept.
    n_sweeps : int
        Sweeps per replica per measurement. Each acceptance is a fraction of
        ``n_sweeps`` proposals, so this sets what the band can resolve.
    band, max_rounds, max_replicas
        As :func:`snakes_and_ladders.opt.schedule.adapt_ladder`.

    Returns
    -------
    AdaptedLadder
    """

    def measure(candidate: tuple[float, ...]) -> list[float]:
        run = parallel_tempering(
            graph, field, candidate, rng, n_sweeps, backend=backend
        )
        return [float(value) for value in run.swap_acceptance]

    return adapt_ladder(measure, ladder, band, max_rounds, max_replicas)


def energies(graph: PottsGraph, field: np.ndarray, states: np.ndarray) -> np.ndarray:
    """Energy of each configuration, ``E = -log W``.

    Taken from :func:`snakes_and_ladders.likelihood.potts.log_weights` rather than written
    again, so the samplers and the exact evaluators cannot disagree about what
    model they are on.
    """
    return -log_weights(graph, field, states)


def _adjacency(graph: PottsGraph) -> list[list[tuple[int, float]]]:
    """Neighbour lists with the coupling on each incident edge."""
    neighbours: list[list[tuple[int, float]]] = [[] for _ in range(graph.n_nodes)]
    for (first, second), coupling in graph.weighted_edges():
        neighbours[first].append((second, coupling))
        neighbours[second].append((first, coupling))
    return neighbours


def _sweep_at(
    graph: PottsGraph,
    rows: np.ndarray,
    neighbours: list[list[tuple[int, float]]],
    backend: Backend,
) -> Callable[[np.ndarray, np.random.Generator, float], None]:
    """One tempered heat-bath sweep, on the backend the caller named.

    Both closures consume exactly ``n_nodes`` uniforms per sweep from the
    generator they are handed, so switching backend changes which arithmetic
    evaluates the conditional and nothing about the stream. Tempering reaches
    the Rust kernel as the model scaling :func:`tempered` states -- field and
    couplings multiplied by ``beta`` -- the identity the Python sweep applies
    to its local field.
    """
    if backend is Backend.PYTHON:

        def python_sweep(
            state: np.ndarray, rng: np.random.Generator, beta: float
        ) -> None:
            _single_site_sweep(state, rows, neighbours, rng, beta=beta)

        return python_sweep
    if backend is Backend.RUST:
        from snakes_and_ladders import oxi_snakes_and_ladders

        if not bool(np.all(rows == rows[0])):
            msg = (
                "the Rust heat-bath sweep takes one field row shared by every "
                "site; this field varies by site. Run Backend.PYTHON, which is "
                "the oracle either way"
            )
            raise ValueError(msg)
        offsets, neighbour_index, couplings = graph.compressed_adjacency()
        contiguous_field = np.ascontiguousarray(rows[0], dtype=np.float64)

        def rust_sweep(
            state: np.ndarray, rng: np.random.Generator, beta: float
        ) -> None:
            draws = np.ascontiguousarray(rng.random(state.shape[0]), dtype=np.float64)
            oxi_snakes_and_ladders.single_site_sweeps(
                state,
                np.ascontiguousarray(beta * contiguous_field),
                offsets,
                neighbour_index,
                np.ascontiguousarray(beta * couplings),
                draws,
                1,
            )

        return rust_sweep
    msg = f"the heat-bath sweep has no {backend} backend"
    raise ValueError(msg)


def _single_site_sweep(
    state: np.ndarray,
    rows: np.ndarray,
    neighbours: list[list[tuple[int, float]]],
    rng: np.random.Generator,
    beta: float = 1.0,
) -> None:
    """One heat-bath sweep: every site redrawn from its exact conditional.

    The baseline the cluster algorithms are measured against, and the update
    `snakes_and_ladders.sim.potts._simulate_gibbs` uses --- restated for a
    single chain rather than shared, since that one is vectorized across many
    independent chains and this one steps a single chain in time.

    ``rows`` is the field as one row per site, widened by
    :func:`snakes_and_ladders.sim.potts.site_field` at the entry point. A
    shared field reaches here as rows that are all equal, so there is one
    code path rather than two, on `sim/potts.py`'s rule (issue #551).

    ``beta`` tempers the conditional in place, for :func:`anneal_potts`, whose
    temperature changes every sweep and would otherwise rebuild the adjacency
    each time. At 1.0 the multiplication is the identity bitwise.
    """
    draws = np.asarray(rng.random(state.shape[0]))
    for node in range(state.shape[0]):
        local = rows[node].copy()
        for neighbour, coupling in neighbours[node]:
            local[state[neighbour]] += coupling
        local *= beta
        local -= local.max()
        cumulative = np.cumsum(np.exp(local))
        # One uniform and a search, rather than `rng.choice` per site: this
        # is the baseline the cluster algorithms are timed against, so its
        # constant factor decides how large a lattice the comparison reaches.
        state[node] = np.searchsorted(cumulative, float(draws[node]) * cumulative[-1])


def _swendsen_wang_sweep(
    state: np.ndarray,
    graph: PottsGraph,
    rows: np.ndarray,
    rng: np.random.Generator,
    counter: ClusterCounter | None = None,
    beta: float = 1.0,
) -> None:
    """Activate bonds, find clusters, recolour each one.

    Every cluster is recoloured independently, so in a field each needs its own
    accept step --- Wolff flips one cluster and needs one. Hence two code paths
    rather than one rule assumed to cover both.

    ``counter``, when given, records every cluster's size and whether its
    field accept step passed; issue #551 measures the acceptance against
    temperature and that is the quantity it reads.

    ``beta`` tempers the bond probability and the accept step together, the
    model scaling :func:`tempered` states, applied here rather than by
    rebuilding the graph per schedule step. At 1.0 it is the identity.
    """
    first = np.fromiter(
        (edge[0] for edge in graph.edges), dtype=np.int64, count=len(graph.edges)
    )
    second = np.fromiter(
        (edge[1] for edge in graph.edges), dtype=np.int64, count=len(graph.edges)
    )
    coupling = beta * np.asarray(graph.coupling, dtype=float)
    like = state[first] == state[second]
    active = like & (rng.random(len(graph.edges)) < 1.0 - np.exp(-coupling))

    parent = np.arange(graph.n_nodes)
    for edge in np.flatnonzero(active):
        _union(parent, int(first[edge]), int(second[edge]))

    labels = np.array([_find(parent, node) for node in range(graph.n_nodes)])
    for root in np.unique(labels):
        members = np.flatnonzero(labels == root)
        outcome = _recolour(state, members, beta * rows, rng)
        if counter is not None:
            counter.record(members, outcome, graph)


def _wolff_sweep(
    state: np.ndarray,
    rows: np.ndarray,
    neighbours: list[list[tuple[int, float]]],
    rng: np.random.Generator,
    counter: ClusterCounter | None = None,
    graph: PottsGraph | None = None,
    beta: float = 1.0,
) -> int:
    """Grow one cluster from a random seed, recolour it, and stop.

    Exactly one cluster per sweep, and the "exactly" is load-bearing. An earlier
    version ran clusters until their cumulative size reached ``n_nodes``, to
    spend the same budget as the other two move sets. That is a
    *state-dependent* stopping rule: an aligned configuration makes large
    clusters, so it reached the budget in fewer steps and received less
    randomization than a disordered one. Each step is correct, but stopping on
    the outcome biases the composition --- measured on a two-site chain at
    ``J = 0.7``, it put 0.384 on each aligned state against an exact 0.334, and
    the chi-square rejected it.

    One flip per sweep is therefore not one single-site sweep's work, and
    :func:`sample_potts` returns the mean cluster size so a comparison can be
    normalized.

    Returns
    -------
    int
        The size of the cluster this step built.
    """
    seed_node = int(rng.integers(state.shape[0]))
    colour = int(state[seed_node])
    cluster = [seed_node]
    in_cluster = np.zeros(state.shape[0], dtype=bool)
    in_cluster[seed_node] = True
    frontier = [seed_node]
    while frontier:
        node = frontier.pop()
        for neighbour, coupling in neighbours[node]:
            if in_cluster[neighbour] or state[neighbour] != colour:
                continue
            if rng.random() < 1.0 - np.exp(-beta * coupling):
                in_cluster[neighbour] = True
                cluster.append(neighbour)
                frontier.append(neighbour)
    members = np.array(cluster, dtype=np.int64)
    outcome = _recolour(state, members, beta * rows, rng)
    if counter is not None:
        counter.record(members, outcome, graph)
    return len(cluster)


def _recolour(
    state: np.ndarray, members: np.ndarray, rows: np.ndarray, rng: np.random.Generator
) -> Recolour:
    """Propose one colour for a whole cluster, accepting on the field alone.

    The proposal is uniform over every colour including the current one, which
    makes it symmetric and leaves the acceptance ratio as the field term alone.
    Drawing from the ``k - 1`` other colours would mix marginally faster and
    would need the proposal ratio carried through the acceptance.

    The bond construction contributes nothing to the ratio: bonds live only
    between like-coloured sites, and every site in the cluster changes colour
    together, so the cluster is exactly as likely to be built in the proposed
    configuration as in the current one. What does not cancel is the field.
    For a field shared by every site that difference is ``|C| * (h[new] -
    h[old])``; for the per-site field of `potts_spots` it is the sum of that
    difference over the cluster's own members, which the shared case is the
    special case of.

    Returns
    -------
    Recolour
        Whether a colour change was proposed at all, and whether it was
        accepted. Issue #551 reads the acceptance against temperature, and a
        run that never proposed is not a run that was rejected.
    """
    current = int(state[members[0]])
    proposed = int(rng.integers(rows.shape[1]))
    if proposed == current:
        return Recolour(proposed=False, accepted=False)
    difference = float(rows[members, proposed].sum() - rows[members, current].sum())
    if difference >= 0.0 or rng.random() < np.exp(difference):
        state[members] = proposed
        return Recolour(proposed=True, accepted=True)
    return Recolour(proposed=True, accepted=False)


def _find(parent: np.ndarray, node: int) -> int:
    """Union-find root, with path compression."""
    root = node
    while parent[root] != root:
        root = int(parent[root])
    while parent[node] != root:
        parent[node], node = root, int(parent[node])
    return root


def _union(parent: np.ndarray, first: int, second: int) -> None:
    """Merge two components."""
    first_root, second_root = _find(parent, first), _find(parent, second)
    if first_root != second_root:
        parent[second_root] = first_root
