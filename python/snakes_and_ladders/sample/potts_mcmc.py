"""Monte Carlo move sets on a Potts lattice: single-site, cluster, gradient-informed.

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

**Two of the six move sets are gradient-informed, and on this energy they are
one kernel.** A locally balanced proposal (Zanella 2020) weights every
single-site change by ``sqrt(pi(s') / pi(s))``; Gibbs-with-gradients
(Grathwohl et al. 2021) weights it by the same function of the *first-order
Taylor estimate* of that ratio at the current one-hot state. On a pairwise
energy the estimate is the ratio --- the relaxed log weight is affine in each
site's row, so a single-site change has no second-order term ---
so the two propose from the same law and differ in what they compute to get
there. That is a property of the Potts energy and not of the implementation,
which is why it is pinned by a test rather than assumed by a shared branch:
:func:`taylor_log_ratios` is the estimate, :func:`autodiff_log_ratios` is the
same quantity from the tape, and the three agree to ``1e-12``.

**Two of the six run where the Fortuin-Kasteleyn construction cannot.** Its
bond probability ``1 - exp(-J)`` is not a probability below zero, so Wolff and
Swendsen-Wang are refused on an antiferromagnet --- the instance a cluster move
is wanted for. :func:`niedermayer_sweep` activates a bond on its energy
relative to a threshold ``E_0`` instead (Niedermayer 1988), which is a
probability for either sign and is Wolff's own where Wolff runs;
:func:`sample_potts_pair` runs two replicas at one temperature and moves them
by Houdayer's isoenergetic cluster swap (2001), whose acceptance is 1 by an
identity rather than by a construction. Neither is a free lunch and the
package does not report one: on the frustrated triangular lattice both
clusters percolate, which
``docs/experiments/022-cluster-moves-for-frustrated-lattices.md`` measures.

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
likelihood is a different object (`snakes_and_ladders.sample.schedule` says why);
here the objective is an energy and the temperature is physical.

See ``docs/tex/textbook.tex``, ``sec:potts`` (Newman &
Barkema chs. 4 and 6 for both algorithms and for Sokal's windowing; Mezard &
Montanari ch. 2).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from enum import StrEnum
from functools import partial
from typing import Any, NamedTuple

import numpy as np

from snakes_and_ladders.backend import Backend
from snakes_and_ladders.sample.accept import accept, accept_at, accept_drawn
from snakes_and_ladders.sample.balanced import (
    draw_change,
    log_balanced_weights,
    log_metropolis_ratio,
    log_normalizer,
    log_ratios,
)
from snakes_and_ladders.sample.schedule import (
    AdaptedLadder,
    TempSchedule,
    adapt_ladder,
    check_ladder,
    ladder,
)
from snakes_and_ladders.sim.graph import PottsGraph
from snakes_and_ladders.sim.potts import (
    energies,
    heat_bath_log_weights,
    local_fields,
    owner_rows,
    site_field,
)

# `current` is aliased: `parallel_tempering` already binds that name to the
# replicas' energies, and one of the two has to give.
from snakes_and_ladders.track import TrackedOptimization
from snakes_and_ladders.track import current as current_tracked

#: Declared so :func:`snakes_and_ladders.sim.potts.energies` re-exports from
#: this module, which `mypy --strict` otherwise refuses: the energy moved to
#: `sim/` in issue #277 so the simulator could score with it, and every
#: caller that had it from here still does.
__all__ = [
    "AnnealedPotts",
    "ClusterCounter",
    "Clusters",
    "MoveKind",
    "PottsChain",
    "PottsMove",
    "PottsPair",
    "Recolour",
    "TemperedChains",
    "TemperedModel",
    "adapt_ladder_potts",
    "anneal_potts",
    "autodiff_log_ratios",
    "cluster_members",
    "energies",
    "find_root",
    "houdayer_cluster",
    "houdayer_move",
    "niedermayer_sweep",
    "niedermayer_threshold",
    "parallel_tempering",
    "refuse_negative_coupling",
    "sample_potts",
    "sample_potts_pair",
    "swap_log_ratio",
    "sweep_for",
    "swendsen_wang_sweep",
    "taylor_log_ratios",
    "tempered",
    "union_roots",
    "wolff_sweep",
]


#: How far from a cumulative boundary a draw must land for the Rust sweep to
#: decide a site itself, in units of the last place per state. NumPy's ``exp``
#: and ``libm``'s differ by at most one such unit, the cumulative sum carries
#: that difference across at most ``n_states`` additions, and scaling the draw
#: by the last entry carries it once more: four units per state bounds it, and
#: this is four times that. The same width, on the same derivation, as
#: `gibbs._GUARD` (issues #561, #599); two consumers, so it is stated in each
#: rather than made a seam.
_GUARD = 16.0


class PottsMove(StrEnum):
    """Which Monte Carlo move set a chain proposes from.

    A ``StrEnum`` for the reason `snakes_and_ladders.search.infer.MoveSet` is one: an
    unrecognized move is rejected by ``mypy --strict`` at the call site rather
    than by a branch that silently falls through to a default.
    """

    SINGLE_SITE = "single-site"
    SWENDSEN_WANG = "swendsen-wang"
    WOLFF = "wolff"
    LOCALLY_BALANCED = "locally-balanced"
    GIBBS_WITH_GRADIENTS = "gibbs-with-gradients"
    NIEDERMAYER = "niedermayer"


#: The move sets built on the Fortuin-Kasteleyn bond construction, which needs
#: every coupling non-negative. Named rather than written as "not single-site":
#: the gradient-informed moves are single-flip and run on an antiferromagnet,
#: and a negation would have refused them with the clusters. Niedermayer's
#: rule builds clusters on a coupling of either sign and is not in the set,
#: which is the whole reason issue #756 adds it.
_CLUSTER_MOVES = frozenset({PottsMove.SWENDSEN_WANG, PottsMove.WOLFF})

#: The move sets :func:`_balanced_sweep_at` serves.
_BALANCED_MOVES = frozenset(
    {PottsMove.LOCALLY_BALANCED, PottsMove.GIBBS_WITH_GRADIENTS}
)


class MoveKind(StrEnum):
    """Which move a learned action applies, over the move set :class:`PottsMove` names.

    Issue #779 moved it here from ``snakes_and_ladders.learn.potts_nd``, which
    spelled it out rather than importing it. It sits beside the moves it names
    so one vocabulary is defined once: ``SWEEP`` is a pass of
    :data:`PottsMove.SINGLE_SITE`, ``FLIP`` is one site of that pass, and
    ``WOLFF`` and ``SWENDSEN_WANG`` are the two cluster moves under their own
    names.
    """

    FLIP = "flip"
    SWEEP = "sweep"
    WOLFF = "wolff"
    SWENDSEN_WANG = "swendsen-wang"
    NIEDERMAYER = "niedermayer"


def refuse_negative_coupling(move: PottsMove, graph: PottsGraph) -> None:
    """Refuse a Fortuin-Kasteleyn cluster move on a graph with a negative coupling.

    One message for every entry point that runs one, rather than a copy of
    it apiece: the bond probability ``1 - exp(-J)`` is not a probability
    below zero, and an antiferromagnet has no like-spin clusters to flip.
    Niedermayer's rule is not in :data:`_CLUSTER_MOVES` and is the move for
    that case (issue #756).
    """
    if move in _CLUSTER_MOVES and min(graph.coupling, default=0.0) < 0.0:
        msg = (
            f"{move} needs every coupling >= 0: the bond probability "
            "1 - exp(-J) is not a probability for J < 0, and an "
            "antiferromagnet has no like-spin clusters to flip"
        )
        raise ValueError(msg)


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
    field: np.ndarray,
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
    field : np.ndarray
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
        :data:`~snakes_and_ladders.backend.Backend.RUST` the default
        (:func:`_sweep_at`, issue #599).
    cluster_backend : Backend
        Which implementation runs the **Swendsen-Wang** pass; the Wolff move
        has one and ignores it. A separate argument rather than the one
        above because the two are not the same decision:
        :data:`~snakes_and_ladders.backend.Backend.PYTHON` is the default
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
    schedule: TempSchedule,
    rng: np.random.Generator,
    *,
    move: PottsMove = PottsMove.SINGLE_SITE,
    backend: Backend = Backend.RUST,
    cluster_backend: Backend = Backend.PYTHON,
) -> AnnealedPotts:
    """Simulated annealing by heat-bath sweeps on a temperature schedule.

    One :func:`_single_site_sweep` per schedule step at that step's
    temperature, tracking the lowest energy seen (Kirkpatrick, Gelatt & Vecchi,
    1983). It is :func:`iterated_conditional_modes` at finite temperature: at
    ``T -> 0`` the heat bath is the argmin over each site's conditional, ICM's
    update, so the two are one search separated by the schedule and a
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
    field : np.ndarray
        External field, shape ``(n_states,)``.
    schedule : TempSchedule
        Temperature per sweep. Its length is the budget.
    rng : np.random.Generator
        Source of every draw, the start included. Passed in rather than
        seeded here, for the reason :func:`sample_potts` gives.

    backend : Backend
        :data:`~snakes_and_ladders.backend.Backend.RUST` runs the
        extension's sweep and is the default;
        :data:`~snakes_and_ladders.backend.Backend.PYTHON` runs the
        oracle that pins it. The two produce the same chain state for state,
        on the same uniforms in the same order (:func:`_sweep_at`).
    cluster_backend : Backend
        Which implementation runs the Swendsen-Wang pass.
        :data:`~snakes_and_ladders.backend.Backend.PYTHON`, the default,
        records every cluster's accept step in ``trace``;
        :data:`~snakes_and_ladders.backend.Backend.RUST` is a chain of the
        same law on another order of draws (:func:`_cluster_pass_rust`) and
        keeps no counter, so its steps leave ``trace`` empty (issue #923).

    Returns
    -------
    AnnealedPotts
    """
    refuse_negative_coupling(move, graph)

    rows = site_field(np.asarray(field, dtype=float), graph.n_nodes)
    state = np.ascontiguousarray(
        rng.integers(0, int(rows.shape[1]), size=graph.n_nodes), dtype=np.int64
    )
    offsets, neighbours, couplings = graph.compressed_adjacency()

    best_state = state.copy()
    best_energy = float(energies(graph, rows, state[None])[0])
    sweep = (
        _balanced_sweep_at(rows, offsets, neighbours, couplings, move)
        if move in _BALANCED_MOVES
        else _sweep_at(rows, offsets, neighbours, couplings, backend)
    )
    # A heat-bath sweep reads every site's label once as a neighbour of each
    # incident edge and writes it once; the bond pass of Swendsen-Wang reads
    # the same two labels per edge. Counting both in one unit is what makes
    # the budget comparable across move sets (issue #551).
    per_sweep = graph.n_nodes + 2 * len(graph.edges)
    visits, trace = 0, []
    # One lookup for the run (`snakes_and_ladders.track`) and one `record` a
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
    walkers : np.ndarray
        ``walkers[t, w]`` is the rung walker ``w`` sat at, at recorded sweep
        ``t``, shape ``(n_sweeps, n_replicas)``. The other reading of the
        same run: a *replica* is a temperature configurations pass through,
        where a *walker* is a configuration followed through the swaps, and
        a round trip is a statement about the second.
        :func:`snakes_and_ladders.sample.tempered.round_trips` and
        :func:`snakes_and_ladders.sample.tempered.up_fraction` read it, an
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
    # Which walker sits at each rung, as `sample.tempered._exchange` tracks
    # it: a swap moves configurations between temperatures, so this is what
    # says a configuration crossed the ladder.
    at_rung = list(range(n_replicas))
    trace = np.empty((n_sweeps, n_replicas), dtype=np.int64)
    proposed = np.zeros(n_replicas - 1)
    accepted = np.zeros(n_replicas - 1)
    current = energies(graph, rows, states)
    best_index = int(np.argmin(current))
    best, best_energy = states[best_index].copy(), float(current[best_index])

    sweep = _sweep_at(rows, offsets, neighbours, couplings, backend)
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

    :func:`snakes_and_ladders.sample.schedule.adapt_ladder`, the measurement being
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
        :func:`~snakes_and_ladders.sample.schedule.ladder` into the same
        floats; its endpoints are kept.
    n_sweeps : int
        Sweeps per replica per measurement. Each acceptance is a fraction of
        ``n_sweeps`` proposals, so this sets what the band can resolve.
    band, max_rounds, max_replicas
        As :func:`snakes_and_ladders.sample.schedule.adapt_ladder`.

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
            _sweep_at(rows, offsets, neighbours, couplings, backend)
            if move is PottsMove.SINGLE_SITE
            else _balanced_sweep_at(rows, offsets, neighbours, couplings, move)
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
            )

        return generalized

    def one_cluster(
        state: np.ndarray, rng: np.random.Generator, beta: float = 1.0
    ) -> int:
        return wolff_sweep(state, rows, offsets, neighbours, couplings, rng, beta=beta)

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
    graph, field, move, rng, n_sweeps, burn_in, thin, temperature, backend
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
    advance = sweep_for(move, graph, rows, offsets, neighbours, couplings, backend)

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


def _sweep_at(
    rows: np.ndarray,
    offsets: np.ndarray,
    neighbours: np.ndarray,
    couplings: np.ndarray,
    backend: Backend,
) -> Callable[[np.ndarray, np.random.Generator, float], None]:
    """One tempered heat-bath sweep, on the backend the caller named.

    Both closures consume exactly ``n_nodes`` uniforms per sweep from the
    generator they are handed, so switching backend changes which arithmetic
    evaluates the conditional and nothing about the stream. Tempering reaches
    the Rust kernel as ``beta`` itself, applied to the accumulated local field
    where the Python sweep applies it, so the two agree bitwise at every
    temperature rather than only at 1.0 (issue #571).

    **The two produce the same chain, not a chain of the same law**, which is
    why :data:`~snakes_and_ladders.backend.Backend.RUST` is the default
    (issue #599). Every step of the conditional is the arithmetic NumPy
    performs, operation for operation, but ``exp`` is not: NumPy computes it
    by its own SIMD polynomial and the kernel by ``libm``, and one draw across
    a boundary that moved in the last place sends two chains apart. So the
    kernel decides a site only where the draw clears every cumulative boundary
    by :data:`_GUARD` units of the last place per state, returns the position
    of the first site it declines, and :func:`_site_update` --- the oracle's
    own update --- decides that one before the kernel resumes.

    The adjacency arrives as the compressed rows both backends read, built
    once by the caller: the Python sweep indexes them and the kernel takes
    them across the boundary without marshalling (issue #277).
    """
    if backend is Backend.PYTHON:

        def python_sweep(
            state: np.ndarray, rng: np.random.Generator, beta: float
        ) -> None:
            _single_site_sweep(
                state, rows, offsets, neighbours, couplings, rng, beta=beta
            )

        return python_sweep
    if backend is Backend.RUST:
        from snakes_and_ladders import oxi_snakes_and_ladders

        # The field crosses as one row per site, which is the shape `rows`
        # already has: `sim.potts.site_field` widened it at the entry point,
        # so a shared field is rows that are all equal and there is one code
        # path rather than two (issue #551, #571). The adjacency is the
        # caller's, built once (issue #277); both arrays are made contiguous
        # here rather than per sweep.
        contiguous_field = np.ascontiguousarray(rows, dtype=np.float64)
        contiguous_couplings = np.ascontiguousarray(couplings, dtype=np.float64)
        # The lists `_site_update` indexes on a hand-back, converted once per
        # run rather than per sweep: the kernel hands a site back so rarely
        # that a per-sweep conversion would cost more than the sweep.
        bounds = offsets.tolist()
        incident, weights = neighbours.tolist(), contiguous_couplings.tolist()

        def rust_sweep(
            state: np.ndarray, rng: np.random.Generator, beta: float
        ) -> None:
            n_nodes = state.shape[0]
            draws = np.ascontiguousarray(rng.random(n_nodes), dtype=np.float64)
            node = 0
            while node < n_nodes:
                # `beta` is passed rather than multiplied into the arguments.
                # The Python sweep scales the accumulated local field, so
                # scaling the parts instead computes `beta * h + sum (beta *
                # J)` against its `(h + sum J) * beta` -- equal in real
                # arithmetic, not bitwise, which cost agreement at every
                # temperature but 1.0 (issue #571). It also drops two
                # whole-array temporaries per sweep, so issue #651 re-opened it
                # under `CLAUDE.md`'s rule that bitwise may back off to a
                # declared tolerance.
                #
                # **Measured and refused.** The reassociation is not a
                # last-place difference: `h + sum J` cancels, so the
                # scaled-parts form's absolute error is set by the magnitudes
                # of the parts while the result is near zero. Over 50,000
                # sites the median is 1.00 unit of the last place and the
                # maximum is 201,145, with 2.9% past `_GUARD` -- and those are
                # the cancelling sites, whose accumulated field is a median
                # 7.4e-02 against 2.8e+00 for the rest. A near-zero field is a
                # near-uniform conditional, so the error concentrates on the
                # sites whose decision it is likeliest to flip. Neither the
                # guard nor a relative tolerance reaches it.
                # `tests/regression/search/test_potts_sweep_reassociation.py`
                # holds those numbers.
                node = oxi_snakes_and_ladders.single_site_sweeps(
                    state,
                    contiguous_field,
                    offsets,
                    neighbours,
                    contiguous_couplings,
                    draws,
                    1,
                    beta,
                    _GUARD,
                    node,
                )
                if node < n_nodes:
                    _site_update(
                        state,
                        rows,
                        incident,
                        weights,
                        bounds,
                        node,
                        float(draws[node]),
                        beta,
                    )
                    node += 1

        return rust_sweep
    msg = f"the heat-bath sweep has no {backend} backend"
    raise ValueError(msg)


def _single_site_sweep(
    state: np.ndarray,
    rows: np.ndarray,
    offsets: np.ndarray,
    neighbours: np.ndarray,
    couplings: np.ndarray,
    rng: np.random.Generator,
    beta: float = 1.0,
) -> None:
    """One heat-bath sweep: every site redrawn from its exact conditional.

    The baseline the cluster algorithms are measured against. The conditional
    is :func:`snakes_and_ladders.sim.potts.heat_bath_log_weights`, shared with
    the vectorized simulator; the loop is not, since that one runs many
    independent chains at once and this one steps a single chain in time
    (issue #277).

    ``rows`` is the field as one row per site, widened by
    :func:`snakes_and_ladders.sim.potts.site_field` at the entry point. A
    shared field reaches here as rows that are all equal, so there is one
    code path rather than two, on `sim/potts.py`'s rule (issue #551).

    ``offsets``, ``neighbours`` and ``couplings`` are the graph's compressed
    rows: site ``i``'s neighbours are ``neighbours[offsets[i]:offsets[i + 1]]``
    and their couplings sit at the same positions.

    ``beta`` tempers the conditional in place, for :func:`anneal_potts`, whose
    temperature changes every sweep and would otherwise rebuild the adjacency
    each time. At 1.0 the multiplication is the identity bitwise, and is
    skipped.
    """
    draws = np.asarray(rng.random(state.shape[0]))
    bounds = offsets.tolist()
    incident, weights = neighbours.tolist(), couplings.tolist()
    for node in range(state.shape[0]):
        _site_update(
            state, rows, incident, weights, bounds, node, float(draws[node]), beta
        )


def _site_update(
    state: np.ndarray,
    rows: np.ndarray,
    incident: Sequence[int],
    weights: Sequence[float],
    bounds: Sequence[int],
    node: int,
    draw: float,
    beta: float,
) -> None:
    """One site redrawn from its exact conditional, in place.

    The whole of the oracle's update, factored out so the Rust sweep's
    hand-back path decides its site by calling this rather than a copy of it
    (issue #599). ``incident``, ``weights`` and ``bounds`` are the compressed
    rows as the lists `sim.potts.heat_bath_log_weights` measured as cheaper to
    index than array rows.

    One uniform and a search, rather than ``rng.choice`` per site: this is the
    baseline the cluster algorithms are timed against, so its constant factor
    decides how large a lattice the comparison reaches.
    """
    local = heat_bath_log_weights(
        rows[node], state, incident, weights, bounds[node], bounds[node + 1], beta
    )
    local -= local.max()
    cumulative = np.cumsum(np.exp(local))
    state[node] = np.searchsorted(cumulative, draw * cumulative[-1])


@dataclass(frozen=True)
class Clusters:
    """A labelling grouped into its clusters, as one sort rather than a scan.

    Parameters
    ----------
    order : np.ndarray
        The nodes, sorted by label and by index within a label.
    bounds : np.ndarray
        Where each cluster starts in ``order``, with ``labels.size`` last, so
        cluster ``k`` is ``order[bounds[k]:bounds[k + 1]]``.
    """

    order: np.ndarray
    bounds: np.ndarray

    def __iter__(self) -> Iterator[Any]:
        """``(order, bounds)``: the order callers unpack.

        ``Any`` for :meth:`TemperedModel.__iter__`'s reason.
        """
        yield from (self.order, self.bounds)


def cluster_members(labels: np.ndarray) -> Clusters:
    """Group a labelling into its clusters: the members, and where each starts.

    ``labels[order[bounds[k]:bounds[k + 1]]]`` is the ``k``-th root in
    increasing order, and the slice of ``order`` is that cluster's members in
    increasing node order --- which is exactly what
    ``np.flatnonzero(labels == root)`` returns for ``root`` walked over
    ``np.unique(labels)``, since a stable sort keeps equal keys in index
    order. So this changes the grouping's cost and not one member of one
    cluster.

    The cost is the point. The form it replaces is ``O(n_clusters * n_nodes)``
    --- one full comparison of the labelling per cluster --- and the bond pass
    at the transition makes a cluster for every 1.7 sites, so the quadratic
    term *is* the pass: at 64x64 it was 10.5 ms of a 41.2 ms
    ``SwendsenWangMove.propose`` against 0.5 ms for this one sort (#754).
    Root ``CLAUDE.md``'s rule that an algorithmic cut outranks a mechanical
    one, taken before the port below rather than ported around.
    """
    order = np.argsort(labels, kind="stable")
    sorted_labels = labels[order]
    starts = np.flatnonzero(
        np.concatenate(([True], sorted_labels[1:] != sorted_labels[:-1]))
    )
    return Clusters(order=order, bounds=np.concatenate((starts, [labels.size])))


def taylor_log_ratios(
    rows: np.ndarray,
    state: np.ndarray,
    neighbours: np.ndarray,
    couplings: np.ndarray,
    owner: np.ndarray,
    beta: float = 1.0,
) -> np.ndarray:
    """Gibbs-with-gradients' first-order estimate of every single-flip change.

    Grathwohl et al. (2021) relax the state to the simplex --- the one-hot
    matrix `learn.relaxed.one_hot` writes --- and estimate
    ``log pi(s') - log pi(s)`` by ``grad(log pi)(x) . (x' - x)``, one gradient
    for the whole neighbourhood against one energy per neighbour.

    **On a Potts energy the estimate is exact.** The relaxed log weight
    ``sum_i h_i . x_i + sum_(ij) J_ij x_i . x_j`` is affine in each site's row
    --- a lattice has no self-coupling, so no term carries ``x_i`` twice ---
    and a single-flip change moves one row, so the first-order term is the
    whole difference. The gradient at a one-hot is then
    :func:`snakes_and_ladders.sim.potts.local_fields`, which is what this
    computes: the tape returns the same numbers
    (:func:`autodiff_log_ratios`, pinned at ``1e-12``) for a tape's cost per
    proposal. Where the estimate is exact this equals
    :func:`~snakes_and_ladders.sample.balanced.log_ratios`, and the two stay
    separate functions because that equality is a property of *this* energy
    and is pinned rather than assumed.

    Parameters
    ----------
    rows, state, neighbours, couplings, owner, beta
        As :func:`snakes_and_ladders.sim.potts.local_fields`.

    Returns
    -------
    np.ndarray
        Shape ``(n_nodes, n_states)``.
    """
    return log_ratios(
        local_fields(rows, state, neighbours, couplings, owner, beta), state
    )


def autodiff_log_ratios(
    graph: PottsGraph, rows: np.ndarray, state: np.ndarray, beta: float = 1.0
) -> np.ndarray:
    """:func:`taylor_log_ratios` from the tape: the definition, not the route.

    The relaxed log weight is built in ``torch`` on the one-hot state and
    differentiated by ``torch.autograd.grad``, which is what Gibbs-with-
    gradients *is*. It is the oracle rather than the sampler's route: the
    closed form :func:`taylor_log_ratios` takes reproduces it to ``1e-12``
    (`tests/regression/search/test_potts_mcmc.py`) and costs one ``bincount``
    against a tape built and walked per proposal.

    ``torch`` is imported here rather than at module scope: it is the
    heaviest import in the package and no chain this module runs needs it.

    Parameters
    ----------
    graph : PottsGraph
        The lattice, read for its edge list and per-edge couplings.
    rows : np.ndarray
        The field as one row per site, shape ``(n_nodes, n_states)``.
    state : np.ndarray
        The current configuration.
    beta : float
        Inverse temperature, scaling the whole log weight.

    Returns
    -------
    np.ndarray
        Shape ``(n_nodes, n_states)``.
    """
    import torch

    # `torch.tensor` rather than `from_numpy`: a graph hands out read-only
    # views of its arrays (#623), which `from_numpy` takes with a warning
    # about undefined behaviour on write. These are read and never written.
    labels = np.asarray(state, dtype=np.int64)
    probabilities = torch.zeros(rows.shape, dtype=torch.float64)
    probabilities[torch.arange(rows.shape[0]), torch.tensor(labels)] = 1.0
    probabilities.requires_grad_(True)

    value = (probabilities * torch.tensor(rows, dtype=torch.float64)).sum()
    if graph.edges:
        ends = graph.edge_index
        first = probabilities[torch.tensor(ends[:, 0], dtype=torch.long)]
        second = probabilities[torch.tensor(ends[:, 1], dtype=torch.long)]
        coupling = torch.tensor(graph.edge_coupling, dtype=torch.float64)
        value = value + (coupling * (first * second).sum(dim=1)).sum()
    (gradient,) = torch.autograd.grad(beta * value, probabilities)
    return log_ratios(gradient.detach().numpy(), labels)


def _balanced_sweep_at(
    rows: np.ndarray,
    offsets: np.ndarray,
    neighbours: np.ndarray,
    couplings: np.ndarray,
    move: PottsMove,
) -> Callable[[np.ndarray, np.random.Generator, float], None]:
    """``n_nodes`` locally balanced proposals per sweep, each Metropolis-corrected.

    The move set Zanella (2020) defines and Grathwohl et al. (2021) take the
    gradient form of; :mod:`snakes_and_ladders.sample.balanced` holds the
    kernel both share and derives the correction. A sweep is ``n_nodes``
    proposals, the heat bath's sweep size, so an autocorrelation time in
    sweeps compares the two without a normalization.

    **The conditional is maintained, not rebuilt.** Every proposal reads every
    site's conditional, and rebuilding it from the adjacency per proposal
    would make a sweep quadratic in the lattice for no new information: a flip
    at ``i`` moves only the rows of ``i``'s neighbours, by the incident
    coupling. Those rows are copied before the flip and restored on a
    rejection, so a rejected proposal leaves the array it found rather than a
    value that has been added to and subtracted from. The array is rebuilt
    once per sweep, which is also what lets ``beta`` change between sweeps
    (:func:`anneal_potts`).

    The two move sets differ in one expression --- which estimate of the
    single-flip differences weights the neighbourhood --- and share the accept
    step, which takes the *exact* difference whichever proposed it.
    """
    owner = owner_rows(offsets)
    bounds = offsets.tolist()
    incident, weights = neighbours.tolist(), couplings.tolist()
    gradient_informed = move is PottsMove.GIBBS_WITH_GRADIENTS

    def sweep(state: np.ndarray, rng: np.random.Generator, beta: float) -> None:
        local = local_fields(rows, state, neighbours, couplings, owner, beta)
        for _ in range(state.shape[0]):
            exact = log_ratios(local, state)
            estimate = (
                taylor_log_ratios(rows, state, neighbours, couplings, owner, beta)
                if gradient_informed
                else exact
            )
            forward_weights = log_balanced_weights(estimate, state)
            forward_total = log_normalizer(forward_weights)
            change = draw_change(forward_weights, forward_total, rng)
            node, colour = change.variable, change.value

            previous = int(state[node])
            log_ratio = float(exact[node, colour])
            forward = float(forward_weights[node, colour])

            # Advanced indexing already copies, so this is the restore buffer
            # and not a view of the rows about to change.
            touched = incident[bounds[node] : bounds[node + 1]]
            restored = local[touched]
            _apply_flip(local, state, node, colour, incident, weights, bounds, beta)
            reverse_estimate = (
                taylor_log_ratios(rows, state, neighbours, couplings, owner, beta)
                if gradient_informed
                else log_ratios(local, state)
            )
            reverse_weights = log_balanced_weights(reverse_estimate, state)
            reverse_total = log_normalizer(reverse_weights)
            reverse = float(reverse_weights[node, previous])

            log_alpha = log_metropolis_ratio(
                log_ratio, forward, forward_total, reverse, reverse_total
            )
            if not accept(log_alpha, rng):
                local[touched] = restored
                state[node] = previous

    return sweep


def _apply_flip(
    local: np.ndarray,
    state: np.ndarray,
    node: int,
    colour: int,
    incident: Sequence[int],
    weights: Sequence[float],
    bounds: Sequence[int],
    beta: float,
) -> None:
    """Relabel one site and carry the change into its neighbours' conditionals.

    The site's own row does not move: its conditional is built from its
    neighbours' labels and its own field, neither of which this touches.
    """
    previous = int(state[node])
    for position in range(bounds[node], bounds[node + 1]):
        neighbour, coupling = incident[position], beta * weights[position]
        local[neighbour, previous] -= coupling
        local[neighbour, colour] += coupling
    state[node] = colour


def swendsen_wang_sweep(
    state: np.ndarray,
    graph: PottsGraph,
    rows: np.ndarray,
    rng: np.random.Generator,
    counter: ClusterCounter | None = None,
    beta: float = 1.0,
    backend: Backend = Backend.PYTHON,
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

    ``backend`` names which implementation runs the pass.
    :data:`~snakes_and_ladders.backend.Backend.PYTHON` is this one, the
    oracle, and is the default for the reason :func:`_cluster_pass_rust`
    states: the two draw the same uniforms in a different order, so the Rust
    pass is a chain of the same law and not the same chain. A ``counter`` is
    refused on the Rust route rather than silently ignored --- the
    instrumentation reads each cluster's members, which is the gather the
    port removes.

    The edge ends are :attr:`~snakes_and_ladders.sim.graph.PottsGraph.edge_index`'s
    rather than two ``np.fromiter`` passes over ``graph.edges``: the graph has
    held the array form since #623, and rebuilding it per sweep was 0.900 ms
    of a 42.2 ms pass against 0.001 ms to read the store (#754). Recompute or
    store, decided as store, and the same ``int64`` indices either way.
    """
    if backend is Backend.RUST:
        if counter is not None:
            msg = (
                "the Rust cluster pass takes no counter: the instrumentation "
                "reads every cluster's members, which is the gather the port "
                "removes (issue #551, #754)"
            )
            raise ValueError(msg)
        _cluster_pass_rust(state, graph, rows, rng, beta)
        return
    if backend is not Backend.PYTHON:
        msg = f"the Swendsen-Wang pass has no {backend} backend"
        raise ValueError(msg)

    first, second = graph.edge_index[:, 0], graph.edge_index[:, 1]
    like = state[first] == state[second]
    active = like & (rng.random(len(graph.edges)) < _bond_probability(graph, beta))

    parent = np.arange(graph.n_nodes)
    for edge in np.flatnonzero(active):
        union_roots(parent, int(first[edge]), int(second[edge]))

    labels = np.array([find_root(parent, node) for node in range(graph.n_nodes)])
    # Scaled once rather than per cluster: the multiply is over the whole
    # field and there are as many clusters as sites at the transition, which
    # made it 10.5 ms of the same 41.2 ms pass. Every entry is the value the
    # per-cluster form produced, so no recolouring moves.
    scaled = beta * rows
    clusters = cluster_members(labels)
    # Bound once: the pass walks a cluster per 1.7 sites, and the lookups
    # would be paid per cluster (#754).
    order, bounds = clusters.order, clusters.bounds
    for cluster in range(bounds.size - 1):
        members = order[bounds[cluster] : bounds[cluster + 1]]
        outcome = _recolour(state, members, scaled, rng)
        if counter is not None:
            counter.record(members, outcome, graph)


def _bond_probability(graph: PottsGraph, beta: float) -> np.ndarray:
    """``1 - exp(-beta J)`` per edge, in the graph's edge order.

    Factored out because both routes evaluate it and only one may: ``exp`` is
    a threshold the bond draw is compared against, so a second evaluation in
    Rust would decide an edge differently in the last place. The kernel takes
    this array and compares against it, which leaves the bond pass the
    oracle's arithmetic exactly (#754).
    """
    return np.asarray(1.0 - np.exp(-(beta * graph.edge_coupling)))


def _cluster_pass_rust(
    state: np.ndarray,
    graph: PottsGraph,
    rows: np.ndarray,
    rng: np.random.Generator,
    beta: float,
) -> None:
    """:func:`swendsen_wang_sweep`'s pass, on the extension.

    **The same law, in a different order of draws.** The oracle draws a
    cluster's colour and then, only where the field difference is negative,
    its accept uniform --- a lazy stream no array can replay, since what the
    next draw *is* depends on the last one's outcome. So this route draws the
    bond uniforms the oracle draws (one array, the same call), and then one
    colour and one uniform per cluster in bulk, each independent and
    identically distributed as the oracle's own: the chain is of the same law
    and is not the same chain, which is why
    :data:`~snakes_and_ladders.backend.Backend.PYTHON` stays the default and
    why the pin is the enumerated law rather than the oracle's stream
    (`tests/regression/search/test_potts_mcmc_cluster_rust.py`).

    **Given the same draws it is the oracle bitwise, by construction.** The
    bond probability and the scaled field cross as arrays NumPy evaluated, so
    the only arithmetic the kernel adds is a cluster's field sum and ``exp``
    of the difference. Both are thresholds, and both are guarded by
    :data:`_GUARD`: a cluster whose decision sits inside the width two
    summation orders and two ``exp`` implementations can move it is handed
    back, and decided here by :func:`_recolour_drawn` --- the oracle's own
    recolouring --- before the kernel resumes. The construction is
    :func:`_sweep_at`'s, per cluster rather than per site.

    One crossing per call, and the boundary carries eight contiguous arrays:
    state and labels out, the flattened edge ends, the bond probabilities and
    their draws, the colour and accept draws, and the scaled field.
    """
    from snakes_and_ladders import oxi_snakes_and_ladders

    n_nodes, n_states = graph.n_nodes, int(rows.shape[1])
    scaled = np.ascontiguousarray(beta * rows, dtype=np.float64)
    edges = np.ascontiguousarray(graph.edge_index, dtype=np.int64).reshape(-1)
    probability = np.ascontiguousarray(_bond_probability(graph, beta), dtype=np.float64)
    bond_draws = np.ascontiguousarray(rng.random(len(graph.edges)), dtype=np.float64)
    # One per cluster, indexed by the cluster's rank in increasing root order.
    # A pass builds at most one cluster per site, so the site count is the
    # bound the caller can know before the bond pass --- and learning the
    # real count would cost a second crossing.
    colour_draws = np.ascontiguousarray(
        rng.integers(0, n_states, size=n_nodes), dtype=np.int64
    )
    accept_draws = np.ascontiguousarray(rng.random(n_nodes), dtype=np.float64)
    labels = np.empty(n_nodes, dtype=np.int64)

    cluster = 0
    while True:
        n_clusters, cluster = oxi_snakes_and_ladders.swendsen_wang_sweep(
            state,
            scaled,
            edges,
            probability,
            bond_draws,
            colour_draws,
            accept_draws,
            labels,
            _GUARD,
            cluster,
        )
        if cluster >= n_clusters:
            return
        clusters = cluster_members(labels)
        order, bounds = clusters.order, clusters.bounds
        members = order[bounds[cluster] : bounds[cluster + 1]]
        # The draw behind a call rather than as a value, for the reason
        # `_recolour_drawn` gives: it is read only where the field difference
        # is negative, and `partial` is what says so without a closure over
        # the loop.
        _recolour_drawn(
            state,
            members,
            scaled,
            int(colour_draws[cluster]),
            partial(float, accept_draws[cluster]),
        )
        cluster += 1


def wolff_sweep(
    state: np.ndarray,
    rows: np.ndarray,
    offsets: np.ndarray,
    neighbours: np.ndarray,
    couplings: np.ndarray,
    rng: np.random.Generator,
    counter: ClusterCounter | None = None,
    graph: PottsGraph | None = None,
    beta: float = 1.0,
    root: int | None = None,
    proposed: int | None = None,
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

    ``root`` and ``proposed``, when given, are the cluster's seed site and the
    colour to recolour it to rather than draws made here. That is issue #706's
    Wolff *action* --- a root, a label and a temperature --- so the choice
    belongs to the policy taking it, leaving the bond construction as the only
    randomness. Omitted, both are drawn as before and every existing caller's
    stream is bitwise unchanged.

    Returns
    -------
    int
        The size of the cluster this step built.
    """
    bounds = offsets.tolist()
    incident, weights = neighbours.tolist(), couplings.tolist()
    seed_node = int(rng.integers(state.shape[0])) if root is None else int(root)
    colour = int(state[seed_node])
    cluster = [seed_node]
    in_cluster = np.zeros(state.shape[0], dtype=bool)
    in_cluster[seed_node] = True
    frontier = [seed_node]
    while frontier:
        node = frontier.pop()
        for position in range(bounds[node], bounds[node + 1]):
            neighbour, coupling = incident[position], weights[position]
            if in_cluster[neighbour] or state[neighbour] != colour:
                continue
            if rng.random() < 1.0 - np.exp(-beta * coupling):
                in_cluster[neighbour] = True
                cluster.append(neighbour)
                frontier.append(neighbour)
    members = np.array(cluster, dtype=np.int64)
    outcome = _recolour(state, members, beta * rows, rng, proposed)
    if counter is not None:
        counter.record(members, outcome, graph)
    return len(cluster)


def niedermayer_threshold(couplings: np.ndarray) -> float:
    """Niedermayer's ``E_0`` for a graph, in the energy units of :func:`energies`.

    ``max(0, -min J)``: the smallest threshold at which every bond probability
    of :func:`niedermayer_sweep` is defined on this graph. On a ferromagnet it
    is 0 and the rule *is* Wolff's, bond for bond and to the last bit; on the
    uniform antiferromagnet it is ``|J|``, where bonds form between unlike
    sites --- which is what an antiferromagnet's satisfied bonds are --- and
    the construction is again exact, the accept step carrying the field alone.

    **It is the smallest, and that is the point.** The bond probabilities rise
    with ``E_0``, so a larger threshold buys nothing and costs the cluster:
    where the couplings are mixed the cluster already percolates here --- 8.94
    sites of 9 on the instance
    `tests/regression/search/test_potts_mcmc.py` measures --- and a single
    cluster that is the whole lattice is a global spin reversal, which is the
    reason Houdayer's pair move exists beside this one.

    Parameters
    ----------
    couplings : np.ndarray
        Edge couplings, as
        :meth:`~snakes_and_ladders.sim.graph.PottsGraph.compressed_adjacency`
        lays them out or in any other order --- only the minimum is read.

    Returns
    -------
    float
        ``E_0``, never negative.
    """
    return max(0.0, -float(np.min(couplings))) if couplings.size else 0.0


def niedermayer_sweep(
    state: np.ndarray,
    rows: np.ndarray,
    offsets: np.ndarray,
    neighbours: np.ndarray,
    couplings: np.ndarray,
    rng: np.random.Generator,
    counter: ClusterCounter | None = None,
    graph: PottsGraph | None = None,
    beta: float = 1.0,
    threshold: float = 0.0,
    root: int | None = None,
    partner: int | None = None,
) -> int:
    """Grow one cluster under Niedermayer's bond rule, transpose two colours on it.

    Niedermayer (1988) generalizes the Fortuin-Kasteleyn construction by
    activating a bond on its *energy relative to a threshold* ``E_0`` rather
    than on its endpoints agreeing. In the energy convention of
    :func:`energies` --- ``E = -h[s] - sum J [s_i = s_j]`` --- a bond's energy
    is ``-J`` where its endpoints agree and ``0`` where they do not, so the
    rule is

    ``p(bond) = 1 - exp(-beta * max(0, E_0 + J [s_i = s_j]))``,

    the ``max`` being what keeps it a probability for any ``E_0`` and any sign
    of ``J``. **Wolff is the case ``E_0 = 0`` on a ferromagnet**: the unlike
    bonds get ``p = 0`` and the like ones ``1 - exp(-beta J)``, which is
    :func:`wolff_sweep`'s own probability.

    ``E_0`` is the one knob, and it runs between two algorithms. At or above
    :func:`niedermayer_threshold` every ``max`` is on its linear branch, the
    boundary terms below cancel exactly, and what is left is a
    Fortuin-Kasteleyn construction for a coupling of *either* sign whose only
    accept step is the field's --- Wolff's, where Wolff runs. Below it the
    like bonds of an antiferromagnet fall to ``p = 0``, the terms survive, and
    at ``E_0 = 0`` on an antiferromagnet no bond forms at all: the cluster is
    its seed and the step is a single-site Metropolis flip on the exact
    difference. The default is the threshold, so the default is the cluster
    algorithm.

    The cluster is then changed by the *transposition* of two colours rather
    than by a recolouring to one. Above ``E_0 = 0`` the cluster is no longer
    monochromatic, and a recolouring would change the agreement of its interior
    bonds --- the one thing the construction needs left alone, since those are
    the factors that cancel between the forward move and the reverse. A
    permutation of the colours cannot change an agreement, which is why it is
    the move that generalizes.

    What does not cancel is the boundary and the field, and that is the accept
    step. Writing ``x`` for ``[s_i = s_j]`` on a boundary bond before the
    transposition and ``x'`` for it after,

    ``delta = sum_C (h[new] - h[old]) + sum_boundary (g(x') - g(x))``,
    ``g(x) = J x - max(0, E_0 + J x)``,

    and the move is accepted with probability ``min(1, exp(beta * delta))``.
    ``g`` is constant in ``x`` wherever ``E_0`` and ``E_0 + J`` are both
    non-negative --- both are then ``-E_0`` --- so at or above the threshold
    every boundary term cancels and the accept step is the field's alone,
    which is what :func:`_recolour` applies for Wolff. Below the threshold
    they do not cancel, and they are what keeps the step a valid
    Metropolis-Hastings move where it is no longer a Fortuin-Kasteleyn one.

    Parameters
    ----------
    state, rows, offsets, neighbours, couplings, rng, counter, graph
        As :func:`wolff_sweep`. ``rows`` is the field at one row per site,
        untempered; ``beta`` scales it here rather than at the call site,
        because the bond rule and the accept step must carry the same one.
    beta : float
        Inverse temperature. ``math.inf`` is admitted and is the ``T = 0``
        limit taken exactly: a bond of positive energy margin is certain
        rather than drawn, and a ``delta`` below zero is refused without
        consuming a uniform.
    threshold : float
        ``E_0``. :func:`niedermayer_threshold` is the value that makes the
        rule Wolff's where Wolff runs.
    root : int | None
        The cluster's seed, or ``None`` to draw it uniformly.
    partner : int | None
        The colour the seed's own colour is transposed with, or ``None`` to
        draw it uniformly from the other ``n_states - 1``. Drawn from the
        colours rather than from the state, so the proposal is symmetric: the
        reverse move must be able to name the same unordered pair.

    Returns
    -------
    int
        The size of the cluster this step built.
    """
    n_nodes = int(state.shape[0])
    n_states = int(rows.shape[1])
    bounds = offsets.tolist()
    incident, weights = neighbours.tolist(), couplings.tolist()
    labels = state.tolist()

    seed_node = int(rng.integers(n_nodes)) if root is None else int(root)
    held = int(labels[seed_node])
    if partner is None:
        swapped = (held + 1 + int(rng.integers(n_states - 1))) % n_states
    else:
        swapped = int(partner)

    in_cluster = np.zeros(n_nodes, dtype=bool)
    in_cluster[seed_node] = True
    cluster = [seed_node]
    frontier = [seed_node]
    while frontier:
        node = frontier.pop()
        for position in range(bounds[node], bounds[node + 1]):
            neighbour = incident[position]
            if in_cluster[neighbour]:
                continue
            margin = threshold + (
                weights[position] if labels[neighbour] == labels[node] else 0.0
            )
            if margin <= 0.0:
                continue
            probability = 1.0 - np.exp(-beta * margin)
            # Only `beta = inf` reaches one, and there the bond is certain
            # rather than drawn: the T = 0 limit without a second branch.
            if probability >= 1.0 or rng.random() < probability:
                in_cluster[neighbour] = True
                cluster.append(neighbour)
                frontier.append(neighbour)

    members = np.array(cluster, dtype=np.int64)
    delta = 0.0
    for node in cluster:
        current = labels[node]
        if current == held:
            moved = swapped
        elif current == swapped:
            moved = held
        else:
            continue
        delta += float(rows[node, moved] - rows[node, current])
        for position in range(bounds[node], bounds[node + 1]):
            neighbour = incident[position]
            if in_cluster[neighbour]:
                continue
            # `g(x') - g(x)` for this boundary bond, with `beta` divided out:
            # the log-density's own term, then the bond probabilities' one.
            coupling = weights[position]
            before = 1.0 if labels[neighbour] == current else 0.0
            after = 1.0 if labels[neighbour] == moved else 0.0
            delta += coupling * (after - before)
            delta -= max(0.0, threshold + coupling * after)
            delta += max(0.0, threshold + coupling * before)

    accepted = _niedermayer_accept(delta, beta, rng)
    if accepted:
        held_members = members[state[members] == held]
        swapped_members = members[state[members] == swapped]
        state[held_members] = swapped
        state[swapped_members] = held
    if counter is not None:
        counter.record(members, Recolour(proposed=True, accepted=accepted), graph)
    return len(cluster)


def _niedermayer_accept(delta: float, beta: float, rng: np.random.Generator) -> bool:
    """:func:`~snakes_and_ladders.sample.accept.accept_at` on ``delta``, under a name.

    A separate function because it is what an ablation replaces:
    `tests/regression/search/test_potts_mcmc.py` swaps in an unconditional
    accept and asserts the enumerated chi-square rejects, which is the
    evidence that the tests above have the power they claim. The ``beta = inf``
    limit it takes is stated where the accept step lives (issue #857).
    """
    return accept_at(beta, delta, rng)


def houdayer_cluster(
    first: np.ndarray, second: np.ndarray, offsets: np.ndarray, neighbours: np.ndarray
) -> np.ndarray:
    """Component index per site over the bonds joining two sites where the replicas disagree.

    The overlap ``q_i = s_i s'_i`` of Houdayer (2001) reads ``-1`` exactly where
    the two replicas disagree, and his cluster is a connected component of that
    region. Sites where the replicas agree are their own singletons here, so
    one array answers both questions a caller has --- which sites are in the
    defect region, and which component each of them is in.

    Uses :func:`find_root` and :func:`union_roots`, so the components are not a second
    reading of what a component is, and walks the compressed rows rather than
    the graph's edge tuples (root `CLAUDE.md`'s layout rule).

    Parameters
    ----------
    first, second : np.ndarray
        The two replicas' labellings, ``(n_nodes,)``.
    offsets, neighbours : np.ndarray
        The compressed adjacency, as
        :meth:`~snakes_and_ladders.sim.graph.PottsGraph.compressed_adjacency`
        lays it out.

    Returns
    -------
    np.ndarray
        ``(n_nodes,)`` of component roots, as :func:`find_root` reports them.
    """
    n_nodes = int(offsets.shape[0]) - 1
    parent = np.arange(n_nodes)
    defect = (first != second).tolist()
    bounds, incident = offsets.tolist(), neighbours.tolist()
    for node in range(n_nodes):
        if not defect[node]:
            continue
        for position in range(bounds[node], bounds[node + 1]):
            neighbour = incident[position]
            if neighbour > node and defect[neighbour]:
                union_roots(parent, node, neighbour)
    return np.array([find_root(parent, node) for node in range(n_nodes)])


def houdayer_move(
    first: np.ndarray,
    second: np.ndarray,
    offsets: np.ndarray,
    neighbours: np.ndarray,
    rng: np.random.Generator,
) -> int:
    """Swap the two replicas' labels on one component of their overlap defect.

    Houdayer's isoenergetic cluster move (2001), in the form his paper states
    for two replicas at one temperature. The cluster is a connected component
    of ``{i : s_i != s'_i}``, and on it the two replicas exchange labels.

    **The acceptance is 1, and it is an identity rather than a cancellation.**
    A bond with both ends in the cluster has its pair of agreements exchanged
    between the replicas, so the pair's energy is unchanged; a bond with one
    end in it has its other end where the replicas agree --- a neighbour that
    disagreed would be in the same component --- so the two terms are again
    exchanged. The field term is exchanged site by site for the same reason.
    So ``E(s) + E(s')`` is invariant, the move is an involution, and the
    defect region it is built from is what the swap leaves alone, which makes
    the reverse proposal exactly as likely as the forward one.

    Parameters
    ----------
    first, second : np.ndarray
        The two replicas, mutated in place.
    offsets, neighbours : np.ndarray
        The compressed adjacency.
    rng : np.random.Generator
        Draws the defect site the component is grown from.

    Returns
    -------
    int
        The size of the cluster swapped; ``0`` where the replicas agree
        everywhere and there is no defect to move.
    """
    defects = np.flatnonzero(first != second)
    if defects.size == 0:
        return 0
    partition = houdayer_cluster(first, second, offsets, neighbours)
    seed_node = int(defects[rng.integers(defects.size)])
    members = np.flatnonzero(partition == partition[seed_node])
    held = first[members].copy()
    first[members] = second[members]
    second[members] = held
    return int(members.size)


def _recolour(
    state: np.ndarray,
    members: np.ndarray,
    rows: np.ndarray,
    rng: np.random.Generator,
    proposed: int | None = None,
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
    h[old])``; for the per-site field of `spatio_only` it is the sum of that
    difference over the cluster's own members, which the shared case is the
    special case of.

    ``proposed``, when given, is the colour to try rather than one drawn here:
    issue #706's Wolff action names the colour it recolours to, so the draw
    belongs to the action rather than to this function. Omitted, the draw is
    unchanged, which is what keeps every existing caller's stream bitwise as it
    was.

    Returns
    -------
    Recolour
        Whether a colour change was proposed at all, and whether it was
        accepted. Issue #551 reads the acceptance against temperature, and a
        run that never proposed is not a run that was rejected.
    """
    if proposed is None:
        proposed = int(rng.integers(rows.shape[1]))
    return _recolour_drawn(state, members, rows, proposed, rng.random)


def _recolour_drawn(
    state: np.ndarray,
    members: np.ndarray,
    rows: np.ndarray,
    proposed: int,
    draw: Callable[[], float],
) -> Recolour:
    """:func:`_recolour` with the colour already chosen and the uniform behind a call.

    The whole of the oracle's recolouring, factored out so the Rust pass's
    hand-back path decides its cluster by calling this rather than a copy of
    it --- :func:`_site_update`'s place in :func:`_sweep_at`, one level up
    (issues #599, #754).

    ``draw`` is a callable and not a float because the acceptance is
    *conditional*: the oracle consumes a uniform only where the field
    difference is negative, and taking one eagerly would advance the
    generator on a cluster that never needed it and move every chain after
    it. The caller passes ``rng.random``; the hand-back passes the draw the
    kernel was given.
    """
    current = int(state[members[0]])
    if proposed == current:
        return Recolour(proposed=False, accepted=False)
    difference = float(rows[members, proposed].sum() - rows[members, current].sum())
    if accept_drawn(difference, draw):
        state[members] = proposed
        return Recolour(proposed=True, accepted=True)
    return Recolour(proposed=True, accepted=False)


def find_root(parent: np.ndarray, node: int) -> int:
    """Union-find root, with path compression."""
    root = node
    while parent[root] != root:
        root = int(parent[root])
    while parent[node] != root:
        parent[node], node = root, int(parent[node])
    return root


def union_roots(parent: np.ndarray, first: int, second: int) -> None:
    """Merge two components."""
    first_root, second_root = find_root(parent, first), find_root(parent, second)
    if first_root != second_root:
        parent[second_root] = first_root
