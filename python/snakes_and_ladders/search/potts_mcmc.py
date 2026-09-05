"""Monte Carlo move sets on a Potts lattice: single-site, Swendsen-Wang, Wolff.

`ROADMAP.md` §1.4 names both cluster algorithms. Single-site flips are the
only Potts move set the repository had, and they slow critically near the
transition --- the autocorrelation time of the energy diverges as the
correlation length does, so no Potts result at a useful lattice size is
reachable through them. Cluster updates flip whole correlated regions at once
and do not.

**The field is the part that is easy to get silently wrong.** The reference
instance is a Potts model *in an external field*, and the Fortuin-Kasteleyn
construction both cluster algorithms rest on is exact only at zero field:
recolouring a cluster changes the field term by ``|C| * (h[new] - h[old])``,
which the bond construction knows nothing about. Left there, the sampler runs,
produces plausible configurations, and converges to the wrong distribution. So
a cluster recolouring carries a Metropolis accept step on exactly that
difference, and the chi-square tests in
`tests/regression/search/test_potts_mcmc.py` are run with and without a field
because only the first of those catches its absence.

These are samplers, not optimizers. They are validated by the distribution
they converge to, and nothing here claims to find a ground state; that belongs
to the classical baseline suite, which has an oracle for it. The one
exception is :func:`anneal_potts`, which is an optimizer built *from* the
sampler: the same sweep on a schedule of falling temperatures (issue #267).

**Temperature is model scaling, and the model says so exactly.** The Potts
coupling absorbs ``beta``: ``exp(-E / T)`` with ``E = -h[s] - J [s = s']`` is
the Boltzmann weight of the model with ``(J / T, h / T)`` at temperature 1.
So a tempered chain runs the untempered sweeps on the scaled model, and there
is no second code path to get wrong --- what :func:`tempered` does is checked
against the energies, and the chain it produces against ``exp(-E / T)``
enumerated from the *unscaled* model, which is the oracle the samplers already
have. Tempering a likelihood is a different object (`snakes_and_ladders.opt.schedule`
says why); here the objective is an energy and the temperature is physical.

See ``docs/tex/main.tex``, "Potts Models in an External Field" (Newman &
Barkema chs. 4 and 6 for both algorithms and for Sokal's windowing; Mezard &
Montanari ch. 2).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from snakes_and_ladders.likelihood.potts import log_weights
from snakes_and_ladders.opt.schedule import Schedule
from snakes_and_ladders.sim.graph import PottsGraph


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
        Sites per cluster flip, averaged over the run. Carried because a
        Wolff sweep flips one cluster while the other two touch every site,
        so an autocorrelation time in sweeps is not comparable across the
        three without it. For the move sets that do not build clusters it is
        ``n_nodes``, which is what they touch per sweep.
    """

    states: np.ndarray
    mean_cluster_size: float


def tempered(
    graph: PottsGraph, field: np.ndarray, temperature: float
) -> tuple[PottsGraph, np.ndarray]:
    """The model whose Boltzmann weight at temperature 1 is this one's at ``temperature``.

    ``(J, h) / T``. At ``T = 1`` the division is the identity bitwise, which
    is what lets every untempered chain be the tempered one at the default
    rather than a separate path.

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
    seed: int,
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
    seed : int
        Seed for ``np.random.default_rng``.
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
        sampler: the chi-square statistic assumes independent draws and the
        correlation inflates it. Thinning by several autocorrelation times is
        what makes the test measure the sampler rather than the correlation.
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
        If a cluster move is asked for on a graph with a negative coupling.
        The bond probability ``1 - exp(-J)`` is not a probability there, and
        an antiferromagnet has no like-spin clusters to flip: the algorithm
        does not apply, rather than applying badly.
    """
    if move is not PottsMove.SINGLE_SITE and min(graph.coupling, default=0.0) < 0.0:
        msg = (
            f"{move} needs every coupling >= 0: the bond probability "
            "1 - exp(-J) is not a probability for J < 0, and an "
            "antiferromagnet has no like-spin clusters to flip"
        )
        raise ValueError(msg)

    graph, field = tempered(graph, field, temperature)
    rng = np.random.default_rng(seed)
    n_states = int(field.shape[0])
    state = rng.integers(0, n_states, size=graph.n_nodes)
    neighbours = _adjacency(graph)

    recorded = np.empty((n_sweeps, graph.n_nodes), dtype=np.int64)
    cluster_total, cluster_count = 0, 0
    for step in range(-burn_in * thin, n_sweeps * thin):
        if move is PottsMove.SINGLE_SITE:
            _single_site_sweep(state, field, neighbours, rng)
        elif move is PottsMove.SWENDSEN_WANG:
            _swendsen_wang_sweep(state, graph, field, rng)
        else:
            cluster_total += _wolff_sweep(state, field, neighbours, rng)
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
        Heat-bath sweeps run, one per schedule step --- the budget, in the
        unit `search/CLAUDE.md` says budgets are counted in.
    """

    labelling: np.ndarray
    energy: float
    final: np.ndarray
    n_sweeps: int


def anneal_potts(
    graph: PottsGraph,
    field: np.ndarray,
    schedule: Schedule,
    seed: int,
) -> AnnealedPotts:
    """Simulated annealing by heat-bath sweeps on a temperature schedule.

    One :func:`_single_site_sweep` per schedule step at that step's
    temperature, tracking the lowest energy seen (Kirkpatrick, Gelatt &
    Vecchi, 1983). It is :func:`iterated_conditional_modes` with a finite
    temperature: at ``T -> 0`` the heat bath is the argmin over each site's
    conditional, which is exactly ICM's update, so the two are the same
    search separated by the schedule alone and a difference between them is a
    statement about the schedule.

    Single-site moves only. The cluster moves are built for sampling near a
    ferromagnetic transition and refuse a negative coupling, and the instances
    worth annealing are frustrated.

    Parameters
    ----------
    graph : PottsGraph
        The instance. Couplings of either sign.
    field : np.ndarray
        External field, shape ``(n_states,)``.
    schedule : Schedule
        Temperature per sweep. Its length is the budget.
    seed : int
        Seed for ``np.random.default_rng``; the start is drawn from it.

    Returns
    -------
    AnnealedPotts
    """
    rng = np.random.default_rng(seed)
    field = np.asarray(field, dtype=float)
    state = rng.integers(0, int(field.shape[0]), size=graph.n_nodes)
    neighbours = _adjacency(graph)

    best_state = state.copy()
    best_energy = float(energies(graph, field, state[None])[0])
    for step in range(schedule.n_steps):
        beta = 1.0 / schedule(step)
        _single_site_sweep(state, field, neighbours, rng, beta=beta)
        energy = float(energies(graph, field, state[None])[0])
        if energy < best_energy:
            best_state, best_energy = state.copy(), energy
    return AnnealedPotts(
        labelling=best_state,
        energy=best_energy,
        final=state,
        n_sweeps=schedule.n_steps,
    )


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


def _single_site_sweep(
    state: np.ndarray,
    field: np.ndarray,
    neighbours: list[list[tuple[int, float]]],
    rng: np.random.Generator,
    beta: float = 1.0,
) -> None:
    """One heat-bath sweep: every site redrawn from its exact conditional.

    The baseline the cluster algorithms are measured against, and the same
    update `snakes_and_ladders.sim.potts._simulate_gibbs` uses --- restated here for a
    single chain rather than shared, because that one is vectorized across
    many independent chains and this one must step a single chain in time.

    ``beta`` tempers the conditional in place, for :func:`anneal_potts`, whose
    temperature changes every sweep and would otherwise rebuild the adjacency
    each time. At 1.0 the multiplication is the identity bitwise.
    """
    draws = np.asarray(rng.random(state.shape[0]))
    for node in range(state.shape[0]):
        local = field.copy()
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
    state: np.ndarray, graph: PottsGraph, field: np.ndarray, rng: np.random.Generator
) -> None:
    """Activate bonds, find clusters, recolour each one.

    Every cluster is recoloured independently, so in a field every cluster
    needs its own accept step --- Wolff flips one cluster and needs one. They
    are different code for that reason rather than one rule assumed to cover
    both.
    """
    first = np.fromiter(
        (edge[0] for edge in graph.edges), dtype=np.int64, count=len(graph.edges)
    )
    second = np.fromiter(
        (edge[1] for edge in graph.edges), dtype=np.int64, count=len(graph.edges)
    )
    coupling = np.asarray(graph.coupling, dtype=float)
    like = state[first] == state[second]
    active = like & (rng.random(len(graph.edges)) < 1.0 - np.exp(-coupling))

    parent = np.arange(graph.n_nodes)
    for edge in np.flatnonzero(active):
        _union(parent, int(first[edge]), int(second[edge]))

    labels = np.array([_find(parent, node) for node in range(graph.n_nodes)])
    for root in np.unique(labels):
        members = np.flatnonzero(labels == root)
        _recolour(state, members, field, rng)


def _wolff_sweep(
    state: np.ndarray,
    field: np.ndarray,
    neighbours: list[list[tuple[int, float]]],
    rng: np.random.Generator,
) -> int:
    """Grow one cluster from a random seed, recolour it, and stop.

    Exactly one cluster per sweep, and the "exactly" is load-bearing. An
    earlier version ran clusters until their cumulative size reached
    ``n_nodes``, to spend the same budget as the other two move sets. That is
    a *state-dependent* stopping rule: an aligned configuration makes large
    clusters, so it reached the budget in fewer steps and received less
    randomization than a disordered one. Each individual step is still
    correct, but stopping on the outcome biases the composition --- measured
    on a two-site chain at ``J = 0.7``, it put 0.384 on each aligned state
    against an exact 0.334, and the chi-square rejected it outright.

    One flip per sweep is therefore not the same amount of work as one
    single-site sweep, and :func:`sample_potts` returns the mean cluster size
    so a comparison can be normalized rather than left to imply that a Wolff
    sweep and a heat-bath sweep cost the same.

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
            if rng.random() < 1.0 - np.exp(-coupling):
                in_cluster[neighbour] = True
                cluster.append(neighbour)
                frontier.append(neighbour)
    _recolour(state, np.array(cluster, dtype=np.int64), field, rng)
    return len(cluster)


def _recolour(
    state: np.ndarray, members: np.ndarray, field: np.ndarray, rng: np.random.Generator
) -> None:
    """Propose one colour for a whole cluster, accepting on the field alone.

    The proposal is uniform over every colour including the current one, which
    makes it symmetric and leaves the acceptance ratio as the field term
    alone. Drawing from the ``k - 1`` other colours would mix marginally
    faster and would need the proposal ratio carried through the acceptance;
    the symmetric version is the one whose correctness a reader can check
    against the code in front of them.

    The bond construction contributes nothing to the ratio: bonds live only
    between like-coloured sites, and every site in the cluster changes colour
    together, so the cluster is exactly as likely to be built in the proposed
    configuration as in the current one. What does not cancel is the field:
    ``|C| * (h[new] - h[old])``.
    """
    current = int(state[members[0]])
    proposed = int(rng.integers(field.shape[0]))
    if proposed == current:
        return
    difference = len(members) * (field[proposed] - field[current])
    if difference >= 0.0 or rng.random() < np.exp(difference):
        state[members] = proposed


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
