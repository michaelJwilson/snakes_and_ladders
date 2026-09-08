"""One Gibbs sampler and one annealer for every problem, over the factor graph (issue #309).

A variable's conditional given everything else is the product of the factors
that touch it, so a heat-bath sweep over a :class:`~snakes_and_ladders.sim.factor_graph.FactorGraph`
serves the Potts lattice, the hidden Markov chain, a tree at one site and the
coupled model alike, through the adapters that already exist -- no sampler
knows what the variables mean. Tempering multiplies every factor's log table
by ``beta``, which is the same graph with its tables scaled; annealing is the
sweep on a :class:`~snakes_and_ladders.opt.schedule.Schedule`, returning the
best state visited.

Three things are held fixed from the specialised samplers. The single-site
update draws one uniform per variable and searches a cumulative sum, the
arithmetic of :func:`snakes_and_ladders.search.potts_mcmc._single_site_sweep`,
so on a Potts graph the two agree draw for draw except where a uniform lands
within rounding of a boundary; the pin is distributional and the agreement is
reported. A sweep never stops on a state-dependent condition
(``search/CLAUDE.md``). And every instance is held to the distribution it
converges to, by goodness-of-fit against enumeration or against the exact
marginals sum-product gives on a tree.

Two moves beyond the single site: an exact block draw of a chain-shaped
subset of variables, by forward filter and backward sample over whatever
factors touch it (the block Gibbs move the coupled model's chains need), and
a Metropolis move over tree topologies whose stationary distribution at
temperature one is the flat-prior weight over fitted likelihoods that
:mod:`snakes_and_ladders.search.support` enumerates.

The generic sweep is Python over factor tables. Its cost against the numba
and Rust Potts kernels is measured in ``tests/benchmarks/test_gibbs_bench.py``
and recorded in ``STATUS.md``; the specialised kernels stay the default for
the Potts lattice, and this sampler is the one for the model none of them
can express.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from snakes_and_ladders.numerics import logsumexp
from snakes_and_ladders.opt.anneal import Extremum, drive
from snakes_and_ladders.opt.schedule import Schedule
from snakes_and_ladders.search.infer import Model, MoveSet, score_topology
from snakes_and_ladders.search.topology import (
    Topology,
    leaf_bipartitions,
    nni_neighbours,
    spr_neighbours,
)
from snakes_and_ladders.sim.factor_graph import Factor, FactorGraph


@dataclass(frozen=True)
class GibbsChain:
    """What a run of the sampler returns.

    Parameters
    ----------
    variables : tuple[str, ...]
        The variables in the order the columns of ``states`` take: the
        graph's own.
    states : np.ndarray
        Integer states, shape ``(n_recorded, n_variables)``, one row per
        recorded sweep after burn-in and thinning.
    log_densities : np.ndarray
        The unnormalized log-density of every recorded state, shape
        ``(n_recorded,)``, at temperature one.
    """

    variables: tuple[str, ...]
    states: np.ndarray
    log_densities: np.ndarray


@dataclass(frozen=True)
class Annealed:
    """What annealing returns: the best state visited, and the path there.

    Parameters
    ----------
    state : np.ndarray
        The state of highest log-density seen, shape ``(n_variables,)``.
    log_density : float
        Its unnormalized log-density at temperature one.
    trajectory : np.ndarray
        The log-density after every step, shape ``(n_steps + 1,)``; the first
        entry is the start.
    """

    state: np.ndarray
    log_density: float
    trajectory: np.ndarray


class _Indexed:
    """A factor graph with its variables numbered and each variable's factors located."""

    def __init__(self, graph: FactorGraph) -> None:
        self.graph = graph
        self.names = tuple(variable.name for variable in graph.variables)
        self.index = {name: position for position, name in enumerate(self.names)}
        self.cardinality = np.array(
            [variable.cardinality for variable in graph.variables], dtype=np.int64
        )
        self.touching: list[list[tuple[Factor, int, np.ndarray]]] = [
            [] for _ in self.names
        ]
        for factor in graph.factors:
            columns = np.array(
                [self.index[name] for name in factor.variables], dtype=np.int64
            )
            for axis, name in enumerate(factor.variables):
                self.touching[self.index[name]].append((factor, axis, columns))

    def conditional(self, state: np.ndarray, position: int) -> np.ndarray:
        """``sum_a log psi_a`` over the states of one variable, the others fixed."""
        local = np.zeros(int(self.cardinality[position]))
        for factor, axis, columns in self.touching[position]:
            key: list[int | slice] = [int(state[column]) for column in columns]
            key[axis] = slice(None)
            local += factor.log_table[tuple(key)]
        return local

    def log_density(self, state: np.ndarray) -> float:
        return self.graph.log_density(
            dict(zip(self.names, map(int, state), strict=True))
        )

    def start(self, rng: np.random.Generator, start: np.ndarray | None) -> np.ndarray:
        if start is None:
            return np.array(
                [int(rng.integers(card)) for card in self.cardinality], dtype=np.int64
            )
        state = np.asarray(start, dtype=np.int64)
        if (
            state.shape != self.cardinality.shape
            or (state < 0).any()
            or (state >= self.cardinality).any()
        ):
            msg = f"start must hold one state per variable inside its cardinality, got {state}"
            raise ValueError(msg)
        return state.copy()


def gibbs_sweep(
    graph: FactorGraph | _Indexed,
    state: np.ndarray,
    rng: np.random.Generator,
    *,
    beta: float = 1.0,
) -> None:
    """One heat-bath update of every variable in graph order, in place.

    One uniform per variable drawn up front and a search of the cumulative
    conditional, the arithmetic of the Potts single-site sweep, so the two
    agree draw for draw on a Potts graph up to rounding.
    """
    indexed = graph if isinstance(graph, _Indexed) else _Indexed(graph)
    draws = np.asarray(rng.random(len(indexed.names)))
    for position in range(len(indexed.names)):
        local = indexed.conditional(state, position)
        local *= beta
        local -= local.max()
        cumulative = np.cumsum(np.exp(local))
        state[position] = np.searchsorted(
            cumulative, float(draws[position]) * cumulative[-1]
        )


def sample_factor_graph(
    graph: FactorGraph,
    rng: np.random.Generator,
    n_sweeps: int,
    burn_in: int = 0,
    thin: int = 1,
    *,
    temperature: float = 1.0,
    start: np.ndarray | None = None,
) -> GibbsChain:
    """Run one chain of single-site sweeps and record its states.

    Parameters
    ----------
    graph : FactorGraph
        Any of the adapters' graphs.
    rng : np.random.Generator
        Passed in rather than seeded here (``sim/CLAUDE.md``).
    n_sweeps : int
        Sweeps after burn-in; every ``thin``-th is recorded.
    burn_in, thin : int
        As :func:`snakes_and_ladders.search.potts_mcmc.sample_potts`.
    temperature : float
        Every factor's log table is scaled by ``1 / temperature``.
    start : np.ndarray | None
        A starting state in the graph's variable order; ``None`` draws one
        uniformly.

    Raises
    ------
    ValueError
        If ``n_sweeps`` or ``thin`` is below one, ``burn_in`` negative, or
        ``temperature`` not positive.
    """
    if n_sweeps < 1 or thin < 1 or burn_in < 0:
        msg = f"n_sweeps {n_sweeps} and thin {thin} must be >= 1 and burn_in {burn_in} >= 0"
        raise ValueError(msg)
    if temperature <= 0.0:
        msg = f"temperature must be positive, got {temperature}"
        raise ValueError(msg)
    indexed = _Indexed(graph)
    beta = 1.0 / temperature
    state = indexed.start(rng, start)
    for _ in range(burn_in):
        gibbs_sweep(indexed, state, rng, beta=beta)
    states = []
    densities = []
    for sweep in range(n_sweeps):
        gibbs_sweep(indexed, state, rng, beta=beta)
        if sweep % thin == 0:
            states.append(state.copy())
            densities.append(indexed.log_density(state))
    return GibbsChain(indexed.names, np.array(states), np.array(densities))


def anneal_factor_graph(
    graph: FactorGraph,
    schedule: Schedule,
    rng: np.random.Generator,
    *,
    start: np.ndarray | None = None,
) -> Annealed:
    """Simulated annealing by heat-bath sweeps: one sweep per schedule step at that step's temperature.

    The generic form of :func:`snakes_and_ladders.search.potts_mcmc.anneal_potts`,
    tracking the state of highest log-density seen. At ``T -> 0`` the heat
    bath is the argmax over each variable's conditional, so the two ends of a
    schedule are single-site descent and free sampling, as there.
    """
    indexed = _Indexed(graph)
    state = indexed.start(rng, start)

    def transition(
        current: np.ndarray, _: float, temperature: float
    ) -> tuple[np.ndarray, float, bool]:
        # The sweep writes through `current`, hence the `copy` below.
        gibbs_sweep(indexed, current, rng, beta=1.0 / temperature)
        return current, indexed.log_density(current), False

    run = drive(
        state,
        indexed.log_density(state),
        transition,
        schedule,
        keep=Extremum.MAXIMUM,
        copy=lambda current: current.copy(),
    )
    return Annealed(run.best, run.best_value, run.trajectory)


def chain_block_sweep(
    graph: FactorGraph,
    state: np.ndarray,
    rng: np.random.Generator,
    chain: Sequence[str],
    *,
    beta: float = 1.0,
) -> None:
    """Redraw a chain-shaped subset of variables from its exact conditional, in place.

    ``chain`` names variables in order; every factor touching one of them
    contributes a unary term, every factor touching two consecutive ones a
    transition, each conditioned on the current states of the variables it
    touches outside the chain. Forward filter, backward sample: the block
    Gibbs move for a hidden Markov chain inside a larger model.

    Raises
    ------
    ValueError
        If a factor touches two non-consecutive chain variables or more than
        two of them, so the subset is not a chain of this graph.
    """
    indexed = _Indexed(graph)
    positions = [indexed.index[name] for name in chain]
    where = {position: step for step, position in enumerate(positions)}
    length = len(positions)
    if length == 0:
        msg = "a chain names at least one variable"
        raise ValueError(msg)
    unary = [np.zeros(int(indexed.cardinality[p])) for p in positions]
    transition: list[np.ndarray | None] = [None] * max(length - 1, 0)
    seen: set[str] = set()
    for position in positions:
        for factor, _, columns in indexed.touching[position]:
            if factor.name in seen:
                continue
            seen.add(factor.name)
            steps = sorted(where[c] for c in columns if int(c) in where)
            key: list[int | slice] = [int(state[c]) for c in columns]
            if len(steps) == 1:
                axis = list(columns).index(positions[steps[0]])
                key[axis] = slice(None)
                unary[steps[0]] += beta * factor.log_table[tuple(key)]
            elif len(steps) == 2 and steps[1] == steps[0] + 1:
                first = list(columns).index(positions[steps[0]])
                second = list(columns).index(positions[steps[1]])
                key[first] = slice(None)
                key[second] = slice(None)
                table = beta * factor.log_table[tuple(key)]
                if first > second:
                    table = table.T
                current = transition[steps[0]]
                transition[steps[0]] = table if current is None else current + table
            else:
                msg = (
                    f"factor {factor.name!r} touches chain variables at steps {steps}, "
                    "which are not consecutive"
                )
                raise ValueError(msg)
    alpha = [unary[0]]
    for t in range(1, length):
        step_table = transition[t - 1]
        if step_table is None:
            step_table = np.zeros((unary[t - 1].shape[0], unary[t].shape[0]))
        alpha.append(logsumexp(alpha[t - 1][:, None] + step_table, axis=0) + unary[t])
    weights = np.exp(alpha[-1] - logsumexp(alpha[-1][None, :], axis=1)[0])
    state[positions[-1]] = rng.choice(weights.shape[0], p=weights / weights.sum())
    for t in range(length - 2, -1, -1):
        step_table = transition[t]
        column = (
            np.zeros(alpha[t].shape[0])
            if step_table is None
            else step_table[:, int(state[positions[t + 1]])]
        )
        scores = alpha[t] + column
        weights = np.exp(scores - logsumexp(scores[None, :], axis=1)[0])
        state[positions[t]] = rng.choice(weights.shape[0], p=weights / weights.sum())


@dataclass(frozen=True)
class AnnealedTopology:
    """What the topology move returns.

    Parameters
    ----------
    topology : Topology
        The topology of highest fitted log-likelihood visited.
    log_likelihood : float
        Its fitted log-likelihood.
    trajectory : np.ndarray
        The current topology's fitted log-likelihood after every step.
    acceptance : float
        The fraction of proposals accepted.
    scores : Mapping[frozenset, float]
        Every topology scored, keyed on its leaf bipartitions: a fit is paid
        once per topology however often the chain returns to it.
    """

    topology: Topology
    log_likelihood: float
    trajectory: np.ndarray
    acceptance: float
    scores: Mapping[frozenset[frozenset[str]], float]


def anneal_topology(
    alignment: Mapping[str, np.ndarray],
    k: int,
    schedule: Schedule,
    rng: np.random.Generator,
    start: Topology,
    *,
    moves: MoveSet = MoveSet.NNI,
    model: Model = Model.JC,
    scores: dict[frozenset[frozenset[str]], float] | None = None,
) -> AnnealedTopology:
    """Metropolis over topologies on the fitted log-likelihood, at the schedule's temperature.

    A proposal is a uniform neighbour under ``moves``; every binary topology
    has the same number of NNI neighbours and the same number of SPR ones, so
    the proposal is symmetric and the acceptance is
    ``min(1, exp((l' - l) / T))``. At ``T = 1`` the stationary distribution
    is the flat-prior weight over fitted likelihoods that
    :func:`snakes_and_ladders.search.support.enumerated_support` computes,
    which is what pins it; at ``T -> 0`` it is hill climbing with a random
    neighbour, which is what the annealed end reaches.

    ``scores`` caches fitted log-likelihoods by leaf bipartitions across
    calls, since the fit is the whole cost.
    """
    cache = {} if scores is None else scores
    score = cached_topology_score(alignment, k, cache, model=model)

    def transition(
        current: Topology, value: float, temperature: float
    ) -> tuple[Topology, float, bool]:
        return topology_step(current, value, temperature, rng, score, moves=moves)

    # A topology is immutable, so the default identity `copy` is right. The
    # loop this replaced nested its best-state test inside `if moved`; that
    # gate was never live, because a rejected step returns the value it was
    # given and `best_value` is an upper bound on every value seen. Dropping
    # it changes no result and is pinned as not changing one.
    run = drive(start, score(start), transition, schedule, keep=Extremum.MAXIMUM)
    return AnnealedTopology(
        run.best,
        run.best_value,
        run.trajectory,
        run.accepted / schedule.n_steps,
        cache,
    )


def cached_topology_score(
    alignment: Mapping[str, np.ndarray],
    k: int,
    cache: dict[frozenset[frozenset[str]], float],
    *,
    model: Model = Model.JC,
) -> Callable[[Topology], float]:
    """A scorer that fits a topology once per leaf bipartitions and reads ``cache`` after.

    The fit is the whole cost of a walk over topologies, so every walk
    shares this: :func:`anneal_topology` and the tempered ensemble of
    :mod:`snakes_and_ladders.search.tempered` alike.
    """

    def score(topology: Topology) -> float:
        key = leaf_bipartitions(topology)
        if key not in cache:
            cache[key] = score_topology(topology, alignment, k, model)
        return cache[key]

    return score


def topology_step(
    current: Topology,
    value: float,
    temperature: float,
    rng: np.random.Generator,
    score: Callable[[Topology], float],
    *,
    moves: MoveSet = MoveSet.NNI,
) -> tuple[Topology, float, bool]:
    """One Metropolis step over topologies at ``temperature``.

    A uniform neighbour under ``moves`` is proposed and accepted with
    ``min(1, exp((l' - l) / T))``, drawing one integer and, where the
    proposal is worse, one uniform from ``rng`` -- the step
    :func:`anneal_topology` takes, so a run there and a replica of the
    tempered ensemble are the same chain draw for draw.

    Returns
    -------
    tuple[Topology, float, bool]
        The topology and fitted log-likelihood after the step, and whether
        the proposal was accepted.
    """
    neighbours = nni_neighbours if moves is MoveSet.NNI else spr_neighbours
    options = list(neighbours(current))
    proposal = options[int(rng.integers(len(options)))]
    proposed = score(proposal)
    difference = (proposed - value) / temperature
    if difference >= 0.0 or rng.random() < np.exp(difference):
        return proposal, proposed, True
    return current, value, False
