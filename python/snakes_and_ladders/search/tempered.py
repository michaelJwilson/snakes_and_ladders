"""A tempered ensemble over structures, composed from moves that already exist (issue #331).

Replica exchange over the factor graph and over tree topologies: replicas at
fixed temperatures, each stepped by a move :mod:`snakes_and_ladders.search.gibbs`
already holds -- the heat-bath sweep for a labelling or a decoding, the
Metropolis move for a topology -- and exchanged on the ratio
:func:`snakes_and_ladders.search.potts_mcmc.parallel_tempering` uses
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

**Every replica draws from its own generator**, spawned from the parent that
then draws only the exchange uniforms, as ``potts_mcmc.parallel_tempering``
does and for the reason it gives. And the estimate is a Monte Carlo one:
:mod:`snakes_and_ladders.search.support` names it a posterior weight only
beside the exchange acceptance and the autocorrelation time that show the
ensemble mixed.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import TypeVar

import numpy as np

from snakes_and_ladders.opt.anneal import exchange
from snakes_and_ladders.search.gibbs import (
    _Indexed,
    cached_topology_score,
    gibbs_sweep,
    topology_step,
)
from snakes_and_ladders.search.infer import Model, MoveSet
from snakes_and_ladders.search.topology import Topology, leaf_bipartitions
from snakes_and_ladders.sim.factor_graph import FactorGraph

S = TypeVar("S")


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
    """

    temperatures: tuple[float, ...]
    keys: tuple[tuple[Hashable, ...], ...]
    log_densities: np.ndarray
    swap_acceptance: np.ndarray
    scores: Mapping[Hashable, float]

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


def _check_ladder(
    temperatures: Sequence[float], n_sweeps: int, thin: int, burn_in: int
) -> None:
    if len(temperatures) < 2:
        msg = (
            f"a tempered ensemble needs at least two temperatures, got "
            f"{len(temperatures)}: a ladder of one has nothing to exchange"
        )
        raise ValueError(msg)
    for temperature in temperatures:
        if not temperature > 0.0:
            msg = f"every temperature must be positive, got {temperature}"
            raise ValueError(msg)
    if n_sweeps < 1 or thin < 1 or burn_in < 0:
        msg = f"n_sweeps {n_sweeps} and thin {thin} must be >= 1 and burn_in {burn_in} >= 0"
        raise ValueError(msg)


def _exchange(
    step: Callable[[S, float, float, np.random.Generator], tuple[S, float]],
    key: Callable[[S], Hashable],
    states: list[S],
    values: list[float],
    temperatures: Sequence[float],
    children: Sequence[np.random.Generator],
    rng: np.random.Generator,
    n_sweeps: int,
    burn_in: int,
    thin: int,
) -> TemperedEnsemble:
    """The replica-exchange loop, over any structure with a step and a key.

    ``step(state, value, temperature, generator)`` advances one replica one
    sweep and returns its new state and log-density at temperature one; the
    exchange ratio takes the energy ``-value``.
    """
    n_replicas = len(temperatures)
    betas = [1.0 / temperature for temperature in temperatures]
    proposed = np.zeros(n_replicas - 1)
    accepted = np.zeros(n_replicas - 1)
    recorded_keys: list[list[Hashable]] = [[] for _ in range(n_replicas)]
    densities: list[list[float]] = []
    scores: dict[Hashable, float] = {}

    def swap(first: int, second: int) -> None:
        states[first], states[second] = states[second], states[first]
        values[first], values[second] = values[second], values[first]

    for sweep in range(burn_in + n_sweeps):
        for replica in range(n_replicas):
            states[replica], values[replica] = step(
                states[replica],
                values[replica],
                temperatures[replica],
                children[replica],
            )
        # The energy is the negated log-density; `exchange` knows only
        # energies, so the negation stays at this call site, where it was
        # before issue #386.
        proposed += 1
        accepted += exchange([-value for value in values], betas, rng.random, swap)
        if sweep >= burn_in and (sweep - burn_in) % thin == 0:
            for replica in range(n_replicas):
                name = key(states[replica])
                scores[name] = values[replica]
                recorded_keys[replica].append(name)
            densities.append(list(values))
    return TemperedEnsemble(
        temperatures=tuple(temperatures),
        keys=tuple(tuple(names) for names in recorded_keys),
        log_densities=np.array(densities),
        swap_acceptance=accepted / proposed,
        scores=scores,
    )


def tempered_factor_graph(
    graph: FactorGraph,
    temperatures: Sequence[float],
    rng: np.random.Generator,
    n_sweeps: int,
    burn_in: int = 0,
    thin: int = 1,
    *,
    start: np.ndarray | None = None,
) -> TemperedEnsemble:
    """Replicas of the heat-bath sweep over ``graph`` at fixed temperatures, exchanging by Metropolis.

    One :func:`snakes_and_ladders.search.gibbs.gibbs_sweep` per replica per
    step at its own ``beta``, then every adjacent pair proposes an exchange.
    Serves a Potts labelling and a hidden path alike, through the adapters
    of :mod:`snakes_and_ladders.sim.factor_graph`.

    Parameters
    ----------
    graph : FactorGraph
        Any of the adapters' graphs.
    temperatures : Sequence[float]
        The ladder, at least two, all positive; the order fixes which pairs
        are adjacent for exchange.
    rng : np.random.Generator
        The parent generator: one child per replica, then the exchange
        uniforms only.
    n_sweeps, burn_in, thin : int
        As :func:`snakes_and_ladders.search.gibbs.sample_factor_graph`, per
        replica.
    start : np.ndarray | None
        A starting state every replica takes; ``None`` draws one per replica
        uniformly from its own child.

    Raises
    ------
    ValueError
        As :func:`snakes_and_ladders.search.gibbs.sample_factor_graph`, and if
        the ladder has fewer than two temperatures or one that is not
        positive.
    """
    _check_ladder(temperatures, n_sweeps, thin, burn_in)
    indexed = _Indexed(graph)
    children = rng.spawn(len(temperatures))
    states = [indexed.start(child, start) for child in children]
    values = [indexed.log_density(state) for state in states]

    def step(
        state: np.ndarray, _: float, temperature: float, child: np.random.Generator
    ) -> tuple[np.ndarray, float]:
        gibbs_sweep(indexed, state, child, beta=1.0 / temperature)
        return state, indexed.log_density(state)

    return _exchange(
        step,
        lambda state: tuple(int(value) for value in state),
        states,
        values,
        temperatures,
        children,
        rng,
        n_sweeps,
        burn_in,
        thin,
    )


def tempered_topologies(
    alignment: Mapping[str, np.ndarray],
    k: int,
    temperatures: Sequence[float],
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

    One :func:`snakes_and_ladders.search.gibbs.topology_step` per replica
    per step on the fitted log-likelihood, every replica from ``start``.
    The energy in the exchange ratio is the negative fitted log-likelihood,
    so the replica at temperature one targets the flat-prior weight over
    maximized likelihoods that
    :func:`snakes_and_ladders.search.support.enumerated_support` computes --
    a weight over fitted likelihoods, not a marginal over branch lengths.

    ``scores`` caches fitted log-likelihoods by leaf bipartitions across
    calls, since the fit is the whole cost; every replica shares it.

    Raises
    ------
    ValueError
        If the ladder has fewer than two temperatures or one that is not
        positive, or ``n_sweeps``, ``thin`` or ``burn_in`` is unusable.
    """
    _check_ladder(temperatures, n_sweeps, thin, burn_in)
    cache = {} if scores is None else scores
    score = cached_topology_score(alignment, k, cache, model=model)
    children = rng.spawn(len(temperatures))
    value = score(start)

    def step(
        state: Topology, current: float, temperature: float, child: np.random.Generator
    ) -> tuple[Topology, float]:
        topology, moved_to, _ = topology_step(
            state, current, temperature, child, score, moves=moves
        )
        return topology, moved_to

    return _exchange(
        step,
        leaf_bipartitions,
        [start] * len(temperatures),
        [value] * len(temperatures),
        temperatures,
        children,
        rng,
        n_sweeps,
        burn_in,
        thin,
    )
