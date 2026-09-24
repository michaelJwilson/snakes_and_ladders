"""The enumerated law a Potts chain is held to, and the fit against it.

Three modules under ``tests/regression/sample/`` pinned a sampler to the
Boltzmann distribution by enumerating the configurations, weighting them and
counting a chain into the cells. `_exact_distribution` was written out twice
byte for byte and inlined a third time, and the counting loop five times
(issue #863, design-audit row R19).

The referee has to be *independent* of the sampler, which is what makes the
copies a hazard rather than a nuisance: a chain is judged against an oracle
whose only check is that it agrees with the other copies of itself, so an
error in the enumeration reads as a passing sampler in every module that
carries it. One oracle, and a test of the oracle is a test of all three.

The expected counts are passed rather than derived from the observations: a
thinned chain records fewer states than it swept, and which of the two a
comparison is scaled by is the caller's claim about its own chain.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable

import numpy as np
from snakes_and_ladders.likelihood.potts import log_weights
from snakes_and_ladders.sample.statistics import chi_square_p_value
from snakes_and_ladders.sim.graph import PottsGraph

#: A configuration, as a chain records it.
Configuration = tuple[int, ...]


def enumerated_law(
    graph: PottsGraph, field: np.ndarray, temperature: float = 1.0
) -> tuple[dict[Configuration, int], np.ndarray]:
    """Every configuration of ``graph`` and its exact Boltzmann probability.

    Parameters
    ----------
    graph : PottsGraph
        The instance. Enumeration is ``n_states ** n_nodes`` rows, so this is
        an oracle for the small declared lattices and nothing larger.
    field : np.ndarray
        The field, shared ``(n_states,)`` or per site ``(n_nodes, n_states)``;
        its last axis is the alphabet.
    temperature : float
        ``exp(-E / T)`` from the *unscaled* model, so a tempered chain is
        held to an oracle that shares nothing with `sample.tempered`. The
        default divides by one, which is exact, so the untempered law is
        bitwise what it was before this function existed.

    Returns
    -------
    tuple[dict[Configuration, int], np.ndarray]
        The cell index of each configuration, and the probabilities in that
        order.
    """
    n_states = int(field.shape[-1])
    configurations = np.array(
        list(itertools.product(range(n_states), repeat=graph.n_nodes)),
        dtype=np.int64,
    )
    log_probability = log_weights(graph, field, configurations) / temperature
    probability = np.exp(log_probability - log_probability.max())
    probability /= probability.sum()
    index = {tuple(row): position for position, row in enumerate(configurations)}
    return index, probability


def cell_counts(
    index: dict[Configuration, int], states: Iterable[Iterable[int]]
) -> np.ndarray:
    """How often a chain visited each enumerated configuration.

    Returns
    -------
    np.ndarray
        One count per cell of ``index``, in its order.
    """
    observed = np.zeros(len(index))
    for row in states:
        observed[index[tuple(row)]] += 1
    return observed


def fit_p_value(
    index: dict[Configuration, int],
    probability: np.ndarray,
    states: Iterable[Iterable[int]],
    draws: int,
) -> float:
    """A chain's realized frequencies against the enumerated law.

    Parameters
    ----------
    index, probability : dict[Configuration, int], np.ndarray
        From :func:`enumerated_law`.
    states : Iterable[Iterable[int]]
        The recorded configurations.
    draws : int
        What the expected counts are scaled by. The caller's, because a
        thinned chain records fewer states than it swept and which number a
        comparison is against is part of what it claims.

    Returns
    -------
    float
        The chi-square goodness-of-fit tail probability.
    """
    return chi_square_p_value(cell_counts(index, states), probability * draws)
