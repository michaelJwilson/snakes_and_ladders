"""The enumerated law a Potts chain is held to, and the fit against it.

One oracle for the samplers under ``tests/regression/sample/``, which carried
the enumeration twice byte for byte, inlined a third time, and the counting
loop five times (issue #863, design-audit row R19). The referee must be
independent of the sampler, so a test of this oracle is a test of all three.
Expected counts are passed, not derived: a thinned chain records fewer states
than it swept, and which count scales a comparison is the caller's claim.
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

    Small lattices only; ``exp(-E / T)`` shares nothing with `sample.tempered`.
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
    """How often a chain visited each enumerated configuration, in ``index`` order."""
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
    """A chain's chi-square p-value against the enumerated law.

    ``draws`` scales the expected counts: a thinned chain records fewer states.
    """
    return chi_square_p_value(cell_counts(index, states), probability * draws)
