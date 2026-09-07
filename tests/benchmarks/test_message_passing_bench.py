"""Benchmarks for message passing on a factor graph.

See tests/regression/likelihood/test_message_passing.py for correctness. Each
cell is paired with the specialised evaluator it generalises, so the cost of
the generality -- a table per factor and a dictionary per message, against a
recursion that knows its shape -- is a measured ratio and not a guess.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from pytest_benchmark.fixture import BenchmarkFixture
from snakes_and_ladders.likelihood.belief_propagation import belief_propagation
from snakes_and_ladders.likelihood.message_passing import MessageSchedule, sum_product
from snakes_and_ladders.opt.hmm import forward_log_likelihood_from_density
from snakes_and_ladders.sim.factor_graph import from_hmm, from_potts
from snakes_and_ladders.sim.graph import BoundaryCondition, lattice_graph

_LENGTH = 200
_STATES = 4
FIELD = np.array([0.3, -0.7, 0.15])
LATTICE = lattice_graph((8, 8), coupling=0.3, boundary=BoundaryCondition.OPEN)


def _chain() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    log_initial = np.log(rng.dirichlet(np.ones(_STATES)))
    log_transition = np.log(rng.dirichlet(np.ones(_STATES), size=_STATES))
    log_density = rng.normal(size=(_LENGTH, _STATES))
    return log_initial, log_transition, log_density


def test_sum_product_on_a_chain_benchmark(benchmark: BenchmarkFixture) -> None:
    """The general algorithm on the chain the forward recursion is written for."""
    graph = from_hmm(*_chain())

    result = benchmark(sum_product, graph)

    # Benchmarks assert finiteness only; correctness is pinned in the regression suite.
    assert math.isfinite(result.log_partition)


def test_forward_recursion_on_the_same_chain_benchmark(
    benchmark: BenchmarkFixture,
) -> None:
    """The specialised evaluator, for the ratio."""
    log_initial, log_transition, log_density = _chain()
    tensors = (
        torch.as_tensor(log_density)[None],
        torch.as_tensor(log_initial),
        torch.as_tensor(log_transition),
    )

    result = benchmark(forward_log_likelihood_from_density, *tensors)

    assert math.isfinite(float(result))


def test_flooding_on_a_lattice_benchmark(benchmark: BenchmarkFixture) -> None:
    """Sum-product with flooding on the loopy graph belief_propagation is written for."""
    graph = from_potts(LATTICE, FIELD)

    result = benchmark(sum_product, graph, schedule=MessageSchedule.FLOODING)

    assert math.isfinite(result.log_partition)


def test_belief_propagation_on_the_same_lattice_benchmark(
    benchmark: BenchmarkFixture,
) -> None:
    """The specialised evaluator, for the ratio."""
    result = benchmark(belief_propagation, LATTICE, FIELD)

    assert math.isfinite(result.bethe_log_partition)
