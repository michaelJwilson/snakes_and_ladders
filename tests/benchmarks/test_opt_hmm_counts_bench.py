"""Benchmarks for the count-emission HMM objectives.

See tests/regression/opt/test_opt_hmm_counts.py for correctness. All four are
measured at one shape, because the comparison between them is the only thing
these numbers say: the recursion is shared, so the spread across the four is
the cost of the emission term alone.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from pytest_benchmark.fixture import BenchmarkFixture
from snakes_and_ladders.emissions import (
    BetaBinomialEmission,
    BinomialEmission,
    NegativeBinomialEmission,
    PoissonEmission,
)
from snakes_and_ladders.opt.hmm import (
    BetaBinomialHmmObjective,
    BinomialHmmObjective,
    NegativeBinomialHmmObjective,
    PoissonHmmObjective,
)
from snakes_and_ladders.sim.hmm import HmmParams, simulate_sequences

CountEmission = (
    PoissonEmission | BinomialEmission | NegativeBinomialEmission | BetaBinomialEmission
)
CountObjective = (
    PoissonHmmObjective
    | BinomialHmmObjective
    | NegativeBinomialHmmObjective
    | BetaBinomialHmmObjective
)

TRIALS = np.array([12, 12, 12])
INITIAL = np.array([0.5, 0.3, 0.2])
TRANSITION = np.array([[0.70, 0.20, 0.10], [0.15, 0.75, 0.10], [0.20, 0.20, 0.60]])


def _truth(name: str) -> CountEmission:
    """One three-state count family at the categorical fixture's shape."""
    families: dict[str, CountEmission] = {
        "poisson": PoissonEmission([2.0, 6.0, 10.0]),
        "binomial": BinomialEmission(TRIALS, [0.2, 0.5, 0.8]),
        "negative_binomial": NegativeBinomialEmission(
            [2.0, 5.0, 9.0], [2.0, 6.0, 10.0]
        ),
        "beta_binomial": BetaBinomialEmission(TRIALS, [1.0, 3.0, 6.0], [6.0, 3.0, 1.0]),
    }
    return families[name]


def _objective(name: str) -> tuple[CountObjective, torch.Tensor]:
    """The objective and a truth point, at 600 sequences of length 15."""
    truth = _truth(name)
    params = HmmParams(
        n_states=3,
        sequence_length=15,
        n_sequences=600,
        initial=INITIAL,
        transition=TRANSITION,
        emissions=truth,
        seed=20260903,
        tolerance=1e-12,
    )
    observations = simulate_sequences(params).observations
    objective: CountObjective
    if name == "poisson":
        objective = PoissonHmmObjective(observations, 3)
    elif name == "binomial":
        objective = BinomialHmmObjective(observations, 3, TRIALS)
    elif name == "negative_binomial":
        objective = NegativeBinomialHmmObjective(observations, 3)
    else:
        objective = BetaBinomialHmmObjective(observations, 3, TRIALS)
    values = [value.numpy() for value in truth.named_parameters().values()]
    return objective, objective.theta_from_truth(INITIAL, TRANSITION, *values)


FAMILIES = ("poisson", "binomial", "negative_binomial", "beta_binomial")


@pytest.mark.parametrize("name", FAMILIES)
def test_count_hmm_objective_benchmark(benchmark: BenchmarkFixture, name: str) -> None:
    objective, theta = _objective(name)

    result = benchmark(objective, theta)

    # Benchmarks only assert finiteness -- correctness is pinned in
    # tests/regression/opt/test_opt_hmm_counts.py.
    assert math.isfinite(float(result))


@pytest.mark.parametrize("name", FAMILIES)
def test_count_hmm_objective_and_gradient_benchmark(
    benchmark: BenchmarkFixture, name: str
) -> None:
    objective, theta = _objective(name)

    def _value_and_gradient() -> float:
        point = theta.detach().clone().requires_grad_(True)
        value = objective(point)
        torch.autograd.grad(value, point)
        return float(value.detach())

    assert math.isfinite(benchmark(_value_and_gradient))
