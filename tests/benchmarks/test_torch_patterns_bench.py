"""The PyTorch patterns of issue #544, each timed against the path in the tree.

Every pattern the ticket itemized was declined, and a decline is worth no more
than the measurement under it. These are the arms `STATUS.md` reports, kept so
the ratios are reproduced rather than remembered: a proposer who reaches for
``inference_mode``, a batched objective, a contiguous transition operand or an
eigendecomposition of ``Q`` runs this module and reads what it costs.

Correctness is pinned in ``tests/regression/likelihood/test_torch_patterns.py``
--- each arm there reproduces the value its counterpart produces --- so this
module asserts only that the value is finite, per `DEV.md`'s rule for this
directory.

``pytest-benchmark`` disables itself under ``pytest-xdist`` (issue #405), so a
number taken from a distributed run is no number; ``tests/conftest.py`` fails a
benchmark that reaches one.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from pytest_benchmark.fixture import BenchmarkFixture
from snakes_and_ladders.likelihood.objective import (
    BranchLengthObjective,
    SubstitutionModelObjective,
)
from snakes_and_ladders.likelihood.pruning_torch import (
    branch_lengths_from_tree,
    log_likelihood,
    transition_probabilities,
)
from snakes_and_ladders.sim.simulate import simulate_alignment
from snakes_and_ladders.sim.tree import Node

from tests._fixtures import EIGHT_TAXA, load_fixture

#: The size ``test_pruning_gradient_bench.py`` reports its routes at, so the
#: two modules' numbers sit on one scale.
_SITES = 20_000

#: Starts in the batched arm of item 2. Four rather than one, because a single
#: start is what the batching is meant to beat.
_BATCH = 4


def _dataset() -> tuple[Node, int, np.ndarray, dict[str, np.ndarray], torch.Tensor]:
    params = load_fixture(EIGHT_TAXA)
    dataset = simulate_alignment(
        tau=params.tau,
        k=params.k,
        pi=params.pi,
        rng=np.random.default_rng(params.seed),
        n_sites=_SITES,
    )
    return (
        params.tau,
        params.k,
        params.pi,
        dataset.alignment,
        branch_lengths_from_tree(params.tau),
    )


@pytest.mark.parametrize("mode", ["no_grad", "inference_mode"])
def test_evaluation_mode_benchmark(benchmark: BenchmarkFixture, mode: str) -> None:
    """Item 1: one gradient-free evaluation under each of the two modes."""
    tau, k, pi, alignment, lengths = _dataset()
    context = torch.no_grad if mode == "no_grad" else torch.inference_mode

    def _evaluate() -> float:
        with context():
            return float(log_likelihood(tau, k, pi, alignment, lengths))

    result = benchmark(_evaluate)
    assert math.isfinite(result)


@pytest.mark.parametrize("arm", ["sequential", "vmap"])
def test_batched_objective_benchmark(benchmark: BenchmarkFixture, arm: str) -> None:
    """Item 2: four starting points at one topology, one at a time and batched."""
    tau, k, pi, alignment, _ = _dataset()
    objective = BranchLengthObjective(tau, k, pi, alignment)
    batch = torch.stack(
        [objective.initial() + 0.01 * offset for offset in range(_BATCH)]
    )

    def _sequential() -> float:
        total = 0.0
        for row in batch:
            point = row.clone().requires_grad_(True)
            value = objective(point)
            value.backward()  # type: ignore[no-untyped-call]
            total += float(value.detach())
        return total

    def _vmapped() -> float:
        point = batch.clone().requires_grad_(True)
        values = torch.vmap(objective)(point)
        values.sum().backward()  # type: ignore[no-untyped-call]
        return float(values.detach().sum())

    result = benchmark(_sequential if arm == "sequential" else _vmapped)
    assert math.isfinite(result)


@pytest.mark.parametrize("operand", ["view", "contiguous"])
def test_transition_operand_benchmark(
    benchmark: BenchmarkFixture, operand: str
) -> None:
    """Item 5: the recursion's matmul against a transposed view and a copy of it."""
    tau, k, _, _, lengths = _dataset()
    transitions = transition_probabilities(lengths, k, None)
    partial = torch.rand((_SITES, k), dtype=torch.float64)
    right = transitions[0].T if operand == "view" else transitions[0].T.contiguous()

    result = benchmark(lambda: float((partial @ right).sum()))
    assert math.isfinite(result)


@pytest.mark.parametrize("route", ["closed_form", "matrix_exp", "eigendecomposition"])
def test_transition_construction_benchmark(
    benchmark: BenchmarkFixture, route: str
) -> None:
    """Item 6: every branch's ``P(t)``, by each of the three constructions."""
    tau, k, _, alignment, lengths = _dataset()
    objective = SubstitutionModelObjective(tau, k, alignment)
    rate_matrix = objective.rate_matrix(objective.initial()).detach()

    def _eigendecomposition() -> torch.Tensor:
        values, vectors = torch.linalg.eig(rate_matrix)
        inverse = torch.linalg.inv(vectors)
        scaled = torch.exp(values[None, :] * lengths[:, None].to(values.dtype))
        transitions: torch.Tensor = (
            (vectors[None] * scaled[:, None, :]) @ inverse[None]
        ).real
        return transitions

    routes = {
        "closed_form": lambda: transition_probabilities(lengths, k, None),
        "matrix_exp": lambda: transition_probabilities(lengths, k, rate_matrix),
        "eigendecomposition": _eigendecomposition,
    }

    result = benchmark(routes[route])
    assert math.isfinite(float(result.sum()))
