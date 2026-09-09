"""Three routes to the gradient of the pruning recursion, timed against each other.

The comparison issue #449 asks for: PyTorch's taped backward (the oracle),
``burn``'s tape in Rust, and the analytic two-pass backward behind one
``torch.autograd.Function``. One evaluation is a forward pass and a
``backward()``, which is what a fitting step costs.

Correctness is pinned in ``tests/regression/likelihood/test_pruning_analytic.py``
and ``test_pruning_burn.py``; these assert only that the value is finite, per
`DEV.md`'s rule for this directory. The through-the-binding numbers are these;
the `burn` kernel alone is `benches/oxi_snakes_and_ladders_bench.rs`, and the
difference between the two is the FFI boundary.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest
import torch
from pytest_benchmark.fixture import BenchmarkFixture
from snakes_and_ladders.likelihood import pruning_analytic, pruning_burn, pruning_torch
from snakes_and_ladders.sim.params import load_simulation_params
from snakes_and_ladders.sim.simulate import simulate_alignment
from snakes_and_ladders.sim.tree import Node

from tests._fixtures import FIXTURES_DIR

_ROUTES: dict[str, Callable[..., torch.Tensor]] = {
    "taped": pruning_torch.log_likelihood,
    "burn": pruning_burn.log_likelihood,
    "analytic": pruning_analytic.log_likelihood,
}


def _dataset(
    fixture_name: str, n_sites: int
) -> tuple[Node, int, np.ndarray, dict[str, np.ndarray], torch.Tensor]:
    params = load_simulation_params(FIXTURES_DIR / fixture_name)
    dataset = simulate_alignment(
        tau=params.tau,
        k=params.k,
        pi=params.pi,
        rng=np.random.default_rng(params.seed),
        n_sites=n_sites,
    )
    return (
        params.tau,
        params.k,
        params.pi,
        dataset.alignment,
        pruning_torch.branch_lengths_from_tree(params.tau),
    )


@pytest.mark.parametrize("route", sorted(_ROUTES))
@pytest.mark.parametrize(
    ("fixture_name", "n_sites"),
    [
        ("tree_jc/ci.yaml", 20_000),  # 4 taxa
        ("tree_jc/release.yaml", 20_000),  # 8 taxa
    ],
)
def test_gradient_benchmark(
    benchmark: BenchmarkFixture, route: str, fixture_name: str, n_sites: int
) -> None:
    """One forward pass and one backward, per route."""
    tau, k, pi, alignment, lengths = _dataset(fixture_name, n_sites)
    evaluate = _ROUTES[route]

    def _gradient_step() -> float:
        branch_lengths = lengths.clone().requires_grad_(True)
        value = evaluate(tau, k, pi, alignment, branch_lengths)
        value.backward()  # type: ignore[no-untyped-call]
        return float(value.detach())

    result = benchmark(_gradient_step)
    assert math.isfinite(result)
    assert result < 0.0  # a log-likelihood, never positive
