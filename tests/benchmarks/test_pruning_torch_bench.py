"""Benchmarks for the PyTorch Felsenstein pruning backend.

See tests/regression/test_pruning_torch.py for correctness. Two shapes are
measured: the forward pass alone, at the same (taxa, site) fixtures as
``test_likelihood_pruning_bench.py`` so the two numbers are comparable, and
one gradient-descent step fitting a general rate matrix Q via
``pruning_torch.log_likelihood``'s ``rate_matrix`` (``torch.matrix_exp``)
path -- the general-Q path issue #70 asks be exercised, even though JC's Q
is fully determined by k. The fitting loop lives in this test only:
``snakes_and_ladders.likelihood`` gains no parameter-fitting feature, out of scope for
issue #70 (that is ``snakes_and_ladders.opt``'s follow-on job).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from pytest_benchmark.fixture import BenchmarkFixture
from snakes_and_ladders.likelihood import pruning, pruning_torch
from snakes_and_ladders.sim.params import load_simulation_params
from snakes_and_ladders.sim.simulate import simulate_alignment

from tests._fixtures import FIXTURES_DIR


@pytest.mark.parametrize(
    "fixture_name",
    [
        "simulation_params.yaml",  # 4 taxa, 200_000 sites
        "simulation_params_small_sites.yaml",  # 4 taxa, 20_000 sites
        "simulation_params_8taxa.yaml",  # 8 taxa, 200_000 sites
    ],
)
def test_torch_log_likelihood_benchmark(
    benchmark: BenchmarkFixture, fixture_name: str
) -> None:
    params = load_simulation_params(FIXTURES_DIR / fixture_name)
    dataset = simulate_alignment(
        tau=params.tau,
        k=params.k,
        pi=params.pi,
        rng=np.random.default_rng(params.seed),
        n_sites=params.n_sites,
    )
    branch_lengths = pruning_torch.branch_lengths_from_tree(params.tau)

    result = benchmark(
        pruning_torch.log_likelihood,
        params.tau,
        params.k,
        params.pi,
        dataset.alignment,
        branch_lengths,
    )

    # Benchmarks only assert finiteness -- numerical correctness is pinned
    # separately in tests/regression/test_pruning_torch.py.
    assert math.isfinite(float(result))
    assert float(result) < 0.0  # a log-likelihood, never positive


def test_numpy_vs_torch_forward_pass(benchmark: BenchmarkFixture) -> None:
    """Torch forward pass against the NumPy reference at a fixed size (report both)."""
    params = load_simulation_params(FIXTURES_DIR / "simulation_params_small_sites.yaml")
    dataset = simulate_alignment(
        tau=params.tau,
        k=params.k,
        pi=params.pi,
        rng=np.random.default_rng(params.seed),
        n_sites=params.n_sites,
    )
    branch_lengths = pruning_torch.branch_lengths_from_tree(params.tau)

    numpy_result = pruning.log_likelihood(
        params.tau, params.k, params.pi, dataset.alignment
    )
    torch_result = benchmark(
        pruning_torch.log_likelihood,
        params.tau,
        params.k,
        params.pi,
        dataset.alignment,
        branch_lengths,
    )

    assert math.isclose(float(torch_result), numpy_result, abs_tol=1e-9)


def test_fit_general_rate_matrix_benchmark(benchmark: BenchmarkFixture) -> None:
    """One Adam step fitting a general Q (``torch.matrix_exp`` path), not just k."""
    params = load_simulation_params(FIXTURES_DIR / "simulation_params_small_sites.yaml")
    dataset = simulate_alignment(
        tau=params.tau,
        k=params.k,
        pi=params.pi,
        rng=np.random.default_rng(params.seed),
        n_sites=params.n_sites,
    )
    branch_lengths = pruning_torch.branch_lengths_from_tree(params.tau).requires_grad_(
        True
    )
    raw_rates = torch.zeros(
        (params.k, params.k), dtype=torch.float64, requires_grad=True
    )
    optimizer = torch.optim.Adam([branch_lengths, raw_rates], lr=1e-2)

    def _fit_step() -> float:
        off_diagonal = torch.nn.functional.softplus(raw_rates) * (
            1.0 - torch.eye(params.k, dtype=torch.float64)
        )
        rate_matrix = off_diagonal - torch.diag(off_diagonal.sum(dim=1))

        optimizer.zero_grad()
        negative_log_likelihood = -pruning_torch.log_likelihood(
            params.tau,
            params.k,
            params.pi,
            dataset.alignment,
            branch_lengths,
            rate_matrix=rate_matrix,
        )
        negative_log_likelihood.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
        return float(negative_log_likelihood.detach())

    result = benchmark(_fit_step)
    assert math.isfinite(result)
