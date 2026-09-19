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
from snakes_and_ladders.sim.tree import balanced_tree

from tests._fixtures import FIXTURES_DIR


@pytest.mark.parametrize(
    "fixture_name",
    [
        "tree_jc/stress.yaml",  # 4 taxa, 200_000 sites
        "tree_jc/ci.yaml",  # 4 taxa, 20_000 sites
        "tree_jc/release.yaml",  # 8 taxa, 200_000 sites
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
    params = load_simulation_params(FIXTURES_DIR / "tree_jc/ci.yaml")
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
    params = load_simulation_params(FIXTURES_DIR / "tree_jc/ci.yaml")
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


#: Leaves whose partials the post-order visits. The hoisted traversal of
#: issue #754 removes a Python frame and two dictionary lookups *per node*,
#: so what it buys scales with this and not with the site count; both are
#: parametrized here so the reader can read that off the table rather than
#: take it on the docstring's word.
@pytest.mark.parametrize("n_sites", [500, 2_000])
@pytest.mark.parametrize("n_taxa", [20, 50])
def test_gradient_at_many_leaves_benchmark(
    benchmark: BenchmarkFixture, n_taxa: int, n_sites: int
) -> None:
    """One value and gradient on a many-leaved tree: the search's inner unit.

    `test_torch_log_likelihood_benchmark` above is four and eight leaves over
    hundreds of thousands of sites, which is the opposite shape: there the
    per-node Python is nothing beside the tensors. A topology search is here
    instead --- tens of leaves, thousands of sites --- and `--tier mid` of
    `profile_hotpaths.py` ranked the post-order at 17.5% of a 20-taxon NNI
    search and 19.1% of an SPR one.
    """
    tau = balanced_tree(n_taxa, 0.5)
    dataset = simulate_alignment(
        tau=tau,
        k=4,
        pi=np.full(4, 0.25),
        rng=np.random.default_rng(754),
        n_sites=n_sites,
    )
    alignment = dict(dataset.alignment)
    branch_lengths = pruning_torch.branch_lengths_from_tree(tau).requires_grad_(True)

    def _value_and_gradient() -> tuple[torch.Tensor, torch.Tensor]:
        value = pruning_torch.log_likelihood(
            tau, 4, np.full(4, 0.25), alignment, branch_lengths
        )
        (gradient,) = torch.autograd.grad(value, branch_lengths)
        return value, gradient

    value, gradient = benchmark(_value_and_gradient)

    assert math.isfinite(float(value))
    assert bool(torch.isfinite(gradient).all())
