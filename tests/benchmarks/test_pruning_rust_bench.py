"""Benchmarks for the Rust CPU Felsenstein pruning backend.

See tests/regression/likelihood/test_pruning_rust.py for correctness. The
fixture cells match test_likelihood_pruning_bench.py and
test_pruning_torch_bench.py, so the NumPy, PyTorch, and Rust numbers are
directly comparable.

Those fixtures sit at 4 and 8 taxa by 200,000 sites -- far past `ROADMAP.md`
§1.2's site range and far short of its taxon range. That is the shape the FFI
copy gap hid in: a cost growing with `n x L` is invisible at 8 taxa. The two
declared-scale cells at the end of this file are where issue #232 measured the
Rust backend returning nothing, and where any later claim about it has to be
made.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.testing import assert_allclose
from pytest_benchmark.fixture import BenchmarkFixture
from sal.fixtures import load_params
from sal.learn.tree import with_uniform_branch_lengths
from sal.likelihood import pruning
from sal.likelihood.device import CROSS_DEVICE_RTOL_FLOAT64
from sal.likelihood.pruning import rust as pruning_rust
from sal.sim.params import SimulationParams
from sal.sim.simulate import simulate_alignment
from sal.sim.simulator import simulate_tree
from sal.sim.topology import random_topology
from sal.sim.tree import Node

from tests._fixtures import FIXTURES_DIR


@pytest.mark.parametrize(
    "fixture_name",
    [
        "tree_jc/stress.yaml",  # 4 taxa, 200_000 sites
        "tree_jc/ci.yaml",  # 4 taxa, 20_000 sites
        "tree_jc/release.yaml",  # 8 taxa, 200_000 sites
    ],
)
def test_rust_log_likelihood_benchmark(
    benchmark: BenchmarkFixture, fixture_name: str
) -> None:
    params = load_params(FIXTURES_DIR / fixture_name, SimulationParams)
    dataset = simulate_tree(params, np.random.default_rng(params.seed))

    result = benchmark(
        pruning_rust.log_likelihood,
        params.tau,
        params.n_states,
        params.pi,
        dataset.alignment,
    )

    # Benchmarks only assert finiteness -- numerical correctness is pinned
    # separately in tests/regression/test_pruning_rust.py.
    assert math.isfinite(result)
    assert result < 0.0  # a log-likelihood, never positive


def test_numpy_vs_rust_forward_pass(benchmark: BenchmarkFixture) -> None:
    """Rust forward pass against the NumPy reference at a fixed size (report both)."""
    params = load_params(FIXTURES_DIR / "tree_jc/ci.yaml", SimulationParams)
    dataset = simulate_tree(params, np.random.default_rng(params.seed))

    numpy_result = pruning.log_likelihood(
        params.tau, params.n_states, params.pi, dataset.alignment
    )
    rust_result = benchmark(
        pruning_rust.log_likelihood,
        params.tau,
        params.n_states,
        params.pi,
        dataset.alignment,
    )

    assert math.isclose(rust_result, numpy_result, abs_tol=1e-9)


# Two cells rather than one, so the *trend* with `n x L` is visible: a cost
# paid at the FFI boundary shows as a ratio that decays between them, which a
# single point cannot report. `n = 1000` is left out -- one call there is a
# second, and the decay is already established by 200.
CELLS = [(20, 11_000), (200, 11_000)]


def _problem(
    cell: tuple[int, int],
) -> tuple[Node, np.ndarray, dict[str, np.ndarray]]:
    n_taxa, n_sites = cell
    rng = np.random.default_rng(n_taxa * 100_000 + n_sites)
    names = [f"t{index}" for index in range(n_taxa)]
    tau = with_uniform_branch_lengths(random_topology(names, rng), 0.1)
    pi = np.full(4, 0.25)
    dataset = simulate_alignment(
        tau=tau, n_states=4, pi=pi, rng=np.random.default_rng(1), n_sites=n_sites
    )
    return tau, pi, dict(dataset.alignment)


@pytest.mark.parametrize(
    "cell", CELLS, ids=lambda cell: f"{cell[0]}taxa_{cell[1]}sites"
)
def test_numpy_pruning_at_declared_scale(
    benchmark: BenchmarkFixture, cell: tuple[int, int]
) -> None:
    """The oracle, and the baseline every port ratio here is against."""
    tau, pi, alignment = _problem(cell)
    result = benchmark(pruning.log_likelihood, tau, 4, pi, alignment)
    assert math.isfinite(result)
    assert result < 0.0


@pytest.mark.parametrize(
    "cell", CELLS, ids=lambda cell: f"{cell[0]}taxa_{cell[1]}sites"
)
def test_rust_pruning_at_declared_scale(
    benchmark: BenchmarkFixture, cell: tuple[int, int]
) -> None:
    """The Rust backend on the same problem, against the oracle.

    Within `CROSS_DEVICE_RTOL_FLOAT64`, not bit-exactly: the oracle reaches
    BLAS for the `(n_sites, k) @ (k, k)` message and the Rust kernel sums each
    row itself, so the two associate the same products differently and land a
    few units in the last place apart. Realized here: 3.5e-15 relative at 20
    taxa and 1.6e-15 at 200, against a bound of 1e-11.
    """
    tau, pi, alignment = _problem(cell)
    expected = pruning.log_likelihood(tau, 4, pi, alignment)
    result = benchmark(pruning_rust.log_likelihood, tau, 4, pi, alignment)
    assert_allclose(result, expected, rtol=CROSS_DEVICE_RTOL_FLOAT64)
