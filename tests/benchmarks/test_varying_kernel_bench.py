"""What the per-step transition kernel costs, in time and in memory (issue #653).

Two cells per size: the ``(K, K)`` matrix 26 call sites pass, and the
``(T - 1, K, K)`` form admitted alongside it. The pair is the point --- the
constant path must not pay for a shape it does not use, and what the varying
path costs must be a number rather than an expectation. Correctness is pinned
in tests/regression/likelihood/test_varying_transition_kernel.py.

The memory is the larger finding and does not appear in a timing table: at
``T = 100,000``, ``K = 8`` the kernel argument goes from 512 B to 48.83 MiB,
which is ``T`` times the same numbers and the reason the constant shape stays.
"""

from __future__ import annotations

import math

import numpy as np
from pytest_benchmark.fixture import BenchmarkFixture
from sal.likelihood.forward_backward import forward_backward

_STATES = 8
_LENGTH = 2000


def _chain() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    log_density = np.log(rng.dirichlet(np.ones(_STATES), size=_LENGTH))
    log_initial = np.log(rng.dirichlet(np.ones(_STATES)))
    log_transition = np.log(rng.dirichlet(np.ones(_STATES), size=_STATES))
    return log_density, log_initial, log_transition


def test_forward_backward_with_one_kernel_benchmark(
    benchmark: BenchmarkFixture,
) -> None:
    """The constant shape: the baseline the varying one is read against."""
    log_density, log_initial, log_transition = _chain()

    result = benchmark(forward_backward, log_density, log_initial, log_transition)

    # Benchmarks assert finiteness only; correctness is pinned in the regression suite.
    assert math.isfinite(result.log_evidence)


def test_forward_backward_with_a_kernel_per_step_benchmark(
    benchmark: BenchmarkFixture,
) -> None:
    """The same chain and the same numbers, addressed one kernel per transition."""
    log_density, log_initial, log_transition = _chain()
    kernels = np.repeat(log_transition[None], _LENGTH - 1, axis=0)

    result = benchmark(forward_backward, log_density, log_initial, kernels)

    assert math.isfinite(result.log_evidence)
