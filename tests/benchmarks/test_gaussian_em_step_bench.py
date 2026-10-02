"""Ten Gaussian EM iterations, mixture and HMM, on both backends (issue #1160).

The collapse check runs once per M step on every route: an ``O(n_states)``
comparison beside the ``O(n_samples * n_states)`` E step. Timed at 10^5 draws,
the size #986 measured the streamed mixture step at, on a fit that never
collapses, so the number is the check's cost on the path every fit takes.
See tests/regression/opt/test_opt_gaussian_collapse.py for correctness.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from pytest_benchmark.fixture import BenchmarkFixture
from sal.backend import Backend
from sal.emissions import GaussianEmission, pooled_variance_floor
from sal.opt.em import EmConfig
from sal.opt.hmm.estimation import baum_welch_family
from sal.opt.mixture import expectation_maximization

MEAN = np.array([-6.0, 0.0, 6.0])
SCALE = np.array([1.0, 1.5, 1.0])
TEN = EmConfig(max_iterations=10, tolerance=0.0)


def _draws() -> np.ndarray:
    """10^5 draws from three separated components, labelled i.i.d."""
    rng = np.random.default_rng(1160)
    labels = rng.integers(0, 3, 100_000)
    return np.asarray(rng.normal(MEAN[labels], SCALE[labels]))


@pytest.mark.parametrize("backend", [Backend.PYTHON, Backend.RUST])
def test_gaussian_mixture_em_benchmark(
    benchmark: BenchmarkFixture, backend: Backend
) -> None:
    draws = _draws()
    start = GaussianEmission(MEAN + 0.5, SCALE * 1.2, pooled_variance_floor(draws))
    weights = torch.full((3,), 1.0 / 3.0, dtype=torch.float64)

    fit = benchmark(
        expectation_maximization, draws, weights, start, TEN, backend=backend
    )

    assert fit.frozen == ()
    assert math.isfinite(fit.log_likelihood)


@pytest.mark.parametrize("backend", [Backend.PYTHON, Backend.RUST])
def test_gaussian_baum_welch_benchmark(
    benchmark: BenchmarkFixture, backend: Backend
) -> None:
    draws = _draws().reshape(100, 1_000)
    start = GaussianEmission(MEAN + 0.5, SCALE * 1.2, pooled_variance_floor(draws))
    uniform = torch.full((3,), -math.log(3.0), dtype=torch.float64)

    fit = benchmark(
        baum_welch_family,
        draws,
        uniform,
        uniform.repeat(3, 1),
        start,
        TEN,
        backend=backend,
    )

    assert fit.frozen == ()
    assert math.isfinite(fit.log_likelihood)
