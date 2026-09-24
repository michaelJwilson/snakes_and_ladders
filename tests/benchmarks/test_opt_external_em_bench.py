"""Baum--Welch and mixture EM beside hmmlearn's and scikit-learn's (issue #975).

Agreement is pinned in `tests/validation/test_hmmlearn.py` and
`test_scikit_learn.py`, and the runtime goals there. Each row times ten
iterations of the package's EM from one start and records beside it the
framework's `fit` time for the same ten on the same data, the median of
`REPEATS` subprocess runs, and the peak resident memory each side's fit adds,
each read in a fresh interpreter (#987).

Sizes: Baum--Welch at the gate fixture (`hmm/ci.yaml`, 9,000 positions) and at
10⁵ and 10⁶ positions, sequences of 100 from the fixture's model; mixture EM
at 4,000 draws, the size `test_opt_mixture_bench.py` uses, and at 10⁵ and 10⁶.
The 10⁶ rows carry `release`.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch
from pytest_benchmark.fixture import BenchmarkFixture
from snakes_and_ladders.emissions import GaussianEmission
from snakes_and_ladders.fixtures import load_params
from snakes_and_ladders.opt.hmm import baum_welch
from snakes_and_ladders.opt.mixture import expectation_maximization
from snakes_and_ladders.sim.hmm import HmmParams, simulate_sequences
from snakes_and_ladders.validation import hmmlearn, scikit_learn

from tests._fixtures import FIXTURES_DIR
from tests._frameworks import requires
from tests.validation._goals import median_package

#: Subprocess runs whose median each framework figure is.
REPEATS = 3

#: EM iterations per timed fit, on both sides.
N_ITER = 10


def _simplex(rng: np.random.Generator, *shape: int) -> np.ndarray:
    draw = rng.random(shape) + 0.5
    return np.asarray(draw / draw.sum(axis=-1, keepdims=True))


@requires("hmmlearn")
@pytest.mark.parametrize(
    "n_sequences",
    [None, 1_000, pytest.param(10_000, marks=pytest.mark.release)],
)
def test_baum_welch_beside_hmmlearn_benchmark(
    benchmark: BenchmarkFixture, n_sequences: int | None
) -> None:
    params = load_params(FIXTURES_DIR / "hmm" / "ci.yaml", HmmParams)
    if n_sequences is not None:
        params = dataclasses.replace(params, lengths=(100,) * n_sequences)
    observations = simulate_sequences(params).observations
    rng = np.random.default_rng(975)
    start = (
        _simplex(rng, params.n_states),
        _simplex(rng, params.n_states, params.n_states),
        _simplex(rng, params.n_states, params.n_symbols),
    )
    fits = [hmmlearn.baum_welch(observations, *start, N_ITER) for _ in range(REPEATS)]
    benchmark.extra_info["hmmlearn_fit_s"] = float(
        np.median([fit.seconds for fit in fits])
    )
    benchmark.extra_info["hmmlearn_peak_bytes"] = float(
        np.median([fit.peak_bytes for fit in fits])
    )
    ours = {
        "observations": observations,
        "initial": start[0],
        "transition": start[1],
        "emission": start[2],
        "n_iter": np.asarray(N_ITER),
    }
    benchmark.extra_info["package_peak_bytes"] = median_package(
        "baum_welch", ours, "peak_bytes", repeats=REPEATS
    )
    logs = [torch.log(torch.as_tensor(p)) for p in start]

    fit = benchmark(
        baum_welch, observations, *logs, max_iterations=N_ITER, tolerance=-np.inf
    )

    assert np.isfinite(fit.log_likelihood)


@requires("scikit_learn")
@pytest.mark.parametrize(
    "n_samples",
    [4_000, 100_000, pytest.param(1_000_000, marks=pytest.mark.release)],
)
def test_mixture_em_beside_scikit_learn_benchmark(
    benchmark: BenchmarkFixture, n_samples: int
) -> None:
    rng = np.random.default_rng(975)
    component = rng.choice(3, size=n_samples, p=[0.3, 0.3, 0.4])
    observations = rng.normal(
        np.array([-4.0, 0.0, 5.0])[component], np.array([1.0, 1.5, 1.0])[component]
    )
    weights = np.array([0.3, 0.3, 0.4])
    mean = np.array([-3.0, 0.5, 4.0])
    scale = np.array([1.2, 1.0, 1.3])
    fits = [
        scikit_learn.expectation_maximization(
            observations, weights, mean, scale, N_ITER
        )
        for _ in range(REPEATS)
    ]
    benchmark.extra_info["scikit_learn_fit_s"] = float(
        np.median([fit.seconds for fit in fits])
    )
    benchmark.extra_info["scikit_learn_peak_bytes"] = float(
        np.median([fit.peak_bytes for fit in fits])
    )
    ours = {
        "observations": observations,
        "weights": weights,
        "mean": mean,
        "scale": scale,
        "n_iter": np.asarray(N_ITER),
    }
    benchmark.extra_info["package_peak_bytes"] = median_package(
        "mixture_em", ours, "peak_bytes", repeats=REPEATS
    )

    fit = benchmark(
        expectation_maximization,
        observations,
        torch.as_tensor(weights),
        GaussianEmission(mean, scale, 1e-12),
        max_iterations=N_ITER,
        tolerance=-np.inf,
    )

    assert np.isfinite(fit.log_likelihood)
