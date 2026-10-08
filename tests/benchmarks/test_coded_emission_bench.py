"""Coded against dense count emissions, and the coded weighted sum against NumPy's (issue #1340).

A negative binomial of seven states at a log-normal exposure, at 1e4 and 1e6
observations; the 1e6 row is the stress size and the only one a speedup is
read at. ``log_emission`` is timed on a :class:`~sal.emissions.coded.Dense`
and on its :class:`~sal.emissions.coded.Coded` form (encoded once, outside
the timer, as a fit encodes once); ``log_emission_sum`` with ``(K, n)``
weights against ``(w * log_emission(Dense)).sum(axis=1)``, the NumPy route
from the same observations; ``log_emission_partials`` in Rust against
its NumPy oracle ``_partials_numpy`` (#1353). ``tests/regression/test_emissions_coded.py``
pins each pair.
"""

from __future__ import annotations

import numpy as np
import pytest
from sal.emissions import NegativeBinomialEmission
from sal.emissions.coded import (
    Coded,
    Dense,
    _partials_numpy,
    encode,
    log_emission,
    log_emission_partials,
    log_emission_partials_sum,
    log_emission_sum,
)

N_STATES = 7

Instance = tuple[NegativeBinomialEmission, Dense, Coded, np.ndarray]


def _instance(n: int) -> Instance:
    rng = np.random.default_rng(1340)
    family = NegativeBinomialEmission(
        rng.uniform(4.0, 30.0, N_STATES), rng.uniform(20.0, 300.0, N_STATES)
    )
    exposure = np.exp(0.5 * rng.standard_normal(n))
    states = rng.integers(0, N_STATES, n)
    counts = np.asarray(
        family.sample(states, rng, covariate=exposure[:, None]), dtype=np.float64
    ).reshape(-1)
    weights = rng.dirichlet(np.ones(N_STATES), n).T.copy()
    return family, Dense(counts, exposure), encode(counts, exposure), weights


@pytest.fixture(scope="module", params=[10_000, 1_000_000], ids=["1e4", "1e6"])
def instance(request: pytest.FixtureRequest) -> Instance:
    return _instance(int(request.param))


def _numpy_sum(
    family: NegativeBinomialEmission, dense: Dense, weights: np.ndarray
) -> np.ndarray:
    return (weights * log_emission(family, dense)).sum(axis=1)


@pytest.mark.release
@pytest.mark.benchmark
def test_coded_bench_dense(benchmark: object, instance: Instance) -> None:
    """The reference: ``log_emission`` on every observation."""
    family, dense, _, _ = instance
    benchmark(log_emission, family, dense)  # type: ignore[operator]


@pytest.mark.release
@pytest.mark.benchmark
def test_coded_bench_coded(benchmark: object, instance: Instance) -> None:
    """Tables at the distinct counts, the covariate per observation, gathered."""
    family, _, coded, _ = instance
    benchmark(log_emission, family, coded)  # type: ignore[operator]


@pytest.mark.release
@pytest.mark.benchmark
def test_coded_bench_numpy_sum(benchmark: object, instance: Instance) -> None:
    """The reference sum: the dense score, weighted and summed by NumPy."""
    family, dense, _, weights = instance
    benchmark(_numpy_sum, family, dense, weights)  # type: ignore[operator]


@pytest.mark.release
@pytest.mark.benchmark
def test_coded_bench_sum(benchmark: object, instance: Instance) -> None:
    """``log_emission_sum``: the coded score and the Rust per-state bincount."""
    family, _, coded, weights = instance
    benchmark(log_emission_sum, family, coded, weights)  # type: ignore[operator]


@pytest.mark.release
@pytest.mark.benchmark
def test_coded_bench_partials_numpy(benchmark: object, instance: Instance) -> None:
    """The oracle: ``_partials_numpy`` over ``digamma_rising``, the reference the port is read against."""
    family, _, coded, _ = instance
    benchmark(_partials_numpy, family, coded)  # type: ignore[operator]


@pytest.mark.release
@pytest.mark.benchmark
def test_coded_bench_partials(benchmark: object, instance: Instance) -> None:
    """``log_emission_partials`` in Rust (#1353): one crossing, the GIL released."""
    family, _, coded, _ = instance
    benchmark(log_emission_partials, family, coded)  # type: ignore[operator]


@pytest.mark.release
@pytest.mark.benchmark
def test_coded_bench_partials_sum(benchmark: object, instance: Instance) -> None:
    """``log_emission_partials_sum`` with ``(K, n)`` weights: the partials and the per-state sum."""
    family, _, coded, weights = instance
    benchmark(log_emission_partials_sum, family, coded, weights)  # type: ignore[operator]
