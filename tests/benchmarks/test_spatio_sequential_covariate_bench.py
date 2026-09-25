"""What the covariate costs the compiled backend (issue #658).

`emission_rows` tabulates by count where there is no covariate and by the
distinct `(count, covariate)` pairs where there is. The second calls
`np.unique` over one pair array per channel, which the first does not, so the
pair is the ratio this change is about. Correctness is pinned in
tests/regression/likelihood/test_spatio_sequential_rust_covariate.py.

The E step is timed too, because the tabulation is per call and the question is
what fraction of a call it is — a ratio on the tabulation alone says nothing
about whether the backend is still worth having (root `CLAUDE.md`, effect size).
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
from pytest_benchmark.fixture import BenchmarkFixture
from sal.likelihood import spatio_sequential_rust as rust
from sal.sim.count_pairs import CountPairInstance
from sal.sim.count_pairs_rust import fine_instance
from sal.sim.fixtures import fixture

PROBLEM = "spatio_sequential_counts"


def _instance() -> CountPairInstance:
    return fine_instance(fixture(PROBLEM, "ci").path)


def _covariate(observations: np.ndarray) -> np.ndarray:
    """A per-vertex exposure and a covering trial count, the case this is for."""
    n_positions, n_nodes = observations.shape[:2]
    rng = np.random.default_rng(4)
    covariate = np.empty((n_positions, n_nodes, 2))
    covariate[..., 0] = rng.uniform(0.5, 2.0, size=(1, n_nodes))
    covariate[..., 1] = int(observations[..., 1].max()) + 3
    return covariate


def test_emission_rows_by_count_benchmark(benchmark: BenchmarkFixture) -> None:
    """Tabulated by count: the row is the count and there is no `np.unique`."""
    instance = _instance()

    rows = benchmark(rust.emission_rows, instance.params, instance.observations)

    # Benchmarks assert finiteness only; correctness is pinned in the regression suite.
    assert np.isfinite(rows.total_table).any()


def test_emission_rows_by_pair_benchmark(benchmark: BenchmarkFixture) -> None:
    """Tabulated by the distinct `(count, covariate)` pairs: the added cost."""
    instance = _instance()
    params = replace(instance.params, covariate=_covariate(instance.observations))

    rows = benchmark(rust.emission_rows, params, instance.observations)

    assert np.isfinite(rows.total_table).any()


def test_class_posteriors_uncovaried_benchmark(benchmark: BenchmarkFixture) -> None:
    """One E step without a covariate: what the tabulation is a fraction of."""
    instance = _instance()

    result = benchmark(
        rust.class_posteriors,
        instance.params,
        instance.observations,
        instance.labels,
    )

    assert math.isfinite(float(result.log_evidence.sum()))


def test_class_posteriors_covaried_benchmark(benchmark: BenchmarkFixture) -> None:
    """The same E step conditioned, which is the number a caller pays."""
    instance = _instance()
    params = replace(instance.params, covariate=_covariate(instance.observations))

    result = benchmark(
        rust.class_posteriors, params, instance.observations, instance.labels
    )

    assert math.isfinite(float(result.log_evidence.sum()))
