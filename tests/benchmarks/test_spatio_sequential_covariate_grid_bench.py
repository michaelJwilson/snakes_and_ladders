"""The coupled E step under a continuous covariate, Rust against NumPy (issue #1064).

At the stress instance of ``spatio_sequential_counts_covariate`` --- 5,041
vertices, ``M = K = 10``, 2,000 positions, a log-normal exposure and trial
count per observation --- the Rust E step factors the exposure and lays the
trial count out in each of the three ``CovariateRows``, and the NumPy oracle
scores every observation. The rows are built once, outside the measurement,
as a fit builds them. A table by count and distinct exposure is not measured:
at 3,634 counts and 1.0e7 exposures it is 3.7e10 rows, past a ``uint32`` index
and 2.9e13 bytes. The numbers are in ``changelog.d/1064.added.md``.

Each case runs once, as ``test_spatio_sequential_rust_bench.py`` states why.
Correctness is pinned in
``tests/regression/likelihood/test_spatio_sequential_rust_covariate_grid.py``.
"""

from __future__ import annotations

from typing import Any

import pytest
from pytest_benchmark.fixture import BenchmarkFixture
from sal.likelihood.spatio_sequential import (
    COVARIATE_ROWS,
    CovariateRows,
    class_posteriors,
    rust,
)
from sal.sim.count_pairs import CountPairInstance
from sal.sim.count_pairs.rust import fine_instance
from sal.sim.fixtures import fixture

PROBLEM = "spatio_sequential_counts_covariate"


def _instance() -> CountPairInstance:
    """The stress instance, simulated once per session."""
    return fine_instance(fixture(PROBLEM, "stress").path)


def _once(
    benchmark: BenchmarkFixture, call: Any, *arguments: Any, **keywords: Any
) -> Any:
    """Time one call, once."""
    return benchmark.pedantic(  # type: ignore[no-untyped-call]
        call, args=arguments, kwargs=keywords, rounds=1, iterations=1
    )


@pytest.mark.stress
@pytest.mark.parametrize("layout", COVARIATE_ROWS)
def test_rust_class_posteriors_under_a_continuous_covariate(
    benchmark: BenchmarkFixture, layout: CovariateRows
) -> None:
    # The tables are built per call, so the build is inside the measurement.
    instance = _instance()
    rows = rust.observation_rows(
        instance.observations, instance.params.covariate, covariate_rows=layout
    )

    _once(
        benchmark,
        rust.class_posteriors,
        instance.params,
        instance.observations,
        instance.labels,
        covariate_rows=rows,
    )


@pytest.mark.stress
@pytest.mark.parametrize("layout", COVARIATE_ROWS)
def test_rust_external_field_under_a_continuous_covariate(
    benchmark: BenchmarkFixture, layout: CovariateRows
) -> None:
    instance = _instance()
    rows = rust.observation_rows(
        instance.observations, instance.params.covariate, covariate_rows=layout
    )
    posterior = rust.class_posteriors(
        instance.params, instance.observations, instance.labels, covariate_rows=rows
    ).posterior

    _once(
        benchmark,
        rust.external_field,
        instance.params,
        instance.observations,
        instance.labels,
        posterior,
        covariate_rows=rows,
    )


@pytest.mark.release
def test_numpy_class_posteriors_under_a_continuous_covariate(
    benchmark: BenchmarkFixture,
) -> None:
    instance = _instance()

    _once(
        benchmark,
        class_posteriors,
        instance.params,
        instance.observations,
        instance.labels,
    )
