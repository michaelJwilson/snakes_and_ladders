"""NumPy against Rust for the coupled E step and its field (issue #399).

Both numbers at every declared bin factor of the 5,041-vertex instance, per
`CLAUDE.md`'s Measurement rule: the NumPy reference is what the port is
judged against, and a kernel timed only against itself reports nothing.

The cases are tiered by what they cost the *NumPy* side, which is the slow
one: at bin factor 10 one field is over three minutes, at factor 5 twice
that, and at factor 1 it is half an hour. So the NumPy field is measured at
factor 10 and the NumPy E step at factors 10 and 5, both under `release`;
the Rust cases run at every factor under `stress`, since the whole of the
Rust side of the sweep is under twenty seconds at the largest.

Each case runs once: `pytest-benchmark`'s default of many rounds would turn a
three-minute measurement into an hour, and the quantity wanted here is the
one sweep the key-fixture budget is spent on.
"""

from __future__ import annotations

from typing import Any

import pytest
from pytest_benchmark.fixture import BenchmarkFixture
from snakes_and_ladders.likelihood import spatio_sequential_rust as rust
from snakes_and_ladders.likelihood.spatio_sequential import (
    class_posteriors,
    external_field,
)
from snakes_and_ladders.sim.count_pairs import CountPairInstance
from snakes_and_ladders.sim.count_pairs_rust import binned_instance
from snakes_and_ladders.sim.fixtures import fixture

PROBLEM = "spatio_sequential_counts"


def _instance(factor: int) -> CountPairInstance:
    """The declared 5K instance at one bin factor, simulated once per session."""
    return binned_instance(fixture(PROBLEM, "stress").path, factor)


def _once(benchmark: BenchmarkFixture, call: Any, *arguments: Any) -> Any:
    """Time one call, once. See this module's docstring for why not more."""
    return benchmark.pedantic(  # type: ignore[no-untyped-call]
        call, args=arguments, rounds=1, iterations=1
    )


@pytest.mark.stress
@pytest.mark.parametrize("factor", [10, 5, 1])
def test_rust_class_posteriors_at_each_bin_factor(
    benchmark: BenchmarkFixture, factor: int
) -> None:
    instance = _instance(factor)

    _once(
        benchmark,
        rust.class_posteriors,
        instance.params,
        instance.observations,
        instance.labels,
    )


@pytest.mark.stress
@pytest.mark.parametrize("factor", [10, 5, 1])
def test_rust_external_field_at_each_bin_factor(
    benchmark: BenchmarkFixture, factor: int
) -> None:
    instance = _instance(factor)
    posterior = rust.class_posteriors(
        instance.params, instance.observations, instance.labels
    ).posterior

    _once(
        benchmark,
        rust.external_field,
        instance.params,
        instance.observations,
        instance.labels,
        posterior,
    )


@pytest.mark.release
@pytest.mark.parametrize("factor", [10, 5])
def test_numpy_class_posteriors_at_each_bin_factor(
    benchmark: BenchmarkFixture, factor: int
) -> None:
    instance = _instance(factor)

    _once(
        benchmark,
        class_posteriors,
        instance.params,
        instance.observations,
        instance.labels,
    )


@pytest.mark.release
def test_numpy_external_field_at_the_coarsest_bin_factor(
    benchmark: BenchmarkFixture,
) -> None:
    instance = _instance(10)
    posterior = rust.class_posteriors(
        instance.params, instance.observations, instance.labels
    ).posterior

    _once(
        benchmark,
        external_field,
        instance.params,
        instance.observations,
        instance.labels,
        posterior,
    )
