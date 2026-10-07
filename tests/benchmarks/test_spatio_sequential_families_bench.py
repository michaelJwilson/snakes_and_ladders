"""NumPy against Rust on every one-channel family of the coupled model (issue #1308).

Both routes of the four entry points at the stress instance of
``spatio_sequential``, a 10x10 lattice of four positions, with each case of
``test_spatio_sequential_rust_families.py`` in its classes and the categorical
model beside them. The ratios decide the ``backend`` defaults under the
owner's rule from #1283: 2x or more on every family a function serves, at
stress size, min of 3 rounds.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture
from sal.backend import Backend
from sal.likelihood.spatio_sequential import (
    class_posteriors,
    external_field,
    labelled_log_likelihood,
)
from sal.search.spatio_sequential import fit_spatio_sequential
from sal.sim.fixtures import fixture
from sal.sim.spatio_sequential import simulate_spatio_sequential

from tests.regression.likelihood.test_spatio_sequential_rust_families import (
    CASES,
    _instance,
)

BACKENDS = [Backend.PYTHON, Backend.RUST]


def _fit(params: Any, observations: np.ndarray, _: np.ndarray, **kwargs: Any) -> Any:
    """The fit from a fixed generator, so every round runs the same blocks."""
    return fit_spatio_sequential(
        params, observations, np.random.default_rng(1), **kwargs
    )


CALLS: dict[str, Callable[..., Any]] = {
    "class_posteriors": class_posteriors,
    "external_field": external_field,
    "labelled_log_likelihood": labelled_log_likelihood,
    "fit_spatio_sequential": _fit,
}


def _case(name: str) -> tuple[Any, np.ndarray, np.ndarray]:
    """One draw of the stress instance under ``name``'s classes."""
    if name == "categorical":
        params = fixture("spatio_sequential", "stress").params
        data = simulate_spatio_sequential(params, np.random.default_rng(0))
        return params, data.observations, data.labels
    return _instance(name)


@pytest.mark.parametrize("backend", BACKENDS, ids=str)
@pytest.mark.parametrize("call", list(CALLS))
@pytest.mark.parametrize("family", ["categorical", *CASES])
def test_one_channel_route_at_stress(
    benchmark: BenchmarkFixture, family: str, call: str, backend: Backend
) -> None:
    # Each call is timed over at least three rounds after one warm-up, and
    # the minimum is what the PR reports.
    params, observations, labels = _case(family)

    benchmark.pedantic(  # type: ignore[no-untyped-call]
        CALLS[call],
        args=(params, observations, labels),
        kwargs={"backend": backend},
        rounds=5 if call == "fit_spatio_sequential" else 50,
        iterations=1,
        warmup_rounds=1,
    )
