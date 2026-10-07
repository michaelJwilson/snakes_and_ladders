"""NumPy against Rust on the categorical coupled model, the fit's default (issue #1298).

Both routes of the four entry points at the declared stress instance of
``spatio_sequential``, a 10x10 lattice of four positions: the size the label
solvers are compared on. The ratio decides the ``backend`` default under the
owner's rule from #1283, 2x or more at stress size, min of 3 rounds.
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


@pytest.mark.parametrize("backend", BACKENDS, ids=str)
@pytest.mark.parametrize("name", list(CALLS))
def test_categorical_route_at_stress(
    benchmark: BenchmarkFixture, name: str, backend: Backend
) -> None:
    # One draw of the stress instance; each call is timed over at least three
    # rounds, and the minimum is what the PR reports.
    params = fixture("spatio_sequential", "stress").params
    data = simulate_spatio_sequential(params, np.random.default_rng(0))

    benchmark.pedantic(  # type: ignore[no-untyped-call]
        CALLS[name],
        args=(params, data.observations, data.labels),
        kwargs={"backend": backend},
        rounds=5 if name == "fit_spatio_sequential" else 50,
        iterations=1,
        warmup_rounds=1,
    )
