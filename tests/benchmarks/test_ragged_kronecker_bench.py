"""The Kronecker switch in its factors, against an explicit ``2K x 2K`` step (issue #1133).

A slow chain of ``K`` states under a fast binary layer, ``2K`` states over 50
segments of 2,000 positions. The baseline is the same kernel's stay-or-move
switch at ``2K`` states, which builds and multiplies an explicit ``2K x 2K``
transition per step: the arithmetic a Kronecker step costs when it is not
factored. ``tests/regression/likelihood/test_ragged_rust.py`` pins both
Kronecker kinds against the materialized stack and against enumeration.
"""

from __future__ import annotations

import numpy as np
import pytest
from sal.likelihood.ragged import SwitchKind, posteriors
from sal.ragged import Ragged

#: Segments and their length.
N_SEGMENTS, LENGTH = 50, 2_000

Instance = tuple[Ragged, np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def _instance(slow: int) -> Instance:
    rng = np.random.default_rng(1133)
    n = 2 * slow
    density = Ragged(
        np.log(rng.random((N_SEGMENTS * LENGTH, n))), (LENGTH,) * N_SEGMENTS
    )
    chain = rng.dirichlet(np.ones(slow), size=slow)
    return (
        density,
        np.log(rng.dirichlet(np.ones(n))),
        np.log(chain),
        np.log(rng.dirichlet(np.ones(n), size=n)),
        rng.uniform(size=N_SEGMENTS * LENGTH),
    )


@pytest.fixture(scope="module", params=[4, 16], ids=lambda k: f"K={k}")
def instance(request: pytest.FixtureRequest) -> Instance:
    """The instance at ``K`` slow states."""
    return _instance(int(request.param))


@pytest.mark.release
@pytest.mark.benchmark
def test_ragged_explicit_step(benchmark: object, instance: Instance) -> None:
    """The baseline: an explicit ``2K x 2K`` transition built and applied per step."""
    density, initial, _, full, switch = instance
    benchmark(posteriors, density, initial, full, switch=switch)  # type: ignore[operator]


@pytest.mark.release
@pytest.mark.benchmark
@pytest.mark.parametrize(
    "kind", [SwitchKind.KRONECKER, SwitchKind.KRONECKER_DIAGONAL], ids=str
)
def test_ragged_kronecker_step(
    benchmark: object, instance: Instance, kind: SwitchKind
) -> None:
    """The Kronecker step in its factors."""
    density, initial, slow, _, switch = instance
    benchmark(posteriors, density, initial, slow, switch=switch, switch_kind=kind)  # type: ignore[operator]
