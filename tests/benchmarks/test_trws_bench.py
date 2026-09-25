"""TRW-S at release size: the compiled kernel against its Python reference (issue #1060).

Correctness is pinned in `tests/regression/search/test_trws.py`. Here the
instance is `spatio_only/release`, 5,041 sites and 14,840 edges at ten
states. Both backends run :data:`ITERATIONS` iterations at zero tolerance, so
they do the same work and the pair's ratio is the kernel's; the whole call is
timed --- the chain layout, the passes, the bound, the decode and the energy
--- since that is what a caller pays. The compiled run to convergence is timed
beside them: that is the measurement's cost.
"""

from __future__ import annotations

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture
from sal.backend import Backend
from sal.search.ground_state import Rung
from sal.search.potts_starts import spatio_rung
from sal.search.trws import trws
from sal.sim.fixtures import fixture

#: Iterations both backends run: a forward and a backward pass, the bound and
#: a decode each.
ITERATIONS = 10


def _rung() -> Rung:
    return spatio_rung(fixture("spatio_only", "release").params, "release")


@pytest.mark.parametrize("backend", [Backend.NUMBA, Backend.PYTHON], ids=str)
def test_trws_fixed_iterations_benchmark(
    benchmark: BenchmarkFixture, backend: Backend
) -> None:
    rung = _rung()

    def run() -> float:
        return trws(
            rung.graph,
            rung.field,
            max_iterations=ITERATIONS,
            tolerance=0.0,
            backend=backend,
        ).bound

    bound = benchmark.pedantic(run, rounds=3, iterations=1, warmup_rounds=1)  # type: ignore[no-untyped-call]

    assert np.isfinite(bound)


def test_trws_to_convergence_benchmark(benchmark: BenchmarkFixture) -> None:
    # The compiled run at its defaults, to the tolerance: 43 iterations here.
    rung = _rung()

    def run() -> float:
        return trws(rung.graph, rung.field).bound

    bound = benchmark.pedantic(run, rounds=5, iterations=1, warmup_rounds=1)  # type: ignore[no-untyped-call]

    assert np.isfinite(bound)
