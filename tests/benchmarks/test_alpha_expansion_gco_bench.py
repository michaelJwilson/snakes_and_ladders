"""The package's alpha expansion beside gco's, on one lattice per size (issue #974).

The properties both must satisfy are pinned in `tests/validation/test_gco.py`.
Each row times the package's expansion on the Rust cut to convergence and
records beside it what gco's script measured on the same model: `gco_build_s`
for its graph, `gco_expand_s` for `expansion()`, the median of `REPEATS`
subprocess runs, and both energies under `sim.potts.energy`. The lattice is
open at q = 10's critical coupling under a standard normal field, the case
#952's spike measured.

Sizes: 16² is the gate size, 71², 142² and 284² the stress sizes a speedup is
read at (root `CLAUDE.md`). The two stress rows above 71² carry `release`:
the package's expansion takes 3.7 s at 142² on `main`'s Rust path.
"""

from __future__ import annotations

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture
from snakes_and_ladders.backend import Backend
from snakes_and_ladders.search.alpha_expansion import alpha_expansion
from snakes_and_ladders.sim.graph import BoundaryCondition, lattice_graph
from snakes_and_ladders.sim.potts import critical_coupling
from snakes_and_ladders.validation import gco
from snakes_and_ladders.validation.runner import available

pytestmark = pytest.mark.skipif(
    not available("gco"), reason="gco is the validation-gco extra"
)

#: Subprocess runs per size whose median gco's figures are.
REPEATS = 3

#: Labels, as #952's spike.
N_STATES = 10

SIDES = [
    16,
    71,
    pytest.param(142, marks=pytest.mark.release),
    pytest.param(284, marks=pytest.mark.release),
]


@pytest.mark.parametrize("side", SIDES)
def test_rust_expansion_beside_gco_benchmark(
    benchmark: BenchmarkFixture, side: int
) -> None:
    graph = lattice_graph(
        (side, side), BoundaryCondition.OPEN, critical_coupling(N_STATES)
    )
    field = np.random.default_rng(974).normal(size=(graph.n_nodes, N_STATES))
    runs = [gco.alpha_expansion(graph, field, N_STATES) for _ in range(REPEATS)]
    benchmark.extra_info["gco_build_s"] = float(
        np.median([one.build_seconds for one in runs])
    )
    benchmark.extra_info["gco_expand_s"] = float(
        np.median([one.seconds for one in runs])
    )
    benchmark.extra_info["gco_energy"] = runs[0].energy

    result = benchmark.pedantic(  # type: ignore[no-untyped-call]
        alpha_expansion,
        args=(graph, field, N_STATES),
        kwargs={"backend": Backend.RUST},
        rounds=3,
        iterations=1,
    )
    benchmark.extra_info["package_energy"] = result.energy

    assert np.isfinite(result.energy)
