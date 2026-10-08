"""What `track` records of the Potts `parallel_tempering`, from its sandbox home (issue #1352).

Moved from `tests/regression/test_track.py` with the sampler; the instance,
seed and assertions are unchanged.
"""

from __future__ import annotations

import numpy as np
import pytest
from sal.backend import Backend
from sal.sandbox.potts_tempering import TemperedChains, parallel_tempering
from sal.sim.graph import BoundaryCondition, lattice_graph
from sal.track import NULL_RUN, MemoryRun, track

SEED = 4242
#: The 2x2 two-state lattice `test_track.py` runs every sampler on.
SHAPE = (2, 2)
COUPLING = 0.8
FIELD = np.zeros(2)
SWEEPS = 30
TEMPERATURES = (0.5, 1.0, 2.0)


def _tempered() -> TemperedChains:
    return parallel_tempering(
        lattice_graph(SHAPE, BoundaryCondition.OPEN, COUPLING),
        FIELD,
        TEMPERATURES,
        np.random.default_rng(SEED),
        SWEEPS,
        backend=Backend.PYTHON,
    )


@pytest.mark.patch
@pytest.mark.smoke
def test_the_null_run_leaves_a_tempered_run_bitwise() -> None:
    outside = _tempered()
    with track(NULL_RUN):
        inside = _tempered()

    assert np.array_equal(inside.states, outside.states)
    assert np.array_equal(inside.swap_acceptance, outside.swap_acceptance)
    assert inside.energy == outside.energy


@pytest.mark.smoke
def test_the_tempered_run_records_the_swap_acceptance_it_returns() -> None:
    with track() as tracked:
        chains = _tempered()

    run = tracked.run
    assert isinstance(run, MemoryRun)
    assert len(run.series("swap_acceptance")) == SWEEPS
    assert run.last("swap_acceptance") == float(np.mean(chains.swap_acceptance))
    assert run.last("state_bytes") == float(chains.states[0].nbytes)
