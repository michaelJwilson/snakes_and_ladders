"""Niedermayer's step under a forbidden label, moved with the move to the sandbox (issues #1146, #1365)."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from sal.backend import Backend
from sal.sample.potts_mcmc import (
    adjacency_lists,
    swendsen_wang_heat_bath_sweep,
    swendsen_wang_sweep,
    wolff_heat_bath_sweep,
    wolff_sweep,
)
from sal.sandbox.potts_moves import niedermayer_sweep

from tests.regression.sample.test_potts_forbidden_labels import (
    LATTICE,
    LATTICE_ALLOWED,
    LATTICE_FIELD,
    LATTICE_MOVES,
    SWEEPS,
)


@pytest.mark.analytic
@pytest.mark.parametrize("move", LATTICE_MOVES)
def test_a_forbidden_start_is_left_and_never_re_entered(move: str) -> None:
    # Every site at label 0, which no site allows. Over 300 sweeps with
    # RuntimeWarning an error: the chain reaches an allowed labelling, and
    # from the first one on every labelling is allowed --- the target puts no
    # mass on a forbidden one.
    offsets, neighbours, couplings = LATTICE.compressed_adjacency()
    lists = adjacency_lists(offsets, neighbours, couplings)
    adjacency = (offsets, neighbours, couplings)
    rng = np.random.default_rng(1139)
    state = np.zeros(LATTICE.n_nodes, dtype=np.int64)
    sites = np.arange(LATTICE.n_nodes)
    entered: int | None = None
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        for sweep in range(SWEEPS):
            if move.startswith("swendsen_wang_") and not move.endswith("heat_bath"):
                backend = Backend.RUST if move.endswith("rust") else Backend.PYTHON
                swendsen_wang_sweep(state, LATTICE, LATTICE_FIELD, rng, backend=backend)
            elif move == "wolff":
                wolff_sweep(state, LATTICE_FIELD, *adjacency, rng, lists=lists)
            elif move == "niedermayer":
                niedermayer_sweep(state, LATTICE_FIELD, *adjacency, rng, lists=lists)
            elif move == "swendsen_wang_heat_bath":
                swendsen_wang_heat_bath_sweep(state, LATTICE, LATTICE_FIELD, rng)
            else:
                wolff_heat_bath_sweep(
                    state, LATTICE_FIELD, *adjacency, rng, lists=lists
                )
            allowed = bool(LATTICE_ALLOWED[sites, state].all())
            if entered is None and allowed:
                entered = sweep
            assert entered is None or allowed, f"re-entered at sweep {sweep}"

    assert entered is not None
