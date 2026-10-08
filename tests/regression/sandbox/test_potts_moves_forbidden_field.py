"""The ghost-spin and label-directed passes under a forbidden label, moved with the moves to the sandbox (issues #1154, #1365)."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from sal.backend import Backend
from sal.sandbox.potts_moves import ghost_spin_sweep, label_directed_sweep

from tests.regression.sample.test_potts_forbidden_field_moves import (
    BETAS,
    ROWS,
    _ghost_kernel,
    _keeps,
    _label_directed_kernel,
    _lattice_run,
    _mixes,
)
from tests.regression.sample.test_potts_forbidden_labels import (
    DRAWS,
    LATTICE,
    LATTICE_ALLOWED,
    SIGMAS,
    _field,
)
from tests.regression.sample.test_potts_heat_bath_cluster import (
    N_STATES,
    _graph,
    _states,
)


@pytest.mark.oracle
@pytest.mark.parametrize("beta", BETAS)
def test_the_enumerated_label_directed_kernel_keeps_the_restricted_law(
    beta: float,
) -> None:
    # Each target's kernel keeps the law; their cycle, the chain
    # `sample_potts` runs, carries every start to it.
    kernels = [_label_directed_kernel(beta, target) for target in range(N_STATES)]
    for kernel in kernels:
        _keeps(kernel, beta)
    _mixes(kernels[0] @ kernels[1] @ kernels[2], beta)


@pytest.mark.oracle
@pytest.mark.parametrize(("move", "start", "beta", "backend", "target"), ROWS, ids=str)
def test_each_move_draws_its_enumerated_row(
    move: str,
    start: tuple[int, ...],
    beta: float,
    backend: Backend,
    target: int,
) -> None:
    # 40,000 steps from `start`, RuntimeWarning an error: every one of the
    # 81 cells within four binomial standard errors of the kernel's row, and
    # a cell the kernel never reaches never reached.
    index = {state: k for k, state in enumerate(_states())}
    kernel = (
        _ghost_kernel(beta) if move == "ghost" else _label_directed_kernel(beta, target)
    )
    exact = kernel[index[start]]
    graph, rows = _graph(), _field()
    rng = np.random.default_rng(1154)
    counts = np.zeros(len(index))
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        for _ in range(DRAWS):
            state = np.array(start, dtype=np.int64)
            if move == "ghost":
                ghost_spin_sweep(state, graph, rows, rng, beta, backend=backend)
            else:
                label_directed_sweep(
                    state, graph, rows, rng, target, beta, backend=backend
                )
            counts[index[tuple(state.tolist())]] += 1
    empirical = counts / DRAWS

    error = np.sqrt(exact * (1.0 - exact) / DRAWS)
    assert np.flatnonzero(np.abs(empirical - exact) > SIGMAS * error).size == 0


@pytest.mark.analytic
@pytest.mark.parametrize("move", ["ghost", "label_directed"])
def test_a_forbidden_start_is_left_and_never_re_entered(move: str) -> None:
    # Every site at label 0, which no site allows: the chain reaches an
    # allowed labelling, and from the first one on every labelling is allowed.
    sites = np.arange(LATTICE.n_nodes)
    visited = _lattice_run(move, np.zeros(LATTICE.n_nodes, dtype=np.int64))
    allowed = [bool(LATTICE_ALLOWED[sites, state].all()) for state in visited]

    assert any(allowed)
    first = allowed.index(True)
    assert all(allowed[first:]), f"re-entered after sweep {first}"
