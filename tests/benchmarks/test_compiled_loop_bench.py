"""The torch loop against the compiled loop, per HMC proposal, on a cheap energy and on an HMM (issue #1220).

The case for running a chain in ``oxisal`` is the Python loop's share of a
proposal. On ``GaussianTarget(diagonal_precision(10))`` the energy is a
ten-term product, so the loop is most of the proposal; on a 4-state Gaussian
HMM (``d = 23``) of equal-length segments, the case the compiled route runs,
the gradient's forward-backward pass is. Each row times ``PROPOSALS[family]``
proposals of four leapfrog steps on one backend; the per-proposal figure is
the mean over that count. Correctness is pinned in
``tests/regression/sample/test_supported_gradient.py`` and
``test_hmc_compiled.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from pytest_benchmark.fixture import BenchmarkFixture
from sal.backend import Backend
from sal.emissions import GaussianEmission
from sal.opt.hmm import EmissionHmmObjective
from sal.opt.objective import Objective
from sal.sample.hmc import sample
from sal.search.hmm_starts import gaussian_quantile_start
from sal.sim.hmm import HmmParams, simulate_sequences
from sal.validation.gaussian import GaussianTarget, diagonal_precision

from tests._scale import at_scale

#: Hidden states, as the HMM-starts instance the issue was measured on.
N_STATES = 4

#: Positions per segment: one length, which the compiled HMM kernel requires.
LENGTH = 60

#: Proposals per timed chain.
PROPOSALS = {"gaussian": 2_000, "hmm": 20}


def _hmm(n_segments: int) -> EmissionHmmObjective:
    """``n_segments`` segments of ``LENGTH`` from a sticky 4-state Gaussian HMM."""
    off = 0.1 / (N_STATES - 1)
    transition = np.full((N_STATES, N_STATES), off) + np.eye(N_STATES) * (0.9 - off)
    data = simulate_sequences(
        HmmParams(
            N_STATES,
            (LENGTH,) * n_segments,
            np.full(N_STATES, 1.0 / N_STATES),
            transition,
            GaussianEmission(
                np.arange(N_STATES, dtype=float), np.full(N_STATES, 0.7), 1e-6
            ),
            1220,
            0.0,
        )
    )
    return EmissionHmmObjective(
        data.batch, gaussian_quantile_start(data.batch.values, N_STATES, 1e-6)
    )


@at_scale("n_segments", 12, 200)
@pytest.mark.parametrize("backend", [Backend.PYTHON, Backend.RUST], ids=str)
@pytest.mark.parametrize("family", ["gaussian", "hmm"])
def test_a_proposal_on_each_loop_benchmark(
    benchmark: BenchmarkFixture, family: str, backend: Backend, n_segments: int
) -> None:
    objective: Objective = (
        GaussianTarget(diagonal_precision(10))
        if family == "gaussian"
        else _hmm(n_segments)
    )
    step = 0.3 if family == "gaussian" else float(5e-3 / np.sqrt(n_segments / 12))

    chain = benchmark(
        sample,
        objective,
        torch.Generator().manual_seed(0),
        PROPOSALS[family],
        step_size=step,
        n_steps=4,
        backend=backend,
    )

    assert torch.isfinite(chain.draws).all()
    assert chain.acceptance_rate > 0.0
