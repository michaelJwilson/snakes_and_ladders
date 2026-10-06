"""Torch-route HMC and MALA on a Gaussian HMM, where ``U`` costs a forward recursion (issue #1217).

A proposal reads ``U`` at both ends. Both are carried from force evaluations
the proposal already made, so its wall time is its gradients' alone; this
times one chain at the CI size and at the stress size, where the forward and
the compiled E step are both linear in the positions. Correctness is pinned
in `tests/regression/sample/test_hmc_carried_energy.py`.
"""

from __future__ import annotations

import numpy as np
import torch
from pytest_benchmark.fixture import BenchmarkFixture
from sal.backend import Backend
from sal.emissions import GaussianEmission
from sal.opt.hmm import EmissionHmmObjective
from sal.sample.hmc import sample
from sal.sample.langevin import mala
from sal.search.hmm_starts import gaussian_quantile_start
from sal.sim.hmm import HmmParams, simulate_sequences

from tests._scale import at_scale

#: Hidden states, as the HMM-starts instance the issue was measured on.
N_STATES = 4

#: Proposals per timed chain.
PROPOSALS = 20


def _objective(n_segments: int) -> EmissionHmmObjective:
    """Segments of 30 to 89 positions from a sticky 4-state Gaussian HMM."""
    lengths = tuple(
        int(one) for one in np.random.default_rng(0).integers(30, 90, size=n_segments)
    )
    off = 0.1 / (N_STATES - 1)
    transition = np.full((N_STATES, N_STATES), off) + np.eye(N_STATES) * (0.9 - off)
    emission = GaussianEmission(
        np.arange(N_STATES, dtype=float), np.full(N_STATES, 0.7), 1e-6
    )
    data = simulate_sequences(
        HmmParams(
            N_STATES,
            lengths,
            np.full(N_STATES, 1.0 / N_STATES),
            transition,
            emission,
            1172,
            0.0,
        )
    )
    return EmissionHmmObjective(
        data.batch, gaussian_quantile_start(data.batch.values, N_STATES, 1e-6)
    )


def _step(n_segments: int) -> float:
    """The step the 12-segment instance takes, scaled as the posterior narrows."""
    return float(5e-3 / np.sqrt(n_segments / 12))


@at_scale("n_segments", 12, 200)
def test_hmc_on_an_hmm_benchmark(benchmark: BenchmarkFixture, n_segments: int) -> None:
    objective = _objective(n_segments)

    chain = benchmark(
        sample,
        objective,
        torch.Generator().manual_seed(0),
        PROPOSALS,
        step_size=_step(n_segments),
        n_steps=4,
        backend=Backend.PYTHON,
    )

    assert torch.isfinite(chain.draws).all()


@at_scale("n_segments", 12, 200)
def test_mala_on_an_hmm_benchmark(benchmark: BenchmarkFixture, n_segments: int) -> None:
    objective = _objective(n_segments)

    chain = benchmark(
        mala,
        objective,
        torch.Generator().manual_seed(0),
        PROPOSALS,
        step_size=_step(n_segments),
        backend=Backend.PYTHON,
    )

    assert torch.isfinite(chain.draws).all()
