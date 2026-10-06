"""Benchmarks for the Hamiltonian ladder with and without its per-rung warm-up (issue #1208).

Correctness is pinned in `tests/regression/sample/test_hmc_tempering_adaptation.py`.
Each row times 200 rounds of a four-rung ladder at five leapfrog steps on
Rosenbrock; the adapted row adds 100 warm-up proposals per rung, so its
gradients are 1.5 times the fixed row's and a ratio near 1.5 says the
warm-up costs what its proposals cost and nothing more.
"""

from __future__ import annotations

import pytest
import torch
from pytest_benchmark.fixture import BenchmarkFixture
from sal.opt.testfunctions import Rosenbrock
from sal.sample.hmc import Adaptation, parallel_tempering

LADDER = (1.0, 4.0, 16.0, 64.0)


@pytest.mark.parametrize("adapted", [False, True], ids=["fixed", "adapted"])
@pytest.mark.parametrize("dimension", [2, 10])
def test_hmc_parallel_tempering_benchmark(
    benchmark: BenchmarkFixture, dimension: int, adapted: bool
) -> None:
    adaptation = (
        Adaptation(warmup=100, target_acceptance=0.65, step_jitter=0.4)
        if adapted
        else None
    )
    run = benchmark(
        parallel_tempering,
        Rosenbrock(dimension=dimension),
        LADDER,
        torch.Generator().manual_seed(1),
        200,
        step_size=0.01,
        n_steps=5,
        adaptation=adaptation,
    )

    assert torch.isfinite(run.positions).all()
