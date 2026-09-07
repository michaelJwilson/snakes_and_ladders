"""Benchmarks for a complete fit on each reference instance.

See tests/regression/test_opt_fit.py for correctness. A whole fit rather than
one step: L-BFGS spends most of its time inside the line search, so a
per-step number would misreport what a fit actually costs. The interval
computation is measured separately because it is a Hessian, not a gradient --
quadratic in the parameter count where the fit is linear, and the reason a
recovery test over replicates costs what it does.
"""

from __future__ import annotations

import math

from pytest_benchmark.fixture import BenchmarkFixture
from snakes_and_ladders.opt.fit import constrained_standard_errors, fit
from snakes_and_ladders.opt.hmm import HmmObjective
from snakes_and_ladders.opt.potts import (
    PottsObjective,
    load_potts_params,
    simulate_chains,
)
from snakes_and_ladders.sim.hmm import load_hmm_params, simulate_sequences

from tests._fixtures import FIXTURES_DIR

POTTS_FIXTURE = FIXTURES_DIR / "potts_params.yaml"
HMM_FIXTURE = FIXTURES_DIR / "hmm_params.yaml"


def test_potts_fit_benchmark(benchmark: BenchmarkFixture) -> None:
    params = load_potts_params(POTTS_FIXTURE)
    objective = PottsObjective(simulate_chains(params), params.n_states)

    result = benchmark(fit, objective)

    # Benchmarks only assert finiteness and that the fit converged --
    # recovery is pinned in tests/regression/test_opt_fit.py.
    assert result.converged
    assert math.isfinite(result.value)


def test_hmm_fit_benchmark(benchmark: BenchmarkFixture) -> None:
    params = load_hmm_params(HMM_FIXTURE)
    objective = HmmObjective(
        simulate_sequences(params).observations, params.n_states, params.n_symbols
    )

    result = benchmark(fit, objective)

    assert result.converged
    assert math.isfinite(result.value)


def test_hmm_interval_benchmark(benchmark: BenchmarkFixture) -> None:
    params = load_hmm_params(HMM_FIXTURE)
    objective = HmmObjective(
        simulate_sequences(params).observations, params.n_states, params.n_symbols
    )
    theta = fit(objective).theta

    errors = benchmark(constrained_standard_errors, objective, theta)

    assert all(math.isfinite(float(e.sum())) for e in errors.values())
