"""Benchmarks for the Jukes-Cantor simulator's hot path.

See tests/regression/test_jc_simulate.py for correctness, pinned separately
per the repo's division of labor between the two test directories. Runs
across a variety of site/taxa sizes, per sim/CLAUDE.md's local rule.
"""

from __future__ import annotations

import pytest
from pytest_benchmark.fixture import BenchmarkFixture
from snakes_and_ladders.sim.params import load_simulation_params
from snakes_and_ladders.sim.simulate import simulate_alignment

from tests._fixtures import FIXTURES_DIR


@pytest.mark.parametrize(
    "fixture_name",
    [
        "simulation_params.yaml",  # 4 taxa, 200_000 sites
        "simulation_params_small_sites.yaml",  # 4 taxa, 20_000 sites
        "simulation_params_8taxa.yaml",  # 8 taxa, 200_000 sites
    ],
)
def test_simulate_alignment_benchmark(
    benchmark: BenchmarkFixture, fixture_name: str
) -> None:
    params = load_simulation_params(FIXTURES_DIR / fixture_name)

    dataset = benchmark(
        simulate_alignment,
        params.tau,
        params.k,
        params.pi,
        params.seed,
        params.n_sites,
    )

    # Benchmarks only assert shape -- numerical correctness is pinned
    # separately in tests/regression/test_jc_simulate.py.
    for states in dataset.alignment.values():
        assert states.shape == (params.n_sites,)
