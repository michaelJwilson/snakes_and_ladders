"""Benchmark for ``snakes_and_ladders.sim.newick.to_newick``.

See tests/regression/test_newick.py for correctness. Runs across the same
site/taxa fixtures as the simulator benchmark, per sim/CLAUDE.md's local
rule, with ancestral states embedded (the more expensive of the two code
paths through ``to_newick``).
"""

from __future__ import annotations

import pytest
from pytest_benchmark.fixture import BenchmarkFixture
from snakes_and_ladders.sim.newick import to_newick
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
def test_to_newick_with_states_benchmark(
    benchmark: BenchmarkFixture, fixture_name: str
) -> None:
    params = load_simulation_params(FIXTURES_DIR / fixture_name)
    dataset = simulate_alignment(
        tau=params.tau, k=params.k, pi=params.pi, seed=params.seed, n_sites=10
    )

    labelled = benchmark(to_newick, dataset.tau, dataset.node_states, 0)

    # Correctness (including the grammar check) is pinned separately in
    # tests/regression/test_newick.py; here just confirm every state
    # embedded and the string terminated.
    assert labelled.endswith(";")
    for name in dataset.node_states:
        assert name in labelled
