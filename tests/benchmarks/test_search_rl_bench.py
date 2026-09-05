"""Benchmarks for the phylogenetic environment.

See tests/regression/test_search_rl.py for correctness. The number that
matters is the ratio between the two reward models, because that ratio is the
entire argument for issue #131's simplification: one candidate scored at
known parameters against the same candidate scored at maximized ones.

It lives here rather than in a caption because it is machine-dependent, and
`DEV.md` forbids ranking performance on CI hardware. The committed caption
states the structural reason instead -- a full optimization against a single
pruning pass -- and this is where the size of that difference is measured.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from phylo.search.rl import RewardModel, TopologyEnvironment
from phylo.search.topology import random_topology
from phylo.sim.params import load_simulation_params
from phylo.sim.simulate import simulate_alignment
from pytest_benchmark.fixture import BenchmarkFixture

from tests._fixtures import FIXTURES_DIR

FIXTURE = FIXTURES_DIR / "simulation_params_5taxa.yaml"
_BRANCH_LENGTH = 0.1629

# The fixture issue #177 added, at 7 taxa and 2000 sites against 5 and 500.
# Measured here because issue #178 counts its training budget in candidate
# scorings, and a budget is only affordable relative to what one costs.
HARD_FIXTURE = FIXTURES_DIR / "simulation_params_hard.yaml"
_HARD_BRANCH_LENGTH = 0.2451


def _alignment(
    fixture: Path = FIXTURE,
) -> tuple[dict[str, np.ndarray], int, np.ndarray]:
    params = load_simulation_params(fixture)
    dataset = simulate_alignment(
        tau=params.tau,
        k=params.k,
        pi=params.pi,
        seed=params.seed,
        n_sites=params.n_sites,
    )
    return dict(dataset.alignment), params.k, params.pi


def _score_once(
    reward: RewardModel,
    benchmark: BenchmarkFixture,
    fixture: Path = FIXTURE,
    branch_length: float = _BRANCH_LENGTH,
) -> float:
    alignment, k, pi = _alignment(fixture)
    topology = random_topology(sorted(alignment), np.random.default_rng(1))

    def run() -> float:
        # A fresh environment per call: the cache is the point of the class,
        # and timing a cache hit would measure a dictionary lookup.
        return TopologyEnvironment(
            alignment, k, pi, branch_length, reward=reward
        ).score(topology)

    return float(benchmark(run))


def test_known_reward_benchmark(benchmark: BenchmarkFixture) -> None:
    """One candidate scored at fixed known parameters."""
    assert np.isfinite(_score_once(RewardModel.KNOWN, benchmark))


def test_fitted_reward_benchmark(benchmark: BenchmarkFixture) -> None:
    """The same candidate scored at maximized branch lengths."""
    assert np.isfinite(_score_once(RewardModel.FITTED, benchmark))


def test_known_reward_on_the_hard_fixture_benchmark(
    benchmark: BenchmarkFixture,
) -> None:
    """One candidate scored at fixed known parameters, at 7 taxa."""
    assert np.isfinite(
        _score_once(RewardModel.KNOWN, benchmark, HARD_FIXTURE, _HARD_BRANCH_LENGTH)
    )


def test_fitted_reward_on_the_hard_fixture_benchmark(
    benchmark: BenchmarkFixture,
) -> None:
    """The same candidate at maximized branch lengths, at 7 taxa."""
    assert np.isfinite(
        _score_once(RewardModel.FITTED, benchmark, HARD_FIXTURE, _HARD_BRANCH_LENGTH)
    )


def test_neighbourhood_benchmark(benchmark: BenchmarkFixture) -> None:
    """Generating and deduplicating the neighbourhood, without scoring it."""
    alignment, k, pi = _alignment()
    environment = TopologyEnvironment(alignment, k, pi, _BRANCH_LENGTH)
    topology = random_topology(sorted(alignment), np.random.default_rng(1))

    actions = benchmark(environment.actions, topology)

    assert len(actions) > 0
