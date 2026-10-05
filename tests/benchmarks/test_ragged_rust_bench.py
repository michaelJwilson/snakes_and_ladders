"""What padding costs, measured rather than asserted (issue #666)."""

from __future__ import annotations

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture
from sal.likelihood.ragged import posteriors
from sal.ragged import Ragged

STATES = 4


def _density(lengths: tuple[int, ...], seed: int = 7) -> Ragged:
    rng = np.random.default_rng(seed)
    return Ragged(np.log(rng.random((sum(lengths), STATES))), lengths)


@pytest.mark.benchmark(group="ragged-posteriors")
@pytest.mark.parametrize(
    "lengths",
    [(500,) * 64, (30,) * 63 + (2000,)],
    ids=["even", "one-long-among-short"],
)
def test_ragged_posteriors_bench(
    benchmark: BenchmarkFixture, lengths: tuple[int, ...]
) -> None:
    """The compiled path pads nothing, so its cost follows the total only.

    The two shapes are chosen to separate the two costs: even lengths waste no
    padding, and one long segment among short ones is 97% padding in a block.
    """
    density = _density(lengths)
    initial = np.log(np.full(STATES, 1.0 / STATES))
    transition = np.log(np.full((STATES, STATES), 1.0 / STATES))
    benchmark(posteriors, density, initial, transition)


@pytest.mark.benchmark(group="ragged-switched")
@pytest.mark.parametrize("route", ["switch", "stack"])
def test_switched_transition_bench(benchmark: BenchmarkFixture, route: str) -> None:
    """A per-site switch (issue #1082): the kernel's per-step build, against the stack.

    `stack` is what a caller did without the switch: materialize one
    `(K, K)` kernel per step and run the oracle's per-step form, one segment
    at a time. 64 segments of 500 at four states.
    """
    from sal.backend import Backend

    lengths = (500,) * 64
    density = _density(lengths)
    initial = np.log(np.full(STATES, 1.0 / STATES))
    transition = np.log(np.random.default_rng(3).dirichlet(np.ones(STATES), STATES))
    switch = np.random.default_rng(4).uniform(size=sum(lengths))
    backend = Backend.RUST if route == "switch" else Backend.PYTHON
    benchmark(posteriors, density, initial, transition, switch=switch, backend=backend)


@pytest.mark.benchmark(group="ragged-viterbi")
@pytest.mark.parametrize("route", ["rust", "numpy"])
@pytest.mark.parametrize("case", ["plain", "switched", "kronecker"])
def test_ragged_viterbi_bench(
    benchmark: BenchmarkFixture, case: str, route: str
) -> None:
    """The compiled Viterbi against its NumPy oracle at stress size (issue #1138).

    200 segments of 1,000 positions, 2e5 in all, at ten states: `plain` takes
    one transition, `switched` a stay-or-move step per position, and
    `kronecker` the `A ⊗ S` step over `2 K = 10` states, which the kernel
    takes in its factors and the oracle as a materialized stack. Three
    rounds, since the oracle's call runs for seconds.
    """
    from sal.backend import Backend
    from sal.likelihood.ragged import SwitchKind, viterbi

    n_states, lengths = 10, (1000,) * 200
    rng = np.random.default_rng(1138)
    density = Ragged(np.log(rng.random((sum(lengths), n_states))), lengths)
    slow = n_states // 2 if case == "kronecker" else n_states
    initial = np.log(np.full(n_states, 1.0 / n_states))
    transition = np.log(rng.dirichlet(np.ones(slow), slow))
    switch = None if case == "plain" else rng.uniform(size=sum(lengths))
    kind = SwitchKind.KRONECKER if case == "kronecker" else SwitchKind.STAY_OR_MOVE
    backend = Backend.RUST if route == "rust" else Backend.PYTHON
    benchmark.pedantic(  # type: ignore[no-untyped-call]
        viterbi,
        args=(density, initial, transition),
        kwargs={"switch": switch, "switch_kind": kind, "backend": backend},
        rounds=3,
        iterations=1,
    )


@pytest.mark.benchmark(group="ragged-sample-paths")
@pytest.mark.parametrize("route", ["rust", "numpy"])
@pytest.mark.parametrize("case", ["plain", "switched", "kronecker"])
def test_ragged_sample_paths_bench(
    benchmark: BenchmarkFixture, case: str, route: str
) -> None:
    """The compiled forward-filter backward-sample against its NumPy oracle (issue #1170).

    The Viterbi bench's stress size and cases: 200 segments of 1,000
    positions at ten states. Each round draws its uniforms from one seed, so
    every round draws the same paths.
    """
    from sal.backend import Backend
    from sal.likelihood.ragged import SwitchKind, sample_paths

    n_states, lengths = 10, (1000,) * 200
    rng = np.random.default_rng(1170)
    density = Ragged(np.log(rng.random((sum(lengths), n_states))), lengths)
    slow = n_states // 2 if case == "kronecker" else n_states
    initial = np.log(np.full(n_states, 1.0 / n_states))
    transition = np.log(rng.dirichlet(np.ones(slow), slow))
    switch = None if case == "plain" else rng.uniform(size=sum(lengths))
    kind = SwitchKind.KRONECKER if case == "kronecker" else SwitchKind.STAY_OR_MOVE
    backend = Backend.RUST if route == "rust" else Backend.PYTHON

    def draw() -> object:
        return sample_paths(
            density,
            initial,
            transition,
            np.random.default_rng(0),
            switch=switch,
            switch_kind=kind,
            backend=backend,
        )

    benchmark.pedantic(draw, rounds=3, iterations=1)  # type: ignore[no-untyped-call]
