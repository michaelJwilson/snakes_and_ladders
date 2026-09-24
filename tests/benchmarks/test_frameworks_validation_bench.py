"""The package's RL estimators, graph surrogate and environment beside TorchRL, PyG and Gymnasium (issue #977).

Agreement is pinned in `tests/validation/test_torchrl.py`,
`test_torch_geometric.py` and `test_gymnasium.py`. Each row times the
package in-process and records beside it the framework's own call, measured
in its subprocess around that call alone, with the ratio. A TorchRL or PyG
row records, on both sides, the median of `REPEATS` calls after one warm-up:
TorchRL's first call in an interpreter cost 3.7x its warm median at 10⁵
decisions. The Gymnasium row times one loop on each side.

Rows and sizes (the first of each is the gate size):

- ``generalized_advantages`` beside TorchRL's ``GAE``: one episode of 10³,
  10⁵ and 10⁶ decisions;
- ``ppo_loss`` and its gradient beside ``ClipPPOLoss``: 10³, 10⁵ and 10⁶
  decisions in episodes of 100 on the Potts chain's features;
- ``surrogate_loss`` and its gradient beside ``ReinforceLoss``: 10³ and 10⁵
  decisions of greedy Potts episodes of 10;
- ``GraphSurrogate``'s forward beside its ``GINConv`` twin on tied weights,
  one open lattice of 16², 71², 142² and 284² nodes;
- decisions through the Gymnasium adapter beside the bare ``rollout``, on the
  Potts chain, 10³ and 10⁵ decisions: what the interface costs, not a
  speedup.

The 10⁶ rows carry `release`.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
import pytest
import torch
from pytest_benchmark.fixture import BenchmarkFixture
from snakes_and_ladders.learn import ppo
from snakes_and_ladders.learn.policy import LinearPolicy
from snakes_and_ladders.learn.reinforce import surrogate_loss
from snakes_and_ladders.learn.rollout import rollout
from snakes_and_ladders.learn.surrogate import Examples, GraphSurrogate, _Batch
from snakes_and_ladders.sim.graph import BoundaryCondition, lattice_graph
from snakes_and_ladders.validation import gymnasium, torch_geometric, torchrl

from tests._frameworks import requires
from tests.regression.learn.conftest import COUPLING, FIELD, potts_environment
from tests.validation._rl import (
    WEIGHTS,
    greedy_episodes,
    potts_decisions,
    ppo_loss_and_gradient,
    reinforce_decisions,
)

#: Warm calls whose median a TorchRL or PyG row records, on each side, after one warm-up.
REPEATS = 5


def _warm(seconds: np.ndarray) -> float:
    """The median of the calls after the first, which pays TorchRL's warm-up."""
    return float(np.median(seconds[1:]))


def _record(benchmark: BenchmarkFixture, framework: float, package: float) -> None:
    benchmark.extra_info["framework_s"] = framework
    benchmark.extra_info["package_s"] = package
    benchmark.extra_info["package_over_framework"] = package / framework


def _once[T](benchmark: BenchmarkFixture, call: Callable[[], T]) -> T:
    return benchmark.pedantic(call, rounds=1, iterations=1)  # type: ignore[no-untyped-call,no-any-return]


def _warm_ours[T](
    benchmark: BenchmarkFixture, call: Callable[[], T]
) -> tuple[T, float]:
    """``call``'s result and its median over `REPEATS` rounds after one warm-up."""
    result: T = benchmark.pedantic(  # type: ignore[no-untyped-call]
        call, rounds=REPEATS, iterations=1, warmup_rounds=1
    )
    assert benchmark.stats is not None
    return result, float(benchmark.stats.stats.median)


@requires("torchrl")
@pytest.mark.parametrize(
    "steps", [1_000, 100_000, pytest.param(1_000_000, marks=pytest.mark.release)]
)
def test_gae_beside_torchrl_benchmark(benchmark: BenchmarkFixture, steps: int) -> None:
    rng = np.random.default_rng(977)
    rewards, values = rng.normal(size=steps), rng.normal(size=steps + 1)
    theirs = torchrl.gae(
        [torchrl.Rollout(rewards.tolist(), values.tolist(), 0.95, True)] * (1 + REPEATS)
    )
    ours, seconds = _warm_ours(
        benchmark,
        lambda: ppo.generalized_advantages(
            rewards.tolist(), values.tolist(), lam=0.95, terminated=True
        ),
    )
    _record(benchmark, _warm(theirs.seconds), seconds)
    np.testing.assert_allclose(theirs.advantages[0], ours, rtol=0.0, atol=1e-8)


@requires("torchrl")
@pytest.mark.parametrize(
    "steps", [1_000, 100_000, pytest.param(1_000_000, marks=pytest.mark.release)]
)
def test_clip_ppo_beside_torchrl_benchmark(
    benchmark: BenchmarkFixture, steps: int
) -> None:
    features, taken = potts_decisions(steps)
    rng = np.random.default_rng(9770)
    old = torch.as_tensor(rng.normal(scale=0.1, size=steps)) - 1.5
    advantages = torch.as_tensor(rng.normal(size=steps))
    theirs = torchrl.clip_ppo_loss(
        features.numpy(),
        taken.numpy(),
        old.numpy(),
        advantages.numpy(),
        np.asarray(WEIGHTS),
        clips=[0.2] * (1 + REPEATS),
        n_episodes=steps // 100,
    )
    (value, gradient), seconds = _warm_ours(
        benchmark,
        lambda: ppo_loss_and_gradient(features, taken, old, advantages, 100, 0.2),
    )
    _record(benchmark, _warm(theirs.seconds), seconds)
    np.testing.assert_allclose(theirs.values[0], float(value.detach()), atol=1e-8)
    np.testing.assert_allclose(theirs.gradients[0], gradient.numpy(), atol=1e-8)


@requires("torchrl")
@pytest.mark.parametrize("steps", [1_000, 100_000])
def test_reinforce_beside_torchrl_benchmark(
    benchmark: BenchmarkFixture, steps: int
) -> None:
    environment = potts_environment()
    weights = torch.tensor(WEIGHTS, dtype=torch.float64)
    episodes = greedy_episodes(environment, weights, steps // 10, 10, 977)
    features, taken, returns = reinforce_decisions(environment, episodes)
    theirs = torchrl.reinforce_loss(
        features.numpy(),
        taken,
        np.tile(returns - 0.5, (1 + REPEATS, 1)),
        np.asarray(WEIGHTS),
        n_episodes=len(episodes),
    )
    policy = LinearPolicy(2)
    policy.set_weights(torch.tensor(WEIGHTS, dtype=torch.float64))

    def loss_and_gradient() -> tuple[torch.Tensor, torch.Tensor]:
        value = surrogate_loss(environment, policy, episodes, 0.5)
        (gradient,) = torch.autograd.grad(value, policy.weights)
        return value, gradient

    (value, gradient), seconds = _warm_ours(benchmark, loss_and_gradient)
    _record(benchmark, _warm(theirs.seconds), seconds)
    np.testing.assert_allclose(theirs.values[0], float(value.detach()), atol=1e-8)
    np.testing.assert_allclose(theirs.gradients[0], gradient.numpy(), atol=1e-8)


@requires("torch_geometric")
@pytest.mark.parametrize("side", [16, 71, 142, 284])
def test_graph_surrogate_beside_gin_benchmark(
    benchmark: BenchmarkFixture, side: int
) -> None:
    graph = lattice_graph((side, side), BoundaryCondition.OPEN, 1.0)
    rng = np.random.default_rng(977)
    tokens = torch.as_tensor(rng.normal(size=(graph.n_nodes, 4)))
    examples = Examples(
        features=torch.as_tensor(rng.normal(size=(1, 3))),
        targets=torch.zeros(1, dtype=torch.float64),
        groups=np.zeros(1, dtype=np.int64),
        tokens=(tokens,),
        adjacency=(np.asarray(graph.edge_index, dtype=np.int64),),
    )
    batch = _Batch(examples)
    torch.manual_seed(977)
    model = GraphSurrogate(3, 4, hidden=8, n_layers=2)
    with torch.no_grad():
        for layer in model.layers:
            layer[0].weight[:, 8:] = layer[0].weight[:, :8]  # type: ignore[index]
    theirs = torch_geometric.gin_forward(
        [model] * (1 + REPEATS),
        n_layers=2,
        hidden=8,
        tokens=batch.tokens.numpy(),
        edges=batch.edges.numpy(),
        owner=batch.owner.numpy(),
        features=batch.features.numpy(),
    )
    with torch.no_grad():
        ours, seconds = _warm_ours(benchmark, lambda: model(batch))
    _record(benchmark, _warm(theirs.seconds), seconds)
    np.testing.assert_allclose(theirs.forward[0], ours.numpy(), atol=1e-10)


@requires("gymnasium")
@pytest.mark.parametrize("steps", [1_000, 100_000])
def test_gymnasium_adapter_beside_rollout_benchmark(
    benchmark: BenchmarkFixture, steps: int
) -> None:
    spec = gymnasium.Spec(
        kind="potts", coupling=COUPLING, field=tuple(FIELD.tolist()), chain_length=4
    )
    theirs = gymnasium.throughput(spec, list(WEIGHTS), n_steps=steps, max_steps=8)
    environment = potts_environment()
    policy = LinearPolicy(2)
    policy.set_weights(torch.tensor(WEIGHTS, dtype=torch.float64))

    def bare() -> int:
        rng = np.random.default_rng(0)
        taken = 0
        while taken < steps:
            taken += len(rollout(environment, policy, rng, 8).actions) or 1
        return taken

    start = time.perf_counter()
    _once(benchmark, bare)
    _record(benchmark, theirs, time.perf_counter() - start)
