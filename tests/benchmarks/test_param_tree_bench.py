"""`sal.param_tree` against the hand-written walk, on an `HmmParams`-sized tree (issue #1129).

Ten states and sixteen channels, four torch leaves nested two deep; the
restart batch is 64 trees. Correctness is `tests/regression/test_param_tree.py`'s.
"""

from __future__ import annotations

import pytest
import torch
from pytest_benchmark.fixture import BenchmarkFixture
from sal import param_tree

#: States, channels and restarts.
K, C, R = 10, 16, 64


def _tree(seed: int) -> dict[str, object]:
    generator = torch.Generator().manual_seed(seed)

    def draw(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator, dtype=torch.float64)

    return {
        "log_initial": draw(K),
        "log_transition": draw(K, K),
        "emissions": {"mean": draw(K, C), "scale": draw(K, C)},
    }


def _by_hand(trees: list[dict[str, object]]) -> dict[str, object]:
    def at(tree: dict[str, object], *path: str) -> torch.Tensor:
        node: object = tree
        for key in path:
            node = node[key]  # type: ignore[index]
        assert isinstance(node, torch.Tensor)
        return node

    paths = [
        ("log_initial",),
        ("log_transition",),
        ("emissions", "mean"),
        ("emissions", "scale"),
    ]
    stacked = [torch.stack([at(tree, *path) for tree in trees]) for path in paths]
    return {
        "log_initial": stacked[0],
        "log_transition": stacked[1],
        "emissions": {"mean": stacked[2], "scale": stacked[3]},
    }


@pytest.mark.benchmark(group="param-tree-stack")
@pytest.mark.parametrize("route", ["param_tree", "by_hand"])
def test_restart_stack_bench(benchmark: BenchmarkFixture, route: str) -> None:
    trees = [_tree(seed) for seed in range(R)]
    stack = param_tree.stack if route == "param_tree" else _by_hand

    stacked = benchmark(stack, trees)

    assert stacked["log_transition"].shape == (R, K, K)


@pytest.mark.benchmark(group="param-tree-round-trip")
def test_round_trip_bench(benchmark: BenchmarkFixture) -> None:
    tree = _tree(0)

    def round_trip() -> object:
        found, shape = param_tree.flatten(tree)
        return param_tree.unflatten(shape, found)

    benchmark(round_trip)
