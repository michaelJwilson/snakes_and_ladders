"""`sal.param_tree` against the hand-written walks it replaces (issue #1129).

Each operation is checked on nests of dicts, lists, registered frozen
dataclasses and ``None`` fields, with torch and NumPy leaves: a round trip
returns the tree unchanged; names and order follow insertion; a stack is the
per-leaf `torch.stack`/`np.stack` written out by hand. Each consumer switched
to the module is bitwise the loop it replaced.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch
from sal import param_tree
from sal.emissions import BetaBinomialEmission, NegativeBinomialEmission
from sal.sim.count_pairs import IndependentCountPair


@param_tree.register
@dataclasses.dataclass(frozen=True)
class _Start:
    """A start as #1085 shapes it: weights, components, an optional note."""

    weights: torch.Tensor
    components: dict[str, np.ndarray]
    note: torch.Tensor | None = None


def _tree(seed: int) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    return {
        "log_transition": torch.as_tensor(rng.normal(size=(3, 3))),
        "emissions": {
            "mean": rng.normal(size=(3, 2)),
            "scale": torch.as_tensor(rng.normal(size=(3, 2))),
        },
        "start": _Start(
            torch.as_tensor(rng.dirichlet(np.ones(3))),
            {"rate": rng.uniform(size=3)},
        ),
        "lags": [torch.as_tensor(rng.normal(size=2)), rng.normal(size=4)],
    }


def _equal(left: object, right: object) -> bool:
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    return bool(np.array_equal(np.asarray(left), np.asarray(right)))


@pytest.mark.smoke
def test_a_round_trip_returns_the_tree() -> None:
    tree = _tree(0)

    found, shape = param_tree.flatten(tree)
    rebuilt = param_tree.unflatten(shape, found)

    assert shape.n_leaves == 7
    assert param_tree.structure(rebuilt) == shape
    assert isinstance(rebuilt["start"], _Start)
    assert rebuilt["start"].note is None
    assert all(
        _equal(a, b)
        for a, b in zip(
            param_tree.leaves(tree), param_tree.leaves(rebuilt), strict=True
        )
    )


@pytest.mark.smoke
def test_names_follow_insertion_order_and_field_order() -> None:
    # optree sorts dict keys by default; the package's walk does not.
    names = list(param_tree.named(_tree(0)))

    assert names == [
        "log_transition",
        "emissions.mean",
        "emissions.scale",
        "start.weights",
        "start.components.rate",
        "lags.0",
        "lags.1",
    ]


@pytest.mark.smoke
def test_a_stack_is_the_per_leaf_stack_written_by_hand() -> None:
    trees = [_tree(seed) for seed in range(5)]

    stacked = param_tree.stack(trees)

    by_hand = [
        torch.stack(parts) if isinstance(parts[0], torch.Tensor) else np.stack(parts)
        for parts in zip(*(param_tree.leaves(tree) for tree in trees), strict=True)
    ]
    got = param_tree.leaves(stacked)
    assert all(_equal(a, b) for a, b in zip(got, by_hand, strict=True))
    assert isinstance(stacked["emissions"]["scale"], torch.Tensor)
    assert isinstance(stacked["emissions"]["mean"], np.ndarray)
    assert got[0].shape == (5, 3, 3)


@pytest.mark.smoke
def test_mismatched_trees_and_counts_are_refused() -> None:
    tree = _tree(0)
    shorter = {key: value for key, value in tree.items() if key != "lags"}

    with pytest.raises(ValueError, match="first tree's structure"):
        param_tree.map_leaves(lambda a, _: a, tree, shorter)
    with pytest.raises(ValueError, match="holds 7 leaves"):
        param_tree.unflatten(param_tree.structure(tree), [1, 2])
    with pytest.raises(ValueError, match="at least one tree"):
        param_tree.stack([])
    with pytest.raises(TypeError, match="not a dataclass"):
        param_tree.register(int)


@pytest.mark.smoke
@pytest.mark.patch
def test_the_count_pair_names_its_parameters_as_the_loop_did() -> None:
    pair = IndependentCountPair(
        NegativeBinomialEmission([4.0, 12.0], [40.0, 150.0]),
        BetaBinomialEmission([20.0, 30.0], [2.0, 6.0], [6.0, 3.0]),
    )
    by_hand: dict[str, torch.Tensor] = {}
    for prefix, family in (
        ("total", pair.total.named_parameters()),
        ("successes", pair.successes.named_parameters()),
    ):
        by_hand.update({f"{prefix}.{name}": value for name, value in family.items()})

    named = pair.named_parameters()

    assert list(named) == list(by_hand)
    assert all(named[key] is by_hand[key] for key in by_hand)


@pytest.mark.smoke
@pytest.mark.patch
def test_a_leaf_map_is_the_dict_comprehension() -> None:
    parameters = {"a": torch.arange(3.0), "b": torch.ones(2, 2)}

    mapped = param_tree.map_leaves(torch.Tensor.numpy, parameters)
    cloned = param_tree.map_leaves(torch.Tensor.clone, parameters)

    assert list(mapped) == list(parameters)
    assert all(
        np.array_equal(mapped[key], value.numpy()) for key, value in parameters.items()
    )
    assert all(
        torch.equal(cloned[key], value) and cloned[key] is not value
        for key, value in parameters.items()
    )
