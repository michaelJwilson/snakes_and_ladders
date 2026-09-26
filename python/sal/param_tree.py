"""Nested parameters as one tree: flatten, rebuild, map, stack and name, with ``optree`` behind it (issue #1129).

A parameter set in this package is a nest --- a ``dict`` of arrays, a
``dict`` inside it for an emission family, a frozen dataclass carrying both
--- and each boundary that walked one wrote its own loop: a dotted name per
leaf, a conversion per leaf, a copy per leaf. This module is that walk once.
Its API is the package's; ``optree`` is the backend, reached only here, so the
choice stays swappable and no caller holds an ``optree`` type.

**The backend, and why.** Measured on an ``HmmParams``-sized tree (ten states,
sixteen channels, a 64-restart batch; #1129): through this module a round
trip takes 12.4 us and :func:`stack` 121 us, against 151 us for the same
stack written out by path. Bare, ``optree`` rebuilt in 4.6 us; the difference
is the insertion-order setting below. ``dm-tree`` took 15 us to rebuild and
327 us to stack, and reads a frozen dataclass as one leaf; ``jax.tree_util``
is as fast but an autodiff import of 1.2 s on paths that take no derivative
(root ``CLAUDE.md``). Against any fit the cost is small either way; what the
tree buys is one walk, and a structure that is checked.

**A frozen dataclass is a node only once registered.** :func:`register` makes
its fields its children, in declaration order; an unregistered one is a leaf,
as every non-container is. Registration is in the ``sal`` namespace, so it
does not change what another library's ``optree`` call sees.

**A dict is walked in insertion order**, not ``optree``'s sorted default,
so a ``named_parameters`` mapping keeps the order its family wrote and a
rebuilt dict iterates as the original did.

**Leaves are anything.** A NumPy array, a torch tensor, a Python scalar: the
walk reads none of them. ``None`` is an empty subtree, as ``optree`` has it,
so an optional field that is unset contributes no leaf.
"""

from __future__ import annotations

import contextlib
import dataclasses
from collections.abc import Callable, Sequence
from typing import Any, cast

import numpy as np
import optree
import torch

#: The ``optree`` namespace every registration and walk here uses.
NAMESPACE = "sal"


def _ordered() -> contextlib.AbstractContextManager[None]:
    """Every walk here reads a dict in insertion order."""
    return optree.dict_insertion_ordered(True, namespace=NAMESPACE)


@dataclasses.dataclass(frozen=True)
class Structure:
    """A tree's shape without its leaves: what :func:`unflatten` rebuilds from.

    Parameters
    ----------
    n_leaves : int
        How many leaves the shape holds.
    """

    n_leaves: int
    _spec: Any = dataclasses.field(repr=False, compare=False)

    def __eq__(self, other: object) -> bool:
        """Whether the two shapes are the same, container for container."""
        return isinstance(other, Structure) and bool(self._spec == other._spec)

    def __hash__(self) -> int:
        """The backend's hash of the shape."""
        return hash(self._spec)


def register[T](cls: type[T]) -> type[T]:
    """Make a dataclass a node: its fields, in order, are its children.

    Usable as a decorator. The class is rebuilt with every field passed by
    keyword, so a frozen dataclass round-trips through :func:`unflatten`.

    Returns
    -------
    type
        ``cls``, registered.

    Raises
    ------
    TypeError
        If ``cls`` is not a dataclass.
    """
    if not dataclasses.is_dataclass(cls):
        msg = f"{cls.__name__} is not a dataclass; only a dataclass is registered"
        raise TypeError(msg)
    names = tuple(field.name for field in dataclasses.fields(cls) if field.init)

    def flatten(node: Any) -> tuple[tuple[Any, ...], None, tuple[str, ...]]:
        return tuple(getattr(node, name) for name in names), None, names

    def rebuild(_: None, children: tuple[Any, ...]) -> Any:
        return cls(**dict(zip(names, children, strict=True)))

    # `optree`'s stubs type a node as a collection; a dataclass is one only
    # through this registration, which is what the cast states.
    optree.register_pytree_node(
        cast("Any", cls), flatten, cast("Any", rebuild), namespace=NAMESPACE
    )
    return cls


def leaves(tree: Any) -> list[Any]:
    """Every leaf, in the walk's order: dict keys as inserted, fields as declared.

    Returns
    -------
    list
    """
    with _ordered():
        return list(optree.tree_leaves(tree, none_is_leaf=False, namespace=NAMESPACE))


def flatten(tree: Any) -> tuple[list[Any], Structure]:
    """The leaves, and the shape :func:`unflatten` puts them back into.

    Returns
    -------
    tuple[list, Structure]
    """
    with _ordered():
        found, spec = optree.tree_flatten(tree, none_is_leaf=False, namespace=NAMESPACE)
    return list(found), Structure(spec.num_leaves, spec)


def unflatten(structure: Structure, values: Sequence[Any]) -> Any:
    """``structure`` with ``values`` as its leaves, in :func:`flatten`'s order.

    Returns
    -------
    Any

    Raises
    ------
    ValueError
        If the count of values is not the shape's.
    """
    if len(values) != structure.n_leaves:
        msg = f"the structure holds {structure.n_leaves} leaves, got {len(values)}"
        raise ValueError(msg)
    with _ordered():
        return optree.tree_unflatten(structure._spec, list(values))


def structure(tree: Any) -> Structure:
    """``tree``'s shape.

    Returns
    -------
    Structure
    """
    return flatten(tree)[1]


def map_leaves(function: Callable[..., Any], tree: Any, *rest: Any) -> Any:
    """``function`` of each leaf, or of the leaves at one position of every tree.

    Returns
    -------
    Any
        The same shape as ``tree``.

    Raises
    ------
    ValueError
        If a tree in ``rest`` is not ``tree``'s shape.
    """
    # `optree` checks every tree against the first as it maps, so the check
    # costs no second walk; its message is restated in the package's words.
    try:
        with _ordered():
            return optree.tree_map(
                function, tree, *rest, none_is_leaf=False, namespace=NAMESPACE
            )
    except ValueError as error:
        if not rest:
            raise
        msg = (
            f"every tree mapped together must have the first tree's structure: {error}"
        )
        raise ValueError(msg) from error


def stack(trees: Sequence[Any], *, axis: int = 0) -> Any:
    """One tree whose leaves are the trees' leaves stacked: restarts, seeds, replicas.

    A torch leaf is stacked by ``torch.stack`` and anything else by
    ``np.stack``, so each leaf keeps its library.

    Returns
    -------
    Any
        The first tree's shape.

    Raises
    ------
    ValueError
        If ``trees`` is empty or two of them differ in shape.
    """
    if not trees:
        msg = "stack needs at least one tree"
        raise ValueError(msg)

    def together(*parts: Any) -> Any:
        if isinstance(parts[0], torch.Tensor):
            return torch.stack(parts, dim=axis)
        return np.stack(parts, axis=axis)

    return map_leaves(together, trees[0], *trees[1:])


def named(tree: Any, *, separator: str = ".") -> dict[str, Any]:
    """Each leaf under the path that reaches it: dict keys and field names joined.

    ``{"total": {"mean": m}}`` names ``m`` as ``"total.mean"``; a sequence's
    position is its index.

    Returns
    -------
    dict[str, Any]
    """
    with _ordered():
        paths, found, _ = optree.tree_flatten_with_path(
            tree, none_is_leaf=False, namespace=NAMESPACE
        )
    return {
        separator.join(str(part) for part in path): leaf
        for path, leaf in zip(paths, found, strict=True)
    }
