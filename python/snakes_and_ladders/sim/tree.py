"""Minimal tree representation for the simulator.

Topology is an input to simulation (drawn from a tree fixture),
never inferred. Newick serialization, parsing, and validation live in
``snakes_and_ladders.sim.newick``, the package's single source of Newick functionality.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass


@dataclass(frozen=True)
class Node:
    """A node in a rooted tree.

    Parameters
    ----------
    name : str
        Label for the node. Leaf names double as alignment taxon names.
    branch_length : float | None
        Length of the edge above this node, in expected substitutions per
        site. ``None`` at the root, where there is no incoming edge.
    children : tuple[Node, ...]
        Child nodes, empty for a leaf.
    """

    name: str
    branch_length: float | None
    children: tuple[Node, ...] = ()

    @property
    def is_leaf(self) -> bool:
        """Whether this node has no children."""
        return not self.children


def preorder(root: Node) -> Iterator[Node]:
    """Yield every node in the tree rooted at ``root``, parent before children.

    Parameters
    ----------
    root : Node
        Root of the tree to walk.

    Returns
    -------
    Iterator[Node]
        Nodes in pre-order.
    """
    yield root
    for child in root.children:
        yield from preorder(child)


def edges(root: Node) -> Iterator[tuple[Node, Node]]:
    """Yield every ``(parent, child)`` edge in the tree rooted at ``root``.

    Parameters
    ----------
    root : Node
        Root of the tree to walk.

    Returns
    -------
    Iterator[tuple[Node, Node]]
        Parent/child pairs, one per edge.
    """
    for child in root.children:
        yield root, child
        yield from edges(child)


def balanced_tree(n_taxa: int, height: float) -> Node:
    """A balanced topology on ``n_taxa`` leaves whose root-to-tip paths are ``height``.

    Deterministic, so a fixture declaring ``(n_taxa, height)`` declares the
    tree: there is no seed to record and no generator to keep in step with a
    committed copy, which is what lets a 200-leaf instance be a fixture at all
    rather than six hundred lines of yaml.

    **Every edge is ``height`` divided by the tree's depth**, so the diameter
    stays put as the leaf count grows and the internal branches shorten
    instead. Holding the *edge* length fixed would grow the diameter with the
    leaf count, and distant pairs then saturate --- at 20 leaves a run that
    did so refused a pair differing on 0.7550 of sites, at or beyond the
    Jukes-Cantor saturation of 0.75. A ladder built that way measures
    divergence wearing a leaf count's label (issue #582).

    The root is trifurcating, the convention
    ``snakes_and_ladders.likelihood.objective`` needs for every branch to be
    separately estimable; a rooted binary root leaves its two branches
    confounded.

    Parameters
    ----------
    n_taxa : int
        Leaves, at least 3 for the trifurcating root to have one each.
    height : float
        Root-to-tip path length, in expected substitutions per site.

    Returns
    -------
    Node
        The root, with leaves named ``t000``, ``t001``, ... in order.

    Raises
    ------
    ValueError
        If ``n_taxa`` is below 3, or ``height`` is not positive.
    """
    if n_taxa < 3:
        msg = f"a trifurcating root needs at least 3 leaves, got {n_taxa}"
        raise ValueError(msg)
    if height <= 0:
        msg = f"height is positive, got {height}"
        raise ValueError(msg)

    names = [f"t{index:03d}" for index in range(n_taxa)]
    third = n_taxa // 3
    groups = (names[:third], names[third : 2 * third], names[2 * third :])
    depth = 1 + max(_balanced_depth(len(group)) for group in groups)
    scale = height / depth
    return Node(
        name="root",
        branch_length=None,
        children=tuple(_balanced(group, scale) for group in groups),
    )


def _balanced_depth(n_leaves: int) -> int:
    """Edges from a balanced subtree's root to its deepest leaf."""
    if n_leaves <= 1:
        return 0
    middle = n_leaves // 2
    return 1 + max(_balanced_depth(middle), _balanced_depth(n_leaves - middle))


def _balanced(names: list[str], scale: float) -> Node:
    """A balanced binary subtree over ``names``, every edge ``scale``."""
    if len(names) == 1:
        return Node(name=names[0], branch_length=scale)
    middle = len(names) // 2
    return Node(
        name=f"internal_{names[0]}_{names[-1]}",
        branch_length=scale,
        children=(_balanced(names[:middle], scale), _balanced(names[middle:], scale)),
    )
