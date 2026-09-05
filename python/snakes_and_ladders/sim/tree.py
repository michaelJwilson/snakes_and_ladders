"""Minimal tree representation for the simulator.

Topology is an input to simulation (drawn from ``simulation_params.yaml``),
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
