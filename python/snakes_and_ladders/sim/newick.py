"""Newick I/O for rooted binary trees: the package's single source of
Newick functionality (root ``CLAUDE.md``, "Package Surface").

Covers the three directions the simulator and its consumers need:
serializing a (optionally state-labelled) tree to Newick, validating that a
string is well-formed Newick for a rooted binary tree, and counting how many
distinct binary topologies exist for a given number of taxa. Every other
module that needs any of these imports this one rather than reimplementing
it -- ``snakes_and_ladders.sim.tree`` only holds the ``Node`` structure and traversals.

The closed-form leaf-labelled rooted binary tree count, ``(2n-3)!!``, is the
standard result for the number of possible tree topologies (Felsenstein,
*Inferring Phylogenies*, ch. 3).

Alongside the rooted grammar this module also validates the *unrooted*,
trifurcating-root convention: an unrooted binary topology drawn with an
arbitrary internal node as a 3-child root, reusing ``Node`` rather than a
parallel type (root ``CLAUDE.md``'s no-premature-abstraction rule). It is
grammar-only -- exactly one root with 3 children, every other internal node
with exactly 2 -- and does not itself identify when two such strings denote
the same topology under a different rooting or child order; that is
``snakes_and_ladders.search.topology``'s ``leaf_bipartitions`` for move-set validation,
and issue #73's canonical Newick key for the general case.
"""

from __future__ import annotations

import numpy as np

from snakes_and_ladders.sim.tree import Node

_TOKEN_BOUNDARY = frozenset("(),:;[")


def count_topologies(n_taxa: int) -> int:
    """Count distinct rooted binary tree topologies for ``n_taxa`` leaves.

    Parameters
    ----------
    n_taxa : int
        Number of labelled leaves, >= 1.

    Returns
    -------
    int
        The closed-form count ``(2 * n_taxa - 3)!!`` (double factorial),
        i.e. 1 for ``n_taxa`` in ``{1, 2}``, 3 for 3 taxa, 15 for 4, etc.
        Counts strictly bifurcating rooted topologies -- the two children of
        every internal node, including the root, are unordered. A tree
        whose root has 3 children (the common "unrooted tree drawn at a
        trifurcating root" convention) is a different, and differently
        counted, object -- see ``validate_newick``.

    Raises
    ------
    ValueError
        If ``n_taxa`` < 1.
    """
    if n_taxa < 1:
        msg = f"n_taxa must be >= 1, got {n_taxa}"
        raise ValueError(msg)

    count = 1
    for odd in range(2 * n_taxa - 3, 0, -2):
        count *= odd
    return count


def to_newick(
    root: Node,
    node_states: dict[str, np.ndarray] | None = None,
    site: int = 0,
) -> str:
    """Serialize a tree to Newick format.

    Parameters
    ----------
    root : Node
        Root of the tree to serialize. Its ``branch_length`` is ignored, per
        the Newick convention that the root carries no incoming edge.
    node_states : dict[str, np.ndarray] | None
        Per-node simulated states, as in ``SimulatedDataset.node_states``
        (leaf name/ancestor name -> states of shape ``(n_sites,)``). When
        given, each node whose name is a key gets its ``site``-th state
        embedded as a ``[&state=<k>]`` comment, so a state-labelled tree
        (leaves plus ancestors) round-trips to a single Newick string.
    site : int
        Alignment column to read the embedded state from. Ignored if
        ``node_states`` is ``None``.

    Returns
    -------
    str
        The tree in Newick format, terminated with ``;``.
    """
    return f"{_to_newick(root, node_states, site)};"


def _to_newick(node: Node, node_states: dict[str, np.ndarray] | None, site: int) -> str:
    if node.is_leaf:
        label = node.name
    else:
        inner = ",".join(
            _to_newick(child, node_states, site) for child in node.children
        )
        label = f"({inner}){node.name}"
    if node_states is not None and node.name in node_states:
        state = int(node_states[node.name][site])
        label = f"{label}[&state={state}]"
    if node.branch_length is None:
        return label
    return f"{label}:{node.branch_length}"


def validate_newick(s: str) -> bool:
    """Check whether ``s`` is a well-formed Newick string for a rooted binary tree.

    A real grammar-level parse (balanced parens, exactly two children per
    internal node including the root, well-formed labels/branch lengths,
    single terminating ``;``), not a substring or trailing-character check.
    A root with 3 children -- the common convention for drawing an unrooted
    tree rooted at a trifurcation -- is rejected: it is not a rooted binary
    tree.

    Parameters
    ----------
    s : str
        Candidate Newick string.

    Returns
    -------
    bool
        Whether ``s`` parses as a rooted binary tree Newick string.
    """
    try:
        _parse_newick(s)
    except ValueError:
        return False
    return True


def validate_unrooted_newick(s: str) -> bool:
    """Check whether ``s`` is well-formed Newick for the trifurcating-root
    unrooted convention.

    A root with exactly 3 children (each itself a rooted binary subtree, or
    a leaf), rather than ``validate_newick``'s exactly 2 -- the standard way
    to draw an unrooted binary topology without a distinguished root edge.
    Below the root the grammar is identical to ``validate_newick``'s: every
    other internal node has exactly 2 children.

    Parameters
    ----------
    s : str
        Candidate Newick string.

    Returns
    -------
    bool
        Whether ``s`` parses as a trifurcating-root unrooted binary
        topology.
    """
    try:
        _parse_unrooted_newick(s)
    except ValueError:
        return False
    return True


def _parse_unrooted_newick(s: str) -> Node:
    if len(s) == 0 or s[0] != "(":
        msg = "expected a trifurcating root starting with '('"
        raise ValueError(msg)
    i = 1
    first, i = _parse_subtree(s, i)
    if i >= len(s) or s[i] != ",":
        msg = f"expected ',' after root's first child at position {i}"
        raise ValueError(msg)
    i += 1
    second, i = _parse_subtree(s, i)
    if i >= len(s) or s[i] != ",":
        msg = f"expected ',' after root's second child at position {i}"
        raise ValueError(msg)
    i += 1
    third, i = _parse_subtree(s, i)
    if i >= len(s) or s[i] != ")":
        msg = f"expected ')' closing the trifurcating root at position {i}"
        raise ValueError(msg)
    i += 1
    name, i = _parse_token(s, i)
    i = _skip_comment(s, i)
    branch_length, i = _parse_branch_length(s, i)
    root = Node(name=name, branch_length=branch_length, children=(first, second, third))
    if i >= len(s) or s[i] != ";":
        msg = "expected terminating ';'"
        raise ValueError(msg)
    i += 1
    if i != len(s):
        msg = f"trailing characters after ';': {s[i:]!r}"
        raise ValueError(msg)
    return root


def _parse_newick(s: str) -> Node:
    root, i = _parse_subtree(s, 0)
    if i >= len(s) or s[i] != ";":
        msg = "expected terminating ';'"
        raise ValueError(msg)
    i += 1
    if i != len(s):
        msg = f"trailing characters after ';': {s[i:]!r}"
        raise ValueError(msg)
    return root


def _parse_subtree(s: str, i: int) -> tuple[Node, int]:
    if i < len(s) and s[i] == "(":
        i += 1
        left, i = _parse_subtree(s, i)
        if i >= len(s) or s[i] != ",":
            msg = f"expected ',' between children at position {i}"
            raise ValueError(msg)
        i += 1
        right, i = _parse_subtree(s, i)
        if i >= len(s) or s[i] != ")":
            msg = f"expected ')' closing internal node at position {i}"
            raise ValueError(msg)
        i += 1
        name, i = _parse_token(s, i)
        i = _skip_comment(s, i)
        branch_length, i = _parse_branch_length(s, i)
        return Node(name=name, branch_length=branch_length, children=(left, right)), i

    name, i = _parse_token(s, i)
    if not name:
        msg = f"empty leaf label at position {i}"
        raise ValueError(msg)
    i = _skip_comment(s, i)
    branch_length, i = _parse_branch_length(s, i)
    return Node(name=name, branch_length=branch_length, children=()), i


def _parse_token(s: str, i: int) -> tuple[str, int]:
    start = i
    while i < len(s) and s[i] not in _TOKEN_BOUNDARY:
        i += 1
    return s[start:i], i


def _skip_comment(s: str, i: int) -> int:
    if i < len(s) and s[i] == "[":
        end = s.find("]", i)
        if end == -1:
            msg = f"unterminated comment starting at position {i}"
            raise ValueError(msg)
        return end + 1
    return i


def _parse_branch_length(s: str, i: int) -> tuple[float | None, int]:
    if i < len(s) and s[i] == ":":
        i += 1
        token, i = _parse_token(s, i)
        try:
            return float(token), i
        except ValueError as exc:
            msg = f"invalid branch length {token!r} at position {i}"
            raise ValueError(msg) from exc
    return None, i
