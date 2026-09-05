"""Rust CPU Felsenstein pruning (`snakes_and_ladders.oxi_snakes_and_ladders.pruning_log_likelihood`) --
pinned against ``snakes_and_ladders.likelihood.pruning``, the NumPy oracle
(``likelihood/CLAUDE.md``, "The NumPy reference is the oracle and it
stays").

Implements the same recursion as the oracle -- eq. (pruning)/(root) of
``docs/tex/textbook.tex`` -- in Rust (``src/pruning.rs``), exposed via PyO3.
There is no autodiff graph to protect here (unlike ``pruning_torch.py``),
but this wrapper still flattens ``tau`` into plain arrays crossing the FFI
boundary in a fixed, defined order, rather than letting Rust read
``Node.branch_length`` mid-recursion -- the same shape of interface
``pruning_torch.py``'s ``branch_order`` convention establishes, kept for
consistency across backends even though Rust has no gradient to keep out of
the topology.

Nodes cross the boundary in post-order (children before parents, root
last): ``snakes_and_ladders.sim.tree`` has no ``postorder`` helper, so this module builds
one locally rather than adding one there for a single caller. Validated to
machine precision against the NumPy oracle
(``tests/regression/test_pruning_rust.py``), per ``likelihood/CLAUDE.md``'s
statement of the Rust-backend tolerance.
"""

from __future__ import annotations

import numpy as np

from snakes_and_ladders import oxi_snakes_and_ladders
from snakes_and_ladders.sim.tree import Node


def _postorder(root: Node) -> list[Node]:
    """Every node in the tree rooted at ``root``, children before parents.

    Parameters
    ----------
    root : Node
        Root of the tree to walk.

    Returns
    -------
    list[Node]
        Nodes in post-order; ``root`` is always last.
    """
    order: list[Node] = []

    def _walk(node: Node) -> None:
        for child in node.children:
            _walk(child)
        order.append(node)

    _walk(root)
    return order


def log_likelihood(
    tau: Node,
    k: int,
    pi: np.ndarray,
    alignment: dict[str, np.ndarray],
    *,
    rescale: bool = True,
) -> float:
    """Total log-likelihood of an alignment under the k-state Jukes-Cantor model.

    Signature matches ``snakes_and_ladders.likelihood.pruning.log_likelihood``; this
    wrapper flattens ``tau`` and ``alignment`` into the arrays
    ``snakes_and_ladders.oxi_snakes_and_ladders.pruning_log_likelihood`` expects and calls the compiled
    Rust kernel.

    Parameters
    ----------
    tau : Node
        Root of the topology, with branch lengths attached to each non-root
        node.
    k : int
        Number of states.
    pi : np.ndarray
        Root state distribution, shape (k,).
    alignment : dict[str, np.ndarray]
        Leaf name to its observed states, each of shape (n_sites,) with
        entries in ``[0, k)``.
    rescale : bool
        Whether to rescale partial likelihoods per node, accumulating the log
        of the scale factor separately, matching
        ``snakes_and_ladders.likelihood.pruning``'s ``rescale`` flag.

    Returns
    -------
    float
        ``sum_s log Pr(data_s | tau, t, Q, pi)``, summed over sites.

    Raises
    ------
    ValueError
        If ``pi`` does not have shape ``(k,)``, ``alignment`` is missing a
        leaf of ``tau``, or a non-root node has no ``branch_length``.
    """
    if pi.shape != (k,):
        msg = f"pi has shape {pi.shape}, expected ({k},)"
        raise ValueError(msg)

    order = _postorder(tau)
    leaves = [node for node in order if node.is_leaf]
    missing = [leaf.name for leaf in leaves if leaf.name not in alignment]
    if missing:
        msg = f"alignment is missing leaf(ves) {missing}"
        raise ValueError(msg)

    index = {id(node): position for position, node in enumerate(order)}
    n_nodes = len(order)

    branch_length: list[float] = []
    children: list[list[int]] = []
    leaf_states: list[list[int]] = []
    for position, node in enumerate(order):
        is_root = position == n_nodes - 1
        if is_root:
            branch_length.append(0.0)
        else:
            if node.branch_length is None:
                msg = f"non-root node {node.name!r} has no branch_length"
                raise ValueError(msg)
            branch_length.append(float(node.branch_length))

        children.append([index[id(child)] for child in node.children])

        if node.is_leaf:
            leaf_states.append(alignment[node.name].tolist())
        else:
            leaf_states.append([])

    result = oxi_snakes_and_ladders.pruning_log_likelihood(
        branch_length, children, leaf_states, k, pi.tolist(), rescale
    )
    return float(result)
