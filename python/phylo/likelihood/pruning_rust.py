"""Rust CPU Felsenstein pruning (`phylo.oxiphylo.pruning_log_likelihood`) --
pinned against ``phylo.likelihood.pruning``, the NumPy oracle
(``likelihood/CLAUDE.md``, "The NumPy reference is the oracle and it
stays").

Implements the same recursion as the oracle -- eq. (pruning)/(root) of
``docs/tex/main.tex`` -- in Rust (``src/pruning.rs``), exposed via PyO3.
There is no autodiff graph to protect here (unlike ``pruning_torch.py``),
but this wrapper still flattens ``tau`` into plain arrays crossing the FFI
boundary in a fixed, defined order, rather than letting Rust read
``Node.branch_length`` mid-recursion -- the same shape of interface
``pruning_torch.py``'s ``branch_order`` convention establishes, kept for
consistency across backends even though Rust has no gradient to keep out of
the topology.

**The alignment crosses as one array, borrowed rather than copied.** The
leaf observations go over as a single C-contiguous ``(n_leaves, n_sites)``
``int64`` block with a per-node row index, so ``rust-numpy`` hands the kernel
the buffer itself. The nested-list form this replaced boxed one Python
integer per observed state, and that cost grows with ``n x L`` while the
kernel's advantage does not -- which is why the caller-visible speedup
decayed to parity at the declared scale (issue #232). It is the same fix
issue #202 applied to the categorical sampler, for the same reason.

Nodes cross the boundary in post-order (children before parents, root
last): ``phylo.sim.tree`` has no ``postorder`` helper, so this module builds
one locally rather than adding one there for a single caller. Validated to
machine precision against the NumPy oracle
(``tests/regression/test_pruning_rust.py``), per ``docs/tex/main.tex``'s
statement of the Rust-backend tolerance.
"""

from __future__ import annotations

import numpy as np

from phylo import oxiphylo
from phylo.sim.tree import Node


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

    Signature matches ``phylo.likelihood.pruning.log_likelihood``; this
    wrapper flattens ``tau`` and ``alignment`` into the arrays
    ``phylo.oxiphylo.pruning_log_likelihood`` expects and calls the compiled
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
        ``phylo.likelihood.pruning``'s ``rescale`` flag.

    Returns
    -------
    float
        ``sum_s log Pr(data_s | tau, t, Q, pi)``, summed over sites.

    Raises
    ------
    ValueError
        If ``pi`` does not have shape ``(k,)``, ``alignment`` is missing a
        leaf of ``tau``, the alignment is ragged, or a non-root node has no
        ``branch_length``.
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

    n_sites = int(alignment[leaves[0].name].shape[0])
    # One row per leaf, filled by C-level assignment. The nested-list form
    # this replaced built a Python integer per observed state, which at
    # `n = 200, L = 11 000` is 2.2 million objects on the way in -- a cost
    # growing with `n * L` while the kernel's advantage does not, and the
    # whole of the gap issue #232 measured.
    leaf_states = np.empty((len(leaves), n_sites), dtype=np.int64)
    leaf_row = np.full(n_nodes, -1, dtype=np.int64)

    branch_length = np.zeros(n_nodes, dtype=np.float64)
    children: list[list[int]] = []
    row = 0
    for position, node in enumerate(order):
        is_root = position == n_nodes - 1
        if not is_root:
            if node.branch_length is None:
                msg = f"non-root node {node.name!r} has no branch_length"
                raise ValueError(msg)
            branch_length[position] = float(node.branch_length)

        children.append([index[id(child)] for child in node.children])

        if node.is_leaf:
            states = alignment[node.name]
            if states.shape != (n_sites,):
                msg = (
                    f"leaf {node.name!r} has shape {states.shape}, "
                    f"expected ({n_sites},) -- the alignment is ragged"
                )
                raise ValueError(msg)
            leaf_states[row] = states
            leaf_row[position] = row
            row += 1

    result = oxiphylo.pruning_log_likelihood(
        branch_length,
        children,
        leaf_states,
        leaf_row.tolist(),
        k,
        np.ascontiguousarray(pi, dtype=np.float64),
        rescale,
    )
    return float(result)
