"""Rust CPU Felsenstein pruning (`snakes_and_ladders.oxi_snakes_and_ladders.pruning_log_likelihood`) --
pinned against ``snakes_and_ladders.likelihood.pruning``, the NumPy oracle
(``likelihood/CLAUDE.md``, "The NumPy reference is the oracle and it
stays").

Implements the same recursion as the oracle -- ``eq:pruning`` and ``eq:root`` of
``docs/tex/textbook.tex`` -- in Rust (``src/pruning.rs``), exposed via PyO3.
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

**The kernel takes no weights, and this wrapper supplies them by grouping.**
``pruning_log_likelihood`` returns one scalar summed over the columns it is
given, so site-pattern weights (``snakes_and_ladders.likelihood.patterns``)
cannot be handed to it without changing the Rust signature. They do not have
to be: the weighted sum is ``sum_w w * (unweighted sum over the patterns of
weight w)``, so one kernel call per *distinct weight* gives the exact value
over the pattern table. The alternative, a weights argument in ``src/``, is
one call rather than a few and is what a later change should do.

Nodes cross the boundary in post-order (children before parents, root
last): ``snakes_and_ladders.sim.tree`` has no ``postorder`` helper, so this module builds
one locally rather than adding one there for a single caller. Validated to
machine precision against the NumPy oracle
(``tests/regression/test_pruning_rust.py``), per ``likelihood/CLAUDE.md``'s
statement of the Rust-backend tolerance.

**Two entry points, and which one a caller wants is a byte count.**
:func:`log_likelihood` rebuilds every argument on every call, including the
alignment block. :class:`PruningProblem` holds the alignment, ``k`` and
``pi`` across passes and rebuilds only what changes -- the branch lengths and
the topology -- so a caller scoring many topologies against one alignment
pays for the alignment once. Issue #444 carries the counts. The two return
the same ``float``, not two values inside a tolerance: same kernel, same
arithmetic, same order, which is what lets one replace the other in a caller
without moving a committed number.

A ``PruningProblem`` is gradient-free. It computes a value and no
derivative, so it belongs on the screening and scoring paths and never
inside a fit: a Rust call in a differentiated path is opaque to
``torch.autograd``, and ``pruning_torch`` remains the only backend for that
(``likelihood/CLAUDE.md``).
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from snakes_and_ladders import oxi_snakes_and_ladders
from snakes_and_ladders.likelihood.patterns import check_weights
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
    weights: np.ndarray | None = None,
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
    weights : np.ndarray | None
        One weight per column, or ``None`` for one occurrence each, matching
        ``snakes_and_ladders.likelihood.pruning``. The kernel takes no
        weights, so a weighted call is split into one kernel call per
        distinct weight -- see the module docstring.
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
        leaf of ``tau``, the alignment is ragged, ``weights`` does not have
        one entry per column, or a non-root node has no ``branch_length``.
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

    weight = check_weights(weights, n_sites)
    row_index = leaf_row.tolist()
    pi_contiguous = np.ascontiguousarray(pi, dtype=np.float64)

    def _kernel(states: np.ndarray) -> float:
        return float(
            oxi_snakes_and_ladders.pruning_log_likelihood(
                branch_length,
                children,
                np.ascontiguousarray(states),
                row_index,
                k,
                pi_contiguous,
                rescale,
            )
        )

    if weight is None:
        return _kernel(leaf_states)
    # The kernel sums its columns unweighted, so the weighted sum is grouped
    # by weight: sum_p w_p l_p = sum_w w * sum_{p : w_p = w} l_p, one kernel
    # call per distinct weight. Exact up to the reassociation of a floating
    # sum, and it crosses the boundary once per group rather than once per
    # column -- the cost issue #232 measured.
    total = 0.0
    for value in np.unique(weight):
        selected = leaf_states[:, weight == value]
        if selected.shape[1]:
            total += float(value) * _kernel(selected)
    return total


class PruningProblem:
    """An alignment held across many pruning passes: it crosses the FFI boundary once.

    Of what :func:`log_likelihood` marshals on every call, only the branch
    lengths change within a fit and only the topology changes between
    candidates of a search. The alignment changes for neither, and issue
    #436's profile is 301 fits and 18,955 forward passes over one alignment
    -- so it crossed about 63 times more often than it changed. This class
    holds it, ``k`` and ``pi`` in Rust and takes only what changes.

    The topology crosses as a flat parent-index array rather than a list of
    lists, so PyO3 borrows one NumPy buffer instead of walking ``n_nodes``
    Python lists and allocating a ``Vec`` per node.

    ``leaf_row`` is rebuilt per call and not held: it maps a node index to a
    row of the alignment, so it is a function of the topology's numbering
    rather than of the alignment, and it changes exactly when the topology
    does. It is ``n_nodes`` ``int64``s against the alignment's
    ``n_leaves * n_sites``.

    Parameters
    ----------
    k : int
        Number of states.
    pi : np.ndarray
        Root state distribution, shape (k,).
    alignment : Mapping[str, np.ndarray]
        Leaf name to its observed states, each of shape (n_sites,) with
        entries in ``[0, k)``. Copied into the handle at construction; later
        edits to the caller's arrays are not seen.

    Raises
    ------
    ValueError
        If ``pi`` does not have shape ``(k,)``, the alignment is empty or
        ragged, or ``k < 2``.
    """

    def __init__(
        self, k: int, pi: np.ndarray, alignment: Mapping[str, np.ndarray]
    ) -> None:
        if pi.shape != (k,):
            msg = f"pi has shape {pi.shape}, expected ({k},)"
            raise ValueError(msg)
        names = sorted(alignment)
        if not names:
            msg = "alignment is empty"
            raise ValueError(msg)
        n_sites = int(alignment[names[0]].shape[0])
        states = np.empty((len(names), n_sites), dtype=np.int64)
        for row, name in enumerate(names):
            observed = alignment[name]
            if observed.shape != (n_sites,):
                msg = (
                    f"leaf {name!r} has shape {observed.shape}, "
                    f"expected ({n_sites},) -- the alignment is ragged"
                )
                raise ValueError(msg)
            states[row] = observed
        self.k = k
        self._row_of = {name: row for row, name in enumerate(names)}
        self._handle = oxi_snakes_and_ladders.PruningProblem(
            states, k, np.ascontiguousarray(pi, dtype=np.float64)
        )

    def log_likelihood(self, tau: Node, *, rescale: bool = True) -> float:
        """Total log-likelihood of the held alignment on ``tau``.

        Returns the same ``float`` as :func:`log_likelihood` called with the
        same tree and alignment.

        Parameters
        ----------
        tau : Node
            Root of the topology, with branch lengths attached to each
            non-root node. Its leaf names must be among the alignment's.
        rescale : bool
            Whether to rescale partial likelihoods per node, as
            :func:`log_likelihood`.

        Returns
        -------
        float
            ``sum_s log Pr(data_s | tau, t, Q, pi)``, summed over sites.

        Raises
        ------
        ValueError
            If the alignment is missing a leaf of ``tau``, or a non-root node
            has no ``branch_length``.
        """
        order = _postorder(tau)
        missing = [
            node.name
            for node in order
            if node.is_leaf and node.name not in self._row_of
        ]
        if missing:
            msg = f"alignment is missing leaf(ves) {missing}"
            raise ValueError(msg)

        index = {id(node): position for position, node in enumerate(order)}
        n_nodes = len(order)
        branch_length = np.zeros(n_nodes, dtype=np.float64)
        # -1 marks the root, which has no parent, and an internal node, which
        # reads no row: the two sentinels the Rust side already expects.
        parent = np.full(n_nodes, -1, dtype=np.int64)
        leaf_row = np.full(n_nodes, -1, dtype=np.int64)
        for position, node in enumerate(order):
            if position != n_nodes - 1:
                if node.branch_length is None:
                    msg = f"non-root node {node.name!r} has no branch_length"
                    raise ValueError(msg)
                branch_length[position] = float(node.branch_length)
            for child in node.children:
                parent[index[id(child)]] = position
            if node.is_leaf:
                leaf_row[position] = self._row_of[node.name]

        return float(
            self._handle.log_likelihood(branch_length, parent, leaf_row, rescale)
        )
