"""Cross the FFI boundary once per search, not once per pass: the answer is 1.34% (issue #444, PR 1).

**The problem, as posed.** `oxi_snakes_and_ladders.pruning_log_likelihood`
takes seven arguments and one of them — the branch lengths — changes between
the forward passes of a fit; the topology changes between candidates of a
search; the alignment changes for neither. Issue #436 found a 28-35%
criterion win on the pruning kernel arriving as +1.8% and -2.2% through the
binding, and named the boundary as where to look next. So: hold what does not
change on the Rust side and re-marshal only what does.

**The measured answer.** 1.34% at 8 taxa and 0.79% at 20. Decomposing one
through-binding call puts marshalling at 1-5% of it and the kernel at 90-100%
(issue #443), so amortizing every crossing a fit would make is worth that and
no more — under the 10% bar `STATUS.md` records ports against. The handle
below does what it was built to do: it cuts the bytes a caller materialises
per pass by 179x to 1,881x and the argument construction from 0.173 ms to
0.023 ms of a 5.606 ms call. The bytes fell by three orders of magnitude and
the wall time did not follow. `STATUS.md` carries every figure, including a
20-candidate sweep that reads -22 to -26% and is not attributed.

**It is not on a hot path, and there is no path for it to be on.** A search's
fits run `likelihood.pruning_torch`, which autograd differentiates and a Rust
kernel cannot; `likelihood.pruning_rust` is reached only by
`qa.backend_agreement` and the tests. So this is a finished answer to a
question that was asked, kept per `sandbox/CLAUDE.md` rather than deleted,
and `likelihood/` is unchanged. Its regression tests pin it against
`likelihood.pruning_rust.log_likelihood`, which returns the same ``float``
bitwise — one kernel, one order of operations. The `#[pyclass]` it fronts
stays in `src/pruning.rs` for the same reason.

**A problem handle is not a generator.** `src/sampling.rs` states that Rust
holds no random generator, so that a chain's reproducibility follows from the
caller's `numpy.random.Generator`. Nothing here is random: the handle is an
immutable copy of arguments the caller already owns, and two calls with the
same arguments return the same ``float`` whatever order they run in.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from snakes_and_ladders import oxi_snakes_and_ladders
from snakes_and_ladders.sim.tree import Node


def _postorder(root: Node) -> list[Node]:
    """Every node in the tree rooted at ``root``, children before parents.

    A copy of ``likelihood.pruning_rust``'s walk rather than an import of it:
    `sandbox/CLAUDE.md` keeps the oracle home free of edges into the hot-path
    packages, and this is six lines.

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


class PruningProblem:
    """An alignment, ``k`` and ``pi`` held across pruning passes.

    Of what ``likelihood.pruning_rust.log_likelihood`` marshals on every call,
    only the branch lengths change between the forward passes of a fit and
    only the topology changes between candidates of a search. This holds the
    rest in Rust and takes the branch lengths, a flat parent-index array and
    ``leaf_row`` as three borrowed buffers, so the topology reaches PyO3 as
    one array rather than ``n_nodes`` Python lists.

    ``leaf_row`` is rebuilt per call and not held: it maps a node index to a
    row of the alignment, so it is a function of the topology's numbering
    rather than of the alignment.

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

        Returns the same ``float`` as
        ``likelihood.pruning_rust.log_likelihood`` called with the same tree
        and alignment --- the same kernel over the same topology in the same
        order.

        Parameters
        ----------
        tau : Node
            Root of the topology, with branch lengths attached to each
            non-root node. Its leaf names must be among the alignment's.
        rescale : bool
            Whether to rescale partial likelihoods per node, accumulating the
            log of the scale factor separately.

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
        # reads no row: the two sentinels the Rust side expects.
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


__all__ = ["PruningProblem"]
