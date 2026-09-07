"""Vectorized NumPy Felsenstein pruning -- the oracle every backend is pinned against.

Implements the pruning recursion, eq. (pruning), and the root marginalization,
eq. (pruning) of ``docs/tex/textbook.tex`` (Sec. "The algorithm: pruning"):
the site log-likelihood
is computed post-order over ``(site, state)`` NumPy arrays, reusing
``snakes_and_ladders.sim.jc.jc_transition_probabilities`` for P(t). Partial likelihoods
underflow for realistic (site, taxa) counts, so they are rescaled per node
with the log of the scale factor accumulated separately -- a transformation
of the same computation, not a different algorithm (``likelihood/CLAUDE.md``,
"Rescaling must stay differentiable") -- so the rescaled and unrescaled paths
must agree wherever both run.

This module is written to be obviously correct, not fast: it is the reference
every accelerated backend (Rust, PyTorch, CUDA, Metal) is validated against,
per ``likelihood/CLAUDE.md``.
"""

from __future__ import annotations

import numpy as np

from snakes_and_ladders.sim.jc import jc_transition_probabilities
from snakes_and_ladders.sim.tree import Node, preorder


def log_likelihood(
    tau: Node,
    k: int,
    pi: np.ndarray,
    alignment: dict[str, np.ndarray],
    *,
    rescale: bool = True,
) -> float:
    """Total log-likelihood of an alignment under the k-state Jukes-Cantor model.

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
        entries in ``[0, k)`` -- the shape
        ``snakes_and_ladders.sim.simulate.SimulatedDataset.alignment`` produces.
    rescale : bool
        Whether to rescale partial likelihoods per node, accumulating the log
        of the scale factor separately (docs/tex/textbook.tex, Sec. "The algorithm:
        pruning").
        Disabling this underflows for realistic (site, taxa) sizes; it exists
        so tests can check the two paths agree on small problems where both
        run.

    Returns
    -------
    float
        ``sum_s log Pr(data_s | tau, t, Q, pi)``, summed over sites (eq.
        site-independence).

    Raises
    ------
    ValueError
        If ``pi`` does not have shape ``(k,)``, or ``alignment`` is missing a
        leaf of ``tau``.
    """
    if pi.shape != (k,):
        msg = f"pi has shape {pi.shape}, expected ({k},)"
        raise ValueError(msg)

    leaves = [node for node in preorder(tau) if node.is_leaf]
    missing = [leaf.name for leaf in leaves if leaf.name not in alignment]
    if missing:
        msg = f"alignment is missing leaf(ves) {missing}"
        raise ValueError(msg)

    n_sites = alignment[leaves[0].name].shape[0]
    log_scale = np.zeros(n_sites)

    def _post_order(node: Node) -> np.ndarray:
        if node.is_leaf:
            states = alignment[node.name]
            partial = np.zeros((n_sites, k))
            partial[np.arange(n_sites), states] = 1.0
            return partial

        partial = np.ones((n_sites, k))
        for child in node.children:
            if child.branch_length is None:
                msg = f"non-root node {child.name!r} has no branch_length"
                raise ValueError(msg)
            child_partial = _post_order(child)
            transition = jc_transition_probabilities(child.branch_length, k=k)
            # message[s, i] = sum_j P_ij(t) * L_child(s, j) -- eq. (pruning).
            partial = partial * (child_partial @ transition.T)

        if rescale:
            scale = partial.max(axis=1)
            # A site with scale == 0 has zero likelihood under the model
            # (an internal node's partial likelihood vanished entirely, e.g.
            # a zero-length branch spanning incompatible states); leave it at
            # 0 rather than dividing, so log(0) = -inf propagates correctly
            # instead of being masked by a spurious log_scale contribution.
            safe_scale = np.where(scale > 0, scale, 1.0)
            partial = partial / safe_scale[:, np.newaxis]
            log_scale[:] += np.log(safe_scale)

        return partial

    root_partial = _post_order(tau)
    site_likelihood = root_partial @ pi  # eq. (root)
    return float(np.sum(np.log(site_likelihood) + log_scale))
