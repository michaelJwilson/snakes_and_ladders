"""Direct marginalization: the independent oracle pruning is tested against.

Exponential in the number of internal nodes, and deliberately so: per
``likelihood/CLAUDE.md``, "correctness comes from brute force, not from
another backend... two backends agreeing proves nothing if both are wrong."
Sums ``eq:site-independence`` of ``docs/tex/textbook.tex`` directly, over every
joint assignment of internal-node states, without ``eq:pruning``'s conditional-independence
factorization. Shares no traversal or accumulation code with
``sal.likelihood.pruning`` -- only the already-independently-validated
substitution model, ``jc_transition_probabilities``.

Cost is ``O(n_sites * k**(internal nodes) * n_edges)``, and the
``k ** internal`` factor is refused above
:data:`sal.enumeration.MAX_ENUMERABLE_CONFIGURATIONS` rather than
attempted. Until issue #230 that limit was a sentence in this docstring
asking callers to keep ``tau`` to ``n <= 6`` taxa, which nothing checked: an
oversized call ran until the kernel stopped it, and read as infrastructure
breaking rather than as a stated limit.
"""

from __future__ import annotations

import itertools
import math

import numpy as np

from sal.enumeration import refuse_oversized
from sal.sim.jc import jc_transition_probabilities
from sal.sim.tree import Node, edges, preorder


def brute_force_log_likelihood(
    tau: Node,
    n_states: int,
    pi: np.ndarray,
    alignment: dict[str, np.ndarray],
) -> float:
    """Total log-likelihood by direct marginalization over internal states.

    Parameters
    ----------
    tau : Node
        Root of the topology, with branch lengths attached to each non-root
        node. Keep to ``n <= 6`` leaves -- this enumerates every joint
        assignment of internal-node states.
    n_states : int
        Number of states.
    pi : np.ndarray
        Root state distribution, shape (k,).
    alignment : dict[str, np.ndarray]
        Leaf name to its observed states, each of shape (n_sites,) with
        entries in ``[0, k)``.

    Returns
    -------
    float
        ``sum_s log Pr(data_s | tau, t, Q, pi)``, computed by summing the
        joint probability of every internal-node state assignment directly,
        rather than via the pruning recursion.

    Raises
    ------
    ValueError
        If ``pi`` does not have shape ``(k,)``, ``alignment`` is missing a
        leaf of ``tau``, or the tree is large enough that ``k ** internal``
        assignments are past
        :data:`sal.enumeration.MAX_ENUMERABLE_CONFIGURATIONS`.
    """
    if pi.shape != (n_states,):
        msg = f"pi has shape {pi.shape}, expected ({n_states},)"
        raise ValueError(msg)

    leaves = [node for node in preorder(tau) if node.is_leaf]
    internal = [node for node in preorder(tau) if not node.is_leaf]
    refuse_oversized(
        n_states ** len(internal),
        what=f"{n_states}**{len(internal)} ancestral-state assignments",
    )
    missing = [leaf.name for leaf in leaves if leaf.name not in alignment]
    if missing:
        msg = f"alignment is missing leaf(ves) {missing}"
        raise ValueError(msg)

    tree_edges = list(edges(tau))
    transitions: dict[str, np.ndarray] = {}
    for _, child in tree_edges:
        if child.branch_length is None:
            msg = f"non-root node {child.name!r} has no branch_length"
            raise ValueError(msg)
        transitions[child.name] = jc_transition_probabilities(
            child.branch_length, n_states=n_states
        )

    n_sites = alignment[leaves[0].name].shape[0]
    total_log_likelihood = 0.0
    for site in range(n_sites):
        site_likelihood = 0.0
        for assignment in itertools.product(range(n_states), repeat=len(internal)):
            state: dict[str, int] = {
                node.name: value
                for node, value in zip(internal, assignment, strict=True)
            }
            for leaf in leaves:
                state[leaf.name] = int(alignment[leaf.name][site])

            prob = float(pi[state[tau.name]])
            for parent, child in tree_edges:
                prob *= transitions[child.name][state[parent.name], state[child.name]]
            site_likelihood += prob

        total_log_likelihood += math.log(site_likelihood)

    return total_log_likelihood
