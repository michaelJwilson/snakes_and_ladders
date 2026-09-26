"""Vectorized NumPy Felsenstein pruning -- the oracle every backend is pinned against.

Implements the pruning recursion, ``eq:pruning``, and the root marginalization,
``eq:root``, of ``docs/tex/textbook.tex``, which ``app:pruning`` derives as the
marginalization over internal states: the site log-likelihood
is computed post-order over ``(site, state)`` NumPy arrays, reusing
``sal.sim.jc.jc_transition_probabilities`` for P(t). Partial likelihoods
underflow for realistic (site, taxa) counts, so they are rescaled per node
with the log of the scale factor accumulated separately -- a transformation
of the same computation, not a different algorithm (``likelihood/CLAUDE.md``,
"Rescaling must stay differentiable") -- so the rescaled and unrescaled paths
must agree wherever both run.

Duplicate alignment columns contribute the same term, so the caller may pass
the distinct columns with their counts as ``weights``
(``sal.likelihood.patterns``) and read the same number off
fewer columns. The identity is exact; ``weights=None`` is every column once.

This module is written to be obviously correct, not fast: it is the reference
every accelerated backend (Rust, PyTorch, CUDA, Metal) is validated against,
per ``likelihood/CLAUDE.md``.

``log_likelihood`` takes a ``backend``, which is a **door and not a rung**
(issue #860): ``Backend.RUST`` calls ``likelihood.pruning.rust``'s own entry point with
the arguments it was given and returns what it returns, and the recursion
below is reached on ``Backend.PYTHON``, the default. No arithmetic moved and
no rung merged --- ``likelihood.pruning.rust`` keeps its entry point and its bitwise
pin, and this oracle gained no code from the route it referees.
"""

from __future__ import annotations

from typing import cast

import numpy as np

from sal.backend import Backend, twin
from sal.likelihood.patterns import check_weights
from sal.sim.jc import jc_transition_probabilities
from sal.sim.tree import Node, preorder


def log_likelihood(
    tau: Node,
    n_states: int,
    pi: np.ndarray,
    alignment: dict[str, np.ndarray],
    *,
    weights: np.ndarray | None = None,
    rescale: bool = True,
    backend: Backend = Backend.PYTHON,
) -> float:
    """Total log-likelihood of an alignment under the k-state Jukes-Cantor model.

    Parameters
    ----------
    tau : Node
        Root of the topology, with branch lengths attached to each non-root
        node.
    n_states : int
        Number of states.
    pi : np.ndarray
        Root state distribution, shape (k,).
    alignment : dict[str, np.ndarray]
        Leaf name to its observed states, each of shape (n_sites,) with
        entries in ``[0, k)`` -- the shape
        ``sal.sim.simulate.SimulatedDataset.alignment`` produces.
    weights : np.ndarray | None
        One weight per column, or ``None`` for one occurrence each. Passing
        the table of ``sal.likelihood.patterns.compress``
        alongside its weights evaluates the same log-likelihood over the
        distinct columns alone, exactly (``eq:site-independence``).
    rescale : bool
        Whether to rescale partial likelihoods per node, accumulating the log
        of the scale factor separately (``docs/tex/textbook.tex``, ``eq:pruning``).
        Disabling this underflows for realistic (site, taxa) sizes; it exists
        so tests can check the two paths agree on small problems where both
        run.
    backend : Backend
        Which implementation runs it. ``PYTHON`` is this module's own
        recursion, the oracle, and is the default: a caller who does not ask
        gets the reference (issue #860). ``RUST`` is
        ``sal.likelihood.pruning.rust``'s call, the one the
        caller made by importing that module, reached through the enum every
        other twin is reached through. ``TORCH`` is
        ``sal.likelihood.pruning.torch`` at the branch lengths ``tau``
        carries, its tape detached: the value alone, for a caller comparing
        backends (issue #1059); a caller that differentiates takes the tensor
        from that module.

    Returns
    -------
    float
        ``sum_s log Pr(data_s | tau, t, Q, pi)``, summed over sites
        (``eq:site-independence``).

    Raises
    ------
    ValueError
        If ``pi`` does not have shape ``(k,)``, ``alignment`` is missing a
        leaf of ``tau``, or ``weights`` does not have one entry per column.
        Or if ``backend`` is not ``PYTHON``, ``TORCH`` or ``RUST``.
    """
    # A door, before any arithmetic: the recursion below is the oracle and
    # gains nothing from the route it referees.
    if backend is Backend.TORCH:
        # Imported here so the oracle's import loads no torch.
        from sal.likelihood.pruning import torch as taped

        return float(
            taped.log_likelihood(
                tau,
                n_states,
                pi,
                alignment,
                taped.branch_lengths_from_tree(tau),
                weights=weights,
                rescale=rescale,
            ).detach()
        )
    if (rust := twin("pruning log_likelihood", backend, __name__)) is not None:
        return cast(
            "float",
            rust.log_likelihood(
                tau, n_states, pi, alignment, weights=weights, rescale=rescale
            ),
        )

    if pi.shape != (n_states,):
        msg = f"pi has shape {pi.shape}, expected ({n_states},)"
        raise ValueError(msg)

    leaves = [node for node in preorder(tau) if node.is_leaf]
    missing = [leaf.name for leaf in leaves if leaf.name not in alignment]
    if missing:
        msg = f"alignment is missing leaf(ves) {missing}"
        raise ValueError(msg)

    n_sites = alignment[leaves[0].name].shape[0]
    weight = check_weights(weights, n_sites)
    log_scale = np.zeros(n_sites)

    def _post_order(node: Node) -> np.ndarray:
        if node.is_leaf:
            states = alignment[node.name]
            partial = np.zeros((n_sites, n_states))
            partial[np.arange(n_sites), states] = 1.0
            return partial

        partial = np.ones((n_sites, n_states))
        for child in node.children:
            if child.branch_length is None:
                msg = f"non-root node {child.name!r} has no branch_length"
                raise ValueError(msg)
            child_partial = _post_order(child)
            transition = jc_transition_probabilities(
                child.branch_length, n_states=n_states
            )
            # message[s, i] = sum_j P_ij(t) * L_child(s, j) -- eq:pruning.
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
    site_likelihood = root_partial @ pi  # eq:root
    site_log_likelihood = np.log(site_likelihood) + log_scale
    if weight is None:
        return float(np.sum(site_log_likelihood))
    return float(np.dot(weight, site_log_likelihood))
