"""Differentiable PyTorch Felsenstein pruning -- pinned against
``snakes_and_ladders.likelihood.pruning``, the NumPy oracle (CLAUDE.md, "The NumPy
reference is the oracle and it stays").

Branch lengths are a tensor kept separate from the
topology: a ``snakes_and_ladders.sim.tree.Node`` here describes only shape (leaf names,
children), never a differentiable quantity, while ``branch_lengths`` --
ordered by ``branch_order(tau)`` -- is the tensor ``torch.autograd``
differentiates through. ``Node.branch_length`` is never read by
``log_likelihood``.

JC transition probabilities default to the closed form of eq. (jc) in
``docs/tex/main.tex``, built from ``torch.exp`` so the branch length stays
in the graph. Passing ``rate_matrix`` switches to the general
``torch.linalg.matrix_exp(Q * t)`` path -- the same path a fitted, non-JC
rate matrix would use -- and must agree with the closed form when ``Q`` is
the JC generator (tests/regression/test_pruning_torch.py).

Rescaling (``likelihood/CLAUDE.md``, "Rescaling must stay differentiable")
accumulates ``log_scale`` by tensor addition, never in place, so it composes
correctly under autograd.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch

from snakes_and_ladders.sim.tree import Node, edges, preorder


def branch_order(tau: Node) -> list[str]:
    """Child-node names in ``edges(tau)`` order.

    Parameters
    ----------
    tau : Node
        Root of the topology.

    Returns
    -------
    list[str]
        The order ``branch_lengths`` passed to ``log_likelihood`` must
        follow: one entry per non-root node, in ``edges(tau)`` order.
    """
    return [child.name for _, child in edges(tau)]


def branch_lengths_from_tree(
    tau: Node,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Read ``tau``'s own branch lengths into a tensor, ordered by ``branch_order``.

    Convenience for seeding a ``branch_lengths`` tensor from a fixture tree
    (e.g. before calling ``.requires_grad_(True)`` for ``gradcheck``);
    ``log_likelihood`` itself never reads ``tau``'s ``branch_length`` fields.

    Parameters
    ----------
    tau : Node
        Root of the topology, with branch lengths attached to each non-root
        node.

    Returns
    -------
    torch.Tensor
        Shape ``(len(branch_order(tau)),)``, in ``dtype`` on ``device``.

    Raises
    ------
    ValueError
        If a non-root node has no ``branch_length``.
    """
    lengths: list[float] = []
    for _, child in edges(tau):
        if child.branch_length is None:
            msg = f"non-root node {child.name!r} has no branch_length"
            raise ValueError(msg)
        lengths.append(child.branch_length)
    return torch.tensor(lengths, dtype=dtype, device=device)


def _jc_transition_probabilities(t: torch.Tensor, k: int) -> torch.Tensor:
    """Closed-form JC P(t) (eq. (jc) of ``docs/tex/main.tex``), differentiable in ``t``."""
    decay = torch.exp(-k * t / (k - 1))
    off_diagonal = (1.0 - decay) / k
    diagonal = 1.0 / k + (k - 1) / k * decay
    eye = torch.eye(k, dtype=t.dtype, device=t.device)
    return off_diagonal * (1.0 - eye) + diagonal * eye


def _transition_probabilities(
    t: torch.Tensor, k: int, rate_matrix: torch.Tensor | None
) -> torch.Tensor:
    if rate_matrix is None:
        return _jc_transition_probabilities(t, k)
    result: torch.Tensor = torch.linalg.matrix_exp(rate_matrix * t)
    return result


def log_likelihood(
    tau: Node,
    k: int,
    pi: np.ndarray | torch.Tensor,
    alignment: Mapping[str, np.ndarray | torch.Tensor],
    branch_lengths: torch.Tensor,
    *,
    rate_matrix: torch.Tensor | None = None,
    rescale: bool = True,
) -> torch.Tensor:
    """Total log-likelihood, differentiable w.r.t. ``branch_lengths``.

    Parameters
    ----------
    tau : Node
        Root of the topology. Its own ``branch_length`` fields are ignored;
        branch lengths come from ``branch_lengths`` instead.
    k : int
        Number of states.
    pi : np.ndarray | torch.Tensor
        Root state distribution, shape ``(k,)``.
    alignment : Mapping[str, np.ndarray | torch.Tensor]
        Leaf name to its observed states, each of shape ``(n_sites,)`` with
        entries in ``[0, k)``.
    branch_lengths : torch.Tensor
        Shape ``(len(branch_order(tau)),)``, in ``dtype`` on ``device``. The
        tensor autograd differentiates through; kept separate from ``tau``.
    rate_matrix : torch.Tensor | None
        If given, shape ``(k, k)``: a general rate matrix ``Q``, and
        transition probabilities use ``torch.linalg.matrix_exp(Q * t)``
        instead of the closed-form JC formula. ``None`` (default) uses the
        closed form.
    rescale : bool
        Whether to rescale partial likelihoods per node, matching
        ``snakes_and_ladders.likelihood.pruning``'s ``rescale`` flag.

    Returns
    -------
    torch.Tensor
        0-dimensional; ``sum_s log Pr(data_s | tau, branch_lengths, Q, pi)``.

    Raises
    ------
    ValueError
        If ``pi`` does not have shape ``(k,)``, ``branch_lengths`` does not
        have shape ``(len(branch_order(tau)),)``, or ``alignment`` is
        missing a leaf of ``tau``.
    """
    # dtype and device follow branch_lengths, so a caller moves the whole
    # recursion by moving one tensor -- and float64 stays the default
    # because branch_lengths_from_tree defaults to it. Metal rejects
    # float64 outright, so an accelerator path must choose float32 here
    # rather than have it chosen silently.
    dtype = branch_lengths.dtype
    device = branch_lengths.device
    pi_t = torch.as_tensor(pi, dtype=dtype, device=device)
    if pi_t.shape != (k,):
        msg = f"pi has shape {tuple(pi_t.shape)}, expected ({k},)"
        raise ValueError(msg)

    order = branch_order(tau)
    if branch_lengths.shape != (len(order),):
        msg = (
            f"branch_lengths has shape {tuple(branch_lengths.shape)}, "
            f"expected ({len(order)},) to match branch_order(tau)"
        )
        raise ValueError(msg)
    index = {name: i for i, name in enumerate(order)}

    leaves = [node for node in preorder(tau) if node.is_leaf]
    missing = [leaf.name for leaf in leaves if leaf.name not in alignment]
    if missing:
        msg = f"alignment is missing leaf(ves) {missing}"
        raise ValueError(msg)

    n_sites = int(torch.as_tensor(alignment[leaves[0].name]).shape[0])
    log_scale = torch.zeros(n_sites, dtype=dtype, device=device)

    def _post_order(node: Node) -> torch.Tensor:
        nonlocal log_scale
        if node.is_leaf:
            states = torch.as_tensor(
                alignment[node.name], dtype=torch.long, device=device
            )
            partial = torch.zeros((n_sites, k), dtype=dtype, device=device)
            partial[torch.arange(n_sites), states] = 1.0
            return partial

        partial = torch.ones((n_sites, k), dtype=dtype, device=device)
        for child in node.children:
            t = branch_lengths[index[child.name]]
            child_partial = _post_order(child)
            transition = _transition_probabilities(t, k, rate_matrix)
            # message[s, i] = sum_j P_ij(t) * L_child(s, j) -- eq. (pruning).
            partial = partial * (child_partial @ transition.T)

        if rescale:
            scale = partial.amax(dim=1)
            # See snakes_and_ladders.likelihood.pruning: a zero scale means the site is
            # genuinely impossible under the model, left at 0 rather than
            # divided so log(0) = -inf propagates instead of being masked.
            safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
            partial = partial / safe_scale.unsqueeze(1)
            log_scale = log_scale + torch.log(safe_scale)

        return partial

    root_partial = _post_order(tau)
    site_likelihood = root_partial @ pi_t  # eq. (root)
    return torch.sum(torch.log(site_likelihood) + log_scale)
