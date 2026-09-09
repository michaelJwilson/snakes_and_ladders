"""Felsenstein pruning taped by `burn`, behind one ``torch.autograd.Function``.

Route A of issue #449: the recursion is rebuilt in Rust out of
``burn_tensor`` operations and differentiated by ``burn-autodiff`` over
``Autodiff<NdArray<f64>>`` (``src/pruning_burn.rs``), so the gradient in the
branch lengths is produced on the Rust side of the boundary and only
``2n - 3`` doubles come back. The value and the gradient are pinned against
``pruning_torch``, which stays the oracle.

**`f64` is the axis the route was adopted on, and it holds.** `burn`'s
`NdArray` backend is generic over its float element, so the tape is `f64`
throughout: the Rust unit tests agree with central differences to 4.4e-10
relative, which is the difference quotient's truncation and not a narrowing
to `f32`.

**No general rate matrix.** `burn` exposes no matrix exponential, so this
route implements the closed-form Jukes-Cantor transition only. A
``rate_matrix`` is refused rather than ignored, and a caller that fits a
general ``Q`` uses ``pruning_torch``.

The flattening mirrors ``pruning_rust``: the topology crosses as a post-order
child list with one branch length per node, and the alignment as one
C-contiguous ``(n_leaves, n_sites)`` block, borrowed rather than copied. The
gradient comes back in that post-order and is permuted here into
``branch_order`` before it reaches autograd, so the tensor a caller
differentiates is ordered as ``pruning_torch``'s is.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from snakes_and_ladders import oxi_snakes_and_ladders
from snakes_and_ladders.likelihood.patterns import check_weights
from snakes_and_ladders.likelihood.pruning_rust import _postorder
from snakes_and_ladders.likelihood.pruning_torch import branch_order
from snakes_and_ladders.sim.tree import Node, preorder


class _Flattened:
    """A topology and an alignment in the arrays the Rust kernel takes.

    Built once per call site rather than per evaluation where a caller keeps
    one; the arrays depend on the topology and the data, never on the branch
    lengths, which cross separately so a fit rebuilds nothing.

    Attributes
    ----------
    children : list[list[int]]
        Per node in post-order, its children's positions.
    leaf_states : np.ndarray
        ``(n_leaves, n_sites)`` int64, one row per leaf.
    leaf_row : list[int]
        Per node, its row of ``leaf_states``, or ``-1`` if internal.
    to_postorder : np.ndarray
        ``branch_order`` position to post-order node position.
    n_sites : int
        Columns of the alignment.
    """

    def __init__(self, tau: Node, alignment: Mapping[str, np.ndarray]) -> None:
        nodes = _postorder(tau)
        position = {id(node): index for index, node in enumerate(nodes)}
        leaves = [node for node in nodes if node.is_leaf]
        missing = [leaf.name for leaf in leaves if leaf.name not in alignment]
        if missing:
            msg = f"alignment is missing leaf(ves) {missing}"
            raise ValueError(msg)

        self.n_sites = int(np.asarray(alignment[leaves[0].name]).shape[0])
        self.children = [
            [position[id(child)] for child in node.children] for node in nodes
        ]
        self.leaf_states = np.empty((len(leaves), self.n_sites), dtype=np.int64)
        self.leaf_row = [-1] * len(nodes)
        row = 0
        for node in nodes:
            if not node.is_leaf:
                continue
            states = np.asarray(alignment[node.name])
            if states.shape != (self.n_sites,):
                msg = (
                    f"leaf {node.name!r} has shape {states.shape}, "
                    f"expected ({self.n_sites},) -- the alignment is ragged"
                )
                raise ValueError(msg)
            self.leaf_states[row] = states
            self.leaf_row[position[id(node)]] = row
            row += 1

        by_name = {node.name: position[id(node)] for node in nodes}
        self.to_postorder = np.array(
            [by_name[name] for name in branch_order(tau)], dtype=np.int64
        )
        self.n_nodes = len(nodes)


class _PruningLogLikelihood(torch.autograd.Function):
    """One graph node whose backward reads the gradient the Rust tape produced.

    The kernel returns the value and the gradient together, so ``forward``
    keeps the gradient and ``backward`` scales it: the boundary is crossed
    once per evaluation rather than once per pass.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        branch_lengths: torch.Tensor,
        flattened: _Flattened,
        k: int,
        pi: torch.Tensor,
        weight: torch.Tensor | None,
        rescale: bool,
    ) -> torch.Tensor:
        lengths = np.zeros(flattened.n_nodes, dtype=np.float64)
        detached = branch_lengths.detach().to(dtype=torch.float64).cpu().numpy()
        lengths[flattened.to_postorder] = detached
        value, gradient = oxi_snakes_and_ladders.pruning_gradient(
            lengths,
            flattened.children,
            flattened.leaf_states,
            flattened.leaf_row,
            k,
            np.ascontiguousarray(pi.detach().cpu().numpy(), dtype=np.float64),
            None if weight is None else np.ascontiguousarray(
                weight.detach().cpu().numpy(), dtype=np.float64
            ),
            rescale,
        )
        ctx.gradient = torch.as_tensor(
            np.asarray(gradient, dtype=np.float64)[flattened.to_postorder],
            dtype=branch_lengths.dtype,
            device=branch_lengths.device,
        )
        return torch.as_tensor(
            value, dtype=branch_lengths.dtype, device=branch_lengths.device
        )

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor | None, ...]:
        gradient: torch.Tensor = ctx.gradient
        return gradient * grad_output, None, None, None, None, None


def log_likelihood(
    tau: Node,
    k: int,
    pi: np.ndarray | torch.Tensor,
    alignment: Mapping[str, np.ndarray],
    branch_lengths: torch.Tensor,
    *,
    weights: np.ndarray | None = None,
    rate_matrix: torch.Tensor | None = None,
    rescale: bool = True,
    flattened: _Flattened | None = None,
) -> torch.Tensor:
    """Total log-likelihood, differentiable w.r.t. ``branch_lengths``.

    Signature and value match
    :func:`snakes_and_ladders.likelihood.pruning_torch.log_likelihood`, which
    stays the oracle; only how the gradient is obtained differs.

    Parameters
    ----------
    tau, k, pi, alignment, branch_lengths, weights, rescale
        As :func:`snakes_and_ladders.likelihood.pruning_torch.log_likelihood`.
        ``pi`` is a constant here: this backward computes no gradient for it.
    rate_matrix : torch.Tensor | None
        Must be ``None``: `burn` has no matrix exponential and this route
        implements the closed-form Jukes-Cantor transition only.
    flattened : _Flattened | None
        The topology and alignment already in the kernel's arrays, for a
        caller evaluating one fixture repeatedly. Built here when omitted.

    Returns
    -------
    torch.Tensor
        0-dimensional; ``sum_s log Pr(data_s | tau, branch_lengths, pi)``.

    Raises
    ------
    ValueError
        If ``pi`` does not have shape ``(k,)``, ``branch_lengths`` does not
        have shape ``(len(branch_order(tau)),)``, ``alignment`` is missing a
        leaf of ``tau``, ``weights`` does not have one entry per column,
        ``pi`` requires a gradient this backward does not produce, or
        ``rate_matrix`` is given.
    """
    if rate_matrix is not None:
        msg = (
            "pruning_burn implements the closed-form Jukes-Cantor transition "
            "only; a general rate matrix needs pruning_torch"
        )
        raise ValueError(msg)
    dtype, device = branch_lengths.dtype, branch_lengths.device
    pi_t = torch.as_tensor(pi, dtype=dtype, device=device)
    if pi_t.shape != (k,):
        msg = f"pi has shape {tuple(pi_t.shape)}, expected ({k},)"
        raise ValueError(msg)
    if pi_t.requires_grad:
        msg = (
            "pruning_burn computes a gradient in branch_lengths only; "
            "pi must be a constant -- use pruning_torch"
        )
        raise ValueError(msg)

    order = branch_order(tau)
    if branch_lengths.shape != (len(order),):
        msg = (
            f"branch_lengths has shape {tuple(branch_lengths.shape)}, "
            f"expected ({len(order)},) to match branch_order(tau)"
        )
        raise ValueError(msg)

    if flattened is None:
        flattened = _Flattened(tau, alignment)
    leaves = [node for node in preorder(tau) if node.is_leaf]
    missing = [leaf.name for leaf in leaves if leaf.name not in alignment]
    if missing:
        msg = f"alignment is missing leaf(ves) {missing}"
        raise ValueError(msg)

    weight = check_weights(weights, flattened.n_sites)
    weight_t = (
        None if weight is None else torch.as_tensor(weight, dtype=dtype, device=device)
    )
    result: torch.Tensor = _PruningLogLikelihood.apply(
        branch_lengths, flattened, k, pi_t, weight_t, rescale
    )
    return result
