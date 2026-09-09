"""Analytic gradient of Felsenstein pruning, behind one ``torch.autograd.Function``.

The forward value is ``pruning_torch.log_likelihood``'s, computed with the
same operations in the same order; the gradient in ``branch_lengths`` comes
from the closed form rather than from a tape. ``alg:pruning-backward`` of
``docs/tex/textbook.tex`` states the recursion; this module implements it.

**Why this exists.** Issue #443 measured the taped backward at 40.5% of a
fit and attributed the cost to the number of autograd graph nodes, which the
post-order builds one of per tree node per operation. A
``torch.autograd.Function`` contributes exactly one node whatever the tree's
size, so the count stops tracking the tree. Issue #449 measures the three
routes against each other; ``STATUS.md`` carries the numbers.

**Rescaling is differentiated as a constant, and that is exact rather than an
approximation.** The recursion divides each partial by a per-site scale and
accumulates its logarithm, so the total is *algebraically independent* of the
scale: for any positive scaling held fixed, the two terms move against each
other and cancel. The derivative may therefore be taken with the scales frozen
at the values the forward pass produced, and the ``log_scale`` term
contributes nothing to it. The taped path differentiates through the
``amax`` instead and reaches the same derivative by a longer route; the two
agree to the tolerance ``tests/regression/likelihood/test_pruning_analytic.py``
states.

**Sibling products are formed without dividing.** The derivative with respect
to a child's branch needs the product of its siblings' messages, and taking it
as the parent's partial divided by the child's own message is wrong wherever a
message is zero -- which happens at any site an observed state forbids. A
prefix/suffix scan over the children gives the same product with no division.

The gradient in the root distribution, in a general rate matrix, or in the
alignment is not computed: those are constants of the fit this attacks
(``likelihood.objective.BranchLengthObjective``), and a caller that needs them
uses ``pruning_torch`` instead. Passing a ``pi`` or ``rate_matrix`` that
requires a gradient is refused rather than silently returning zero for it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

from snakes_and_ladders.likelihood.patterns import check_weights
from snakes_and_ladders.likelihood.pruning_torch import (
    branch_order,
    transition_probabilities,
)
from snakes_and_ladders.sim.tree import Node, preorder


def _transition_derivatives(
    t: torch.Tensor,
    k: int,
    rate_matrix: torch.Tensor | None,
    transitions: torch.Tensor,
) -> torch.Tensor:
    """``dP(t)/dt`` for every branch length in ``t``, shape ``(*t.shape, k, k)``.

    Parameters
    ----------
    t : torch.Tensor
        Branch lengths, shape ``(n_branches,)``.
    k : int
        Number of states.
    rate_matrix : torch.Tensor | None
        ``None`` for the closed-form Jukes-Cantor derivative of ``eq:jc``;
        otherwise ``Q``, whose transition matrix satisfies ``dP/dt = Q P(t)``.
    transitions : torch.Tensor
        ``P(t)`` at the same branch lengths, shape ``(n_branches, k, k)``.
        Reused rather than recomputed for the general path.

    Returns
    -------
    torch.Tensor
        One ``(k, k)`` derivative matrix per entry of ``t``.
    """
    if rate_matrix is None:
        # d/dt [1/k + (delta_ij - 1/k) exp(-k t / (k - 1))].
        decay = torch.exp(-k * t / (k - 1))[..., None, None]
        eye = torch.eye(k, dtype=t.dtype, device=t.device)
        return (eye - 1.0 / k) * (-k / (k - 1)) * decay
    return rate_matrix @ transitions


def _postorder(root: Node) -> list[Node]:
    """Every node of ``root``'s tree, children before parents, root last."""
    order: list[Node] = []

    def _walk(node: Node) -> None:
        for child in node.children:
            _walk(child)
        order.append(node)

    _walk(root)
    return order


def _sibling_products(messages: Sequence[torch.Tensor]) -> list[torch.Tensor]:
    """Per child, the product of every *other* child's message.

    A prefix/suffix scan, so no message is divided by -- a message is zero at
    any site the child's observation forbids, and dividing there produces a
    NaN where the derivative is finite.

    Parameters
    ----------
    messages : Sequence[torch.Tensor]
        One ``(n_sites, k)`` message per child, in ``node.children`` order.

    Returns
    -------
    list[torch.Tensor]
        ``result[i]`` is the elementwise product of ``messages`` without
        ``messages[i]``; an all-ones tensor where the node has one child.
    """
    count = len(messages)
    if count == 1:
        return [torch.ones_like(messages[0])]
    if count == 2:
        # Every internal node of a binary tree, so it is worth not allocating
        # the scan's two identity tensors to reach the same answer.
        return [messages[1], messages[0]]
    prefix: list[torch.Tensor | None] = [None] * count
    for position in range(1, count):
        running = messages[position - 1]
        prefix[position] = running if position == 1 else prefix[position - 1] * running
    result: list[torch.Tensor] = [messages[0]] * count
    suffix: torch.Tensor | None = None
    for position in range(count - 1, -1, -1):
        head = prefix[position]
        if head is None:
            assert suffix is not None
            result[position] = suffix
        elif suffix is None:
            result[position] = head
        else:
            result[position] = head * suffix
        suffix = messages[position] if suffix is None else suffix * messages[position]
    return result


class _PruningLogLikelihood(torch.autograd.Function):
    """``log_likelihood`` as one graph node, with the two-pass backward.

    ``forward`` runs the post-order recursion under ``no_grad`` and keeps the
    scaled partials and the messages it produced; ``backward`` walks the same
    tree root-to-tip, carrying the adjoint of each node's partial and reading
    off one branch derivative per edge (``alg:pruning-backward`` of
    ``docs/tex/textbook.tex``).
    """

    @staticmethod
    def forward(
        ctx: Any,
        branch_lengths: torch.Tensor,
        tau: Node,
        k: int,
        pi: torch.Tensor,
        alignment: Mapping[str, torch.Tensor],
        weight: torch.Tensor | None,
        rate_matrix: torch.Tensor | None,
        rescale: bool,
    ) -> torch.Tensor:
        dtype, device = branch_lengths.dtype, branch_lengths.device
        order = branch_order(tau)
        index = {name: position for position, name in enumerate(order)}
        nodes = _postorder(tau)
        leaves = [node for node in nodes if node.is_leaf]
        n_sites = int(alignment[leaves[0].name].shape[0])

        with torch.no_grad():
            transitions = transition_probabilities(branch_lengths, k, rate_matrix)
            partials: dict[str, torch.Tensor] = {}
            messages: dict[str, torch.Tensor] = {}
            scales: dict[str, torch.Tensor] = {}
            log_scale = torch.zeros(n_sites, dtype=dtype, device=device)

            for node in nodes:
                if node.is_leaf:
                    states = torch.as_tensor(
                        alignment[node.name], dtype=torch.long, device=device
                    )
                    partial = torch.zeros((n_sites, k), dtype=dtype, device=device)
                    partial[torch.arange(n_sites, device=device), states] = 1.0
                    partials[node.name] = partial
                    continue

                partial = torch.ones((n_sites, k), dtype=dtype, device=device)
                for child in node.children:
                    transition = transitions[index[child.name]]
                    # message[s, i] = sum_j P_ij(t) L_child(s, j) -- eq:pruning.
                    message = partials[child.name] @ transition.T
                    messages[child.name] = message
                    partial = partial * message

                if rescale:
                    scale = partial.amax(dim=1)
                    safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
                    partial = partial / safe_scale.unsqueeze(1)
                    log_scale = log_scale + torch.log(safe_scale)
                    scales[node.name] = safe_scale
                partials[node.name] = partial

            site_likelihood = partials[tau.name] @ pi  # eq:root
            site_log_likelihood = torch.log(site_likelihood) + log_scale
            total = (
                torch.sum(site_log_likelihood)
                if weight is None
                else torch.dot(weight, site_log_likelihood)
            )

        ctx.branch_lengths_value = branch_lengths.detach()
        ctx.tau = tau
        ctx.k = k
        ctx.pi = pi
        ctx.weight = weight
        ctx.rate_matrix = rate_matrix
        ctx.rescale = rescale
        ctx.index = index
        ctx.transitions = transitions
        ctx.partials = partials
        ctx.messages = messages
        ctx.scales = scales
        ctx.site_likelihood = site_likelihood
        ctx.n_branches = len(order)
        return total

    @staticmethod
    def backward(
        ctx: Any, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor | None, ...]:
        tau: Node = ctx.tau
        k: int = ctx.k
        pi: torch.Tensor = ctx.pi
        transitions: torch.Tensor = ctx.transitions
        index: dict[str, int] = ctx.index
        partials: dict[str, torch.Tensor] = ctx.partials
        messages: dict[str, torch.Tensor] = ctx.messages
        scales: dict[str, torch.Tensor] = ctx.scales
        derivatives = _transition_derivatives(
            ctx.branch_lengths_value, k, ctx.rate_matrix, transitions
        )
        gradient = torch.zeros(
            ctx.n_branches,
            dtype=transitions.dtype,
            device=transitions.device,
        )

        # d total / d L_root[s, i]. The rescaling contributes nothing: the
        # total is algebraically independent of the scales, so they are held
        # at the values the forward pass produced (module docstring).
        adjoint_root = pi.unsqueeze(0) / ctx.site_likelihood.unsqueeze(1)
        if ctx.weight is not None:
            adjoint_root = adjoint_root * ctx.weight.unsqueeze(1)
        adjoints: dict[str, torch.Tensor] = {tau.name: adjoint_root * grad_output}

        for node in preorder(tau):
            if node.is_leaf:
                continue
            adjoint = adjoints[node.name]
            if ctx.rescale:
                adjoint = adjoint / scales[node.name].unsqueeze(1)
            siblings = _sibling_products(
                [messages[child.name] for child in node.children]
            )
            for child, sibling in zip(node.children, siblings, strict=True):
                position = index[child.name]
                # d total / d message_child[s, i].
                outer = adjoint * sibling
                # d total / d t_child = sum_ij dP_ij/dt * sum_s outer[s, i]
                # L_child[s, j].
                gradient[position] = torch.sum(
                    derivatives[position] * (outer.T @ partials[child.name])
                )
                if not child.is_leaf:
                    adjoints[child.name] = outer @ transitions[position]

        return gradient, None, None, None, None, None, None, None


def log_likelihood(
    tau: Node,
    k: int,
    pi: np.ndarray | torch.Tensor,
    alignment: Mapping[str, np.ndarray | torch.Tensor],
    branch_lengths: torch.Tensor,
    *,
    weights: np.ndarray | None = None,
    rate_matrix: torch.Tensor | None = None,
    rescale: bool = True,
) -> torch.Tensor:
    """Total log-likelihood, differentiable w.r.t. ``branch_lengths``.

    Signature and value match
    :func:`snakes_and_ladders.likelihood.pruning_torch.log_likelihood`, which
    stays the oracle; only how the gradient is obtained differs.

    Parameters
    ----------
    tau, k, pi, alignment, branch_lengths, weights, rate_matrix, rescale
        As :func:`snakes_and_ladders.likelihood.pruning_torch.log_likelihood`.
        ``pi`` and ``rate_matrix`` are constants here: this backward computes
        no gradient for them.

    Returns
    -------
    torch.Tensor
        0-dimensional; ``sum_s log Pr(data_s | tau, branch_lengths, Q, pi)``.

    Raises
    ------
    ValueError
        If ``pi`` does not have shape ``(k,)``, ``branch_lengths`` does not
        have shape ``(len(branch_order(tau)),)``, ``alignment`` is missing a
        leaf of ``tau``, ``weights`` does not have one entry per column, or
        ``pi`` or ``rate_matrix`` requires a gradient this backward does not
        produce.
    """
    dtype, device = branch_lengths.dtype, branch_lengths.device
    pi_t = torch.as_tensor(pi, dtype=dtype, device=device)
    if pi_t.shape != (k,):
        msg = f"pi has shape {tuple(pi_t.shape)}, expected ({k},)"
        raise ValueError(msg)
    if pi_t.requires_grad or (rate_matrix is not None and rate_matrix.requires_grad):
        msg = (
            "pruning_analytic computes a gradient in branch_lengths only; "
            "pi and rate_matrix must be constants -- use pruning_torch"
        )
        raise ValueError(msg)

    order = branch_order(tau)
    if branch_lengths.shape != (len(order),):
        msg = (
            f"branch_lengths has shape {tuple(branch_lengths.shape)}, "
            f"expected ({len(order)},) to match branch_order(tau)"
        )
        raise ValueError(msg)

    leaves = [node for node in preorder(tau) if node.is_leaf]
    missing = [leaf.name for leaf in leaves if leaf.name not in alignment]
    if missing:
        msg = f"alignment is missing leaf(ves) {missing}"
        raise ValueError(msg)

    n_sites = int(torch.as_tensor(alignment[leaves[0].name]).shape[0])
    weight = check_weights(weights, n_sites)
    weight_t = (
        None if weight is None else torch.as_tensor(weight, dtype=dtype, device=device)
    )
    result: torch.Tensor = _PruningLogLikelihood.apply(  # type: ignore[no-untyped-call]
        branch_lengths, tau, k, pi_t, alignment, weight_t, rate_matrix, rescale
    )
    return result
