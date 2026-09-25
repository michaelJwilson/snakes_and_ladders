"""The pruning routes' shared plumbing: one post-order, one leaf indicator, one rescale, one set of checks (issue #858).

``sal.likelihood.pruning`` imports nothing from here and never
will. It is the oracle every other route is pinned against
(``likelihood/CLAUDE.md``, "The reference implementation is the oracle and it
stays"), so its arithmetic, its traversal and its checks stay written out in
that file: a referee sharing a body with what it referees stops refereeing,
and the rungs of ``infra/ladder.py`` are pinned pairwise against each other
rather than against a common ancestor. What is folded here is plumbing below
the rungs --- the walk that orders the nodes, the indicator that reads the
data, the rescaling step and the four validations --- and each rung keeps its
own entry point above it.

**Nothing here unifies arithmetic that differed.** The two rescaling copies
replaced a vanished scale with different objects: ``pruning_torch`` builds one
scalar per call, ``pruning_analytic`` and ``log_likelihood_cached`` a tensor
of ones per node. Which object is passed stays the caller's, as ``fallback``,
because a fold that picks one for both is a fold that has to prove the bits
agree rather than leave them alone. The leaf indicator is the same case: two
of the three call sites build the index on the tensors' device and the third
on the default one, so ``index_device`` is a parameter and each call keeps the
device behaviour it had.

The fifth copy of the checks, in ``brute_force``, stays where it is for the
first reason above: it is the referee that pins the NumPy oracle.

**The module imports no torch** (issue #1011). The checks, the post-order and
the NumPy indicator serve routes that take no derivative --- ``pruning_rust``,
``likelihood.blocks``, ``likelihood.surrogate`` --- and loading torch for them
cost 1.3 s and 590 MB. :func:`leaf_indicator` imports it where it builds a
tensor, and :func:`rescale_partial` reaches it through its argument's methods.
"""

from __future__ import annotations

from collections.abc import Container, Iterable
from typing import TYPE_CHECKING

import numpy as np

from sal.sim.tree import Node

if TYPE_CHECKING:
    import torch


def postorder(root: Node) -> list[Node]:
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


def check_pi_shape(shape: tuple[int, ...], k: int) -> None:
    """Refuse a root distribution that is not ``(k,)``.

    Parameters
    ----------
    shape : tuple[int, ...]
        ``pi``'s shape, as a tuple whatever library holds it.
    k : int
        Number of states.

    Raises
    ------
    ValueError
        If ``shape`` is not ``(k,)``.
    """
    if shape != (k,):
        msg = f"pi has shape {shape}, expected ({k},)"
        raise ValueError(msg)


def check_alignment_covers(
    leaf_names: Iterable[str], alignment: Container[str]
) -> None:
    """Refuse an alignment missing a leaf of the topology.

    Parameters
    ----------
    leaf_names : Iterable[str]
        The topology's leaf names.
    alignment : Container[str]
        The alignment, tested by membership so a mapping or its keys serve.

    Raises
    ------
    ValueError
        If any leaf is absent, listing every one that is.
    """
    missing = [name for name in leaf_names if name not in alignment]
    if missing:
        msg = f"alignment is missing leaf(ves) {missing}"
        raise ValueError(msg)


def check_branch_lengths_shape(shape: tuple[int, ...], n_branches: int) -> None:
    """Refuse a branch-length tensor that does not match ``branch_order(tau)``.

    Parameters
    ----------
    shape : tuple[int, ...]
        The tensor's shape.
    n_branches : int
        ``len(branch_order(tau))``.

    Raises
    ------
    ValueError
        If ``shape`` is not ``(n_branches,)``.
    """
    if shape != (n_branches,):
        msg = (
            f"branch_lengths has shape {shape}, "
            f"expected ({n_branches},) to match branch_order(tau)"
        )
        raise ValueError(msg)


def require_branch_length(node: Node) -> float:
    """``node``'s branch length, refusing a non-root node that carries none.

    Parameters
    ----------
    node : Node
        A non-root node, whose branch length the recursion needs.

    Returns
    -------
    float
        The branch length.

    Raises
    ------
    ValueError
        If the node has no ``branch_length``.
    """
    if node.branch_length is None:
        msg = f"non-root node {node.name!r} has no branch_length"
        raise ValueError(msg)
    return float(node.branch_length)


def leaf_indicator_array(states: np.ndarray, n_sites: int, k: int) -> np.ndarray:
    """The one-hot partial of a leaf's observed states, as a NumPy array.

    Parameters
    ----------
    states : np.ndarray
        The leaf's observed states, shape ``(n_sites,)`` with entries in
        ``[0, k)``.
    n_sites : int
        Columns in the alignment.
    k : int
        Alphabet size.

    Returns
    -------
    np.ndarray
        ``(n_sites, k)``, one at the observed state of each site --- the data
        indicator of ``eq:pruning``.
    """
    observed = np.asarray(states, dtype=np.int64)
    table = np.zeros((n_sites, k))
    table[np.arange(n_sites), observed] = 1.0
    return table


def leaf_indicator(
    states: object,
    n_sites: int,
    k: int,
    dtype: torch.dtype,
    device: torch.device,
    *,
    index_device: torch.device | None,
) -> torch.Tensor:
    """The one-hot partial of a leaf's observed states, as a tensor.

    Parameters
    ----------
    states : object
        The leaf's observed states, as the alignment holds them.
    n_sites : int
        Columns in the alignment.
    k : int
        Alphabet size.
    dtype : torch.dtype
        Tensor type, taken from the branch lengths.
    device : torch.device
        Where the recursion runs.
    index_device : torch.device | None
        Where the scatter's row index is built. ``None`` builds it on the
        default device, which is what
        :func:`sal.likelihood.pruning_torch.log_likelihood_cached`
        does and keeps doing; the other callers pass ``device``. A parameter
        rather than a choice made here, so no call site's device behaviour
        moves.

    Returns
    -------
    torch.Tensor
        ``(n_sites, k)``, one at the observed state of each site. Constant in
        the branch lengths, so it carries no gradient.
    """
    import torch

    observed = torch.as_tensor(states, dtype=torch.long, device=device)
    partial = torch.zeros((n_sites, k), dtype=dtype, device=device)
    partial[torch.arange(n_sites, device=index_device), observed] = 1.0
    return partial


def rescale_partial(
    partial: torch.Tensor,
    log_scale: torch.Tensor,
    fallback: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Divide a partial by its per-site maximum, accumulating the logarithm.

    ``likelihood/CLAUDE.md``, "Rescaling must stay differentiable": the total
    is the logarithm of the rescaled likelihood plus the accumulated scale, a
    transformation of the same computation. ``log_scale`` is returned rather
    than added in place, so the sum composes under autograd.

    A site whose scale is zero has zero likelihood under the model --- an
    internal node's partial vanished entirely, e.g. a zero-length branch
    spanning incompatible states. It is left at 0 rather than divided, so
    ``log(0) = -inf`` propagates instead of being masked by a spurious
    ``log_scale`` contribution.

    Parameters
    ----------
    partial : torch.Tensor
        ``(n_sites, k)`` partial likelihoods of one node.
    log_scale : torch.Tensor
        ``(n_sites,)`` accumulated log scale, added to rather than mutated.
    fallback : torch.Tensor | None
        What replaces a vanished scale. ``None`` builds ``ones_like(scale)``,
        which is what :mod:`sal.likelihood.pruning_analytic`
        and ``log_likelihood_cached`` build per node;
        :func:`sal.likelihood.pruning_torch.log_likelihood`
        passes the one scalar it builds per call instead. The selected value
        is the same either way; which object is allocated stays the caller's.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        The rescaled partial, the new accumulated log scale, and the scale
        divided by --- which the analytic backward keeps per node.
    """
    # Tensor methods, which are the same kernels as `torch.where`, `ones_like`
    # and `torch.log`, so this module names no torch at import (issue #1011).
    scale = partial.amax(dim=1)
    safe_scale = scale.where(
        scale > 0, scale.new_ones(scale.shape) if fallback is None else fallback
    )
    return (
        partial / safe_scale.unsqueeze(1),
        log_scale + safe_scale.log(),
        safe_scale,
    )
