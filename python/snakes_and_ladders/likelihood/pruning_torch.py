"""Differentiable PyTorch Felsenstein pruning -- pinned against
``snakes_and_ladders.likelihood.pruning``, the NumPy oracle (CLAUDE.md, "The NumPy
reference is the oracle and it stays").

Branch lengths are a tensor kept separate from the
topology: a ``snakes_and_ladders.sim.tree.Node`` here describes only shape (leaf names,
children), never a differentiable quantity, while ``branch_lengths`` --
ordered by ``branch_order(tau)`` -- is the tensor ``torch.autograd``
differentiates through. ``Node.branch_length`` is never read by
``log_likelihood``.

JC transition probabilities default to the closed form ``eq:jc`` of
``docs/tex/textbook.tex``, built from ``torch.exp`` so the branch length stays
in the graph. Passing ``rate_matrix`` switches to the general
``torch.linalg.matrix_exp(Q * t)`` path -- the same path a fitted, non-JC
rate matrix would use -- and must agree with the closed form when ``Q`` is
the JC generator (tests/regression/test_pruning_torch.py).

Rescaling (``likelihood/CLAUDE.md``, "Rescaling must stay differentiable;
its factor must not be")
accumulates ``log_scale`` by tensor addition, never in place, so it composes
correctly under autograd. The scaling *factor* is detached: it cancels between
the division and the ``log`` it is added to, so it carries no derivative, and
the gradient of the rescaled path is the gradient of the unrescaled one
(tests/regression/likelihood/test_pruning_torch.py).
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
    """Closed-form JC P(t), ``eq:jc`` of ``docs/tex/textbook.tex``, differentiable in ``t``.

    ``t`` may be a scalar or a vector of branch lengths; the result carries
    one ``(k, k)`` matrix per entry of ``t`` in its leading dimensions. The
    arithmetic is elementwise, so a matrix taken from the batched result is
    the matrix the scalar call returns, bitwise -- a test pins it.
    """
    decay = torch.exp(-k * t / (k - 1))[..., None, None]
    off_diagonal = (1.0 - decay) / k
    diagonal = 1.0 / k + (k - 1) / k * decay
    eye = torch.eye(k, dtype=t.dtype, device=t.device)
    return off_diagonal * (1.0 - eye) + diagonal * eye


def _transition_probabilities(
    t: torch.Tensor, k: int, rate_matrix: torch.Tensor | None
) -> torch.Tensor:
    """``P(t)`` for every branch length in ``t``, shape ``(*t.shape, k, k)``.

    One call per likelihood evaluation rather than one per branch: the
    hill-climb profile behind #264 charged 8,730 scalar calls per 970
    evaluations to this function, a Python-level call per child per node that
    root ``CLAUDE.md``'s inlining rule names. Batched, the closed form is one
    ``exp`` over the branch vector and the matrix exponential one batched
    ``matrix_exp``.
    """
    if rate_matrix is None:
        return _jc_transition_probabilities(t, k)
    result: torch.Tensor = torch.linalg.matrix_exp(rate_matrix * t[..., None, None])
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
    # Every branch's transition matrix at once, indexed by branch_order.
    transitions = _transition_probabilities(branch_lengths, k, rate_matrix)

    # Contiguous, because a leaf's message is a gather along its rows and a
    # gather from a transposed view walks them at stride: measured on the
    # eight-taxon fixture, 79 us per leaf against 17 us contiguous.
    transposed = transitions.transpose(-2, -1).contiguous()

    def _message(node: Node) -> torch.Tensor:
        """What ``node`` contributes to its parent: ``P(t)`` applied to its partial.

        ``message[s, i] = sum_j P_ij(t) * L_node(s, j)`` -- ``eq:pruning``.
        """
        transition_t = transposed[index[node.name]]
        if node.is_leaf:
            # A leaf's partial is one-hot, so the sum over ``j`` keeps one
            # term and drops products that are exactly zero: the message is
            # row ``states[s]`` of ``P(t).T``, gathered rather than formed
            # and multiplied. Bitwise the matmul's result, and the one-hot
            # matrix is never materialized.
            states = torch.as_tensor(
                alignment[node.name], dtype=torch.long, device=device
            )
            return transition_t.index_select(0, states)
        return _partial(node) @ transition_t

    def _partial(node: Node) -> torch.Tensor:
        nonlocal log_scale
        # Seeded from the first child rather than from ones: multiplying by a
        # ones tensor is exact, so dropping it is bitwise, and it is one
        # allocation and one product per internal node per evaluation.
        children = node.children
        partial = _message(children[0])
        for child in children[1:]:
            partial = partial * _message(child)

        if rescale:
            # Detached because the scale cancels: this node's partial is
            # divided by ``s`` and ``log s`` is added to the running total, so
            # the total is what it was and ``s`` contributes nothing to the
            # derivative. Differentiating it would compute a zero through the
            # amax, the comparison and the log -- half of the backward pass on
            # the eight-taxon fixture (issue #397). The rescaling itself stays
            # in the graph; what leaves it is a constant.
            scale = partial.amax(dim=1).detach()
            # See snakes_and_ladders.likelihood.pruning: a zero scale means the site is
            # genuinely impossible under the model, left at 0 rather than
            # divided so log(0) = -inf propagates instead of being masked.
            safe_scale = torch.where(scale > 0, scale, 1.0)
            partial = partial / safe_scale.unsqueeze(1)
            log_scale = log_scale + torch.log(safe_scale)

        return partial

    root_partial = _partial(tau)
    site_likelihood = root_partial @ pi_t  # eq:root
    return torch.sum(torch.log(site_likelihood) + log_scale)


class PartialCache:
    """Partial likelihoods of subtrees, keyed by what determines them.

    A subtree's partial likelihood is a function of the subtree's shape, the
    branch lengths inside it, and the leaf data -- and of nothing above it.
    So when a search evaluates a neighbour topology at the lengths its parent
    was fitted with, every subtree the move did not touch has the partial the
    parent already computed (issue #289). The key is the subtree's structure
    with its lengths, built recursively and order-independent, so the same
    subtree reached under a different rooting or child order still hits.

    Two things are deliberate. The cache holds detached tensors and serves only
    :func:`log_likelihood_cached`, which runs without gradients: inside a fit
    every length moves, so nothing would hit and the graph would be pinned
    alive. And hits are counted, because a cache that never hits is a cost.

    Parameters
    ----------
    max_entries : int
        Entries kept before the oldest is evicted; each is one
        ``(n_sites, k)`` tensor and one ``(n_sites,)`` tensor.
    """

    def __init__(self, max_entries: int = 4096) -> None:
        self._store: dict[object, tuple[torch.Tensor, torch.Tensor]] = {}
        self._max_entries = max_entries
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        return len(self._store)

    def get(self, key: object) -> tuple[torch.Tensor, torch.Tensor] | None:
        found = self._store.get(key)
        if found is None:
            self.misses += 1
        else:
            self.hits += 1
        return found

    def put(self, key: object, value: tuple[torch.Tensor, torch.Tensor]) -> None:
        if len(self._store) >= self._max_entries:
            del self._store[next(iter(self._store))]
        self._store[key] = value


def log_likelihood_cached(
    tau: Node,
    k: int,
    pi: np.ndarray | torch.Tensor,
    alignment: Mapping[str, np.ndarray | torch.Tensor],
    branch_lengths: torch.Tensor,
    cache: PartialCache,
    *,
    rate_matrix: torch.Tensor | None = None,
) -> float:
    """:func:`log_likelihood` without gradients, reusing cached subtree partials.

    The same recursion and the same arithmetic per subtree, so a partial taken
    from the cache is the partial the recursion would have computed, bitwise;
    only the per-subtree rescaling sums are accumulated bottom-up rather than
    in one running total, which is why this is a separate function and not a
    flag on :func:`log_likelihood`, whose arithmetic stays exactly as it was.
    A test pins the two within the float64 agreement tolerance.

    Parameters
    ----------
    tau, k, pi, alignment, branch_lengths, rate_matrix
        As :func:`log_likelihood`.
    cache : PartialCache
        Shared across the calls that should reuse each other's work -- one per
        search, typically.

    Returns
    -------
    float
        The total log-likelihood.
    """
    dtype, device = branch_lengths.dtype, branch_lengths.device
    order = branch_order(tau)
    index = {name: i for i, name in enumerate(order)}
    lengths = branch_lengths.detach()
    with torch.no_grad():
        pi_t = torch.as_tensor(pi, dtype=dtype, device=device)
        transitions = _transition_probabilities(lengths, k, rate_matrix)
        leaves = [node for node in preorder(tau) if node.is_leaf]
        n_sites = int(torch.as_tensor(alignment[leaves[0].name]).shape[0])

        def visit(node: Node) -> tuple[object, torch.Tensor, torch.Tensor]:
            if node.is_leaf:
                key: object = ("leaf", node.name)
                found = cache.get(key)
                if found is not None:
                    return key, *found
                states = torch.as_tensor(
                    alignment[node.name], dtype=torch.long, device=device
                )
                partial = torch.zeros((n_sites, k), dtype=dtype, device=device)
                partial[torch.arange(n_sites), states] = 1.0
                scale = torch.zeros(n_sites, dtype=dtype, device=device)
                cache.put(key, (partial, scale))
                return key, partial, scale

            parts = []
            for child in node.children:
                child_key, child_partial, child_scale = visit(child)
                length = float(lengths[index[child.name]])
                parts.append((child_key, length, child_partial, child_scale))
            key = ("node", tuple(sorted((repr(ck), ln) for ck, ln, _, _ in parts)))
            found = cache.get(key)
            if found is not None:
                return key, *found

            partial = torch.ones((n_sites, k), dtype=dtype, device=device)
            log_scale = torch.zeros(n_sites, dtype=dtype, device=device)
            for child, (_, _, child_partial, child_scale) in zip(
                node.children, parts, strict=True
            ):
                transition = transitions[index[child.name]]
                partial = partial * (child_partial @ transition.T)
                log_scale = log_scale + child_scale
            scale = partial.amax(dim=1)
            safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
            partial = partial / safe_scale.unsqueeze(1)
            log_scale = log_scale + torch.log(safe_scale)
            cache.put(key, (partial, log_scale))
            return key, partial, log_scale

        _, root_partial, root_scale = visit(tau)
        site_likelihood = root_partial @ pi_t  # eq:root
        return float(torch.sum(torch.log(site_likelihood) + root_scale))
