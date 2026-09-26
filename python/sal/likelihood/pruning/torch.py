"""Differentiable PyTorch Felsenstein pruning -- pinned against
``sal.likelihood.pruning``, the NumPy oracle (CLAUDE.md, "The NumPy
reference is the oracle and it stays").

Branch lengths are a tensor kept separate from the topology: a
``sal.sim.tree.Node`` here describes only shape (leaf names,
children), while ``branch_lengths`` -- ordered by ``branch_order(tau)`` -- is
the tensor ``torch.autograd`` differentiates through. ``Node.branch_length``
is never read by ``log_likelihood``.

JC transition probabilities default to the closed form ``eq:jc`` of
``docs/tex/textbook.tex``, built from ``torch.exp`` so the branch length stays
in the graph. Passing ``rate_matrix`` switches to the general
``torch.linalg.matrix_exp(Q * t)`` path -- what a fitted, non-JC rate matrix
would use -- which must agree with the closed form when ``Q`` is the JC
generator (tests/regression/test_pruning_torch.py).

Rescaling (``likelihood/CLAUDE.md``, "Rescaling must stay differentiable")
accumulates ``log_scale`` by tensor addition, never in place, so it composes
correctly under autograd.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import NamedTuple

import numpy as np
import torch

from sal.likelihood.patterns import check_weights
from sal.likelihood.pruning_common import (
    check_alignment_covers,
    check_branch_lengths_shape,
    check_pi_shape,
    leaf_indicator,
    require_branch_length,
    rescale_partial,
)
from sal.sim.tree import Node, edges, preorder


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

    Seeds a ``branch_lengths`` tensor from a fixture tree (e.g. before
    ``.requires_grad_(True)`` for ``gradcheck``); ``log_likelihood`` never
    reads ``tau``'s ``branch_length`` fields.

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
    lengths = [require_branch_length(child) for _, child in edges(tau)]
    return torch.tensor(lengths, dtype=dtype, device=device)


def _jc_transition_probabilities(t: torch.Tensor, n_states: int) -> torch.Tensor:
    """Closed-form JC P(t), ``eq:jc`` of ``docs/tex/textbook.tex``, differentiable in ``t``.

    ``t`` may be a scalar or a vector of branch lengths; the result carries
    one ``(k, k)`` matrix per entry of ``t`` in its leading dimensions. The
    arithmetic is elementwise, so a matrix taken from the batched result is
    the matrix the scalar call returns, bitwise -- a test pins it.
    """
    decay = torch.exp(-n_states * t / (n_states - 1))[..., None, None]
    off_diagonal = (1.0 - decay) / n_states
    diagonal = 1.0 / n_states + (n_states - 1) / n_states * decay
    eye = torch.eye(n_states, dtype=t.dtype, device=t.device)
    return off_diagonal * (1.0 - eye) + diagonal * eye


def transition_probabilities(
    t: torch.Tensor, n_states: int, rate_matrix: torch.Tensor | None
) -> torch.Tensor:
    """``P(t)`` for every branch length in ``t``, shape ``(*t.shape, k, k)``.

    One call per likelihood evaluation rather than one per branch: the
    hill-climb profile behind #264 charged 8,730 scalar calls per 970
    evaluations here, the per-child call root ``CLAUDE.md``'s inlining rule
    names. Batched, the closed form is one ``exp`` over the branch vector and
    the matrix exponential one batched ``matrix_exp``.
    """
    if rate_matrix is None:
        return _jc_transition_probabilities(t, n_states)
    result: torch.Tensor = torch.linalg.matrix_exp(rate_matrix * t[..., None, None])
    return result


#: Leaf indicator partials, keyed by the states array's identity with the
#: alphabet and the tensor type. A leaf's partial is the data indicator of
#: ``eq:pruning``: it depends on the alignment and on nothing being fitted, so
#: it is the same tensor at every evaluation of a fit -- and a fit evaluates it
#: thousands of times. At 20 taxa, 20 of the tree's 38 nodes are leaves, and
#: rebuilding each one cost a ``zeros``, an ``arange``, an ``as_tensor`` and a
#: scatter per evaluation (issue #443's Python post-order, measured at 21.8% of
#: a 20-taxon fit's self time).
#:
#: The states array is held in the value so its ``id`` cannot be recycled onto
#: a different array while the entry lives, which is what makes the identity
#: key sound rather than merely usually right.
_LEAF_PARTIALS: dict[
    tuple[int, int, torch.dtype, str], tuple[object, torch.Tensor]
] = {}

#: Entries kept before the oldest is dropped. One entry is an
#: ``(n_sites, k)`` tensor; a 20-taxon fit needs 20 of them.
_LEAF_PARTIAL_LIMIT = 512


def _leaf_partial(
    states: object,
    n_sites: int,
    n_states: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """The one-hot partial for a leaf's observed states, built once per array.

    Parameters
    ----------
    states : object
        The leaf's observed states, as the alignment holds them.
    n_sites : int
        Columns in the alignment.
    n_states : int
        Alphabet size.
    dtype : torch.dtype
        Tensor type, taken from the branch lengths.
    device : torch.device
        Where the recursion runs.

    Returns
    -------
    torch.Tensor
        ``(n_sites, k)``, one at the observed state of each site. Constant in
        the branch lengths, so it carries no gradient and is shared rather
        than copied.
    """
    key = (id(states), n_states, dtype, str(device))
    hit = _LEAF_PARTIALS.get(key)
    if hit is not None and hit[0] is states:
        return hit[1]

    partial = leaf_indicator(
        states, n_sites, n_states, dtype, device, index_device=device
    )
    if len(_LEAF_PARTIALS) >= _LEAF_PARTIAL_LIMIT:
        _LEAF_PARTIALS.pop(next(iter(_LEAF_PARTIALS)))
    _LEAF_PARTIALS[key] = (states, partial)
    return partial


#: One step of a topology's post-order schedule: the slot the node's partial
#: lands in, the alignment key if the node is a leaf, and one
#: ``(child slot, branch index)`` pair per child if it is not. A tuple rather
#: than a class because the evaluation loop unpacks one per node.
_Step = tuple[int, str | None, tuple[tuple[int, int], ...]]


class Traversal(NamedTuple):
    """What a topology fixes, so an evaluation reads it instead of walking.

    Parameters
    ----------
    order : list[str]
        ``branch_order(tau)``, which ``branch_lengths`` is checked against.
    leaf_names : tuple[str, ...]
        Leaf names, left to right --- the order ``preorder`` reports them in,
        which post-order also reports them in, the two differing only on
        internal nodes.
    steps : tuple[_Step, ...]
        The nodes in post-order: children before their parent, children in
        the topology's own order, so the sequence of arithmetic is the
        recursion's exactly.
    root_slot : int
        The slot the root's partial lands in, which is the last step's.
    """

    order: list[str]
    leaf_names: tuple[str, ...]
    steps: tuple[_Step, ...]
    root_slot: int


#: Post-order schedules, keyed by the topology's identity with the topology
#: held in the value, as `_LEAF_PARTIALS` holds its states array and for the
#: same reason: the key is sound only while the object it names is alive.
#:
#: The traversal is a function of the topology and of nothing that is fitted,
#: so a fit evaluating one tree hundreds of times walks it once. At the
#: enumerable tier the Python recursion was 15.0% of an NNI search (0.75 s of
#: 4.97 s) and 14.9% of an SPR one, one frame and two dictionary lookups per
#: node per evaluation (issue #754).
_TRAVERSALS: dict[int, tuple[Node, Traversal]] = {}

#: Schedules kept before the oldest is dropped. One entry is a few hundred
#: integers; a 20-taxon search holds one per candidate topology it is still
#: fitting.
_TRAVERSAL_LIMIT = 256


def _build_traversal(tau: Node) -> Traversal:
    """The post-order schedule of ``tau``, with each branch's index baked in."""
    order = branch_order(tau)
    index = {name: i for i, name in enumerate(order)}
    steps: list[_Step] = []
    leaf_names: list[str] = []

    def _visit(node: Node) -> int:
        children = tuple((_visit(child), index[child.name]) for child in node.children)
        slot = len(steps)
        if node.is_leaf:
            leaf_names.append(node.name)
            steps.append((slot, node.name, ()))
        else:
            steps.append((slot, None, children))
        return slot

    root_slot = _visit(tau)
    return Traversal(order, tuple(leaf_names), tuple(steps), root_slot)


def traversal(tau: Node) -> Traversal:
    """``_build_traversal(tau)``, built once per topology object."""
    key = id(tau)
    hit = _TRAVERSALS.get(key)
    if hit is not None and hit[0] is tau:
        return hit[1]

    built = _build_traversal(tau)
    if len(_TRAVERSALS) >= _TRAVERSAL_LIMIT:
        _TRAVERSALS.pop(next(iter(_TRAVERSALS)))
    _TRAVERSALS[key] = (tau, built)
    return built


def log_likelihood(
    tau: Node,
    n_states: int,
    pi: np.ndarray | torch.Tensor,
    alignment: Mapping[str, np.ndarray | torch.Tensor],
    branch_lengths: torch.Tensor,
    *,
    weights: np.ndarray | None = None,
    rate_matrix: torch.Tensor | None = None,
    rescale: bool = True,
) -> torch.Tensor:
    """Total log-likelihood, differentiable w.r.t. ``branch_lengths``.

    Parameters
    ----------
    tau : Node
        Root of the topology. Its own ``branch_length`` fields are ignored;
        branch lengths come from ``branch_lengths`` instead.
    n_states : int
        Number of states.
    pi : np.ndarray | torch.Tensor
        Root state distribution, shape ``(k,)``.
    alignment : Mapping[str, np.ndarray | torch.Tensor]
        Leaf name to its observed states, each of shape ``(n_sites,)`` with
        entries in ``[0, k)``.
    branch_lengths : torch.Tensor
        Shape ``(len(branch_order(tau)),)``, in ``dtype`` on ``device``. The
        tensor autograd differentiates through; kept separate from ``tau``.
    weights : np.ndarray | None
        One weight per column, or ``None`` for one occurrence each. The
        compressed alignment of
        ``sal.likelihood.patterns.compress`` with its weights
        gives the same value over the distinct columns alone. The weights are
        constants of the data, so the gradient in ``branch_lengths`` is the
        weighted sum of the site gradients.
    rate_matrix : torch.Tensor | None
        If given, shape ``(k, k)``: a general rate matrix ``Q``, and
        transition probabilities use ``torch.linalg.matrix_exp(Q * t)``
        instead of the closed-form JC formula. ``None`` (default) uses the
        closed form.
    rescale : bool
        Whether to rescale partial likelihoods per node, matching
        ``sal.likelihood.pruning``'s ``rescale`` flag.

    Returns
    -------
    torch.Tensor
        0-dimensional; ``sum_s log Pr(data_s | tau, branch_lengths, Q, pi)``.

    Raises
    ------
    ValueError
        If ``pi`` does not have shape ``(k,)``, ``branch_lengths`` does not
        have shape ``(len(branch_order(tau)),)``, ``alignment`` is missing a
        leaf of ``tau``, or ``weights`` does not have one entry per column.
    """
    # dtype and device follow branch_lengths, so a caller moves the whole
    # recursion by moving one tensor, and float64 stays the default because
    # branch_lengths_from_tree defaults to it. Metal rejects float64, so an
    # accelerator path must choose float32 here rather than have it chosen
    # silently.
    dtype = branch_lengths.dtype
    device = branch_lengths.device
    pi_t = torch.as_tensor(pi, dtype=dtype, device=device)
    check_pi_shape(tuple(pi_t.shape), n_states)

    schedule = traversal(tau)
    check_branch_lengths_shape(tuple(branch_lengths.shape), len(schedule.order))
    check_alignment_covers(schedule.leaf_names, alignment)

    n_sites = int(torch.as_tensor(alignment[schedule.leaf_names[0]]).shape[0])
    weight = check_weights(weights, n_sites)
    log_scale = torch.zeros(n_sites, dtype=dtype, device=device)
    # The scalar the rescale falls back to, built once per call rather
    # than as a fresh `ones_like` per internal node.
    one = torch.ones((), dtype=dtype, device=device)
    # Every branch's transition matrix at once, indexed by branch_order.
    transitions = transition_probabilities(branch_lengths, n_states, rate_matrix)

    # The post-order, flat: one pass over the schedule the topology fixes,
    # rather than a Python frame and two dictionary lookups per node per
    # evaluation. Each partial is popped by the parent that consumes it, so
    # what is held at once is what the recursion held (issue #754). The
    # arithmetic, its operands and its order are the recursion's, so the
    # log-likelihood is bitwise unchanged.
    partials: dict[int, torch.Tensor] = {}
    for slot, leaf_name, children in schedule.steps:
        if leaf_name is not None:
            partials[slot] = _leaf_partial(
                alignment[leaf_name], n_sites, n_states, dtype, device
            )
            continue

        # The first child's message seeds the product rather than a tensor of
        # ones being multiplied by it: the ones were allocated and multiplied
        # once per internal node per evaluation, and a product over one term is
        # that term. The value is unchanged, so this is bitwise.
        partial: torch.Tensor | None = None
        for child_slot, branch in children:
            # message[s, i] = sum_j P_ij(t) * L_child(s, j) -- eq:pruning.
            message = partials.pop(child_slot) @ transitions[branch].T
            partial = message if partial is None else partial * message
        if partial is None:
            # A childless non-leaf constrains nothing.
            partial = torch.ones((n_sites, n_states), dtype=dtype, device=device)

        if rescale:
            # The replacement for a vanished scale is the scalar one rather
            # than a tensor of ones, which is the same value without the
            # per-node allocation.
            partial, log_scale, _ = rescale_partial(partial, log_scale, one)

        partials[slot] = partial

    root_partial = partials.pop(schedule.root_slot)
    site_likelihood = root_partial @ pi_t  # eq:root
    site_log_likelihood = torch.log(site_likelihood) + log_scale
    if weight is None:
        return torch.sum(site_log_likelihood)
    return torch.dot(
        torch.as_tensor(weight, dtype=dtype, device=device), site_log_likelihood
    )


class PartialCache:
    """Partial likelihoods of subtrees, keyed by what determines them.

    A subtree's partial likelihood is a function of the subtree's shape, the
    branch lengths inside it, and the leaf data -- and of nothing above it. So
    when a search evaluates a neighbour topology at the lengths its parent was
    fitted with, every subtree the move did not touch has the partial the
    parent already computed (issue #289). The key is the subtree's structure
    with its lengths, order-independent, so the same subtree reached under a
    different rooting or child order still hits.

    The cache holds detached tensors and serves only
    :func:`log_likelihood_cached`, which runs without gradients: inside a fit
    every length moves, so nothing would hit and the graph would be pinned
    alive. Hits are counted, because a cache that never hits is a cost.

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
    n_states: int,
    pi: np.ndarray | torch.Tensor,
    alignment: Mapping[str, np.ndarray | torch.Tensor],
    branch_lengths: torch.Tensor,
    cache: PartialCache,
    *,
    rate_matrix: torch.Tensor | None = None,
) -> float:
    """:func:`log_likelihood` without gradients, reusing cached subtree partials.

    The same recursion and the same arithmetic per subtree, so a partial taken
    from the cache is the partial the recursion would have computed, bitwise.
    Only the per-subtree rescaling sums accumulate bottom-up rather than in one
    running total, which is why this is a separate function and not a flag on
    :func:`log_likelihood`, whose arithmetic is unchanged. A test pins the two
    within the float64 agreement tolerance.

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
        transitions = transition_probabilities(lengths, n_states, rate_matrix)
        leaves = [node for node in preorder(tau) if node.is_leaf]
        n_sites = int(torch.as_tensor(alignment[leaves[0].name]).shape[0])

        def visit(node: Node) -> tuple[object, torch.Tensor, torch.Tensor]:
            if node.is_leaf:
                key: object = ("leaf", node.name)
                found = cache.get(key)
                if found is not None:
                    return key, *found
                partial = leaf_indicator(
                    alignment[node.name],
                    n_sites,
                    n_states,
                    dtype,
                    device,
                    index_device=None,
                )
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

            partial = torch.ones((n_sites, n_states), dtype=dtype, device=device)
            log_scale = torch.zeros(n_sites, dtype=dtype, device=device)
            for child, (_, _, child_partial, child_scale) in zip(
                node.children, parts, strict=True
            ):
                transition = transitions[index[child.name]]
                partial = partial * (child_partial @ transition.T)
                log_scale = log_scale + child_scale
            partial, log_scale, _ = rescale_partial(partial, log_scale, None)
            cache.put(key, (partial, log_scale))
            return key, partial, log_scale

        _, root_partial, root_scale = visit(tau)
        site_likelihood = root_partial @ pi_t  # eq:root
        return float(torch.sum(torch.log(site_likelihood) + root_scale))
