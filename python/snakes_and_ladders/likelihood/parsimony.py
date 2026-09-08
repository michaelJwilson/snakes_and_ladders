"""Parsimony: the criterion that is provably wrong in a known place.

Parsimony scores a topology by the fewest state changes that explain the data,
with no model, no branch lengths and no probabilities --- an integer where the
likelihood is a float. It is here to be **wrong**, in the one region where the
failure is a theorem rather than a defect.

That region is the *Felsenstein zone*: four taxa, two long branches placed
non-adjacently. Convergent change on the two long branches is cheaper to
explain by grouping them than by the true topology, so as sites increase
parsimony's error rate converges to 1 rather than 0 --- it is statistically
**inconsistent** there, while maximum likelihood under the generating model is
consistent (Felsenstein 1978). Moving the same two long branches to be
adjacent gives the *Farris zone*, where parsimony is consistent and fast, and
the pair together is what makes the first interpretable: an implementation
that is merely broken fails both.

**Small parsimony only.** This scores a *given* topology. Searching for the
most parsimonious tree is large parsimony (``sec:parsimony``) and belongs to
the search machinery (:func:`snakes_and_ladders.search.infer.parsimony_search`),
which consumes either score exactly as it consumes a likelihood.

Two scores over the same post-order pass. :func:`fitch_score` is Fitch's
algorithm for unordered characters (``eq:fitch``): every change costs 1 and
any state may follow any other. :func:`sankoff_score` is Sankoff's dynamic
programme under a weighted step matrix (``eq:sankoff``), of which Fitch is
the unit-cost case --- :func:`unit_step_matrix` recovers it, and the suite
pins that they agree exactly.

See Fitch (1971); Sankoff (1975); Felsenstein, *Inferring Phylogenies*,
ch. 2, 7 and 9.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from snakes_and_ladders.sim.tree import Node


def _check_lengths(alignment: Mapping[str, np.ndarray]) -> int:
    """The one sequence length, refusing an alignment whose rows disagree."""
    lengths = {int(states.shape[0]) for states in alignment.values()}
    if len(lengths) > 1:
        msg = f"alignment sequences differ in length: {sorted(lengths)}"
        raise ValueError(msg)
    return lengths.pop() if lengths else 0


def unit_step_matrix(k: int) -> np.ndarray:
    """The step matrix under which :func:`sankoff_score` is :func:`fitch_score`.

    Parameters
    ----------
    k : int
        Number of states.

    Returns
    -------
    np.ndarray
        Shape ``(k, k)``: 1 off the diagonal, 0 on it.
    """
    return np.ones((k, k)) - np.eye(k)


def fitch_score(tau: Node, alignment: Mapping[str, np.ndarray], k: int) -> int:
    """Fewest state changes on ``tau`` explaining ``alignment``, summed over sites (``eq:fitch``).

    One post-order pass. Each node carries the set of states achievable at it
    for the minimum cost so far, held as a bitmask per site so a whole
    alignment is one vectorized pass rather than a Python loop per column. At
    an internal node the children's sets are intersected; where the
    intersection is empty the union is taken instead and one change is
    counted. That the two cases are exactly "no change needed" and "one change
    needed" is what makes the count minimal, and it is why the criterion needs
    no model.

    Parameters
    ----------
    tau : Node
        The topology to score. Branch lengths are ignored --- parsimony does
        not use them, which is half of what makes it a different criterion
        rather than an approximation to the likelihood.
    alignment : Mapping[str, np.ndarray]
        Leaf name to integer states, shape ``(n_sites,)``, values in
        ``[0, k)``.
    k : int
        Number of states, ``<= 63`` so a mask fits in a signed 64-bit integer.

    Returns
    -------
    int
        The parsimony score: total changes over all sites.

    Raises
    ------
    ValueError
        If a leaf of ``tau`` is missing from ``alignment``, if the sequences
        differ in length, or if ``k`` exceeds what a bitmask holds. A missing
        leaf would otherwise score a strict subtree and return a number that
        is smaller for the wrong reason.
    """
    if not 2 <= k <= 63:
        msg = f"k must be in [2, 63] to fit a bitmask, got {k}"
        raise ValueError(msg)

    _check_lengths(alignment)

    changes = 0

    def visit(node: Node) -> np.ndarray:
        nonlocal changes
        if node.is_leaf:
            if node.name not in alignment:
                msg = f"leaf {node.name!r} is not in the alignment"
                raise ValueError(msg)
            return (1 << alignment[node.name].astype(np.int64)).astype(np.int64)

        masks = [visit(child) for child in node.children]
        combined = masks[0]
        for mask in masks[1:]:
            intersection = combined & mask
            empty = intersection == 0
            changes += int(empty.sum())
            combined = np.where(empty, combined | mask, intersection)
        return combined

    visit(tau)
    return changes


def sankoff_score(
    tau: Node, alignment: Mapping[str, np.ndarray], step_matrix: np.ndarray
) -> float:
    """Minimum weighted change on ``tau`` explaining ``alignment``, summed over sites (``eq:sankoff``).

    Sankoff's dynamic programme (Sankoff 1975; Felsenstein 2004, ch. 2). Each
    node carries, per site, the cost of the cheapest labelling of its subtree
    given its own state --- a ``(k, n_sites)`` array where Fitch carries a
    bitmask --- and a parent's cost at state ``i`` sums over its children the
    cheapest ``step_matrix[i, j]`` plus the child's cost at ``j``. A leaf costs
    0 at its observed state and infinity elsewhere; the site's score is the
    minimum over the root's states. It is the min-plus form of the pruning
    recursion, and with :func:`unit_step_matrix` it returns exactly
    :func:`fitch_score`, which is the reduction the suite pins.

    The tree is scored as rooted: ``step_matrix[i, j]`` is the cost of a
    change from the parent's state ``i`` to the child's ``j``, so an
    asymmetric matrix scores each rooting of the same unrooted tree
    differently. That is the definition and not a defect; the search, which
    walks unrooted topologies, is what refuses a matrix whose score depends
    on the rooting (:func:`snakes_and_ladders.search.infer.parsimony_search`).

    Parameters
    ----------
    tau : Node
        The topology to score. Branch lengths are ignored.
    alignment : Mapping[str, np.ndarray]
        Leaf name to integer states, shape ``(n_sites,)``, values in
        ``[0, k)`` for ``k`` the side of ``step_matrix``.
    step_matrix : np.ndarray
        Shape ``(k, k)``, finite and non-negative; entry ``[i, j]`` is the
        cost of a change from ``i`` to ``j``.

    Returns
    -------
    float
        The weighted parsimony score: minimum total cost over all sites.
        An integer-valued matrix gives an integer-valued float.

    Raises
    ------
    ValueError
        If ``step_matrix`` is not square, finite and non-negative, if a leaf
        of ``tau`` is missing from ``alignment``, if the sequences differ in
        length, or if a state lies outside ``[0, k)``.
    """
    step = np.asarray(step_matrix, dtype=np.float64)
    if step.ndim != 2 or step.shape[0] != step.shape[1] or step.shape[0] < 2:
        msg = f"step_matrix must be square with side >= 2, got shape {step.shape}"
        raise ValueError(msg)
    if not np.all(np.isfinite(step)) or bool(np.any(step < 0.0)):
        msg = "step_matrix entries must be finite and non-negative"
        raise ValueError(msg)
    k = int(step.shape[0])
    n_sites = _check_lengths(alignment)
    for name, states in alignment.items():
        if states.shape[0] and (int(states.min()) < 0 or int(states.max()) >= k):
            msg = f"leaf {name!r} carries a state outside [0, {k})"
            raise ValueError(msg)

    def visit(node: Node) -> np.ndarray:
        if node.is_leaf:
            if node.name not in alignment:
                msg = f"leaf {node.name!r} is not in the alignment"
                raise ValueError(msg)
            cost = np.full((k, n_sites), np.inf)
            cost[alignment[node.name].astype(np.int64), np.arange(n_sites)] = 0.0
            return cost

        total = np.zeros((k, n_sites))
        for child in node.children:
            # (k, k, n_sites): parent state, child state, site. Reduced over
            # the child's state at once rather than in a Python loop per site.
            total += (step[:, :, None] + visit(child)[None, :, :]).min(axis=1)
        return total

    return float(visit(tau).min(axis=0).sum())


def brute_force_parsimony_score(
    tau: Node, alignment: Mapping[str, np.ndarray], k: int
) -> int:
    """The same score, by enumerating every internal-node labelling.

    The oracle :func:`fitch_score` is pinned against, and deliberately
    exponential: ``k ** internal_nodes`` per site. It shares no traversal with
    Fitch --- it assigns states to internal nodes directly and counts
    disagreeing edges --- which is what makes it an independent check rather
    than a second spelling of the same recursion. The same relationship
    :func:`snakes_and_ladders.likelihood.brute_force.brute_force_log_likelihood` has to the
    pruning recursion.

    Callers must keep the tree small: cost is
    ``O(n_sites * k ** internal_nodes)``.

    Parameters
    ----------
    tau, alignment, k
        As :func:`fitch_score`.

    Returns
    -------
    int
        The minimum number of changes, summed over sites.
    """
    from snakes_and_ladders.enumeration import assignments
    from snakes_and_ladders.sim.tree import edges, preorder

    internal = [node.name for node in preorder(tau) if not node.is_leaf]
    position = {name: index for index, name in enumerate(internal)}
    edge_list = [(parent.name, child.name) for parent, child in edges(tau)]
    n_sites = int(next(iter(alignment.values())).shape[0])
    labellings = list(
        assignments(k, len(internal), what=f"{k}**{len(internal)} internal labellings")
    )

    total = 0
    for site in range(n_sites):
        # The observed state at every leaf for this site, so the cost loop
        # below reads one flat mapping and never closes over the loop
        # variable.
        observed = {name: int(states[site]) for name, states in alignment.items()}
        best = min(
            sum(
                1
                for parent, child in edge_list
                if _state(parent, labelling, position, observed)
                != _state(child, labelling, position, observed)
            )
            for labelling in labellings
        )
        total += best
    return total


def _state(
    name: str,
    labelling: tuple[int, ...],
    position: dict[str, int],
    observed: dict[str, int],
) -> int:
    """The state at ``name`` under one internal-node labelling."""
    index = position.get(name)
    return labelling[index] if index is not None else observed[name]
