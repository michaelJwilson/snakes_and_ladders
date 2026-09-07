"""Fitch parsimony: the criterion that is provably wrong in a known place.

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
most parsimonious tree is large parsimony and belongs to the search machinery,
which can consume this score exactly as it consumes a likelihood.

Fitch's algorithm for unordered characters, so every state change costs 1 and
any state may follow any other. Sankoff's generalization to a weighted step
matrix is out of scope; nothing here needs it.

See Fitch (1971); Felsenstein, *Inferring Phylogenies*, ch. 7 and 9.
"""

from __future__ import annotations

import numpy as np

from snakes_and_ladders.sim.tree import Node


def fitch_score(tau: Node, alignment: dict[str, np.ndarray], k: int) -> int:
    """Fewest state changes on ``tau`` explaining ``alignment``, summed over sites.

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
    alignment : dict[str, np.ndarray]
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

    lengths = {int(states.shape[0]) for states in alignment.values()}
    if len(lengths) > 1:
        msg = f"alignment sequences differ in length: {sorted(lengths)}"
        raise ValueError(msg)

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


def brute_force_parsimony_score(
    tau: Node, alignment: dict[str, np.ndarray], k: int
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
    import itertools

    from snakes_and_ladders.sim.tree import edges, preorder

    internal = [node.name for node in preorder(tau) if not node.is_leaf]
    position = {name: index for index, name in enumerate(internal)}
    edge_list = [(parent.name, child.name) for parent, child in edges(tau)]
    n_sites = int(next(iter(alignment.values())).shape[0])
    labellings = list(itertools.product(range(k), repeat=len(internal)))

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
