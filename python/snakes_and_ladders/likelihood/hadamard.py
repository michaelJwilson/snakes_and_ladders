"""Hadamard conjugation: the exact spectral inversion for the two-state symmetric model.

The tree's spectral method in its literal form (issue #364). Hendy and Penny
(1993) showed that for the symmetric two-state model on ``n`` taxa the
sequence spectrum ``s`` --- the frequency of each of the ``2^(n-1)``
site patterns, a pattern being the set of taxa differing from a reference
taxon --- and the edge spectrum ``q`` --- the branch length on each split
of the leaf set, zero where the tree has no such branch --- are related
invertibly by the Sylvester--Hadamard matrix ``H`` of order ``2^(n-1)``
(``eq:hadamard`` of ``docs/tex/textbook.tex``)::

    s = H^-1 exp(H q),  so  q = H^-1 log(H s),

with ``q`` indexed by the same subsets as ``s`` and ``q_empty = -sum`` of the
rest. Every ``2^(n-1)`` pattern probability of a two-state tree is a linear
combination of ``exp`` of linear combinations of the branch lengths, and
``H`` is the change of basis. On the exact spectrum the inversion returns
the tree's own branch lengths on its splits and zero elsewhere, which is
what the test pins to ``1e-12``; on an estimated spectrum the weights
converge with the site count, and the *closest tree* (Hendy and Penny) is
the compatible set of the largest, :func:`closest_tree`.

The model is the two-state Jukes--Cantor of ``eq:jc`` at ``k = 2``, whose
probability of change over a branch ``t`` is ``(1 - exp(-2 t)) / 2``, so the
edge weight is the branch length in the repository's units. A four-state
Jukes--Cantor alignment reduces to it exactly by grouping the states in
pairs, :func:`binary_recoding`: a change of group has probability ``(1 -
exp(-4 t / 3)) / 2``, the two-state form at ``2 t / 3``, so the recoded
spectrum returns ``2/3`` of every branch length. The Kimura three-parameter
model has its own conjugation over the Klein four-group of order ``4^(n-1)``
(Hendy, Penny and Steel 1994); it is not implemented here.

``n`` is capped at 12 because the spectrum has ``2^(n-1)`` entries and the
fast Walsh--Hadamard transform is ``O(N log N)`` in that size: 2,048 entries
at the cap, and every doubling of ``n`` squares it.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from snakes_and_ladders.sim.tree import Node, edges

#: Largest taxon count the transform is offered at; the spectrum has
#: ``2 ** (MAX_TAXA - 1)`` entries.
MAX_TAXA = 12
_MIN_TAXA = 3


def walsh_hadamard(vector: np.ndarray) -> np.ndarray:
    """``H v`` for the Sylvester--Hadamard matrix of order ``len(v)``, in ``O(N log N)``.

    ``H`` is the Kronecker power of ``[[1, 1], [1, -1]]``, symmetric, with
    ``H H = N I``, so ``H^-1 = H / N``. Entry ``(A, B)`` is ``(-1)^|A and
    B|`` for subsets read as bit masks, which is the indexing the spectra
    below use.

    Parameters
    ----------
    vector : np.ndarray
        Length a power of two.

    Returns
    -------
    np.ndarray
        ``H v``, the same length.

    Raises
    ------
    ValueError
        If the length is not a power of two.
    """
    size = vector.shape[0]
    if size == 0 or size & (size - 1):
        msg = f"the Walsh-Hadamard transform needs a power-of-two length, got {size}"
        raise ValueError(msg)
    result = np.array(vector, dtype=np.float64)
    half = 1
    while half < size:
        shaped = result.reshape(-1, 2 * half)
        left = shaped[:, :half].copy()
        right = shaped[:, half:].copy()
        shaped[:, :half] = left + right
        shaped[:, half:] = left - right
        half *= 2
    return result


def _check_names(names: list[str]) -> None:
    if len(names) < _MIN_TAXA or len(names) > MAX_TAXA:
        msg = f"the Hadamard conjugation is offered for {_MIN_TAXA} to {MAX_TAXA} taxa, got {len(names)}"
        raise ValueError(msg)


def binary_recoding(
    alignment: Mapping[str, np.ndarray], k: int
) -> dict[str, np.ndarray]:
    """A ``k``-state Jukes--Cantor alignment as a two-state symmetric one.

    States ``0`` to ``k/2 - 1`` become ``0`` and the rest ``1``. Under
    ``eq:jc`` every state is exchangeable, so the grouping is exact: a
    change of group over a branch ``t`` has probability ``(1 - exp(-k t /
    (k - 1))) / 2``, the two-state model at ``k t / (2 (k - 1))``, which
    :func:`recoding_scale` reports so a caller can rescale the weights back.

    Parameters
    ----------
    alignment : Mapping[str, np.ndarray]
        Taxon name to states in ``[0, k)``.
    k : int
        Number of states, even.

    Returns
    -------
    dict[str, np.ndarray]
        The same taxa, states in ``{0, 1}``.

    Raises
    ------
    ValueError
        If ``k`` is odd, where no grouping of the states is exchangeable.
    """
    if k % 2:
        msg = f"the two-state recoding needs an even state count, got {k}"
        raise ValueError(msg)
    return {
        name: (states >= k // 2).astype(np.int64) for name, states in alignment.items()
    }


def recoding_scale(k: int) -> float:
    """The factor by which :func:`binary_recoding` scales a branch length: ``k / (2 (k - 1))``."""
    return k / (2.0 * (k - 1))


def sequence_spectrum(
    alignment: Mapping[str, np.ndarray],
) -> tuple[list[str], np.ndarray]:
    """The pattern frequencies of a two-state alignment, indexed by subset.

    Taxa are taken in sorted order; the last is the reference. A site's
    pattern is the set of the first ``n - 1`` taxa whose state differs from
    the reference's, read as a bit mask with taxon ``i`` at bit ``i``.

    Parameters
    ----------
    alignment : Mapping[str, np.ndarray]
        Taxon name to states in ``{0, 1}``, 3 to :data:`MAX_TAXA` taxa.

    Returns
    -------
    tuple[list[str], np.ndarray]
        The sorted names and the spectrum of length ``2^(n-1)``, summing
        to 1.

    Raises
    ------
    ValueError
        If the taxon count is outside the offered range, or a state is
        outside ``{0, 1}``.
    """
    names = sorted(alignment)
    _check_names(names)
    states = np.stack([alignment[name] for name in names])
    if states.min() < 0 or states.max() > 1:
        msg = "the sequence spectrum is defined on two-state data"
        raise ValueError(msg)
    differs = states[:-1] != states[-1]
    weights = 1 << np.arange(len(names) - 1)
    index = (differs * weights[:, np.newaxis]).sum(axis=0)
    counts = np.bincount(index, minlength=1 << (len(names) - 1))
    return names, counts / states.shape[1]


def hadamard_conjugation(spectrum: np.ndarray) -> np.ndarray:
    """``q = H^-1 log(H s)``: the edge spectrum a sequence spectrum implies.

    Parameters
    ----------
    spectrum : np.ndarray
        Pattern frequencies of length ``2^(n-1)``, summing to 1.

    Returns
    -------
    np.ndarray
        The same length. Entry ``A`` is the weight of the split ``A`` against
        its complement; entry ``0`` is minus the sum of the others.

    Raises
    ------
    ValueError
        If ``H s`` has a non-positive entry, where the logarithm is
        undefined --- a saturated or too-short alignment.
    """
    transformed = walsh_hadamard(spectrum)
    if transformed.min() <= 0.0:
        msg = (
            "H s has a non-positive entry; the spectrum is saturated or too short "
            "for the conjugation"
        )
        raise ValueError(msg)
    return np.asarray(walsh_hadamard(np.log(transformed)) / spectrum.shape[0])


def expected_spectrum(edge_spectrum: np.ndarray) -> np.ndarray:
    """``s = H^-1 exp(H q)``: the forward direction, the exact spectrum of an edge spectrum."""
    return np.asarray(
        walsh_hadamard(np.exp(walsh_hadamard(edge_spectrum))) / edge_spectrum.shape[0]
    )


def _split_index(split: frozenset[str], names: list[str]) -> int:
    """The subset index of a split, on the side not containing the reference taxon."""
    reference = names[-1]
    side = split if reference not in split else frozenset(names) - split
    return sum(1 << names.index(name) for name in side)


def edge_spectrum(tau: Node) -> tuple[list[str], np.ndarray]:
    """A tree's branch lengths as an edge spectrum, zero on every split it lacks.

    Parameters
    ----------
    tau : Node
        A tree with a branch length on every non-root node, 3 to
        :data:`MAX_TAXA` leaves.

    Returns
    -------
    tuple[list[str], np.ndarray]
        The sorted leaf names and the spectrum, ``q_empty = -sum`` of the
        rest. The two branches below a rooted binary root induce one split
        and are summed.

    Raises
    ------
    ValueError
        If a non-root node has no branch length, or the leaf count is
        outside the offered range.
    """
    names = sorted(_leaves(tau))
    _check_names(names)
    spectrum = np.zeros(1 << (len(names) - 1))
    for _, child in edges(tau):
        if child.branch_length is None:
            msg = f"non-root node {child.name!r} has no branch_length"
            raise ValueError(msg)
        spectrum[_split_index(_leaves(child), names)] += child.branch_length
    spectrum[0] = -spectrum[1:].sum()
    return names, spectrum


def split_weights(
    edge_spectrum: np.ndarray, names: list[str]
) -> dict[frozenset[str], float]:
    """The edge spectrum as split weights, keyed as :func:`~snakes_and_ladders.search.topology.leaf_bipartitions` keys splits.

    Parameters
    ----------
    edge_spectrum : np.ndarray
        Length ``2^(n-1)``, from :func:`hadamard_conjugation` or
        :func:`edge_spectrum`.
    names : list[str]
        The sorted taxon names the spectrum was indexed by.

    Returns
    -------
    dict[frozenset[str], float]
        Every non-empty split to its weight, canonicalized to the side not
        containing the lexicographically smallest name.
    """
    all_names = frozenset(names)
    weights: dict[frozenset[str], float] = {}
    for index in range(1, edge_spectrum.shape[0]):
        side = frozenset(
            name for bit, name in enumerate(names[:-1]) if index >> bit & 1
        )
        canonical = side if names[0] not in side else all_names - side
        weights[canonical] = float(edge_spectrum[index])
    return weights


def _compatible(first: frozenset[str], second: frozenset[str]) -> bool:
    """Two canonical splits (neither containing the anchor) are compatible iff nested or disjoint."""
    return first <= second or second <= first or not (first & second)


def closest_tree(
    weights: Mapping[frozenset[str], float],
    names: list[str],
    minimum_length: float,
) -> Node:
    """The binary tree on the ``n - 3`` largest mutually compatible split weights.

    Hendy and Penny's closest tree, in the greedy form: non-trivial splits
    are taken in decreasing weight and kept when compatible with every one
    already kept, until the tree is resolved. A weight at or below
    ``minimum_length`` --- a split the data does not support, or a pendant
    branch estimated at zero --- is floored there, because the tree is a
    start for a fit in log coordinates and a zero length has none.

    Parameters
    ----------
    weights : Mapping[frozenset[str], float]
        Split weights from :func:`split_weights`.
    names : list[str]
        The taxa, at least 3.
    minimum_length : float
        The floor, positive.

    Returns
    -------
    Node
        An unrooted binary tree in the trifurcating-root convention with a
        branch length on every non-root node; internal nodes are named
        ``h<index>``.

    Raises
    ------
    ValueError
        If the floor is not positive.
    """
    if minimum_length <= 0.0:
        msg = f"minimum_length must be positive, got {minimum_length}"
        raise ValueError(msg)
    all_names = frozenset(names)
    anchor = min(all_names)
    n = len(names)
    internal = [
        split for split in weights if 1 < len(split) < n - 1 and anchor not in split
    ]
    chosen: list[frozenset[str]] = []
    for split in sorted(internal, key=lambda split: -weights[split]):
        if len(chosen) == n - 3:
            break
        if all(_compatible(split, kept) for kept in chosen):
            chosen.append(split)

    def length(split: frozenset[str]) -> float:
        return max(weights.get(split, 0.0), minimum_length)

    # The kept clusters are laminar and none contains the anchor, so each
    # nests in the smallest kept cluster containing it; the anchor's own
    # pendant branch hangs from the root beside the maximal clusters.
    clusters = sorted(chosen, key=len)
    counter = 0

    def build(members: frozenset[str], candidates: list[frozenset[str]]) -> Node:
        nonlocal counter
        if len(members) == 1:
            (name,) = members
            return Node(name=name, branch_length=length(members))
        maximal = [
            cluster
            for cluster in candidates
            if cluster < members
            and not any(cluster < other < members for other in candidates)
        ]
        covered: frozenset[str] = (
            frozenset().union(*maximal) if maximal else frozenset()
        )
        children = [
            build(cluster, [c for c in candidates if c < cluster])
            for cluster in maximal
        ] + [build(frozenset((name,)), []) for name in sorted(members - covered)]
        counter += 1
        return Node(
            name=f"h{counter}", branch_length=length(members), children=tuple(children)
        )

    rest = all_names - {anchor}
    top = build(rest, [cluster for cluster in clusters if cluster < rest])
    return Node(
        name="root",
        branch_length=None,
        children=(*top.children, Node(name=anchor, branch_length=length(rest))),
    )


def _leaves(node: Node) -> frozenset[str]:
    if node.is_leaf:
        return frozenset((node.name,))
    return frozenset().union(*(_leaves(child) for child in node.children))
