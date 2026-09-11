"""Pairwise distances from an alignment, with the variance a caller needs.

The moment estimator the tree has in place of the HMM's spectral method
(issue #364). Spectral learning of an HMM inverts observable moments to
parameters with a consistency guarantee (Hsu, Kakade and Zhang 2012); Mossel
and Roch (2006) carry the argument to a phylogeny, where it becomes a distance
per pair of taxa inverted from the pair co-occurrence matrix, followed by
neighbor joining (:mod:`snakes_and_ladders.search.neighbor_joining`). This
module is the first half: the distances and their delta-method variances, so a
caller can state whether the errors sit inside Atteson's radius.

Two distances, ``eq:jc-distance`` and ``eq:logdet`` of
``docs/tex/textbook.tex``. The Jukes--Cantor one inverts ``eq:jc`` in closed
form: with ``p`` the fraction of sites at which the pair differs,
``d = -((k-1)/k) log(1 - k p / (k-1))``. The log-det distance (Steel 1994)
reads the full ``k x k`` pair frequency matrix ``F`` and is additive along the
tree under any Markov model, stationary or not; under a rate matrix of trace
``-k``, which ``eq:normalization`` gives the Jukes--Cantor model, it equals
the branch length exactly, and under a general time-reversible model it is
``-tr(Q)/k`` times it, which is all neighbor joining needs.

Both variances are the delta method over the multinomial site counts, stated
in :func:`multinomial_delta_variance` once, so the two estimators share one
derivation and the test that pins its coverage covers both. A saturated pair
--- ``p`` at or beyond ``(k-1)/k``, or a singular ``F`` --- has no finite
distance and is refused rather than clamped, per ``likelihood/CLAUDE.md``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from snakes_and_ladders.sim.tree import Node


class DistanceKind(StrEnum):
    """Which estimator :func:`distance_matrix` applies to every pair."""

    JUKES_CANTOR = "jukes-cantor"
    LOG_DET = "log-det"


@dataclass(frozen=True)
class Distance:
    """One pair's distance estimate and its delta-method variance.

    Parameters
    ----------
    value : float
        Expected substitutions per site along the path between the pair.
    variance : float
        Delta-method variance of ``value`` over the multinomial site counts.
    """

    value: float
    variance: float

    @property
    def standard_error(self) -> float:
        """Square root of the variance."""
        return float(np.sqrt(self.variance))


def pair_counts(first: np.ndarray, second: np.ndarray, k: int) -> np.ndarray:
    """The ``k x k`` joint count matrix of two aligned sequences.

    Parameters
    ----------
    first, second : np.ndarray
        Integer states in ``[0, k)``, the same length.
    k : int
        Number of states.

    Returns
    -------
    np.ndarray
        Shape ``(k, k)``; entry ``[i, j]`` counts the sites at which
        ``first`` is ``i`` and ``second`` is ``j``.

    Raises
    ------
    ValueError
        If the sequences differ in length, are empty, or carry a state
        outside ``[0, k)``.
    """
    if first.shape != second.shape or first.ndim != 1:
        msg = f"sequences have shapes {first.shape} and {second.shape}"
        raise ValueError(msg)
    if first.shape[0] == 0:
        msg = "a distance needs at least one site"
        raise ValueError(msg)
    for sequence in (first, second):
        if sequence.min() < 0 or sequence.max() >= k:
            msg = (
                f"states must lie in [0, {k}), got [{sequence.min()}, {sequence.max()}]"
            )
            raise ValueError(msg)
    counts = np.zeros((k, k), dtype=np.int64)
    np.add.at(counts, (first, second), 1)
    return counts


def multinomial_delta_variance(
    frequencies: np.ndarray, gradient: np.ndarray, n_sites: int
) -> float:
    """Delta-method variance of a statistic of multinomial proportions.

    For proportions ``F`` of ``n_sites`` draws, ``Cov(F_a, F_b) =
    (delta_ab F_a - F_a F_b) / n_sites``, so a statistic with gradient ``g``
    at ``F`` has variance ``(sum_a F_a g_a^2 - (sum_a F_a g_a)^2) /
    n_sites``. One derivation for every distance here.

    Parameters
    ----------
    frequencies : np.ndarray
        The proportions, any shape, summing to 1.
    gradient : np.ndarray
        The statistic's partial derivatives, the same shape.
    n_sites : int
        Draws the proportions were formed from.

    Returns
    -------
    float
        The variance, non-negative.
    """
    weighted = frequencies * gradient
    return float(
        max(0.0, (np.sum(weighted * gradient) - np.sum(weighted) ** 2) / n_sites)
    )


def jukes_cantor_distance(first: np.ndarray, second: np.ndarray, k: int) -> Distance:
    """The closed-form inversion of ``eq:jc`` from the fraction of differing sites, ``eq:jc-distance``.

    ``d = -((k-1)/k) log(1 - k p / (k-1))`` for ``p`` the observed fraction
    of sites at which the pair differs, which is the value of ``t`` at which
    the off-diagonal mass of ``eq:jc``, ``(k-1)/k (1 - exp(-k t/(k-1)))``,
    equals ``p``. Variance by the delta method on ``p``: ``p (1-p) / (L (1 -
    k p/(k-1))^2)``.

    Parameters
    ----------
    first, second : np.ndarray
        Integer states in ``[0, k)``, the same length ``L``.
    k : int
        Number of states.

    Returns
    -------
    Distance
        The distance and its variance.

    Raises
    ------
    ValueError
        If the pair is saturated --- ``p >= (k-1)/k`` --- where no finite
        distance exists.
    """
    counts = pair_counts(first, second, k)
    n_sites = int(counts.sum())
    p = 1.0 - float(np.trace(counts)) / n_sites
    saturation = (k - 1) / k
    if p >= saturation:
        msg = (
            f"the pair differs at a fraction {p:.4f} of sites, at or beyond the "
            f"saturation {saturation:.4f} of the {k}-state model; no finite "
            "distance exists"
        )
        raise ValueError(msg)
    inside = 1.0 - p / saturation
    value = -saturation * np.log(inside)
    variance = p * (1.0 - p) / (n_sites * inside**2)
    return Distance(float(value), float(variance))


def log_det_distance(first: np.ndarray, second: np.ndarray, k: int) -> Distance:
    """The log-det distance of Steel (1994) from the pair frequency matrix, ``eq:logdet``.

    ``d = -(1/k) [log det F - (log det Pi_x + log det Pi_y) / 2]`` for ``F``
    the ``k x k`` joint frequency matrix and ``Pi_x``, ``Pi_y`` the diagonal
    matrices of its row and column sums. Additive along a tree under any
    Markov process on the branches; equal to the branch length when the
    rate matrix has trace ``-k``, which ``eq:normalization`` gives the
    Jukes--Cantor model, and ``-tr(Q)/k`` times it under a general
    time-reversible ``Q``.

    The variance is :func:`multinomial_delta_variance` with the gradient
    ``dd/dF_ij = -(1/k) [(F^-1)_ji - 1/(2 r_i) - 1/(2 c_j)]``, ``r`` and
    ``c`` the row and column sums.

    Parameters
    ----------
    first, second : np.ndarray
        Integer states in ``[0, k)``, the same length.
    k : int
        Number of states.

    Returns
    -------
    Distance
        The distance and its variance.

    Raises
    ------
    ValueError
        If ``F`` is singular, or either marginal has a state with no sites,
        where the logarithm is undefined.
    """
    counts = pair_counts(first, second, k)
    n_sites = int(counts.sum())
    frequencies = counts / n_sites
    rows = frequencies.sum(axis=1)
    columns = frequencies.sum(axis=0)
    sign, log_det = np.linalg.slogdet(frequencies)
    if sign <= 0 or rows.min() <= 0 or columns.min() <= 0:
        msg = (
            "the pair frequency matrix is singular or a state is unobserved; "
            "the log-det distance is undefined"
        )
        raise ValueError(msg)
    value = -(log_det - 0.5 * (np.sum(np.log(rows)) + np.sum(np.log(columns)))) / k
    inverse = np.linalg.inv(frequencies)
    gradient = (
        -(inverse.T - 0.5 / rows[:, np.newaxis] - 0.5 / columns[np.newaxis, :]) / k
    )
    return Distance(
        float(value), multinomial_delta_variance(frequencies, gradient, n_sites)
    )


def distance_matrix(
    alignment: Mapping[str, np.ndarray],
    k: int,
    kind: DistanceKind = DistanceKind.JUKES_CANTOR,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Every pair's distance and variance, over the taxa in sorted order.

    Parameters
    ----------
    alignment : Mapping[str, np.ndarray]
        Taxon name to its states, as
        :func:`snakes_and_ladders.sim.simulate.simulate_alignment` returns.
    k : int
        Number of states.
    kind : DistanceKind
        Which estimator to apply.

    Returns
    -------
    tuple[list[str], np.ndarray, np.ndarray]
        The taxon names in sorted order, the symmetric distance matrix with a
        zero diagonal, and the matrix of variances, both of shape
        ``(n, n)``.

    Raises
    ------
    ValueError
        If fewer than two taxa are given, or any pair is refused by the
        estimator.
    """
    names = sorted(alignment)
    if len(names) < 2:
        msg = f"a distance matrix needs at least 2 taxa, got {len(names)}"
        raise ValueError(msg)
    estimator = (
        jukes_cantor_distance if kind is DistanceKind.JUKES_CANTOR else log_det_distance
    )
    distances = np.zeros((len(names), len(names)))
    variances = np.zeros((len(names), len(names)))
    for row, first in enumerate(names):
        for column in range(row + 1, len(names)):
            estimate = estimator(alignment[first], alignment[names[column]], k)
            distances[row, column] = distances[column, row] = estimate.value
            variances[row, column] = variances[column, row] = estimate.variance
    return names, distances, variances


def tree_distances(tau: Node) -> tuple[list[str], np.ndarray]:
    """The additive distances a tree's branch lengths imply, leaf to leaf.

    The expected value every estimator above converges to, and the input on
    which neighbor joining is exact. Path length is the sum of branch
    lengths between two leaves; the root contributes nothing of its own, so
    a rooted binary tree and its unrooted form give the same matrix.

    Parameters
    ----------
    tau : Node
        A tree with a branch length on every non-root node.

    Returns
    -------
    tuple[list[str], np.ndarray]
        The leaf names in sorted order and the ``(n, n)`` matrix of path
        lengths.

    Raises
    ------
    ValueError
        If a non-root node carries no branch length.
    """
    height: dict[str, float] = {}
    lineage: dict[str, list[str]] = {}

    def visit(node: Node, above: list[str], depth: float) -> None:
        if node is not tau:
            if node.branch_length is None:
                msg = f"non-root node {node.name!r} has no branch_length"
                raise ValueError(msg)
            depth += node.branch_length
        height[node.name] = depth
        path = [*above, node.name]
        if node.is_leaf:
            lineage[node.name] = path
        for child in node.children:
            visit(child, path, depth)

    visit(tau, [], 0.0)
    names = sorted(lineage)
    matrix = np.zeros((len(names), len(names)))
    for row, first in enumerate(names):
        for column in range(row + 1, len(names)):
            second = names[column]
            shared = 0
            while (
                shared < min(len(lineage[first]), len(lineage[second]))
                and lineage[first][shared] == lineage[second][shared]
            ):
                shared += 1
            ancestor = height[lineage[first][shared - 1]]
            length = height[first] + height[second] - 2.0 * ancestor
            matrix[row, column] = matrix[column, row] = length
    return names, matrix
