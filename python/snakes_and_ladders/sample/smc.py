"""Sequential Monte Carlo over topologies: trees built by coalescence, weighted by partial likelihoods.

Every other sampler over topologies here *mixes*: :func:`~snakes_and_ladders.sample.gibbs.anneal_topology`
and :func:`~snakes_and_ladders.sample.tempered.tempered_topologies` walk the
neighbours of a whole tree. This one *constructs* (Bouchard-Côté,
Sankararaman & Jordan 2012): a particle is a forest over the taxa, a step
coalesces two of its subtrees into one, and the weight the step earns is the
ratio of the new subtree's likelihood to its two children's --- the partial
likelihood Felsenstein pruning already computes at every internal node. After
``n - 1`` steps every particle is a rooted binary tree, and the product of the
steps' mean weights is an unbiased estimate of the evidence the target is
normalized by, which no mixing sampler here can report without a second
method.

**What the target is, stated exactly.** The proposal picks a uniformly random
pair at every step, so every one of the ``prod_m C(m, 2)`` coalescence
sequences is equally likely under it, and the sampler targets the law over
sequences proportional to the likelihood of the tree each one builds, with the
declared branch length on every child edge. That law is enumerable at five
taxa (180 sequences) and is what `tests/regression/sample/test_smc.py` holds
the particles and the evidence to. It is *not* the flat prior over unrooted
topologies with every edge at the declared length: a rooted coalescence tree
projects to an unrooted one with the root's two edges joined into one of twice
the length, and distinct sequences build the same tree in different orders.
Both are corrections a later ticket can carry as weights; this module says what
it samples rather than approximating what it does not (issue #824).

The cost unit is partial-likelihood evaluations, one per subtree the cache has
not seen, which is the unit the search spends and the one every number here is
stated in.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from snakes_and_ladders.likelihood.pruning import log_likelihood
from snakes_and_ladders.numerics import logsumexp
from snakes_and_ladders.sim.topology import leaf_bipartitions
from snakes_and_ladders.sim.tree import Node


@dataclass(frozen=True)
class SmcTopologies:
    """What one run returns: the particles, their weights, the evidence and the cost.

    Parameters
    ----------
    trees : tuple[Node, ...]
        One rooted binary tree per particle, every child edge at
        ``branch_length``; :func:`~snakes_and_ladders.sim.topology.leaf_bipartitions`
        is its unrooted projection.
    log_weights : np.ndarray
        Normalized log weights of the final particles, ``(n_particles,)``,
        summing to one in probability.
    log_evidence : float
        ``log`` of the mean likelihood over coalescence sequences, the
        normalizer of the target this module states, estimated by the product
        of the steps' mean weights times the leaves' own likelihood.
    ess : np.ndarray
        Effective sample size ``1 / sum w^2`` of the normalized weights at
        each of the ``n - 1`` steps, before resampling.
    evaluations : int
        Partial-likelihood evaluations spent: subtrees scored once each.
    branch_length : float
        The declared length on every child edge.
    """

    trees: tuple[Node, ...]
    log_weights: np.ndarray
    log_evidence: float
    ess: np.ndarray
    evaluations: int
    branch_length: float


class _Partials:
    """Log partial likelihoods per subtree, scored once each and keyed by structure."""

    def __init__(
        self, alignment: Mapping[str, np.ndarray], k: int, pi: np.ndarray
    ) -> None:
        self._alignment = {
            name: np.asarray(states) for name, states in alignment.items()
        }
        self._k = k
        self._pi = np.asarray(pi, dtype=np.float64)
        self._cache: dict[str, float] = {}
        self.evaluations = 0

    def key(self, tree: Node) -> str:
        if tree.is_leaf:
            return tree.name
        return "(" + ",".join(sorted(self.key(child) for child in tree.children)) + ")"

    def leaves(self, tree: Node) -> list[str]:
        if tree.is_leaf:
            return [tree.name]
        return [leaf for child in tree.children for leaf in self.leaves(child)]

    def score(self, tree: Node) -> float:
        """``log sum_x pi(x) P(subtree data | root state x)``, summed over sites."""
        key = self.key(tree)
        found = self._cache.get(key)
        if found is not None:
            return found
        self.evaluations += 1
        if tree.is_leaf:
            value = float(np.sum(np.log(self._pi[self._alignment[tree.name]])))
        else:
            restricted = {name: self._alignment[name] for name in self.leaves(tree)}
            value = log_likelihood(tree, self._k, self._pi, restricted)
        self._cache[key] = value
        return value

    def recomputed(self, tree: Node) -> float:
        """The score without the cache, for the pin that the cache returns what it would."""
        if tree.is_leaf:
            return float(np.sum(np.log(self._pi[self._alignment[tree.name]])))
        restricted = {name: self._alignment[name] for name in self.leaves(tree)}
        return log_likelihood(tree, self._k, self._pi, restricted)


def _systematic(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Systematic resampling: one uniform, ``n`` evenly spaced positions."""
    n = weights.size
    positions = (rng.random() + np.arange(n)) / n
    return np.asarray(np.searchsorted(np.cumsum(weights), positions).clip(0, n - 1))


def smc_topologies(
    alignment: Mapping[str, np.ndarray],
    k: int,
    rng: np.random.Generator,
    n_particles: int,
    *,
    branch_length: float,
    pi: Sequence[float] | np.ndarray | None = None,
) -> SmcTopologies:
    """Build ``n_particles`` trees by coalescence, weighted by partial likelihoods.

    Parameters
    ----------
    alignment : Mapping[str, np.ndarray]
        Observed states per taxon, each ``(n_sites,)``.
    k : int
        Number of states.
    rng : np.random.Generator
        Draws every pair choice and every resampling; passed in, never seeded
        here (`sim/CLAUDE.md`).
    n_particles : int
        Forests carried. Resampled systematically before every step after the
        first, so the final weights are the last step's.
    branch_length : float
        The length of every child edge a coalescence creates, the fixed-length
        reward :class:`~snakes_and_ladders.learn.tree.TreeEnvironment` scores
        under ``KNOWN``.
    pi : Sequence[float] | np.ndarray | None
        Root distribution; uniform over ``k`` states when omitted.

    Returns
    -------
    SmcTopologies

    Raises
    ------
    ValueError
        If fewer than three taxa or one particle are given.
    """
    names = sorted(alignment)
    if len(names) < 3:
        msg = f"need at least 3 taxa to build a topology, got {len(names)}"
        raise ValueError(msg)
    if n_particles < 1:
        msg = f"n_particles must be >= 1, got {n_particles}"
        raise ValueError(msg)
    root_pi = np.full(k, 1.0 / k) if pi is None else np.asarray(pi, dtype=np.float64)
    partials = _Partials(alignment, k, root_pi)

    forests: list[list[Node]] = [
        [Node(name, branch_length) for name in names] for _ in range(n_particles)
    ]
    log_w = np.zeros(n_particles)
    log_evidence = float(sum(partials.score(Node(name, None)) for name in names))
    ess_per_step: list[float] = []
    counter = 0
    for size in range(len(names), 1, -1):
        if size < len(names):
            kept = _systematic(np.exp(log_w), rng)
            forests = [list(forests[index]) for index in kept]
        n_pairs = size * (size - 1) // 2
        picks = rng.integers(n_pairs, size=n_particles)
        increments = np.empty(n_particles)
        for index, forest in enumerate(forests):
            first, second = _pair(size, int(picks[index]))
            left, right = forest[first], forest[second]
            counter += 1
            joined = Node(
                f"c{counter}",
                branch_length,
                (
                    Node(left.name, branch_length, left.children),
                    Node(right.name, branch_length, right.children),
                ),
            )
            increments[index] = (
                partials.score(joined) - partials.score(left) - partials.score(right)
            )
            forest[first] = joined
            del forest[second]
        log_mean = float(logsumexp(increments, axis=0)) - np.log(n_particles)
        log_evidence += log_mean
        log_w = increments - float(logsumexp(increments, axis=0))
        ess_per_step.append(1.0 / float(np.sum(np.exp(2.0 * log_w))))

    trees = tuple(Node(forest[0].name, None, forest[0].children) for forest in forests)
    return SmcTopologies(
        trees=trees,
        log_weights=log_w,
        log_evidence=log_evidence,
        ess=np.array(ess_per_step),
        evaluations=partials.evaluations,
        branch_length=branch_length,
    )


def _pair(size: int, index: int) -> tuple[int, int]:
    """The ``index``-th unordered pair of ``range(size)``, in lexicographic order."""
    first = 0
    remaining = index
    while remaining >= size - first - 1:
        remaining -= size - first - 1
        first += 1
    return first, first + 1 + remaining


def topology_weights(result: SmcTopologies) -> dict[frozenset[frozenset[str]], float]:
    """The particles' weight on each unrooted topology, by its bipartitions."""
    weights: dict[frozenset[frozenset[str]], float] = {}
    for tree, log_weight in zip(result.trees, result.log_weights, strict=True):
        key = leaf_bipartitions(tree)
        weights[key] = weights.get(key, 0.0) + float(np.exp(log_weight))
    return weights


def split_support(result: SmcTopologies) -> dict[frozenset[str], float]:
    """The particles' weight on each split: a posterior split support for free."""
    support: dict[frozenset[str], float] = {}
    for key, weight in topology_weights(result).items():
        for split in key:
            support[split] = support.get(split, 0.0) + weight
    return support


__all__ = ["SmcTopologies", "smc_topologies", "split_support", "topology_weights"]
