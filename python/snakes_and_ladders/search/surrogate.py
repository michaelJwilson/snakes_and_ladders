"""Learned surrogates for the two searches: the data they are fitted to, and the seam that ranks by them (issue #308).

``snakes_and_ladders.learn.surrogate`` sees tensors; this module, which may import both
halves, turns topologies and lattices into them. :func:`tree_examples` scores
every topology it is handed by an exact target -- the maximized
log-likelihood, or the log-likelihood at given lengths -- and pairs it with
the features and tokens of ``likelihood.features``. :func:`lattice_examples`
does the same for ``log Z`` or the ground-state energy of a lattice.
:class:`LearnedTreeSurrogate` wraps a fit as the ``Surrogate`` the search's
``infer(..., surrogate=)`` ranks a neighbourhood by, and
:func:`shuffle_children` is the symmetry a tree has that its spelling does
not: the transform training data is augmented by, and the one every feature
is checked invariant under.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np
import torch

from snakes_and_ladders.bound import Bound
from snakes_and_ladders.learn.surrogate import CalibratedBound, Examples, Fitted
from snakes_and_ladders.likelihood.features import (
    lattice_features,
    lattice_tokens,
    tree_adjacency,
    tree_features,
    tree_tokens,
)
from snakes_and_ladders.likelihood.pruning_torch import branch_order, log_likelihood
from snakes_and_ladders.search.infer import Model, score_topology
from snakes_and_ladders.search.topology import Topology
from snakes_and_ladders.sim.graph import PottsGraph
from snakes_and_ladders.sim.tree import Node

TreeTarget = Callable[[Topology, Mapping[str, np.ndarray]], float]
LatticeTarget = Callable[[PottsGraph, np.ndarray], float]


def maximized_target(k: int, model: Model = Model.JC) -> TreeTarget:
    """The maximized log-likelihood of a topology: a fit per call."""

    def target(topology: Topology, alignment: Mapping[str, np.ndarray]) -> float:
        return score_topology(topology, alignment, k, model)

    return target


def fixed_length_target(k: int, pi: np.ndarray, length: float) -> TreeTarget:
    """The log-likelihood of a topology with every branch at ``length``: one pruning pass per call."""

    def target(topology: Topology, alignment: Mapping[str, np.ndarray]) -> float:
        lengths = torch.full(
            (len(branch_order(topology)),), length, dtype=torch.float64
        )
        return float(log_likelihood(topology, k, pi, alignment, lengths))

    return target


def tree_node_tokens(
    topology: Topology, alignment: Mapping[str, np.ndarray], k: int
) -> tuple[torch.Tensor, np.ndarray]:
    """Per-node tokens and the tree's edges, for the graph model: a node carries its branch's token, the root zeros."""
    names, adjacency = tree_adjacency(topology)
    by_branch = dict(
        zip(branch_order(topology), tree_tokens(topology, alignment, k), strict=True)
    )
    width = next(iter(by_branch.values())).shape[0]
    rows = torch.stack(
        [by_branch.get(name, torch.zeros(width, dtype=torch.float64)) for name in names]
    )
    return rows, adjacency


def tree_examples(
    alignments: Sequence[Mapping[str, np.ndarray]],
    topologies: Sequence[Sequence[Topology]],
    k: int,
    pi: np.ndarray,
    target: TreeTarget,
) -> Examples:
    """One example per (alignment, topology), grouped by alignment, scored by ``target``.

    The offset is the plug-in lower bound, so a model learns the gap between
    it and the target and falls back to the bound where it has learned
    nothing.
    """
    features, targets, groups, tokens, adjacency = [], [], [], [], []
    for group, (alignment, candidates) in enumerate(
        zip(alignments, topologies, strict=True)
    ):
        for topology in candidates:
            features.append(tree_features(topology, alignment, k, pi))
            targets.append(target(topology, alignment))
            groups.append(group)
            rows, edges = tree_node_tokens(topology, alignment, k)
            tokens.append(rows)
            adjacency.append(edges)
    return Examples(
        torch.stack(features),
        torch.tensor(targets, dtype=torch.float64),
        np.array(groups, dtype=np.int64),
        tuple(tokens),
        tuple(adjacency),
        offset=plug_in_offset(torch.stack(features)),
    )


def plug_in_offset(features: torch.Tensor) -> torch.Tensor:
    """The plug-in bound in total, from its per-site feature and the site count."""
    return features[:, 0] * features[:, 6]


def mean_field_offset(features: torch.Tensor) -> torch.Tensor:
    """The mean-field bound in total, from its per-node feature and the node count."""
    return features[:, 0] * features[:, 4]


def lattice_examples(
    graphs: Sequence[PottsGraph],
    fields: Sequence[np.ndarray],
    target: LatticeTarget,
    *,
    groups: Sequence[int] | None = None,
) -> Examples:
    """One example per (graph, field), scored by ``target``, grouped as given or one group each.

    The offset is the mean-field lower bound on ``log Z``; a caller learning
    the ground-state energy instead replaces it with :meth:`Examples.with_offset`.
    """
    features, targets, tokens, adjacency = [], [], [], []
    for graph, field in zip(graphs, fields, strict=True):
        features.append(lattice_features(graph, field))
        targets.append(target(graph, field))
        tokens.append(lattice_tokens(graph))
        adjacency.append(np.array(graph.edges, dtype=np.int64).reshape(-1, 2))
    return Examples(
        torch.stack(features),
        torch.tensor(targets, dtype=torch.float64),
        np.arange(len(graphs)) if groups is None else np.array(groups, dtype=np.int64),
        tuple(tokens),
        tuple(adjacency),
        offset=mean_field_offset(torch.stack(features)),
    )


def shuffle_children(topology: Topology, rng: np.random.Generator) -> Topology:
    """The same tree with every node's children in a random order: a symmetry of the problem, not of its spelling."""
    if topology.is_leaf:
        return topology
    children = [shuffle_children(child, rng) for child in topology.children]
    order = rng.permutation(len(children))
    return Node(
        topology.name, topology.branch_length, tuple(children[i] for i in order)
    )


class LearnedTreeSurrogate:
    """A fit, or a calibrated bound made from one, as the ``Surrogate`` a search ranks by."""

    def __init__(
        self, fitted: Fitted | CalibratedBound, k: int, pi: np.ndarray
    ) -> None:
        self.fitted = fitted
        self.k = k
        self.pi = np.asarray(pi, dtype=float)

    @property
    def kind(self) -> Bound:
        return self.fitted.kind

    def __call__(self, structure: object, data: object) -> torch.Tensor:
        if not isinstance(structure, Node) or not isinstance(data, Mapping):
            msg = "a tree surrogate takes a topology and an alignment"
            raise TypeError(msg)
        rows, edges = tree_node_tokens(structure, data, self.k)
        features = tree_features(structure, data, self.k, self.pi)[None, :]
        examples = Examples(
            features,
            torch.zeros(1, dtype=torch.float64),
            np.zeros(1, dtype=np.int64),
            (rows,),
            (edges,),
            offset=plug_in_offset(features),
        )
        return self.fitted.predict(examples)[0]


__all__ = [
    "LatticeTarget",
    "LearnedTreeSurrogate",
    "TreeTarget",
    "fixed_length_target",
    "lattice_examples",
    "maximized_target",
    "mean_field_offset",
    "plug_in_offset",
    "shuffle_children",
    "tree_examples",
    "tree_node_tokens",
]
