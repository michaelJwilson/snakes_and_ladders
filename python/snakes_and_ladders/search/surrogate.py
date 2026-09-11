"""Learned surrogates for the two searches: the data they are fitted to, and the seam that ranks by them (issue #308).

``snakes_and_ladders.learn.surrogate`` sees tensors; this module, which may import both
halves, turns topologies and lattices into them. :func:`tree_examples` scores
every topology it is handed by an exact target -- the maximized
log-likelihood, or the log-likelihood at given lengths -- and pairs it with
the features and tokens of ``likelihood.features``. :func:`lattice_examples`
does the same for ``log Z`` --- :func:`enumerated_log_partition_target` and
:func:`strip_log_partition_target` --- or for the energy the discrete
solvers reach, :func:`ground_state_target`, which is the target at the sizes
no exact ``log Z`` reaches.
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
    LATTICE_FEATURE_NAMES,
    lattice_features,
    lattice_tokens,
    tree_adjacency,
    tree_features,
    tree_tokens,
)
from snakes_and_ladders.likelihood.potts import enumerate_potts, strip_log_partition
from snakes_and_ladders.likelihood.pruning_torch import branch_order, log_likelihood
from snakes_and_ladders.likelihood.surrogate import decoupled_ground_energy
from snakes_and_ladders.search.alpha_expansion import alpha_expansion
from snakes_and_ladders.search.backend import Backend
from snakes_and_ladders.search.infer import Model, score_topology
from snakes_and_ladders.search.topology import Topology
from snakes_and_ladders.sim.graph import PottsGraph
from snakes_and_ladders.sim.potts import SpatioOnlyParams, spatio_only_field
from snakes_and_ladders.sim.tree import Node

TreeTarget = Callable[[Topology, Mapping[str, np.ndarray]], float]
LatticeTarget = Callable[[PottsGraph, np.ndarray], float]

#: Feature positions by name, so an offset reads the vector rather than an
#: index that a reordering would silently move.
_FEATURE = {name: index for index, name in enumerate(LATTICE_FEATURE_NAMES)}


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
    return (
        features[:, _FEATURE["mean_field_per_node"]] * features[:, _FEATURE["n_nodes"]]
    )


def ground_state_offset(
    graphs: Sequence[PottsGraph], fields: Sequence[np.ndarray]
) -> torch.Tensor:
    """One decoupled energy bound per instance: what a ground-state target is predicted above.

    Read from the instances rather than from a feature vector, since the
    bound needs the couplings and the field and no summary of them carries
    it. Learning the gap above it is what makes a poor fit fall back to a
    bound rather than to nothing, exactly as
    :func:`mean_field_offset` does for ``log Z``.
    """
    return torch.stack(
        [
            decoupled_ground_energy(graph, torch.as_tensor(np.asarray(field, float)))
            for graph, field in zip(graphs, fields, strict=True)
        ]
    )


def lattice_instances(
    params: SpatioOnlyParams, n_groups: int, per_group: int
) -> tuple[list[PottsGraph], list[np.ndarray], list[int]]:
    """``n_groups * per_group`` instances of the declared family, grouped by draw.

    The lattice, the coupling and ``alpha`` are the fixture's; what varies is
    the per-site covariate, redrawn as ``lognormal(0, s)`` for ``s`` the
    spread of the fixture's own log sizes. One rule for all three rungs,
    including the one whose sizes are written out, so an instance at 9 sites
    and one at 5,041 differ in size and geometry and in nothing else.

    A group is a draw seed, which is what :func:`split_by_group` assigns
    whole and what :func:`argmax_agreement` ranks within; members of one
    group are independent draws, so ranking inside a group is the same
    question as ranking across one.
    """
    log_sigma = float(np.std(np.log(params.sizes)))
    graphs, fields, groups = [], [], []
    for group in range(n_groups):
        for member in range(per_group):
            rng = np.random.default_rng([params.seed, group, member])
            sizes = np.exp(rng.normal(0.0, log_sigma, params.graph.n_nodes))
            graphs.append(params.graph)
            fields.append(spatio_only_field(params.alpha, sizes))
            groups.append(group)
    return graphs, fields, groups


def enumerated_log_partition_target() -> LatticeTarget:
    """``log Z`` by enumeration: exact, and affordable only where ``q ** N`` is."""

    def target(graph: PottsGraph, field: np.ndarray) -> float:
        return enumerate_potts(graph, field).log_partition

    return target


def strip_log_partition_target() -> LatticeTarget:
    """``log Z`` by the column transfer matrix: exact on an open 2-D strip past enumeration."""

    def target(graph: PottsGraph, field: np.ndarray) -> float:
        shape, boundary = graph.shape, graph.boundary
        if shape is None or boundary is None or len(shape) != 2:
            msg = f"the strip oracle needs a 2-D lattice, got shape {shape}"
            raise ValueError(msg)
        couplings = np.asarray(graph.coupling, dtype=float)
        if couplings.size and float(couplings.max() - couplings.min()) > 0.0:
            msg = "the strip oracle takes one uniform coupling"
            raise ValueError(msg)
        return strip_log_partition(
            (shape[0], shape[1]), boundary, float(couplings[0]), field
        )

    return target


def ground_state_target(
    n_states: int, *, backend: Backend = Backend.PYTHON
) -> LatticeTarget:
    """The energy alpha-expansion reaches from the per-site data optimum: one discrete solve per call.

    The release rung's target, because it is what the discrete solvers
    compute at 5,041 sites where no exact ``log Z`` is available. It is an
    upper bound on the ground-state energy, not the ground state, so a test
    reporting it states the gap against
    :func:`~snakes_and_ladders.likelihood.surrogate.decoupled_ground_energy`
    below it.

    ``backend`` reaches :func:`~snakes_and_ladders.search.alpha_expansion.expand`'s
    minimum cut and nothing else. Both settle on the same labelling, the cut
    being deterministic, and on ``spatio_only/release.yaml`` both return
    ``-10454.156290057308``.
    """

    def target(graph: PottsGraph, field: np.ndarray) -> float:
        return alpha_expansion(graph, field, n_states, backend=backend).energy

    return target


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
        tokens.append(lattice_tokens(graph, field))
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
    "enumerated_log_partition_target",
    "fixed_length_target",
    "ground_state_offset",
    "ground_state_target",
    "lattice_examples",
    "lattice_instances",
    "maximized_target",
    "mean_field_offset",
    "plug_in_offset",
    "shuffle_children",
    "strip_log_partition_target",
    "tree_examples",
    "tree_node_tokens",
]
