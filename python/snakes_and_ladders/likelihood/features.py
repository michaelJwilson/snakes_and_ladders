"""Features a learned surrogate reads: cheap, model-aware, and invariant to how a structure is spelt (issue #308).

A surrogate that learns from raw structure has to rediscover what the
analytic bounds already know, so the features here *are* those bounds and
their by-products, at a cost the search can pay per candidate: for a
topology, the plug-in and parsimony bounds of
:mod:`snakes_and_ladders.likelihood.surrogate`, the Fitch score, and the
least-squares residual; for a lattice, the mean-field and spanning-tree
bounds, the Bethe free energy, and the energy alpha-expansion reaches.

Two shapes, for two families of model. :func:`tree_features` and
:func:`lattice_features` are one vector per structure, for a linear model or
an MLP. :func:`tree_tokens` and :func:`lattice_tokens` are one row per
branch or per node, for the set, attention and graph models that must not
care in what order the rows come. Every feature is a function of the
unrooted topology, so it is unchanged by re-rooting or by swapping a node's
children; ``tests/regression/likelihood/test_surrogate.py`` pins that.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch

from snakes_and_ladders.likelihood.belief_propagation import (
    ConvergenceError,
    belief_propagation,
)
from snakes_and_ladders.likelihood.surrogate import (
    ParsimonyUpperBound,
    PlugInLikelihood,
    jc_distances,
    least_squares_lengths,
    least_squares_residual,
    mean_field_log_partition,
    site_fitch_scores,
    spanning_tree_log_partition,
)
from snakes_and_ladders.search.alpha_expansion import alpha_expansion
from snakes_and_ladders.search.topology import Topology, branch_splits
from snakes_and_ladders.sim.graph import PottsGraph

TREE_FEATURE_NAMES = (
    "plug_in_per_site",
    "parsimony_bound_per_site",
    "fitch_per_site",
    "least_squares_residual",
    "total_length",
    "n_taxa",
    "n_sites",
)
TREE_TOKEN_NAMES = ("length", "balance", "across", "within")
LATTICE_FEATURE_NAMES = (
    "mean_field_per_node",
    "spanning_tree_per_node",
    "bethe_per_node",
    "expansion_energy_per_node",
    "n_nodes",
    "edges_per_node",
    "mean_coupling",
    "field_spread",
)
LATTICE_TOKEN_NAMES = ("degree", "coupling_sum", "coupling_abs_sum")


def tree_features(
    topology: Topology, alignment: Mapping[str, np.ndarray], k: int, pi: np.ndarray
) -> torch.Tensor:
    """One vector per topology, in ``TREE_FEATURE_NAMES`` order, per-site where it scales with sites."""
    n_sites = int(next(iter(alignment.values())).shape[0])
    distances = jc_distances(alignment, k)
    lengths = least_squares_lengths(topology, distances)
    plug_in = float(PlugInLikelihood(k, pi)(topology, alignment))
    bound = float(ParsimonyUpperBound(k, pi)(topology, alignment))
    fitch = float(site_fitch_scores(topology, alignment).sum())
    return torch.tensor(
        [
            plug_in / n_sites,
            bound / n_sites,
            fitch / n_sites,
            least_squares_residual(topology, distances, lengths),
            float(lengths.sum()),
            float(len(alignment)),
            float(n_sites),
        ],
        dtype=torch.float64,
    )


def tree_tokens(
    topology: Topology, alignment: Mapping[str, np.ndarray], k: int
) -> torch.Tensor:
    """One row per branch, in ``TREE_TOKEN_NAMES`` order: its least-squares length, how evenly its split divides the taxa, and the mean distance across and within the split."""
    distances = jc_distances(alignment, k)
    lengths = least_squares_lengths(topology, distances)
    names = sorted(alignment)
    rows = []
    for column, split in enumerate(branch_splits(topology)):
        inside = {name for name in names if name in split}
        across = [
            distance for pair, distance in distances.items() if len(pair & inside) == 1
        ]
        within = [
            distance for pair, distance in distances.items() if len(pair & inside) != 1
        ]
        rows.append(
            [
                float(lengths[column]),
                min(len(inside), len(names) - len(inside)) / len(names),
                float(np.mean(across)) if across else 0.0,
                float(np.mean(within)) if within else 0.0,
            ]
        )
    return torch.tensor(rows, dtype=torch.float64)


def lattice_features(graph: PottsGraph, field: np.ndarray) -> torch.Tensor:
    """One vector per lattice, in ``LATTICE_FEATURE_NAMES`` order, per node where it scales with size.

    The Bethe free energy is belief propagation's estimate of ``log Z``,
    exact on a tree and neither bound elsewhere; where sum-product does not
    converge the mean-field value stands in, so the feature is always
    defined. The expansion energy is what alpha-expansion from the uniform
    labelling reaches, an upper bound on the ground-state energy.
    """
    field = np.asarray(field, dtype=float)
    field_tensor = torch.as_tensor(field)
    mean_field = float(mean_field_log_partition(graph, field_tensor))
    spanning = float(spanning_tree_log_partition(graph, field_tensor))
    try:
        bethe = float(belief_propagation(graph, field).bethe_log_partition)
    except ConvergenceError:
        bethe = mean_field
    per_node_field = np.tile(field, (graph.n_nodes, 1))
    expansion = alpha_expansion(
        graph,
        per_node_field,
        field.shape[0],
        start=np.zeros(graph.n_nodes, dtype=np.int64),
    )
    n_nodes = float(graph.n_nodes)
    return torch.tensor(
        [
            mean_field / n_nodes,
            spanning / n_nodes,
            bethe / n_nodes,
            expansion.energy / n_nodes,
            n_nodes,
            len(graph.edges) / n_nodes,
            float(np.mean(graph.coupling)),
            float(np.max(field) - np.min(field)),
        ],
        dtype=torch.float64,
    )


def lattice_tokens(graph: PottsGraph) -> torch.Tensor:
    """One row per node, in ``LATTICE_TOKEN_NAMES`` order: its degree and the sum and absolute sum of its couplings."""
    rows = np.zeros((graph.n_nodes, 3))
    for (first, second), coupling in graph.weighted_edges():
        for node in (first, second):
            rows[node, 0] += 1.0
            rows[node, 1] += coupling
            rows[node, 2] += abs(coupling)
    return torch.as_tensor(rows)


def tree_adjacency(topology: Topology) -> tuple[list[str], np.ndarray]:
    """Node names and the undirected edge list of the topology, by index into the names.

    The rooted spelling's edges, which is the unrooted tree's edges plus a
    degree-two root; a graph model that sums over neighbours reads the same
    structure whichever spelling it is handed, up to that root.
    """
    names: list[str] = []
    pairs: list[tuple[int, int]] = []

    def visit(node: Topology) -> int:
        index = len(names)
        names.append(node.name)
        for child in node.children:
            pairs.append((index, visit(child)))
        return index

    visit(topology)
    return names, np.array(pairs, dtype=np.int64).reshape(-1, 2)


__all__ = [
    "LATTICE_FEATURE_NAMES",
    "LATTICE_TOKEN_NAMES",
    "TREE_FEATURE_NAMES",
    "TREE_TOKEN_NAMES",
    "lattice_features",
    "lattice_tokens",
    "tree_adjacency",
    "tree_features",
    "tree_tokens",
]
