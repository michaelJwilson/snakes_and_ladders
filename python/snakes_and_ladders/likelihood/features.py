"""Features a learned surrogate reads: cheap, model-aware, and invariant to how a structure is spelt (issue #308).

A surrogate that learns from raw structure has to rediscover what the
analytic bounds already know, so the features here *are* those bounds and
their by-products, at a cost the search can pay per candidate: for a
topology, the plug-in and parsimony bounds of
:mod:`snakes_and_ladders.likelihood.surrogate`, the Fitch score, and the
least-squares residual; for a lattice, the mean-field, decoupled and
saturated bounds on ``log Z``.

**Every lattice feature costs ``O(N + E)``, and that is a constraint rather
than a preference** (issue #365). The spanning-tree bound is one exact tree
pass per edge and does not reach the 5,041 sites of
``spatio_only/release.yaml``; belief propagation takes a shared field only;
and the energy alpha-expansion reaches *is* the target a release-size
surrogate is fitted to, so reading it as a feature would predict the target
from itself. All three were features here before that fixture existed, and
none is a feature now. Each remains where it is computed, and the tests
bracket every fitted value between the bounds that reach the size in hand.

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

from snakes_and_ladders.likelihood.surrogate import (
    ParsimonyUpperBound,
    PlugInLikelihood,
    decoupled_log_partition,
    jc_distances,
    least_squares_lengths,
    least_squares_residual,
    mean_field_log_partition,
    saturated_log_partition,
    site_fitch_scores,
    site_rows,
)
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
    "decoupled_per_node",
    "saturated_per_node",
    "n_nodes",
    "edges_per_node",
    "mean_coupling",
    "field_spread",
    "field_spread_sd",
)
#: Unrolled mean-field iterations the lattice feature takes. The bound holds
#: at every iterate, so this trades tightness for cost: measured against 400
#: iterations on the three ``spatio_only`` instances, 80 leaves 2.1e-10 at
#: 5,041 sites and nothing at 9 or 72, at two fifths of the 200 the bound's
#: own default takes.
MEAN_FIELD_ITERATIONS = 80

LATTICE_TOKEN_NAMES = (
    "degree",
    "coupling_sum",
    "coupling_abs_sum",
    "field_max",
    "field_min",
    "field_mean",
)


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

    Three bounds on ``log Z`` and five statistics of the instance. ``field``
    is shared or per site, widened once by
    :func:`~snakes_and_ladders.likelihood.surrogate.site_rows`; the last two
    entries are the mean and the standard deviation across sites of the
    field's per-site range, which are what tell a model how hard the
    covariate pushes and how unevenly. A shared field makes the second of
    them zero, which is the statement that there is no covariate.
    """
    rows = site_rows(torch.as_tensor(np.asarray(field, dtype=float)), graph.n_nodes)
    n_nodes = float(graph.n_nodes)
    span = (rows.max(dim=1).values - rows.min(dim=1).values).numpy()
    return torch.tensor(
        [
            float(
                mean_field_log_partition(
                    graph, rows, n_iterations=MEAN_FIELD_ITERATIONS
                )
            )
            / n_nodes,
            float(decoupled_log_partition(graph, rows)) / n_nodes,
            float(saturated_log_partition(graph, rows)) / n_nodes,
            n_nodes,
            len(graph.edges) / n_nodes,
            float(np.mean(graph.coupling)),
            float(np.mean(span)),
            float(np.std(span)),
        ],
        dtype=torch.float64,
    )


def lattice_tokens(graph: PottsGraph, field: np.ndarray) -> torch.Tensor:
    """One row per node, in ``LATTICE_TOKEN_NAMES`` order: its degree, the sum and absolute sum of its couplings, and three summaries of its own field.

    The field enters as ``max``, ``min`` and ``mean`` over classes rather
    than as the row itself, so a token is the same width at three classes
    and at ten and one fitted model reads both. Under
    ``h[n, m] = alpha[m] log(size_n / size_bar)`` the row is that scalar
    covariate times a vector fixed across the lattice, so the three
    summaries carry it exactly.
    """
    rows = site_rows(torch.as_tensor(np.asarray(field, dtype=float)), graph.n_nodes)
    tokens = np.zeros((graph.n_nodes, len(LATTICE_TOKEN_NAMES)))
    for (first, second), coupling in graph.weighted_edges():
        for node in (first, second):
            tokens[node, 0] += 1.0
            tokens[node, 1] += coupling
            tokens[node, 2] += abs(coupling)
    tokens[:, 3] = rows.max(dim=1).values.numpy()
    tokens[:, 4] = rows.min(dim=1).values.numpy()
    tokens[:, 5] = rows.mean(dim=1).numpy()
    return torch.as_tensor(tokens)


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
    "MEAN_FIELD_ITERATIONS",
    "TREE_FEATURE_NAMES",
    "TREE_TOKEN_NAMES",
    "lattice_features",
    "lattice_tokens",
    "tree_adjacency",
    "tree_features",
    "tree_tokens",
]
