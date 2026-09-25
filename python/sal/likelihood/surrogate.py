"""Surrogates for an exact evaluation: bounds with proofs, and the seam a learned one fills (issue #308).

A surrogate is a function of a discrete structure and the data, at fixed
continuous parameters, standing in for an evaluation the search pays for. Its
:class:`~sal.bound.Bound` is part of the object: a **lower**
bound, an **upper** bound, or a **point** prediction. An analytic bound
carries a proof in the textbook's derivations appendix and is certified by
:func:`~sal.bound.certify`, which refuses a bound violated once
against the exact value; a learned one is calibrated to a stated coverage.
A tree surrogate is called at a structure and the data, takes no continuous
argument to differentiate in, and returns a ``float`` (issue #1011). Of the
lattice bounds, the mean-field and spanning-tree bounds are differentiable in
the field and the couplings through ``torch``; the three ``O(N + E)`` bounds
take no derivative, take arrays and return a ``float``. The module imports no
``torch``: the differentiable bounds, and the one pruning evaluation the
plug-in bound makes, import it at the call.

Two hard evaluations, two families of surrogate:

* **Trees.** The maximized log-likelihood of a topology costs a fit of
  tens of pruning evaluations. :class:`PlugInLikelihood` is a lower bound
  on it -- one pruning evaluation at branch lengths solved by least squares
  from pairwise distances, feasible lengths being all a lower bound needs
  (``eq:plugin-bound``). :class:`ParsimonyUpperBound` is an upper bound on
  the likelihood at *every* branch length, hence on the maximum: under
  Jukes--Cantor the site likelihood is multilinear in one variable per
  branch, so its maximum over lengths sits at a vertex where every branch
  is zero or infinite, and at a vertex it is at most ``pi(x_1) k^{-F_s}``
  for the site's Fitch score ``F_s`` (``eq:parsimony-bound``). One Fitch
  pass, no parameters, and both bounds apply to a subtree as to the tree.
  The entrywise limit of ``P(t)`` gives a weaker bound this one dominates;
  :func:`prune_with_matrices` is kept because the vertex argument is checked
  by enumerating the vertices with it.
* **Lattice Potts.** ``log Z`` is what enumeration cannot reach.
  :func:`mean_field_log_partition` is the naive mean-field lower bound,
  Gibbs' inequality at the fixed point of a product distribution
  (``eq:mean-field-bound``). :func:`spanning_tree_log_partition` is the
  Jensen upper bound over a distribution on spanning trees, each tree's
  partition function exact and the parameters split so they average to the
  original (``eq:spanning-tree-bound``). From any pair of bounds on
  ``log Z`` at inverse temperature ``beta``, :func:`ground_state_energy_bounds`
  brackets the ground-state energy (``eq:ground-state-bounds``).

The surrogates that are learned live in :mod:`sal.learn.surrogate`;
the features they read are assembled in :mod:`sal.likelihood.features`.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from sal.bound import Bound, Surrogate
from sal.likelihood.pruning_common import leaf_indicator_array
from sal.numerics import logsumexp
from sal.sim.graph import PottsGraph
from sal.sim.potts import site_field
from sal.sim.topology import Topology, branch_splits
from sal.sim.tree import Node

if TYPE_CHECKING:
    import torch

# --- trees ----------------------------------------------------------------


def jc_distances(
    alignment: Mapping[str, np.ndarray], k: int
) -> dict[frozenset[str], float]:
    """Pairwise Jukes--Cantor distances, ``-(k-1)/k log(1 - k p/(k-1))``, saturating where undefined."""
    names = sorted(alignment)
    limit = (k - 1) / k
    distances: dict[frozenset[str], float] = {}
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            mismatch = float(np.mean(alignment[first] != alignment[second]))
            inside = 1.0 - mismatch / limit
            distances[frozenset((first, second))] = (
                -limit * float(np.log(inside))
                if inside > 1e-6
                else -limit * float(np.log(1e-6))
            )
    return distances


def least_squares_lengths(
    topology: Topology,
    distances: Mapping[frozenset[str], float],
    *,
    n_iterations: int = 20,
) -> np.ndarray:
    """Non-negative branch lengths whose path lengths best fit ``distances``, in ``branch_order``.

    The unconstrained least-squares solution clamped to non-negative, then
    ``n_iterations`` of projected gradient on the convex objective. Any
    non-negative lengths are a feasible point of the fit, which is what
    makes :class:`PlugInLikelihood` a bound; the refinement only tightens it.

    NumPy, since no derivative is taken through the fit: the lengths are a
    function of float distances alone. ``numpy.linalg.lstsq`` (LAPACK
    ``gelsd``) returns one answer bitwise on repeated calls, where
    ``torch.linalg.lstsq``'s CPU default ``gelsy`` returned up to 42
    ulp-different answers in 1,000 calls on one 10x7 system (issue #1026).
    """
    splits = branch_splits(topology)
    pairs = list(distances)
    design = np.zeros((len(pairs), len(splits)))
    for row, pair in enumerate(pairs):
        first, second = tuple(pair)
        for column, split in enumerate(splits):
            if (first in split) != (second in split):
                design[row, column] = 1.0
    target = np.array([distances[pair] for pair in pairs], dtype=np.float64)
    lengths = np.maximum(np.linalg.lstsq(design, target, rcond=None)[0], 1e-6)
    step = 1.0 / float(np.linalg.norm(design, ord=2) ** 2)
    for _ in range(n_iterations):
        gradient = design.T @ (design @ lengths - target)
        lengths = np.maximum(lengths - step * gradient, 1e-6)
    return lengths


def least_squares_residual(
    topology: Topology, distances: Mapping[frozenset[str], float], lengths: np.ndarray
) -> float:
    """``sum (path length - distance)^2`` at the given lengths: how tree-like the distances are."""
    splits = branch_splits(topology)
    total = 0.0
    for pair, distance in distances.items():
        first, second = tuple(pair)
        path = sum(
            float(lengths[column])
            for column, split in enumerate(splits)
            if (first in split) != (second in split)
        )
        total += (path - distance) ** 2
    return total


class PlugInLikelihood(Surrogate):
    """Lower bound on the maximized log-likelihood: one pruning evaluation at least-squares lengths.

    Any feasible lengths give a value at most the maximum, and one pruning
    evaluation costs a fiftieth of a fit. Exact where the least-squares
    lengths are the optimum.
    """

    kind = Bound.LOWER

    def __init__(self, k: int, pi: np.ndarray) -> None:
        self.k = k
        self.pi = np.asarray(pi, dtype=float)

    def lengths(
        self, topology: Topology, alignment: Mapping[str, np.ndarray]
    ) -> np.ndarray:
        """The feasible branch lengths the bound is evaluated at, in ``branch_order``."""
        return least_squares_lengths(topology, jc_distances(alignment, self.k))

    def __call__(self, structure: object, data: object) -> float:
        topology, alignment = _tree_arguments(structure, data)
        # `likelihood.torch.pruning.log_likelihood` takes tensors, since the fits
        # differentiate through it: the lengths cross into it here, once.
        import torch

        from sal.likelihood.torch.pruning import log_likelihood

        return float(
            log_likelihood(
                topology,
                self.k,
                self.pi,
                alignment,
                torch.from_numpy(self.lengths(topology, alignment)),
            )
        )


def prune_with_matrices(
    tau: Node,
    k: int,
    pi: np.ndarray,
    alignment: Mapping[str, np.ndarray],
    matrices: Mapping[str, np.ndarray],
) -> float:
    """Pruning with an arbitrary matrix per branch, summed over sites in the log domain.

    The recursion of ``eq:pruning`` with ``P(t)`` replaced by whatever
    ``matrices`` holds for each child, which is what the elementwise bound
    needs and what the fitted evaluators cannot do.
    """
    n_sites = next(iter(alignment.values())).shape[0]

    def partial(node: Node) -> np.ndarray:
        if node.is_leaf:
            return leaf_indicator_array(alignment[node.name], n_sites, k)
        table = np.ones((n_sites, k))
        for child in node.children:
            table = table * (partial(child) @ np.asarray(matrices[child.name]).T)
        return table

    site = partial(tau) @ np.asarray(pi, dtype=float)
    with np.errstate(divide="ignore"):
        return float(np.sum(np.log(site)))


def site_fitch_scores(tau: Node, alignment: Mapping[str, np.ndarray]) -> np.ndarray:
    """Fitch's minimum change count per site, the recursion of ``fitch_score`` kept per column."""
    changes: np.ndarray | None = None

    def visit(node: Node) -> np.ndarray:
        nonlocal changes
        if node.is_leaf:
            return (1 << np.asarray(alignment[node.name]).astype(np.int64)).astype(
                np.int64
            )
        combined = visit(node.children[0])
        for child in node.children[1:]:
            mask = visit(child)
            intersection = combined & mask
            empty = intersection == 0
            changes = empty.astype(np.int64) if changes is None else changes + empty
            combined = np.where(empty, combined | mask, intersection)
        return combined

    combined = visit(tau)
    if changes is None:
        return np.zeros(combined.shape[0], dtype=np.int64)
    return changes


class ParsimonyUpperBound(Surrogate):
    """Upper bound on the Jukes--Cantor log-likelihood at every branch length, hence on its maximum.

    ``P(t) = a I + (1 - a) J / k`` with ``a = exp(-k t / (k - 1))``, so the
    site likelihood is multilinear in the ``a`` of the branches and its
    maximum over the box ``[0, 1]^B`` is at a vertex. At a vertex every
    branch either forbids a change or forgets the parent's state at cost
    ``1/k``, and rooting at the first leaf, the site likelihood is
    ``pi(x_1) k^{-c}`` with ``c`` at least the site's Fitch score. Summed
    over sites: ``sum_s log pi(x_{1,s}) - F(tau) log k``. One Fitch pass.
    Exact only where every site fits with no change.
    """

    kind = Bound.UPPER

    def __init__(self, k: int, pi: np.ndarray) -> None:
        self.k = k
        self.pi = np.asarray(pi, dtype=float)

    def __call__(self, structure: object, data: object) -> float:
        topology, alignment = _tree_arguments(structure, data)
        first = np.asarray(alignment[sorted(alignment)[0]], dtype=np.int64)
        changes = int(site_fitch_scores(topology, alignment).sum())
        return float(np.sum(np.log(self.pi[first]))) - changes * float(np.log(self.k))


def _tree_arguments(
    structure: object, data: object
) -> tuple[Topology, Mapping[str, np.ndarray]]:
    if not isinstance(structure, Node) or not isinstance(data, Mapping):
        msg = "a tree surrogate takes a topology and an alignment"
        raise TypeError(msg)
    return structure, data


# --- lattice Potts --------------------------------------------------------


def _edge_tensors(
    graph: PottsGraph, couplings: torch.Tensor | None
) -> tuple[np.ndarray, torch.Tensor]:
    import torch

    pairs = graph.edge_index
    if couplings is None:
        # `.copy()`: the graph's cached array is unwritable, and a tensor
        # sharing it would reach back into the cache on any in-place op.
        couplings = torch.as_tensor(graph.edge_coupling.copy())
    if couplings.shape != (pairs.shape[0],):
        msg = f"one coupling per edge: expected {pairs.shape[0]}, got {tuple(couplings.shape)}"
        raise ValueError(msg)
    return pairs, couplings


def site_rows(field: torch.Tensor, n_nodes: int) -> torch.Tensor:
    """``field`` as one row per site, widened in ``torch`` so the gradient survives.

    :func:`sal.sim.potts.site_field` is the same widening over
    NumPy, and the two agree on what a shape means; a bound differentiable in
    its field cannot route through NumPy to get there, which is the whole of
    why this exists beside it.

    Raises
    ------
    ValueError
        If ``field`` is neither ``(n_states,)`` nor ``(n_nodes, n_states)``,
        refused for the reason ``site_field`` refuses it: a shared field on a
        graph whose node count equals its state count would otherwise be read
        as per-site and score a different model.
    """
    import torch

    field = torch.as_tensor(field, dtype=torch.float64)
    if field.dim() == 1:
        return field.expand(n_nodes, field.shape[0])
    if field.dim() == 2 and field.shape[0] == n_nodes:
        return field
    msg = (
        f"field has shape {tuple(field.shape)}; expected (n_states,) or "
        f"({n_nodes}, n_states)"
    )
    raise ValueError(msg)


def mean_field_log_partition(
    graph: PottsGraph,
    field: torch.Tensor,
    *,
    couplings: torch.Tensor | None = None,
    n_iterations: int = 200,
) -> torch.Tensor:
    """Naive mean field: ``log Z >= E_q[-E] + H(q)`` for the product ``q`` at its fixed point.

    Gibbs' inequality holds for every product distribution, so the bound is
    valid at any iterate and tightest at the fixed point; the iteration is
    unrolled, which is what makes the value differentiable in ``field`` and
    ``couplings`` (the latter defaulting to the graph's own). ``field`` is
    shared or per site, widened once by :func:`site_rows`.
    """
    import torch

    pairs, couplings = _edge_tensors(graph, couplings)
    rows = site_rows(field, graph.n_nodes)
    n_states = rows.shape[1]
    q = torch.full((graph.n_nodes, n_states), 1.0 / n_states, dtype=torch.float64)
    first = torch.as_tensor(pairs[:, 0])
    second = torch.as_tensor(pairs[:, 1])
    for _ in range(n_iterations):
        local = rows.clone()
        local = local.index_add(0, first, couplings[:, None] * q[second])
        local = local.index_add(0, second, couplings[:, None] * q[first])
        q = torch.softmax(local, dim=1)
    energy = (q * rows).sum() + (couplings * (q[first] * q[second]).sum(dim=1)).sum()
    entropy = -(q * torch.log(q.clamp_min(1e-300))).sum()
    return energy + entropy


def _spanning_trees_covering_every_edge(graph: PottsGraph) -> list[list[int]]:
    """One spanning tree per edge, containing it: a distribution whose support covers the graph."""
    n_edges = len(graph.edges)
    adjacency: list[list[tuple[int, int]]] = [[] for _ in range(graph.n_nodes)]
    for index, (first, second) in enumerate(graph.edges):
        adjacency[first].append((second, index))
        adjacency[second].append((first, index))
    trees = []
    for forced in range(n_edges):
        first, second = graph.edges[forced]
        parent = np.full(graph.n_nodes, -1, dtype=np.int64)
        seen = np.zeros(graph.n_nodes, dtype=bool)
        seen[[first, second]] = True
        chosen = [forced]
        frontier = [first, second]
        while frontier:
            node = frontier.pop(0)
            for neighbour, index in adjacency[node]:
                if not seen[neighbour]:
                    seen[neighbour] = True
                    parent[neighbour] = node
                    chosen.append(index)
                    frontier.append(neighbour)
        if not seen.all():
            msg = "the graph is disconnected; a spanning tree does not exist"
            raise ValueError(msg)
        trees.append(chosen)
    return trees


def tree_log_partition(
    n_nodes: int,
    tree_edges: Sequence[tuple[int, int]],
    couplings: torch.Tensor,
    field: torch.Tensor,
) -> torch.Tensor:
    """Exact ``log Z`` of a Potts model on a tree by one leaf-to-root pass, in the log domain.

    ``field`` is shared or per site, widened once by :func:`site_rows`.
    """
    import torch

    adjacency: list[list[tuple[int, torch.Tensor]]] = [[] for _ in range(n_nodes)]
    for (first, second), coupling in zip(tree_edges, couplings, strict=True):
        adjacency[first].append((second, coupling))
        adjacency[second].append((first, coupling))
    rows = site_rows(field, n_nodes)
    identity = torch.eye(rows.shape[1], dtype=torch.float64)

    def message(node: int, parent: int) -> torch.Tensor:
        total = rows[node].clone()
        for child, coupling in adjacency[node]:
            if child == parent:
                continue
            incoming = message(child, node)  # over child's states
            total = total + torch.logsumexp(
                coupling * identity + incoming[None, :], dim=1
            )
        return total

    return torch.logsumexp(message(0, -1), dim=0)


def spanning_tree_log_partition(
    graph: PottsGraph,
    field: torch.Tensor,
    *,
    couplings: torch.Tensor | None = None,
) -> torch.Tensor:
    """Jensen's upper bound on ``log Z`` over a distribution on spanning trees.

    ``log Z`` is convex in the parameters, so for trees ``T`` with weights
    ``rho(T)`` and parameters ``theta(T)`` supported on ``T`` whose weighted
    mean is ``theta``, ``log Z(theta) <= sum_T rho(T) log Z(theta(T))``. The
    trees are one per edge, uniform, each built to contain its edge; an edge
    in a fraction ``rho_e`` of them carries ``J_e / rho_e`` on each tree it is
    in, so the means agree. Every tree term is exact. Tight only where the
    graph is a tree.
    """
    import torch

    pairs, couplings = _edge_tensors(graph, couplings)
    field = site_rows(field, graph.n_nodes)
    trees = _spanning_trees_covering_every_edge(graph)
    appearances = torch.zeros(len(graph.edges), dtype=torch.float64)
    for tree in trees:
        appearances[tree] += 1.0
    appearances = appearances / len(trees)
    total = torch.zeros((), dtype=torch.float64)
    for tree in trees:
        tree_pairs = [tuple(pairs[index]) for index in tree]
        tree_couplings = torch.stack(
            [couplings[index] / appearances[index] for index in tree]
        )
        total = total + tree_log_partition(
            graph.n_nodes, tree_pairs, tree_couplings, field
        )
    return total / len(trees)


def decoupled_ground_energy(graph: PottsGraph, field: npt.ArrayLike) -> float:
    """``-sum_n max_m h[n, m] - sum_e max(J_e, 0)``: no configuration's energy lies below it.

    The energy is a sum of site terms and edge terms, so its minimum is at
    least the sum of each term's own minimum; the labelling attaining every
    minimum at once need not exist, which is the slack. ``O(N + E)``, which
    is what makes it the bracket's lower end at sizes
    :func:`spanning_tree_log_partition` does not reach
    (``eq:decoupled-energy-bound``). No derivative is taken through it, so
    it is NumPy: ``field`` is shared or per site, widened by
    :func:`~sal.sim.potts.site_field` (issue #1011).
    """
    rows = site_field(np.asarray(field, dtype=np.float64), graph.n_nodes)
    return float(-rows.max(axis=1).sum() - np.maximum(graph.edge_coupling, 0.0).sum())


def decoupled_log_partition(graph: PottsGraph, field: npt.ArrayLike) -> float:
    """``sum_n logsumexp_m h[n, m]``: ``log Z`` with every coupling dropped.

    A lower bound wherever every coupling is non-negative, since dropping a
    non-negative term from each configuration's exponent lowers every
    summand of ``Z``. ``O(N)``, and looser than mean field, which recovers
    it at zero coupling. NumPy, as :func:`decoupled_ground_energy` is.
    """
    rows = site_field(np.asarray(field, dtype=np.float64), graph.n_nodes)
    return float(logsumexp(rows, axis=1).sum())


def saturated_log_partition(graph: PottsGraph, field: npt.ArrayLike) -> float:
    """``N log q - E_decoupled``: an upper bound on ``log Z`` at any size.

    ``Z <= q^N exp(-E_min)`` and :func:`decoupled_ground_energy` bounds
    ``E_min`` below, so the two compose. ``O(N + E)`` against
    :func:`spanning_tree_log_partition`'s one exact tree pass per edge,
    which is why this is the upper end of the bracket on a lattice of
    thousands of sites and the spanning-tree bound the tighter one where it
    runs (``eq:saturated-bound``). NumPy, as :func:`decoupled_ground_energy`
    is.
    """
    rows = site_field(np.asarray(field, dtype=np.float64), graph.n_nodes)
    n_states = float(rows.shape[1])
    return graph.n_nodes * float(np.log(n_states)) - decoupled_ground_energy(
        graph, field
    )


@dataclass(frozen=True)
class EnergyBounds:
    """A bracket on the ground-state energy.

    Parameters
    ----------
    lower : float
        ``-U / beta``, from the upper bound ``U`` on ``log Z(beta)``.
    upper : float
        ``(N log q - L) / beta``, from the lower bound ``L``.
    """

    lower: float
    upper: float

    def __iter__(self) -> Iterator[float]:
        """``(lower, upper)``: the order callers unpack."""
        yield from (self.lower, self.upper)


def ground_state_energy_bounds(
    graph: PottsGraph, field: np.ndarray, beta: float
) -> EnergyBounds:
    """``[-U/beta, (N log q - L)/beta]`` brackets the ground-state energy, from ``L <= log Z(beta) <= U``.

    ``Z(beta) >= exp(-beta E_min)`` gives the lower end and
    ``Z(beta) <= q^N exp(-beta E_min)`` the upper, with ``log Z`` at inverse
    temperature ``beta`` bracketed by the mean-field and spanning-tree
    bounds on the scaled model. Tighter as ``beta`` grows, until the bounds
    on ``log Z`` themselves loosen.
    """
    if beta <= 0.0:
        msg = f"beta must be positive, got {beta}"
        raise ValueError(msg)
    import torch

    scaled_field = site_rows(
        torch.as_tensor(np.asarray(field, dtype=float) * beta), graph.n_nodes
    )
    scaled_couplings = torch.as_tensor(graph.edge_coupling * beta)
    lower_log_z = float(
        mean_field_log_partition(graph, scaled_field, couplings=scaled_couplings)
    )
    upper_log_z = float(
        spanning_tree_log_partition(graph, scaled_field, couplings=scaled_couplings)
    )
    q = int(scaled_field.shape[1])
    return EnergyBounds(
        -upper_log_z / beta, (graph.n_nodes * np.log(q) - lower_log_z) / beta
    )


__all__ = [
    "EnergyBounds",
    "ParsimonyUpperBound",
    "PlugInLikelihood",
    "decoupled_ground_energy",
    "decoupled_log_partition",
    "ground_state_energy_bounds",
    "jc_distances",
    "least_squares_lengths",
    "least_squares_residual",
    "mean_field_log_partition",
    "prune_with_matrices",
    "saturated_log_partition",
    "site_fitch_scores",
    "site_rows",
    "spanning_tree_log_partition",
    "tree_log_partition",
]
