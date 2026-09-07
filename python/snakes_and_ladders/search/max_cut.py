"""Max-Cut, which is the antiferromagnetic Ising ground state under another name.

Maximizing the weight of edges whose endpoints differ *is* minimizing
``-sum_(ij) J_ij [s_i == s_j]`` with every ``J`` negative, so no new model is
introduced here --- only a different reading of the one
:func:`snakes_and_ladders.likelihood.potts.log_weights` already defines. That is also
exactly where :mod:`snakes_and_ladders.search.maxflow` stops: a negative coupling is not
submodular, and this is the NP-hard side of the boundary that module refuses
to cross.

**What this adds is a bound that is not enumeration.** Every discrete claim in
this repository rests on exhaustive enumeration and therefore stops at about
twenty sites --- except alpha expansion's factor of 2 (issue #207) and this.
The Goemans-Williamson rounding of a semidefinite relaxation is
0.87856-approximate *in expectation*, and the relaxation's own optimum upper
bounds the true maximum cut, so a run can be certified where enumeration
cannot reach.

**The certificate here is weaker than the theorem, and that is stated rather
than glossed.** Goemans-Williamson assumes the semidefinite program is solved
to optimality. This repository carries no SDP solver and root ``CLAUDE.md``
requires a dependency be justified before it lands, so the relaxation is
solved by Burer-Monteiro factorization in ``torch``: the Gram matrix is
written as ``V V^T`` with unit-norm rows and maximized by gradient ascent.
That converges to the SDP optimum at sufficient rank (Boumal, Voroninski &
Bandeira 2016) but is solved *approximately* here, so the value returned may
sit **below** the true relaxation optimum. A ratio measured against it is
therefore optimistic, and the honest statement is that it certifies the
rounding rather than the relaxation. Where enumeration reaches, the realized
ratio is measured against the true optimum instead, and that is the number to
trust.

See Goemans & Williamson (1995); Burer & Monteiro (2003).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from snakes_and_ladders.enumeration import refuse_oversized
from snakes_and_ladders.sim.graph import PottsGraph

# The Goemans-Williamson constant: the expected ratio of the rounded cut to
# the relaxation's optimum, and hence to the true maximum cut.
GOEMANS_WILLIAMSON_RATIO = 0.87856


@dataclass(frozen=True)
class MaxCutResult:
    """A cut, and the certificate that comes with it.

    Parameters
    ----------
    assignment : np.ndarray
        Side per node, 0 or 1.
    value : float
        Total weight of edges whose endpoints differ.
    relaxation : float
        The value the semidefinite relaxation reached. An upper bound on the
        true maximum cut *when the relaxation is solved to optimality*; see
        the module docstring for why that qualifier is load-bearing here.
    ratio : float
        ``value / relaxation``. Compared against
        :data:`GOEMANS_WILLIAMSON_RATIO`, this is a computable certificate at
        sizes where the true optimum is unknown.
    """

    assignment: np.ndarray
    value: float
    relaxation: float
    ratio: float


def cut_value(graph: PottsGraph, assignment: np.ndarray) -> float:
    """Total weight of the edges this assignment separates."""
    total = 0.0
    for (first, second), weight in graph.weighted_edges():
        if assignment[first] != assignment[second]:
            total += abs(weight)
    return total


def enumerate_max_cut(
    graph: PottsGraph, *, max_nodes: int = 20
) -> tuple[np.ndarray, float]:
    """The true maximum cut, by trying every assignment.

    Exponential and deliberately so --- this is the reference the rounded
    solution is measured against wherever it is affordable. One side is fixed
    to break the global two-fold symmetry, which halves the work and is exact
    rather than a heuristic: complementing an assignment gives the same cut.

    Raises
    ------
    ValueError
        Above ``max_nodes``, where ``2 ** n`` stops being affordable. Refused
        rather than attempted, so the failure is a stated limit rather than a
        test killed for memory.
    """
    refuse_oversized(
        2**graph.n_nodes,
        what=f"2**{graph.n_nodes} assignments",
        limit=2**max_nodes,
    )

    best_value = -1.0
    best = np.zeros(graph.n_nodes, dtype=np.int64)
    for code in range(2 ** (graph.n_nodes - 1)):
        assignment = np.array(
            [(code >> bit) & 1 for bit in range(graph.n_nodes - 1)] + [0],
            dtype=np.int64,
        )
        value = cut_value(graph, assignment)
        if value > best_value:
            best_value, best = value, assignment
    return best, best_value


def goemans_williamson(
    graph: PottsGraph,
    seed: int,
    *,
    rank: int | None = None,
    iterations: int = 600,
    learning_rate: float = 0.05,
    roundings: int = 200,
) -> MaxCutResult:
    """Relax to unit vectors, maximize, then round with a random hyperplane.

    The relaxation replaces each ``s_i in {-1, +1}`` with a unit vector
    ``v_i``, so ``s_i s_j`` becomes ``<v_i, v_j>`` and the cut becomes
    ``sum w_ij (1 - <v_i, v_j>) / 2``. Rounding draws a random hyperplane and
    takes the side each vector falls on; the 0.87856 constant is the expected
    ratio that produces.

    Parameters
    ----------
    graph : PottsGraph
        Edge weights are ``abs(coupling)``, so a graph written with the
        antiferromagnetic sign this problem corresponds to and one written
        with positive weights give the same cut.
    seed : int
        Seeds both the initial vectors and the rounding hyperplanes.
    rank : int | None
        Dimension of the vectors. ``None`` uses ``ceil(sqrt(2 n))``, the rank
        at which the factorization provably admits the relaxation's optimum.
    iterations, learning_rate : int, float
        Gradient ascent budget. Reported through ``relaxation`` rather than
        hidden: an under-solved relaxation makes the certificate weaker, and
        the caller should be able to see which one they got.
    roundings : int
        Hyperplanes drawn. The best is kept, which is standard and is why the
        realized ratio typically beats the expected one by a wide margin.
    """
    weights = torch.zeros((graph.n_nodes, graph.n_nodes), dtype=torch.float64)
    for (first, second), coupling in graph.weighted_edges():
        weights[first, second] += abs(coupling)
        weights[second, first] += abs(coupling)

    dimension = rank if rank is not None else int(np.ceil(np.sqrt(2 * graph.n_nodes)))
    generator = torch.Generator().manual_seed(seed)
    vectors = torch.randn(
        (graph.n_nodes, dimension), generator=generator, dtype=torch.float64
    )
    vectors = vectors / vectors.norm(dim=1, keepdim=True)
    vectors.requires_grad_(True)

    optimizer = torch.optim.Adam([vectors], lr=learning_rate)
    for _ in range(iterations):
        optimizer.zero_grad()
        gram = vectors @ vectors.T
        # Maximize the relaxed cut, so minimize its negation.
        objective = -0.5 * (weights * (1.0 - gram)).sum() / 2.0
        objective.backward()
        optimizer.step()
        with torch.no_grad():
            vectors /= vectors.norm(dim=1, keepdim=True)

    with torch.no_grad():
        gram = vectors @ vectors.T
        relaxation = float(0.5 * (weights * (1.0 - gram)).sum() / 2.0)

        best_value, best = -1.0, np.zeros(graph.n_nodes, dtype=np.int64)
        for _ in range(roundings):
            hyperplane = torch.randn(
                dimension, generator=generator, dtype=torch.float64
            )
            assignment = (vectors @ hyperplane > 0).numpy().astype(np.int64)
            value = cut_value(graph, assignment)
            if value > best_value:
                best_value, best = value, assignment

    return MaxCutResult(
        assignment=best,
        value=best_value,
        relaxation=relaxation,
        ratio=best_value / relaxation if relaxation > 0.0 else 1.0,
    )


def complete_bipartite(
    first_size: int, second_size: int, weight: float = 1.0
) -> PottsGraph:
    """A graph whose maximum cut is every edge, known without solving anything.

    Every edge joins the two parts, so the assignment that separates them cuts
    all of them and no assignment can do better. It is the one instance here
    whose answer is exact at any size, which makes it the check that a solver
    is not merely self-consistent.
    """
    edges = tuple(
        (first, first_size + second)
        for first in range(first_size)
        for second in range(second_size)
    )
    return PottsGraph(
        n_nodes=first_size + second_size,
        edges=edges,
        coupling=(weight,) * len(edges),
    )
