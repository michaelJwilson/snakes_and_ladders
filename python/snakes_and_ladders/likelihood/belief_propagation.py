"""Sum-product belief propagation on a Potts MRF, and the Bethe free energy.

Exact on a tree, approximate on a loop. Both statements are load-bearing here:
the tree case carries the correctness claim, because it is the only regime
where equality against :func:`snakes_and_ladders.likelihood.potts.enumerate_potts` is the
right assertion; the loopy case is *reported* against
:func:`snakes_and_ladders.likelihood.potts.strip_log_partition` as a measured deviation,
because asserting agreement there would assert something false and asserting
only that it ran would be coverage theatre.

Messages are held in the log domain and normalized every sweep. A Potts
coupling of ``J = 2`` on a 4x4 lattice puts ``exp(32)`` inside a product of
sixteen messages, and the linear-domain recursion loses it; the log domain
costs a ``logsumexp`` per edge and does not.

Non-convergence raises. A Bethe free energy computed from messages that never
settled is not an estimate of anything, and the caller cannot tell it from one
that is -- `docs/CLAUDE.md`'s rule that an unstable number is not a
measurement, applied where the instability is in the algorithm rather than in
the machine.

See ``docs/tex/main.tex``, "Belief Propagation and the Bethe Approximation"
(Yedidia, Freeman & Weiss for the free energy; Mezard & Montanari ch. 14;
Koller & Friedman ch. 11).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from snakes_and_ladders.numerics import logsumexp
from snakes_and_ladders.sim.graph import PottsGraph

# Parallel (flooding) updates with damping. Sequential schedules converge on
# more graphs, but the order then decides the answer, and a fixture that
# reproduces only under one traversal is a worse oracle than one that refuses.
DEFAULT_DAMPING = 0.5
# 1e-12 rather than 1e-10 because the beliefs, not the messages, are what a
# caller reads: on the tree fixture 1e-10 leaves the single-site marginals
# 4.1e-11 from exact, outside `likelihood/CLAUDE.md`'s 1e-11 relative bound,
# and 1e-12 brings them to 2.8e-13 for eight more sweeps out of forty-five.
DEFAULT_TOLERANCE = 1e-12
DEFAULT_MAX_ITERATIONS = 2_000


class ConvergenceError(RuntimeError):
    """Raised when the messages did not settle within the iteration cap.

    Carries the residual so a caller tuning damping or the cap can see how
    far off it was, rather than only that it failed.
    """

    def __init__(self, iterations: int, residual: float, tolerance: float) -> None:
        super().__init__(
            f"belief propagation did not converge in {iterations} iterations: "
            f"largest message change {residual:.3e} against a tolerance of "
            f"{tolerance:.3e}"
        )
        self.iterations = iterations
        self.residual = residual
        self.tolerance = tolerance


@dataclass(frozen=True)
class BeliefPropagationResult:
    """Beliefs and the Bethe estimate, from a run that converged.

    Parameters
    ----------
    single_site : np.ndarray
        Belief ``b_i(a)``, shape ``(n_nodes, n_states)``. Exact marginals on
        a tree; an approximation on a loopy graph.
    pairwise : np.ndarray
        Belief ``b_ij(a, b)`` per edge, in the graph's own edge order, shape
        ``(n_edges, n_states, n_states)``.
    bethe_log_partition : float
        ``-F_Bethe``, which equals ``log Z`` exactly on a tree.
    iterations : int
        Sweeps taken to reach the tolerance. Reported beside every deviation
        so a run that only just converged is visible rather than inferred.
    residual : float
        The largest message change on the final sweep.
    """

    single_site: np.ndarray
    pairwise: np.ndarray
    bethe_log_partition: float
    iterations: int
    residual: float


def belief_propagation(
    graph: PottsGraph,
    field: np.ndarray,
    *,
    damping: float = DEFAULT_DAMPING,
    tolerance: float = DEFAULT_TOLERANCE,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
) -> BeliefPropagationResult:
    """Run sum-product to convergence, or refuse.

    Parameters
    ----------
    graph : PottsGraph
        The graph. Each undirected edge carries two messages.
    field : np.ndarray
        External field ``h``, shape ``(n_states,)``. The node potential is
        ``exp(h)``, matching :func:`snakes_and_ladders.likelihood.potts.log_weights`.
    damping : float
        Fraction of the previous message retained, in the log domain, so
        ``0`` is undamped and ``1`` never updates. Damping in logs is a
        geometric mean of the two messages rather than an arithmetic one,
        which keeps a damped message normalizable without a second pass.
    tolerance : float
        Converged when the largest absolute change in any log message falls
        to or below this.
    max_iterations : int
        Sweeps before refusing.

    Returns
    -------
    BeliefPropagationResult
        The beliefs and the Bethe estimate.

    Raises
    ------
    ValueError
        If ``damping`` is outside ``[0, 1)``. At ``1`` the messages never
        move and every graph would "converge" at iteration one with the
        residual identically zero -- a silent wrong answer rather than a
        loud one.
    ConvergenceError
        If the residual is still above ``tolerance`` at ``max_iterations``.
    """
    if not 0.0 <= damping < 1.0:
        msg = (
            f"damping must be in [0, 1), got {damping}: at 1 no message ever "
            "updates and the residual is zero on the first sweep"
        )
        raise ValueError(msg)

    n_states = int(field.shape[0])
    n_edges = len(graph.edges)
    if n_edges == 0:
        return _disconnected(graph, field)

    # Directed edge `2 * e` runs first -> second, `2 * e + 1` runs the other
    # way. `source`/`target` index them, and `reverse` finds a directed edge's
    # opposite, which is the one message excluded from its own recomputation.
    source = np.empty(2 * n_edges, dtype=np.int64)
    target = np.empty(2 * n_edges, dtype=np.int64)
    for position, (first, second) in enumerate(graph.edges):
        source[2 * position], target[2 * position] = first, second
        source[2 * position + 1], target[2 * position + 1] = second, first
    reverse = np.arange(2 * n_edges) ^ 1
    edge_coupling = np.repeat(np.asarray(graph.coupling, dtype=float), 2)

    # log psi_ij(a, b) for each directed edge: J on the diagonal, 0 elsewhere.
    identity = np.eye(n_states)
    log_edge = edge_coupling[:, np.newaxis, np.newaxis] * identity[np.newaxis, :, :]

    messages = np.zeros((2 * n_edges, n_states))
    residual = np.inf
    taken = 0
    for iteration in range(1, max_iterations + 1):
        inbox = np.zeros((graph.n_nodes, n_states))
        np.add.at(inbox, target, messages)

        # Everything the source node knows except what the target told it.
        exclusive = inbox[source] - messages[reverse]
        proposal = logsumexp((field + exclusive)[:, :, np.newaxis] + log_edge, axis=1)
        proposal = proposal - logsumexp(proposal, axis=1)[:, np.newaxis]

        updated = damping * messages + (1.0 - damping) * proposal
        updated = updated - logsumexp(updated, axis=1)[:, np.newaxis]
        residual = float(np.abs(updated - messages).max())
        messages = updated
        if residual <= tolerance:
            taken = iteration
            break
    else:
        raise ConvergenceError(max_iterations, residual, tolerance)

    inbox = np.zeros((graph.n_nodes, n_states))
    np.add.at(inbox, target, messages)
    log_single = field + inbox
    log_single = log_single - logsumexp(log_single, axis=1)[:, np.newaxis]
    single_site = np.exp(log_single)

    exclusive = inbox[source] - messages[reverse]
    forward = np.arange(0, 2 * n_edges, 2)
    backward = forward + 1
    log_pair = (
        log_edge[forward]
        + (field + exclusive[forward])[:, :, np.newaxis]
        + (field + exclusive[backward])[:, np.newaxis, :]
    )
    log_pair = (
        log_pair
        - logsumexp(log_pair.reshape(n_edges, -1), axis=1)[:, np.newaxis, np.newaxis]
    )
    pairwise = np.exp(log_pair)

    return BeliefPropagationResult(
        single_site=single_site,
        pairwise=pairwise,
        bethe_log_partition=-_bethe_free_energy(
            graph, field, single_site, pairwise, log_pair, log_single
        ),
        iterations=taken,
        residual=residual,
    )


def _bethe_free_energy(
    graph: PottsGraph,
    field: np.ndarray,
    single_site: np.ndarray,
    pairwise: np.ndarray,
    log_pair: np.ndarray,
    log_single: np.ndarray,
) -> float:
    """``F_Bethe``, in the form whose negation is ``log Z`` on a tree.

        F = sum_(ij) sum_ab b_ij(ab) log[ b_ij(ab) / (psi_ij(ab) psi_i(a) psi_j(b)) ]
          - sum_i (d_i - 1) sum_a b_i(a) log[ b_i(a) / psi_i(a) ]

    The degree correction is what makes it a *Bethe* free energy rather than a
    naive mean field: a node of degree ``d`` appears in ``d`` edge terms, so
    ``d - 1`` copies of its own entropy are subtracted back off. On a tree the
    two terms telescope and ``-F`` is exactly ``log Z``; the tree test is what
    pins that, and it would fail on a sign or an off-by-one here.
    """
    identity = np.eye(field.shape[0])
    degree = np.zeros(graph.n_nodes)
    edge_total = 0.0
    for position, ((first, second), coupling) in enumerate(graph.weighted_edges()):
        degree[first] += 1
        degree[second] += 1
        log_psi = coupling * identity + field[:, np.newaxis] + field[np.newaxis, :]
        edge_total += float((pairwise[position] * (log_pair[position] - log_psi)).sum())

    node_total = float(
        ((degree - 1.0) * (single_site * (log_single - field)).sum(axis=1)).sum()
    )
    return edge_total - node_total


def _disconnected(graph: PottsGraph, field: np.ndarray) -> BeliefPropagationResult:
    """The edgeless case, where every site is independent and BP is trivial.

    Handled separately rather than as a degenerate loop: with no edges there
    are no messages, so the residual is undefined and the general path would
    have to invent a convergence claim. ``log Z`` is ``n_nodes`` copies of the
    single-site normalizer, exactly.
    """
    log_single = field - logsumexp(field[np.newaxis, :], axis=1)[0]
    single_site = np.tile(np.exp(log_single), (graph.n_nodes, 1))
    return BeliefPropagationResult(
        single_site=single_site,
        pairwise=np.zeros((0, field.shape[0], field.shape[0])),
        bethe_log_partition=float(
            graph.n_nodes * logsumexp(field[np.newaxis, :], axis=1)[0]
        ),
        iterations=0,
        residual=0.0,
    )
