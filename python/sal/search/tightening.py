"""A dual bound on a Potts ground state, and the plaquettes that tighten it.

Issue #696. The exact routes reach a long way --- minimum cut is exact for two
states with attractive couplings, and alpha expansion carries the standard
factor-two guarantee on a metric prior --- and what defeats both is a
**non-submodular** coupling: the frustrated triangular lattice and the planted
glass. There the question "how far from optimal is this labelling" has no
answer in this repository, and that is what a dual bound is for.

**The bound is valid by construction, which is the whole design.** Split the
energy into subproblems whose shares sum to the original,
``E(x) = sum_s E_s(x_s)``. Then
``max_x E(x) <= sum_s max_{x_s} E_s(x_s)``, because the right side maximizes
each share independently and so cannot be beaten by any single ``x``. Every
*redistribution* of shares keeps that true, so the tightening below can be
wrong about how good a bound it finds and cannot be wrong about it **being** a
bound --- and the tests assert exactly that, at every iteration, against
exhaustive enumeration.

**Kikuchi is not this.** A region-based free energy
(:mod:`sal.sandbox.region_graph`, declined and conserved)
approximates ``log Z`` and
is not a bound in either direction; this approximates nothing and bounds the
maximum from above. Two objects, two questions, and the measurement in #689
argues the difference: Kikuchi's advantage over Bethe *decays* as the coupling
grows, which is the regime a ground state lives in.

**What the gap means.** ``energy(decoded) - bound`` is what is not yet
established. At zero the labelling is **provably optimal** and no oracle was
needed to say so, which is the one thing an approximation cannot offer. Above
zero it is an honest interval, reported and never assumed small.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dataclass_field

import numpy as np

from sal.opt.termination import Termination, check_cap
from sal.sim.graph import PottsGraph
from sal.sim.potts import SiteField, energy, log_weight_of, site_field


@dataclass(frozen=True)
class Certificate:
    """A labelling, an upper bound on the optimum, and the gap between them.

    Parameters
    ----------
    labelling : np.ndarray
        The decoded labelling, one state per site.
    energy : float
        Its energy, in the sign `sim.potts.energy` uses: a ground
        state **minimizes** it, so this is an **upper** bound on the minimum
        --- some labelling achieves it, so the best does at least this well.
    bound : float
        A **lower** bound on the minimum, valid at every iteration by the
        construction above. The dual bounds the maximum of the log-weight from
        above and the energy is its negative, so the sign turns once here and
        nowhere else.
    iterations : int
        Coordinate-ascent sweeps taken.
    termination : Termination | None
        Whether the sweeps settled --- the dual moved by at most ``tolerance`` ---
        or the count ran out (issue #860). Not ``optimal``, which is a
        statement about the gap and so about the instance, not about the
        loop.

    Notes
    -----
    ``optimal`` is the only certificate here, and it is a real one: a labelling
    whose energy meets the bound cannot be beaten, because nothing can exceed
    the bound and this labelling attains it.
    """

    labelling: np.ndarray
    energy: float
    bound: float
    iterations: int
    termination: Termination = dataclass_field(kw_only=True)

    @property
    def gap(self) -> float:
        """``energy - bound``, what is not established. Zero certifies."""
        return self.energy - self.bound

    @property
    def optimal(self) -> bool:
        """Whether the gap has closed to floating-point noise."""
        return bool(self.gap <= 1e-9 * max(1.0, abs(self.bound)))


def _edge_tables(graph: PottsGraph, n_states: int) -> np.ndarray:
    """``theta_e(a, b)`` per edge: the coupling on the diagonal, zero off it."""
    identity = np.eye(n_states)
    return graph.edge_coupling[:, np.newaxis, np.newaxis] * identity[np.newaxis, :, :]


def _clusters(
    graph: PottsGraph,
    n_states: int,
    plaquettes: tuple[tuple[int, ...], ...],
) -> list[tuple[tuple[int, ...], np.ndarray]]:
    """The subproblems: one per plaquette, one per edge no plaquette took.

    Each edge's table goes to **exactly one** subproblem, which is what keeps
    the shares summing to the energy and so keeps the bound a bound. An edge
    inside several plaquettes is split evenly between them --- an even split
    is arbitrary and any split is valid, so the arbitrary one is stated rather
    than tuned.
    """
    tables = _edge_tables(graph, n_states)
    edges = [tuple(sorted(edge)) for edge in graph.edges]
    owners: dict[int, list[int]] = {index: [] for index in range(len(edges))}
    for position, sites in enumerate(plaquettes):
        inside = set(sites)
        for index, (left, right) in enumerate(edges):
            if left in inside and right in inside:
                owners[index].append(position)

    built: list[tuple[tuple[int, ...], np.ndarray]] = []
    for position, sites in enumerate(plaquettes):
        ordered = tuple(sorted(sites))
        table = np.zeros((n_states,) * len(ordered))
        for index, (left, right) in enumerate(edges):
            if position not in owners[index]:
                continue
            axes = [ordered.index(left), ordered.index(right)]
            shape = [n_states if axis in axes else 1 for axis in range(len(ordered))]
            moved = tables[index] / len(owners[index])
            if axes[0] > axes[1]:
                moved = moved.T
            table = table + moved.reshape(
                [n_states if axis in axes else 1 for axis in range(len(ordered))]
                if len(ordered) > 2
                else shape
            )
        built.append((ordered, table))
    for index, (left, right) in enumerate(edges):
        if not owners[index]:
            built.append(((left, right), tables[index]))
    return built


def dual_bound(
    graph: PottsGraph,
    field: SiteField | np.ndarray,
    *,
    max_iterations: int = 200,
    tolerance: float = 1e-12,
    plaquettes: tuple[tuple[int, ...], ...] = (),
) -> Certificate:
    """Bound the Potts ground-state energy from below, and decode under it.

    The decomposition: one subproblem per site holding its field, and one per
    cluster holding the couplings of the edges inside it --- edges alone for
    the pairwise relaxation, plaquettes for the tightened one. Coordinate
    descent then moves share between a cluster and its sites, each move
    leaving the total unchanged and so leaving the bound valid.

    Parameters
    ----------
    graph : PottsGraph
        The lattice or graph. Couplings may be of either sign: a repulsive one
        is where this earns its place, since minimum cut cannot take it.
    field : SiteField | np.ndarray
        The external field, ``(n_states,)`` or ``(n_nodes, n_states)``.
    max_iterations : int
        Coordinate-descent sweeps. More can only raise the lower bound, never
        lower it, so this trades time for tightness and never for validity.
    tolerance : float
        The sweeps have settled where one moves the dual by at most this,
        absolute: the dual is an energy, and a relative change would stall
        where it crosses zero.
    plaquettes : tuple[tuple[int, ...], ...]
        Sites of each higher-order cluster, typically a lattice's unit cells or
        its triangles. Empty runs the pairwise relaxation, whose bound on a
        frustrated graph does not depend on the coupling at all: the
        relaxation believes every edge can be satisfied at every site's
        preferred label, which no labelling achieves.

    Returns
    -------
    Certificate
        The labelling, its energy, the lower bound, and the gap between them.
    """
    field = log_weight_of(field)
    values = site_field(field, graph.n_nodes)
    n_nodes, n_states = values.shape
    clusters = _clusters(graph, n_states, plaquettes)

    # `messages[c][k]` is what cluster c lends its k-th site. The dual is
    #
    #   D = sum_i max_a [theta_i(a) + sum_{c in i} mu_{c->i}(a)]
    #     + sum_c max_{x_c} [theta_c(x_c) - sum_{i in c} mu_{c->i}(x_i)]
    #
    # and it bounds the maximum of the log-weight **for any messages at all**:
    # summing the bracketed expressions at one labelling returns it exactly,
    # and each term is maximized independently, so no labelling can exceed it.
    # That is the one-line proof, and it is why the tests assert validity at
    # every iteration rather than only at convergence.
    messages = [np.zeros((len(sites), n_states)) for sites, _ in clusters]

    def node_shares() -> np.ndarray:
        shares = values.copy()
        for (sites, _), lent in zip(clusters, messages, strict=True):
            for position, site in enumerate(sites):
                shares[site] = shares[site] + lent[position]
        return shares

    def spread(vector: np.ndarray, position: int, width: int) -> np.ndarray:
        shape = [1] * width
        shape[position] = vector.shape[0]
        return vector.reshape(shape)

    def dual_value() -> float:
        total = float(node_shares().max(axis=1).sum())
        for (sites, table), lent in zip(clusters, messages, strict=True):
            residual = table.copy()
            for position in range(len(sites)):
                residual = residual - spread(lent[position], position, len(sites))
            total += float(residual.max())
        return total

    # Each block update is the exact minimizer of its own block --- checked
    # against a numerical minimum over the block's messages --- so the sweep
    # descends: over 480 updates on the frustrated triangular lattice, none
    # raised the dual. The tightest iterate is kept anyway, because every
    # iterate is a valid bound by the construction above and retaining the
    # best costs one comparison, which is cheaper than relying on that.
    best = np.inf
    best_shares = node_shares()
    taken = 0
    settled = False
    check_cap("max_iterations", max_iterations)
    for sweep in range(1, max_iterations + 1):
        taken = sweep
        before = dual_value()
        shares = node_shares()
        for index, (sites, table) in enumerate(clusters):
            width = len(sites)
            rest = [
                shares[site] - messages[index][position]
                for position, site in enumerate(sites)
            ]
            folded = table.copy()
            for position in range(width):
                folded = folded + spread(rest[position], position, width)
            for position, site in enumerate(sites):
                others = tuple(axis for axis in range(width) if axis != position)
                # The site's own share is **excluded** before maximizing over
                # the others: the block minimizer is
                # `max_{x_-i} [theta_c + sum_{j != i} rest_j]`, and leaving
                # `rest_i` inside adds half of it to every update, which reads
                # as a coordinate descent that does not descend.
                promise = (folded - spread(rest[position], position, width)).max(
                    axis=others
                )
                updated = promise / width - rest[position] * (width - 1) / width
                shares[site] = rest[position] + updated
                messages[index][position] = updated
        current = dual_value()
        if current < best:
            best, best_shares = current, node_shares()
        if abs(before - current) <= tolerance:
            settled = True
            break

    labelling = np.asarray(best_shares.argmax(axis=1), dtype=np.int64)
    return Certificate(
        labelling=labelling,
        energy=float(energy(graph, field, labelling)),
        # The dual bounds the log-weight from above; the energy is its
        # negative, so an upper bound there is a lower bound here.
        bound=-float(best),
        iterations=taken,
        termination=Termination.after(taken, converged=settled),
    )
