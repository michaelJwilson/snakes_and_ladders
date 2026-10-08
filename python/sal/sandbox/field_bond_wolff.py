"""A Wolff move whose bonds read the field: ``p = 1 - exp(-beta (J - dh)_+)`` (issue #1390, move mix).

Declined: the naive Metropolis-Hastings acceptance does not keep the Potts
law, shown by enumeration on a tiny instance, so the move was not measured.

**The move.** From state ``x``, a seed site ``s`` uniformly, ``a = x_s``, a
target ``b != a`` uniformly; each edge ``e = (i, j)`` with
``x_i = x_j = a`` is bonded independently with

``p_e(a, b) = 1 - exp(-beta * max(J_e - dh_e(a, b), 0))``,
``dh_e(a, b) = ((h_i[a] - h_i[b]) + (h_j[a] - h_j[b])) / 2``,

the mean field cost of moving the edge's two ends from ``a`` to ``b``; the
cluster ``C`` is the seed's bonded component, proposed onto ``b`` and
accepted on Wolff's ratio, which counts only the boundary edges:

``log A = -beta dE + sum_rev-boundary log(1 - p_e(b, a)) - sum_fwd-boundary log(1 - p_e(a, b))``.

With ``dh = 0`` (``field_aware=False``) that is Wolff's move with a
Metropolis recolour, and the boundary terms cancel the coupling part of
``dE`` exactly.

**Why it fails.** The probability of proposing ``C`` is the boundary product
times the probability that the bonds *inside* ``C`` connect it. With a
J-only bond the inside edges carry the same ``p`` forward (``a -> b``) and
back (``b -> a``), so the connectivity factor cancels; with
``dh_e(b, a) = -dh_e(a, b)`` it does not, and the exact ratio is a sum over
connected bond subsets of ``C``: tractable by enumeration on a few sites, not
on a lattice. :func:`transition_matrix` enumerates the kernel; on a 4-cycle at
``q = 3`` the J-only control satisfies detailed balance to rounding and the
field-aware move violates it
(``tests/regression/sandbox/test_field_bond_wolff.py``).
"""

from __future__ import annotations

import itertools
import math

import numpy as np


def bond_probabilities(
    edges: np.ndarray,
    coupling: np.ndarray,
    rows: np.ndarray,
    a: int,
    b: int,
    beta: float,
    *,
    field_aware: bool = True,
) -> np.ndarray:
    """``p_e(a, b)`` per edge, as the module defines it; ``field_aware=False`` is Wolff's ``1 - exp(-beta J)``.

    Returns
    -------
    np.ndarray
        Shape ``(n_edges,)``.
    """
    strength = np.asarray(coupling, dtype=float).copy()
    if field_aware:
        cost = rows[:, a] - rows[:, b]
        strength -= (cost[edges[:, 0]] + cost[edges[:, 1]]) / 2.0
    return np.asarray(1.0 - np.exp(-beta * np.maximum(strength, 0.0)), dtype=float)


def energy(
    edges: np.ndarray, coupling: np.ndarray, rows: np.ndarray, state: np.ndarray
) -> float:
    """``E(x) = -sum_i h_i[x_i] - sum_(ij) J_ij [x_i = x_j]``."""
    agree = state[edges[:, 0]] == state[edges[:, 1]]
    return float(-rows[np.arange(len(state)), state].sum() - np.sum(coupling[agree]))


def log_acceptance(
    edges: np.ndarray,
    coupling: np.ndarray,
    rows: np.ndarray,
    state: np.ndarray,
    cluster: np.ndarray,
    b: int,
    beta: float,
    *,
    field_aware: bool = True,
) -> float:
    """Wolff's log acceptance of moving ``cluster`` (a boolean mask) from its label to ``b``: boundary edges only."""
    a = int(state[np.flatnonzero(cluster)[0]])
    moved = state.copy()
    moved[cluster] = b
    d_energy = energy(edges, coupling, rows, moved) - energy(
        edges, coupling, rows, state
    )
    inside, outside = cluster[edges[:, 0]], cluster[edges[:, 1]]
    boundary = inside != outside
    far = np.where(inside, edges[:, 1], edges[:, 0])
    forward = boundary & (state[far] == a)
    backward = boundary & (state[far] == b)
    p_forward = bond_probabilities(
        edges, coupling, rows, a, b, beta, field_aware=field_aware
    )
    p_backward = bond_probabilities(
        edges, coupling, rows, b, a, beta, field_aware=field_aware
    )
    return float(
        -beta * d_energy
        + np.sum(np.log1p(-p_backward[backward]))
        - np.sum(np.log1p(-p_forward[forward]))
    )


def _component(n: int, edges: np.ndarray, bonded: np.ndarray, seed: int) -> np.ndarray:
    """The seed's component over the bonded edges, as a boolean mask."""
    member = np.zeros(n, dtype=bool)
    member[seed] = True
    grown = True
    while grown:
        grown = False
        for (i, j), on in zip(edges, bonded, strict=True):
            if on and member[i] != member[j]:
                member[i] = member[j] = True
                grown = True
    return member


def step(
    edges: np.ndarray,
    coupling: np.ndarray,
    rows: np.ndarray,
    state: np.ndarray,
    beta: float,
    rng: np.random.Generator,
    *,
    field_aware: bool = True,
) -> np.ndarray:
    """One move from ``state``, returned as a new array; every label allowed (a finite field)."""
    n, q = rows.shape
    seed = int(rng.integers(n))
    a = int(state[seed])
    b = int((a + rng.integers(1, q)) % q)
    p = bond_probabilities(edges, coupling, rows, a, b, beta, field_aware=field_aware)
    same = (state[edges[:, 0]] == a) & (state[edges[:, 1]] == a)
    bonded = same & (rng.random(len(edges)) < p)
    cluster = _component(n, edges, bonded, seed)
    log_a = log_acceptance(
        edges, coupling, rows, state, cluster, b, beta, field_aware=field_aware
    )
    moved = state.copy()
    if math.log(rng.random()) < min(0.0, log_a):
        moved[cluster] = b
    return moved


def transition_matrix(
    edges: np.ndarray,
    coupling: np.ndarray,
    rows: np.ndarray,
    beta: float,
    *,
    field_aware: bool = True,
) -> np.ndarray:
    """The exact kernel of :func:`step` over all ``q ** n`` states, by enumerating seeds, targets and bond subsets.

    States are indexed as ``np.ravel_multi_index`` over ``(q,) * n``.

    Returns
    -------
    np.ndarray
        Shape ``(q ** n, q ** n)``, rows summing to one.
    """
    n, q = rows.shape
    shape = (q,) * n
    kernel = np.zeros((q**n, q**n))
    for index in range(q**n):
        state = np.array(np.unravel_index(index, shape), dtype=np.int64)
        for seed in range(n):
            a = int(state[seed])
            for b in (c for c in range(q) if c != a):
                weight = 1.0 / (n * (q - 1))
                p = bond_probabilities(
                    edges, coupling, rows, a, b, beta, field_aware=field_aware
                )
                same = np.flatnonzero(
                    (state[edges[:, 0]] == a) & (state[edges[:, 1]] == a)
                )
                for chosen in itertools.product((False, True), repeat=len(same)):
                    bonded = np.zeros(len(edges), dtype=bool)
                    bonded[same] = chosen
                    law = float(np.prod(np.where(chosen, p[same], 1.0 - p[same])))
                    cluster = _component(n, edges, bonded, seed)
                    accept = math.exp(
                        min(
                            0.0,
                            log_acceptance(
                                edges,
                                coupling,
                                rows,
                                state,
                                cluster,
                                b,
                                beta,
                                field_aware=field_aware,
                            ),
                        )
                    )
                    moved = state.copy()
                    moved[cluster] = b
                    target = int(np.ravel_multi_index(tuple(moved), shape))
                    kernel[index, target] += weight * law * accept
                    kernel[index, index] += weight * law * (1.0 - accept)
    return kernel


def boltzmann(
    edges: np.ndarray, coupling: np.ndarray, rows: np.ndarray, beta: float
) -> np.ndarray:
    """``exp(-beta E(x)) / Z`` over all ``q ** n`` states, :func:`transition_matrix`'s order."""
    n, q = rows.shape
    log_weight = np.array(
        [
            -beta * energy(edges, coupling, rows, np.array(labels, dtype=np.int64))
            for labels in itertools.product(range(q), repeat=n)
        ]
    )
    weight = np.exp(log_weight - log_weight.max())
    return np.asarray(weight / weight.sum(), dtype=float)
