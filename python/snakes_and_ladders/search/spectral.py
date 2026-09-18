"""A spectral lower bound on the two-state ground-state energy, and the start it gives (issue #718).

**The bound.** With spins ``sigma_i = 2 s_i - 1`` the Potts energy
``E(s) = -sum_i h_i[s_i] - sum_(ij) J_ij [s_i == s_j]`` of
:func:`snakes_and_ladders.sim.potts.energies` is a quadratic form on the
hypercube,

    E = c - (1/4) sigma^T A sigma - b^T sigma,

with ``A`` the coupling matrix (both directions of every edge, so
``sum_(ij) J_ij sigma_i sigma_j = sigma^T A sigma / 2``),
``b_i = (h_i[1] - h_i[0]) / 2`` and
``c = -sum_i (h_i[0] + h_i[1]) / 2 - (1/2) sum_(ij) J_ij``. Lifting the linear
term onto a ghost spin ``sigma_0``, whose sign the global flip fixes, makes it
one quadratic form on ``{-1, +1}^(n+1)`` with the matrix
``A~ = [[0, 2 b^T], [2 b, A]]``, and ``||sigma~||^2 = n + 1`` gives

    E >= c - (1/4) (n + 1) lambda_max(A~).

With no field the ghost spin is dropped and the bound is
``c - (1/4) n lambda_max(A)``. It holds for every coupling sign, which is
where it earns its place: minimum cut is exact only for attractive couplings,
and on the frustrated lattice and the planted glass the only other lower bound
here is :func:`snakes_and_ladders.search.tightening.dual_bound`. The bound is
a theorem, not a measurement; what is measured is its gap.

**The correction, and why it is not optional.** On the cube
``sigma~^T diag(u) sigma~ = sum(u)``, so any ``u`` with ``sum(u) = 0`` leaves
the quadratic form unchanged and the bound holds with ``A~ + diag(u)`` in
place of ``A~`` (Delorme and Poljak 1993). ``lambda_max`` is convex in ``u``
and its subgradient is the top eigenvector squared, so projected subgradient
descent lowers it and every iterate is a valid bound; the best one is kept.
Without it the ghost spin absorbs the field's norm and the bound is worthless
where there is a field --- 393% below the exact cut at 16x16 with a unit
normal field, 1,646% at 64x64; with the default correction the gap is 12%,
9% and 12% at 16x16, 32x32 and 64x64. What remains is the gap of the
relaxation, which a subgradient method reaches only slowly (Delorme and
Poljak's is a bundle method). At zero field on a vertex-transitive graph the
uncorrected ``u = 0`` is already optimal by symmetry.

**The start.** The top eigenvector of ``A~`` is the relaxation's minimizer on
the sphere, and its signs are the labelling closest to it on the cube; the
ghost component's sign gauges the global flip. It is a start and never a
result --- what it buys is measured against random starts at equal budget.

**Two states only.** The lift is the Ising one, so a field with more than two
columns is refused, as :mod:`snakes_and_ladders.search.maxflow` refuses it,
and for the same reason: more than two states is alpha expansion's problem.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse
import scipy.sparse.linalg

from snakes_and_ladders.sim.graph import PottsGraph
from snakes_and_ladders.sim.potts import site_field
from snakes_and_ladders.sim.spectrum import (
    SPARSE_TOLERANCE,
    coupling_matrix,
    extreme_eigenpairs,
)

#: Correction steps by default. Each costs one eigensolve per multiplier in
#: :data:`STEP_MULTIPLIERS`, 48 in all: on the open lattices measured, the
#: last four steps move the bound under 0.5%.
DEFAULT_CORRECTIONS = 16

#: The trial step lengths of a correction step, as multiples of the last
#: accepted length. Measured at 32x32 over 51 eigensolves: three multipliers
#: reach a 9.1% gap to the exact cut, five (0.25 to 4) 9.9%, and a single
#: backtracking step lands anywhere from 6.9% to 29% with the initial length.
STEP_MULTIPLIERS = (0.5, 1.0, 2.0)

#: The first step length as a fraction of ``|lambda_max|``; the bracket
#: adapts it from there.
INITIAL_STEP = 0.3

#: Node count below which every eigensolve of the correction is dense. Lower
#: than :data:`snakes_and_ladders.sim.spectrum.DENSE_BELOW` because the
#: correction pays for 49 solves, not one: at 32x32 the bound is 2.9 s sparse
#: against 6.8 s dense, the same number to 1e-9.
CORRECTION_DENSE_BELOW = 512

#: Lanczos vectors per warm-started step above the dense threshold, and the
#: implicit restarts allowed before the trial is given up: a trial step that
#: makes the top of the spectrum degenerate can hold ARPACK for minutes at
#: its default of ten restarts per node.
WARM_KRYLOV = 60
WARM_RESTARTS = 100


@dataclass(frozen=True)
class SpectralBound:
    """The bound and what it was read from.

    Parameters
    ----------
    bound : float
        A lower bound on ``min_s E(s)``, the best over the corrections run.
    largest_eigenvalue : float
        ``lambda_max`` of the corrected lifted matrix at the best step.
    lifted : bool
        Whether a ghost spin carried the field; ``False`` when the field is
        identically zero across sites.
    uncorrected : float
        The bound at ``u = 0``, so the correction's worth is on the record.
    """

    bound: float
    largest_eigenvalue: float
    lifted: bool
    uncorrected: float


def _corrected_top(
    matrix: scipy.sparse.csr_matrix, corrections: int, dense_below: int
) -> tuple[float, np.ndarray, float]:
    """``(lambda_best, vector_best, lambda_uncorrected)`` over the correction steps.

    Projected subgradient on ``u`` with a bracketed line search: each step
    tries the subgradient direction at :data:`STEP_MULTIPLIERS` times the
    last accepted length and keeps the best trial if it lowers
    ``lambda_max``, else shrinks the length. The accepted values decrease and
    every one is a valid bound. Above the dense threshold each eigensolve is
    ARPACK started from the last accepted vector.
    """
    size = matrix.shape[0]
    values, vectors = extreme_eigenpairs(matrix, largest=True, dense_below=dense_below)
    uncorrected = float(values[0])
    best_value, best_vector = uncorrected, vectors[:, 0].copy()
    shift = np.zeros(size)
    step = INITIAL_STEP * (abs(uncorrected) if uncorrected != 0.0 else 1.0)
    for _ in range(corrections):
        gradient = best_vector**2
        gradient -= gradient.mean()
        norm = float(np.linalg.norm(gradient))
        if norm == 0.0:
            break
        direction = gradient / norm * np.sqrt(size)
        trials = []
        for multiplier in STEP_MULTIPLIERS:
            candidate = shift - multiplier * step * direction
            shifted = scipy.sparse.csr_matrix(matrix + scipy.sparse.diags(candidate))
            value, vector = _warm_top(shifted, best_vector, dense_below)
            trials.append((value, multiplier, candidate, vector))
        value, multiplier, candidate, vector = min(trials, key=lambda trial: trial[0])
        if value < best_value:
            best_value, best_vector, shift = value, vector, candidate
            step *= multiplier
        else:
            step *= 0.5 * min(STEP_MULTIPLIERS)
    return best_value, best_vector, uncorrected


def _warm_top(
    matrix: scipy.sparse.csr_matrix, start: np.ndarray, dense_below: int
) -> tuple[float, np.ndarray]:
    """The top eigenpair, dense below the threshold and ARPACK from ``start`` above."""
    if matrix.shape[0] < dense_below:
        values, vectors = extreme_eigenpairs(
            matrix, largest=True, dense_below=dense_below
        )
        return float(values[0]), vectors[:, 0]
    # The correction flattens the top of the spectrum, and a wider Krylov
    # space is what pays for that: at 128x128 the six-step cost is 4.5 s at
    # `ncv=60` against 13.5 s at ARPACK's default of 20.
    try:
        values, vectors = scipy.sparse.linalg.eigsh(
            matrix,
            k=1,
            which="LA",
            v0=start,
            tol=SPARSE_TOLERANCE,
            ncv=min(WARM_KRYLOV, matrix.shape[0] - 1),
            maxiter=WARM_RESTARTS,
        )
    except scipy.sparse.linalg.ArpackNoConvergence:
        # A trial whose eigenvalue is not converged is not a bound; it reads
        # as a failed trial and the step shrinks.
        return np.inf, start
    vector = np.asarray(vectors[:, 0], dtype=np.float64)
    nonzero = np.flatnonzero(np.abs(vector) > 1e-12)
    if nonzero.size and vector[nonzero[0]] < 0.0:
        vector = -vector
    return float(values[0]), vector


def _lift(
    graph: PottsGraph, field_values: np.ndarray
) -> tuple[scipy.sparse.csr_matrix, float, bool]:
    """``(A~, c, lifted)`` for a two-state field; see the module docstring."""
    values = site_field(field_values, graph.n_nodes)
    if values.shape[1] != 2:
        msg = (
            f"the spectral lift is the two-state one, got a field with "
            f"{values.shape[1]} states: more than two is alpha expansion's problem"
        )
        raise ValueError(msg)
    adjacency = coupling_matrix(graph)
    offset = -float(values.sum()) / 2.0 - float(graph.edge_coupling.sum()) / 2.0
    # The border carries `2 b`, so the lifted form is `sigma~^T A~ sigma~ / 4`.
    linear = values[:, 1] - values[:, 0]
    if not np.any(linear):
        return adjacency, offset, False
    n = graph.n_nodes
    ghost = scipy.sparse.csr_matrix(
        (linear, (np.arange(n), np.zeros(n, dtype=np.int64))), shape=(n, 1)
    )
    lifted = scipy.sparse.bmat([[None, ghost.T], [ghost, adjacency]], format="csr")
    return lifted, offset, True


def spectral_bound(
    graph: PottsGraph,
    field_values: np.ndarray,
    *,
    corrections: int = DEFAULT_CORRECTIONS,
    dense_below: int = CORRECTION_DENSE_BELOW,
) -> SpectralBound:
    """A lower bound on the two-state ground-state energy from one eigenvalue per step.

    Parameters
    ----------
    graph : PottsGraph
        Any coupling sign.
    field_values : np.ndarray
        ``(2,)`` or ``(n_nodes, 2)``.
    corrections : int
        Diagonal-correction steps, each ``len(STEP_MULTIPLIERS)``
        eigensolves; ``0`` is the raw lifted bound. More can only tighten,
        never loosen, since the best iterate is kept.
    dense_below : int
        Node count below which every eigensolve is dense, passed to
        :func:`snakes_and_ladders.sim.spectrum.extreme_eigenpairs`.

    Returns
    -------
    SpectralBound
    """
    if corrections < 0:
        msg = f"corrections must be non-negative, got {corrections}"
        raise ValueError(msg)
    matrix, offset, lifted = _lift(graph, field_values)
    largest, _, uncorrected = _corrected_top(matrix, corrections, dense_below)
    size = graph.n_nodes + (1 if lifted else 0)
    return SpectralBound(
        bound=offset - 0.25 * size * largest,
        largest_eigenvalue=largest,
        lifted=lifted,
        uncorrected=offset - 0.25 * size * uncorrected,
    )


def spectral_start(
    graph: PottsGraph,
    field_values: np.ndarray,
    *,
    corrections: int = DEFAULT_CORRECTIONS,
    dense_below: int = CORRECTION_DENSE_BELOW,
) -> np.ndarray:
    """The labelling the corrected top eigenvector's signs give; ``int64`` in ``{0, 1}``.

    A component at exactly zero takes label 1, so the map is total; the
    ghost spin's sign, where a field lifted one, gauges the global flip.
    """
    matrix, _, lifted = _lift(graph, field_values)
    _, vector, _ = _corrected_top(matrix, corrections, dense_below)
    if lifted:
        gauge = 1.0 if vector[0] >= 0.0 else -1.0
        vector = gauge * vector[1:]
    return (vector >= 0.0).astype(np.int64)
