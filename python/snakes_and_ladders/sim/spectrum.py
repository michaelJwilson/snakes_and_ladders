"""The spectrum of a coupling graph: the seam every spectral method reads (issue #718).

A :class:`~snakes_and_ladders.sim.graph.PottsGraph` is an edge list with a
coupling per edge; every spectral method wants the same two matrices from it
and one or two eigenpairs at an end of the spectrum. Both live here so a
bound, a start or a certificate downstream is a few lines over one seam
rather than its own sparse assembly.

**Two matrices.** :func:`coupling_matrix` is the symmetric ``A`` with
``A_ij = J_ij``, duplicate edges summed --- a periodic lattice two sites
wide carries the same pair twice, and both couplings act. :func:`laplacian`
is ``L = D - A`` with ``D`` the row sums of ``A``, so the constant vector is
in its kernel whatever the couplings' signs.

**Two routes to an eigenpair, one oracle.** :func:`extreme_eigenpairs` takes
``numpy.linalg.eigh`` below :data:`DENSE_BELOW` nodes and
``scipy.sparse.linalg.eigsh`` above it; the dense route is the oracle the
sparse one is pinned against, to ``1e-10``, since ARPACK's answer depends on
a start vector and a tolerance and the dense one on neither. The start vector
is fixed so a call is a function of its input, and each eigenvector's sign is
fixed by its first non-zero component, since an eigenvector is a line and a
caller reading a sign off it needs one convention.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse
import scipy.sparse.linalg

from snakes_and_ladders.sim.graph import PottsGraph

#: Node count below which the dense eigensolver is used. `eigh` on 2,048
#: nodes is 0.9 s on the 4-core host; `eigsh` for one pair at that size is
#: 30 ms, and their agreement is asserted at every size the suite reaches.
DENSE_BELOW = 2048

#: ARPACK's tolerance; the dense-sparse pin is stated against it.
SPARSE_TOLERANCE = 1e-10


def coupling_matrix(graph: PottsGraph) -> scipy.sparse.csr_matrix:
    """The symmetric coupling matrix ``A``, ``A_ij = J_ij``, duplicates summed.

    Parameters
    ----------
    graph : PottsGraph
        Any coupling sign.

    Returns
    -------
    scipy.sparse.csr_matrix
        ``(n_nodes, n_nodes)``, ``float64``, symmetric.
    """
    first, second, coupling = graph.endpoints
    n = graph.n_nodes
    rows = np.concatenate([first, second])
    columns = np.concatenate([second, first])
    data = np.concatenate([coupling, coupling])
    matrix = scipy.sparse.coo_matrix((data, (rows, columns)), shape=(n, n))
    return scipy.sparse.csr_matrix(matrix)


def laplacian(graph: PottsGraph) -> scipy.sparse.csr_matrix:
    """The weighted Laplacian ``L = D - A``; the constant vector is in its kernel."""
    adjacency = coupling_matrix(graph)
    degrees = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    return scipy.sparse.csr_matrix(scipy.sparse.diags(degrees) - adjacency)


def extreme_eigenpairs(
    matrix: scipy.sparse.csr_matrix,
    *,
    largest: bool,
    count: int = 1,
    dense_below: int = DENSE_BELOW,
) -> tuple[np.ndarray, np.ndarray]:
    """The ``count`` largest or smallest eigenpairs of a symmetric matrix.

    Parameters
    ----------
    matrix : scipy.sparse.csr_matrix
        Symmetric; nothing here symmetrizes it.
    largest : bool
        Which end of the spectrum.
    count : int
        Pairs to return, ordered from the end inward.
    dense_below : int
        Node count below which ``eigh`` runs on the dense matrix. A caller
        pins the sparse route by passing ``0`` on a small graph.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``values`` of shape ``(count,)`` and ``vectors`` of shape
        ``(n, count)``, each column unit-norm with a positive first non-zero
        entry.
    """
    n = matrix.shape[0]
    if not 1 <= count < n:
        msg = f"count must lie in [1, {n}), got {count}"
        raise ValueError(msg)
    if n < dense_below:
        values, vectors = np.linalg.eigh(matrix.toarray())
        order = np.argsort(values)[::-1] if largest else np.argsort(values)
        keep = order[:count]
        values, vectors = values[keep], vectors[:, keep]
    else:
        start = np.full(n, 1.0 / np.sqrt(n))
        values, vectors = scipy.sparse.linalg.eigsh(
            matrix,
            k=count,
            which="LA" if largest else "SA",
            v0=start,
            tol=SPARSE_TOLERANCE,
        )
        order = np.argsort(values)[::-1] if largest else np.argsort(values)
        values, vectors = values[order], vectors[:, order]
    for column in range(vectors.shape[1]):
        nonzero = np.flatnonzero(np.abs(vectors[:, column]) > 1e-12)
        if nonzero.size and vectors[nonzero[0], column] < 0.0:
            vectors[:, column] *= -1.0
    return np.asarray(values, dtype=np.float64), np.asarray(vectors, dtype=np.float64)
