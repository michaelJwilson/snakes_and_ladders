"""Canonical problem instances whose answer is known from outside this suite.

Three constructions, admitted under the two rules ``sim/CLAUDE.md`` states.
The first is what makes a fixture worth having: **its answer comes from
outside this repository** --- a closed form, a published result, or an
enumeration that shares no code with the thing being tested. The second is
what makes it belong *here* rather than in a module's own ticket: **more than
one module consumes it.**

Nothing in this module performs inference. Each entry is a construction plus
the quantity that is known about it in advance; the oracles that consume them
live in :mod:`snakes_and_ladders.likelihood` and :mod:`snakes_and_ladders.search`, which is the point of
the second rule.

The three:

1. :func:`frustrated_triangular_lattice` --- geometric frustration, with an
   exact ground-state energy at every size (:func:`minimum_frustrated_edges`).
   Generated here, its energy enumerated by :mod:`snakes_and_ladders.likelihood.potts`, its
   optimum reachable by :func:`snakes_and_ladders.search.max_cut.enumerate_max_cut`, and
   the instance on which :func:`snakes_and_ladders.search.potts_mcmc.sample_potts` must
   *refuse* its cluster moves.
2. :func:`planted_spin_glass` --- a Viana-Bray instance carrying a state of
   known energy, which is the only oracle that survives past the size
   enumeration reaches.
3. :func:`ambiguous_hmm` --- an HMM on which Viterbi and posterior decoding
   return different answers, so a decoder that computes one and reports the
   other is caught.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from snakes_and_ladders.sim.graph import (
    BoundaryCondition,
    PottsGraph,
    erdos_renyi_graph,
    triangular_lattice_graph,
)
from snakes_and_ladders.sim.hmm import HmmParams

#: Wannier's (1950) zero-temperature entropy per site of the infinite
#: triangular Ising antiferromagnet, in units of ``k_B``. Quoted for
#: orientation and **not** asserted anywhere: it is a thermodynamic limit, and
#: the measurements in ``tests/regression/sim/test_canonical.py`` show the
#: finite-size entropy per site is not even monotone on the way to it.
WANNIER_RESIDUAL_ENTROPY = 0.3230659


@dataclass(frozen=True)
class PlantedSpinGlass:
    """A ``+/- J`` instance and the state planted in it.

    Parameters
    ----------
    graph : PottsGraph
        Sparse random graph with couplings of both signs.
    planted : np.ndarray
        The state the couplings were built around, shape ``(n_nodes,)``,
        values in ``{0, 1}``.
    planted_energy : float
        ``E(planted)`` under
        :func:`snakes_and_ladders.likelihood.potts.log_weights`'s convention with a zero
        field. An **upper bound** on the ground-state energy at any size ---
        which is the whole reason to plant a state, and is a weaker claim than
        "the ground state", deliberately: see :func:`planted_spin_glass`.
    unsatisfied : int
        Edges whose coupling opposes ``planted``. Zero makes the instance
        gauge-equivalent to a ferromagnet and therefore trivial.
    """

    graph: PottsGraph
    planted: np.ndarray
    planted_energy: float
    unsatisfied: int


def frustrated_triangular_lattice(
    shape: tuple[int, int] = (3, 3),
    boundary: BoundaryCondition = BoundaryCondition.PERIODIC,
    coupling: float = -1.0,
) -> PottsGraph:
    """The triangular Ising antiferromagnet: frustration with a closed form.

    Every 3-cycle needs at least one edge whose endpoints agree, because a
    triangle is not 2-colourable. On the periodic lattice that counting
    argument closes exactly --- see :func:`minimum_frustrated_edges` --- so
    the ground-state energy is known at **every** size rather than only where
    enumeration reaches.

    Parameters
    ----------
    shape : tuple[int, int]
        ``(rows, columns)``. Both must be ``>= 3`` under
        ``BoundaryCondition.PERIODIC``: at extent 2 the ``+1`` and ``-1``
        neighbours coincide and the wrap produces a doubled bond, which
        :class:`~snakes_and_ladders.sim.graph.PottsGraph` permits but which doubles that
        edge's weight and breaks the counting argument.
    boundary : BoundaryCondition
        ``PERIODIC`` is the case with the closed form. ``OPEN`` is still
        frustrated but its boundary triangles are not covered by the count,
        and it is measured rather than predicted.
    coupling : float
        Must be negative. A positive coupling on this graph is an ordinary
        unfrustrated ferromagnet and nothing here applies to it, so it is
        refused rather than silently returned.

    Raises
    ------
    ValueError
        If ``coupling >= 0``, or an extent is below 3 under ``PERIODIC``.
    """
    if coupling >= 0.0:
        msg = (
            f"coupling must be negative for an antiferromagnet, got {coupling}: "
            "a positive coupling on this graph is unfrustrated"
        )
        raise ValueError(msg)
    if boundary is BoundaryCondition.PERIODIC and min(shape) < 3:
        msg = (
            f"periodic extents must be >= 3, got {shape}: at extent 2 the wrap "
            "produces a doubled bond and the counting argument does not hold"
        )
        raise ValueError(msg)
    return triangular_lattice_graph(shape, boundary, coupling)


def minimum_frustrated_edges(graph: PottsGraph) -> int:
    """Edges that must agree in any two-state ground state, exactly.

    For the periodic triangular lattice on ``N`` sites the answer is ``N``,
    and the argument is a double count rather than a search:

    * the lattice has ``3N`` edges and ``2N`` triangles;
    * each triangle contains at least one agreeing edge, since a 3-cycle is
      not 2-colourable;
    * each edge lies in exactly 2 triangles.

    Counting ``(triangle, agreeing edge)`` pairs gives ``2 * agreeing >= 2N``,
    so ``agreeing >= N``, or one edge in three. Enumeration reaches it at
    ``N = 9, 12, 16, 20``, so the bound is tight as well as valid.

    With ``coupling = -|J|`` and no field the ground-state energy is therefore
    ``|J| * N`` --- known at any size, which is what separates this from every
    other discrete claim in the repository.

    Raises
    ------
    ValueError
        If ``graph`` is not a periodic triangular lattice, where the count
        does not close. An open lattice's boundary triangles are incomplete
        and its answer must be enumerated instead.
    """
    if graph.shape is None or len(graph.shape) != 2:
        msg = "minimum_frustrated_edges needs a 2-D lattice"
        raise ValueError(msg)
    if graph.boundary is not BoundaryCondition.PERIODIC:
        msg = (
            "the counting argument closes only under periodic boundaries; an "
            "open lattice's boundary triangles are incomplete"
        )
        raise ValueError(msg)
    if len(graph.edges) != 3 * graph.n_nodes:
        msg = (
            f"expected 3N = {3 * graph.n_nodes} edges for a periodic "
            f"triangular lattice, got {len(graph.edges)}"
        )
        raise ValueError(msg)
    return graph.n_nodes


def planted_spin_glass(
    n_nodes: int,
    mean_degree: float,
    frustration: float,
    rng: np.random.Generator,
    *,
    magnitude: float = 1.0,
) -> PlantedSpinGlass:
    """A ``+/- J`` spin glass on a sparse random graph, built around a state.

    Viana-Bray rather than Edwards-Anderson: the couplings sit on an
    Erdos-Renyi graph rather than a lattice, so connectivity is a knob instead
    of being fixed at 4 by the geometry. That matters because hardness here is
    tuned by ``mean_degree`` and ``frustration`` together, and a fixture whose
    difficulty cannot be raised cannot referee a claim --- which is exactly
    how issue #177's tree fixture stopped being useful once #198 measured
    random-restart greedy solving it at 1.000.

    **What the planted state is and is not.** It is a state of *known energy*,
    so it upper bounds the ground-state energy at any size and a solver that
    fails to match it has demonstrably not found the optimum. It is **not**
    guaranteed to be the ground state: at ``frustration > 0`` some other state
    may do better, and at small sizes enumeration shows this happening. The
    weaker claim is the one that survives past enumeration, and it is the one
    made.

    **``frustration = 0`` is trivial and must not be used as a hard case.**
    With every edge satisfied the instance is a gauge transform of the
    ferromagnet --- flip the spins in one class of the planted state and every
    coupling becomes positive --- so any local search solves it. The parameter
    exists to be turned up.

    Parameters
    ----------
    n_nodes : int
        Sites, ``>= 2``.
    mean_degree : float
        Target expected degree; the edge probability is
        ``mean_degree / (n_nodes - 1)``.
    frustration : float
        Fraction of edges whose coupling is set *against* the planted state,
        in ``[0, 1]``.
    rng : np.random.Generator
        Passed in, never seeded here, for the reason
        :func:`~snakes_and_ladders.sim.graph.erdos_renyi_graph` states.
    magnitude : float
        ``|J|`` on every edge. The ``+/- J`` model gives every coupling the
        same magnitude and only the sign varies.

    Returns
    -------
    PlantedSpinGlass

    Raises
    ------
    ValueError
        If ``n_nodes < 2``, ``mean_degree`` is negative, ``frustration`` is
        outside ``[0, 1]``, or ``magnitude <= 0``.
    """
    if n_nodes < 2:
        msg = f"n_nodes must be at least 2, got {n_nodes}"
        raise ValueError(msg)
    if mean_degree < 0.0:
        msg = f"mean_degree must be non-negative, got {mean_degree}"
        raise ValueError(msg)
    if not 0.0 <= frustration <= 1.0:
        msg = f"frustration must lie in [0, 1], got {frustration}"
        raise ValueError(msg)
    if magnitude <= 0.0:
        msg = f"magnitude must be positive, got {magnitude}"
        raise ValueError(msg)

    probability = min(1.0, mean_degree / (n_nodes - 1))
    skeleton = erdos_renyi_graph(n_nodes, probability, magnitude, rng)
    planted = rng.integers(0, 2, size=n_nodes)

    # A coupling *satisfies* the planted state when it rewards what the state
    # already does: positive where the endpoints agree, negative where they
    # differ. `frustration` is the rate at which that sign is inverted.
    inverted = rng.random(len(skeleton.edges)) < frustration
    coupling: list[float] = []
    for (first, second), flip in zip(skeleton.edges, inverted, strict=True):
        satisfied = magnitude if planted[first] == planted[second] else -magnitude
        coupling.append(-satisfied if flip else satisfied)

    graph = PottsGraph(n_nodes=n_nodes, edges=skeleton.edges, coupling=tuple(coupling))
    energy = -sum(
        weight
        for (first, second), weight in graph.weighted_edges()
        if planted[first] == planted[second]
    )
    return PlantedSpinGlass(
        graph=graph,
        planted=planted,
        planted_energy=float(energy),
        unsatisfied=int(inverted.sum()),
    )


#: Observations on which :func:`ambiguous_hmm`'s two decoders disagree.
AMBIGUOUS_OBSERVATIONS = np.array([0, 1, 0, 1, 0], dtype=np.int64)


def ambiguous_hmm() -> HmmParams:
    """A two-state HMM whose Viterbi and posterior decodings differ.

    Emissions are reliable (0.83) and the chain is sticky (0.72 to stay), and
    :data:`AMBIGUOUS_OBSERVATIONS` alternates. Those pull in opposite
    directions and the two decoders resolve the conflict differently:

    * **Viterbi** returns ``(0, 0, 0, 0, 0)`` --- the chain never moves,
      because alternating costs four transitions at 0.28 each and that is
      more than the emissions buy back. Unique, and by a margin: the
      runner-up ``(0, 1, 1, 1, 0)`` is 0.3033 nats behind, so no tie-break
      decides the answer.
    * **Posterior decoding** returns ``(0, 1, 0, 1, 0)`` --- the observations
      themselves, because each site's marginal follows its own emission. Every
      one of those marginals is at least 0.6256, so this is not a coin flip
      either.

    The two answers differ at two of five sites. The posterior-decoded
    sequence is the **5th** most likely path of 32, 0.6066 nats below the
    Viterbi path, which is the point: a per-site maximum is not a maximum over
    paths, and no amount of confidence in the marginals makes it one. See
    Durbin et al., *Biological Sequence Analysis*, section 3.2.

    All numbers above are measured by
    :func:`snakes_and_ladders.likelihood.hmm_paths.enumerate_hidden_paths` over all
    ``2 ** 5`` paths and pinned in ``tests/regression/sim/test_canonical.py``.

    Returns
    -------
    HmmParams
        ``sequence_length`` is 5, matching :data:`AMBIGUOUS_OBSERVATIONS`.
    """
    return HmmParams(
        n_states=2,
        n_symbols=2,
        sequence_length=int(AMBIGUOUS_OBSERVATIONS.shape[0]),
        n_sequences=1,
        initial=np.array([0.5, 0.5]),
        transition=np.array([[0.72, 0.28], [0.28, 0.72]]),
        emission=np.array([[0.83, 0.17], [0.17, 0.83]]),
        seed=20260904,
        tolerance=1e-12,
    )
