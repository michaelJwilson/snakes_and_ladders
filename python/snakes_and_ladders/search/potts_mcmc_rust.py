"""Rust single-site heat-bath sampling, beside `potts_mcmc` rather than
replacing it.

The Python sweep stays as the oracle, per root ``CLAUDE.md`` ("Every
accelerated kernel keeps its pure Python/NumPy implementation as an oracle").

**Why this is a second backend and not a replacement.** Issue #232 profiled
`_single_site_sweep` as the one place a Python-level loop dominates -- 100
sweeps on a 32x32 periodic lattice take 1.05 s against 0.064 s at 8x8, linear
in nodes with the constant set by interpreter overhead. But the sweep calls
``np.exp`` and ``np.searchsorted``, and Rust's ``f64::exp`` agrees with NumPy's
SIMD implementation to within a unit in the last place rather than bit-exactly.
``searchsorted`` is a threshold, so one draw landing across a boundary that
moved by 1 ulp picks a different state, and from that step the two chains are
unrelated rather than approximately equal. Switching `sample_potts` to this
path would move every autocorrelation figure ``STATUS.md`` pins, every
committed notebook output that reads a chain, and the goodness-of-fit fixtures'
chain lengths. That is a decision with its own evidence, not a side effect of a
performance ticket (issue #246).

**Agreement is therefore distributional, never bitwise.** The two backends are
refereed by the distribution they converge to, against exhaustive enumeration
at an enumerable size -- which is what ``search/CLAUDE.md`` requires of any
sampler, and the only comparison that means anything between two
implementations of a stochastic process.

**The uniforms are drawn here and passed down.** The Rust module holds no
generator: ``snakes_and_ladders.sim``'s reproducibility contract is that a
seeded generator determines the result, and a second stream inside Rust would
break it silently.
"""

from __future__ import annotations

import numpy as np

from snakes_and_ladders import oxi_snakes_and_ladders
from snakes_and_ladders.search.potts_mcmc import PottsChain, _adjacency
from snakes_and_ladders.sim.graph import PottsGraph


def flatten_adjacency(
    graph: PottsGraph,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The graph's adjacency in compressed-row form, for the kernel.

    Built from :func:`snakes_and_ladders.search.potts_mcmc._adjacency` rather
    than from the edges directly, so the two backends cannot disagree about
    which neighbours a node has: the oracle's own structure is what crosses.

    Parameters
    ----------
    graph : PottsGraph
        The graph to flatten.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        ``offsets`` of length ``n_nodes + 1``, and ``neighbours`` and
        ``couplings`` of length ``2 * n_edges``, indexed in step.
    """
    neighbours = _adjacency(graph)
    offsets = np.zeros(graph.n_nodes + 1, dtype=np.int64)
    for node, incident in enumerate(neighbours):
        offsets[node + 1] = offsets[node] + len(incident)

    flat_neighbours = np.empty(int(offsets[-1]), dtype=np.int64)
    flat_couplings = np.empty(int(offsets[-1]), dtype=np.float64)
    position = 0
    for incident in neighbours:
        for neighbour, coupling in incident:
            flat_neighbours[position] = neighbour
            flat_couplings[position] = coupling
            position += 1
    return offsets, flat_neighbours, flat_couplings


def sample_potts(
    graph: PottsGraph,
    field: np.ndarray,
    rng: np.random.Generator,
    n_sweeps: int,
    burn_in: int = 0,
    thin: int = 1,
) -> PottsChain:
    """Draw a single-site chain on ``graph``, stepping in Rust.

    Signature matches
    :func:`snakes_and_ladders.search.potts_mcmc.sample_potts` less its
    ``move`` argument, which has one value here: the cluster moves are not
    ported, because #232 did not profile them as dominant and
    ``search/CLAUDE.md`` requires a cluster move in a field to carry an accept
    step this kernel does not implement.

    Parameters
    ----------
    graph : PottsGraph
        The graph to sample on.
    field : np.ndarray
        External field ``h``, shape ``(n_states,)``.
    rng : np.random.Generator
        Passed in rather than seeded here. Every uniform the kernel consumes is
        drawn from it, in the order the oracle draws them.
    n_sweeps : int
        Sweeps to record.
    burn_in : int
        Sweeps run and discarded before recording starts.
    thin : int
        Record one sweep in every ``thin``. Successive sweeps are correlated,
        so a goodness-of-fit test run on every sweep rejects a *correct*
        sampler.

    Returns
    -------
    PottsChain
        The recorded configurations. ``mean_cluster_size`` is the node count,
        as it is for the oracle's single-site move: this move set builds no
        clusters, and reporting anything else would invent a statistic.

    Raises
    ------
    ValueError
        If the kernel refuses its arguments -- a state outside the alphabet, a
        ragged adjacency, or a draw count that does not match the sweeps.
    """
    n_states = int(field.shape[0])
    offsets, neighbours, couplings = flatten_adjacency(graph)
    state = np.ascontiguousarray(
        rng.integers(0, n_states, size=graph.n_nodes), dtype=np.int64
    )
    contiguous_field = np.ascontiguousarray(field, dtype=np.float64)

    recorded = np.empty((n_sweeps, graph.n_nodes), dtype=np.int64)
    for step in range(-burn_in * thin, n_sweeps * thin):
        # One sweep at a time, so the recorded stride matches the oracle's and
        # the uniforms are consumed in the same order.
        draws = np.ascontiguousarray(rng.random(graph.n_nodes), dtype=np.float64)
        oxi_snakes_and_ladders.single_site_sweeps(
            state, contiguous_field, offsets, neighbours, couplings, draws, 1
        )
        if step >= 0 and (step + 1) % thin == 0:
            recorded[step // thin] = state

    return PottsChain(states=recorded, mean_cluster_size=float(graph.n_nodes))
