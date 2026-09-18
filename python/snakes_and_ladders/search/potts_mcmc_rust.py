"""The Rust single-site heat-bath chain, named for callers that want it by name.

One line of dispatch onto :func:`snakes_and_ladders.search.potts_mcmc.sample_potts`
with :data:`~snakes_and_ladders.backend.Backend.RUST`, which is now that
function's default. It stays because it is the name issue #246 published and
`tests/benchmarks/` and `STATUS.md` still cite, and because it fixes the one
move set the kernel implements; what it no longer carries is a second loop
over sweeps, which is the thing that could have drifted from the one it
duplicated (issue #599).

**The chain is the oracle's, state for state.** ``f64::exp`` and NumPy's SIMD
``exp`` differ in the last place, and ``searchsorted`` is a threshold, so the
kernel decides a site only where the draw clears every cumulative boundary by
more than that difference can move it and hands the rest back for the NumPy
path to decide. :func:`snakes_and_ladders.search.potts_mcmc._sweep_at` states
the bound. The oracle stays, per root ``CLAUDE.md``, and pins this.

**The uniforms are drawn in Python and passed down.** The Rust module holds no
generator: ``snakes_and_ladders.sim``'s reproducibility contract is that a
seeded generator determines the result, and a second stream inside Rust would
break it silently.
"""

from __future__ import annotations

import numpy as np

from snakes_and_ladders.backend import Backend
from snakes_and_ladders.search.potts_mcmc import PottsChain, PottsMove
from snakes_and_ladders.search.potts_mcmc import sample_potts as _sample_potts
from snakes_and_ladders.sim.graph import PottsGraph


def sample_potts(
    graph: PottsGraph,
    field: np.ndarray,
    rng: np.random.Generator,
    n_sweeps: int,
    burn_in: int = 0,
    thin: int = 1,
) -> PottsChain:
    """Draw a single-site chain on ``graph``, stepping in Rust.

    :func:`snakes_and_ladders.search.potts_mcmc.sample_potts` less its ``move``
    and ``backend`` arguments, which have one value each here: the cluster
    moves are not ported, because #232 did not profile them as dominant and
    ``search/CLAUDE.md`` requires a cluster move in a field to carry an accept
    step this kernel does not implement.

    Parameters
    ----------
    graph, field, rng, n_sweeps, burn_in, thin
        As :func:`snakes_and_ladders.search.potts_mcmc.sample_potts`.

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
    return _sample_potts(
        graph,
        field,
        PottsMove.SINGLE_SITE,
        rng,
        n_sweeps,
        burn_in,
        thin,
        backend=Backend.RUST,
    )
