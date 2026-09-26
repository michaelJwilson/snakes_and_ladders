"""The fixture design: seeded tiling fields, mixed, planted and noisy (issues #1067, #1074).

Every draw is a :class:`~sal.sim.potts.SpatioTilingParams` on
``spatio_tiling/release``'s lattice family --- open triangular, ``q = 10``
--- built by :func:`~sal.sim.potts.tile_partition` and
:func:`~sal.sim.potts.tiling_field`, and a planted draw clears
:func:`~sal.search.potts_starts.recovery_bound` through
:func:`~sal.search.potts_starts.tiling_rung`. Nothing here restates them.

**Three draws.** *Mixed*: tile strengths log-uniform over
``[0.1 J0, 10 J0]``, so some tiles hold their state and others give way;
labelled by a solver. *Planted*: every strength ``1.05`` to ``2.5`` times its
tile's recovery bound, so the planted labelling is the ground state by proof.
*Noisy*: a mixed draw at coupling ``m J0``, plus ``sigma`` times Gaussian
noise per (site, state), smoothed by ``k`` rounds of neighbour averaging and
rescaled to unit standard deviation.

**Why the noisy family.** On mixed fields alpha-expansion sits 1.29 above the
TRW-S bound at 71x71 (#1067), which leaves a surrogate no energy to win. A
grid over ``m in {1, 2, 4}``, ``sigma in {0.5, 1.5, 3}``, ``k in {0, 4}`` at
41x41, ten fields per point, chose ``m = 2, sigma = 1.5, k = 4``: ICM from the
field argmax 312.13 above the bound, alpha-expansion 8.54, the paired
difference 14.5 SE. At 75x75 on the twenty validation seeds, ICM is
1010.91 +- 27.63 above the bound and alpha-expansion 25.68 +- 2.41, in a median
93.0 ms against ICM's 6.8 ms.

**The tile strengths are drawn against ``J0``, not ``m J0``.** The scratch
trial rescaled one module constant for both the model's features and the draw,
so its training fields carried twice the strengths its labels were computed
at; here the draw and the coupling are separate arguments.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sal.search.potts_starts import recovery_bound, tiling_rung
from sal.sim.graph import BoundaryCondition, PottsGraph, triangular_lattice_graph
from sal.sim.potts import SpatioTilingParams, tile_partition, tiling_field

#: ``q``, `spatio_tiling/release`'s state count.
N_STATES = 10

#: ``J0``, `spatio_tiling/release`'s coupling; the tile strengths span it.
BASE_COUPLING = 0.7

#: ``m``: the noisy family's coupling is ``m J0``.
COUPLING_SCALE = 2.0

#: ``sigma``: the noise's standard deviation after smoothing.
NOISE_SCALE = 1.5

#: ``k``: rounds of neighbour averaging applied to the noise.
SMOOTHING_ROUNDS = 4

#: The planted strengths' range, as multiples of each tile's recovery bound:
#: above one so the bound's strict inequality holds with a margin.
PLANTED_MARGIN = (1.05, 2.5)

#: The mixed strengths' range, as multiples of ``J0``, drawn log-uniform.
MIXED_RANGE = (0.1, 10.0)


@dataclass(frozen=True)
class SeedRange:
    """Seeds ``base + per_side * side + index``: one block per lattice side.

    Parameters
    ----------
    base : int
        The range's first seed.
    per_side : int
        The block width per side; ``index`` stays below it, or zero where the
        range serves one side.
    """

    base: int
    per_side: int

    def seed(self, side: int, index: int) -> int:
        """The seed of draw ``index`` at ``side``.

        Returns
        -------
        int
        """
        return self.base + self.per_side * side + index


#: Mixed and planted training draws; side 99 ends below the validation base.
MIXED_TRAIN = SeedRange(0, 10**5)
#: Held-out mixed draws the arms are read on (#1067's table).
MIXED_VALIDATION = SeedRange(10**8, 1000)
#: Held-out mixed draws the curriculum's gates read, disjoint from validation.
MIXED_GATE = SeedRange(10**8 + 5 * 10**6, 1000)
#: The noisy family's parameter grid, at 41x41.
NOISY_GRID = SeedRange(3 * 10**8, 0)
#: The noisy family's validation draws, at 75x75.
NOISY_VALIDATION = SeedRange(4 * 10**8, 0)
#: The noisy family's gate draws.
NOISY_GATE = SeedRange(5 * 10**8, 1000)
#: The noisy family's training draws.
NOISY_TRAIN = SeedRange(6 * 10**8, 10**5)


def lattice(side: int, coupling_scale: float = 1.0) -> PottsGraph:
    """The ``side x side`` open triangular lattice at coupling ``coupling_scale * J0``.

    Returns
    -------
    PottsGraph
    """
    return triangular_lattice_graph(
        (side, side), BoundaryCondition.OPEN, BASE_COUPLING * coupling_scale
    )


def _params(
    graph: PottsGraph,
    tiles: np.ndarray,
    states: np.ndarray,
    strengths: np.ndarray,
    seed: int,
) -> SpatioTilingParams:
    """The fixture's own type, so ``tiling_rung`` and ``recovery_bound`` read it."""
    return SpatioTilingParams(
        graph=graph,
        n_states=N_STATES,
        tiles=tiles,
        states=states,
        strengths=strengths,
        field=tiling_field(tiles, states, strengths, N_STATES),
        tiling_seed=seed,
    )


def draw_tiling(graph: PottsGraph, seed: int, *, planted: bool) -> SpatioTilingParams:
    """One seeded tiling: its tiles, favoured states and strengths.

    The tile count is drawn from ``[max(3, n // 400), max(5, n // 100)]``,
    then the tiles, then one state per tile, then the strengths: log-uniform
    over :data:`MIXED_RANGE` times ``J0`` for a mixed draw, and
    :data:`PLANTED_MARGIN` times the tile's recovery bound, in ``graph``'s
    coupling, for a planted one.

    Returns
    -------
    SpatioTilingParams
    """
    rng = np.random.default_rng(seed)
    n = graph.n_nodes
    k = int(rng.integers(max(3, n // 400), max(5, n // 100) + 1))
    tiles = tile_partition(graph, k, rng)
    states = rng.integers(0, N_STATES, size=k)
    if planted:
        unit = _params(graph, tiles, states, np.ones(k), seed)
        bound = recovery_bound(tiling_rung(unit, "draw"))
        strengths = bound * rng.uniform(*PLANTED_MARGIN, size=k)
    else:
        low, high = (np.log(r * BASE_COUPLING) for r in MIXED_RANGE)
        strengths = np.exp(rng.uniform(low, high, size=k))
    return _params(graph, tiles, states, strengths, seed)


def neighbour_average(graph: PottsGraph, values: np.ndarray) -> np.ndarray:
    """One round of neighbour averaging: each site's row replaced by the mean over it and its neighbours.

    Edge by edge from :attr:`~sal.sim.graph.PottsGraph.edge_index`, uniform
    weights whatever the coupling.

    Returns
    -------
    np.ndarray
        ``values``' shape.
    """
    ends = graph.edge_index
    total = np.array(values, dtype=float)
    np.add.at(total, ends[:, 0], values[ends[:, 1]])
    np.add.at(total, ends[:, 1], values[ends[:, 0]])
    count = 1 + np.bincount(ends.ravel(), minlength=graph.n_nodes)
    return np.asarray(total / count[:, None])


def smoothed_noise(graph: PottsGraph, seed: int, rounds: int) -> np.ndarray:
    """Gaussian noise per (site, state), averaged ``rounds`` times and rescaled to unit std.

    Drawn from ``default_rng([seed, 1])``, a stream the tiling's
    ``default_rng(seed)`` does not share.

    Returns
    -------
    np.ndarray
        Shape ``(n_nodes, N_STATES)``, standard deviation one over all entries.
    """
    noise = np.random.default_rng([seed, 1]).standard_normal((graph.n_nodes, N_STATES))
    for _ in range(rounds):
        noise = neighbour_average(graph, noise)
    return np.asarray(noise / noise.std())


def noisy_field(
    side: int,
    seed: int,
    *,
    coupling_scale: float = COUPLING_SCALE,
    noise_scale: float = NOISE_SCALE,
    rounds: int = SMOOTHING_ROUNDS,
) -> tuple[PottsGraph, np.ndarray]:
    """The noisy, strongly coupled tiling field: ``tiling_field + sigma * noise`` at ``m J0``.

    The tiling is a mixed draw on the ``m J0`` lattice, its strengths against
    ``J0`` (module docstring).

    Returns
    -------
    tuple[PottsGraph, np.ndarray]
        The lattice at ``m J0``, and the field, shape ``(side**2, N_STATES)``.
    """
    graph = lattice(side, coupling_scale)
    params = draw_tiling(graph, seed, planted=False)
    return graph, params.field + noise_scale * smoothed_noise(graph, seed, rounds)


def mixed_field(side: int, seed: int) -> tuple[PottsGraph, np.ndarray]:
    """A mixed draw's lattice at ``J0`` and its field, the pair :func:`noisy_field` returns.

    Returns
    -------
    tuple[PottsGraph, np.ndarray]
    """
    graph = lattice(side)
    return graph, draw_tiling(graph, seed, planted=False).field
