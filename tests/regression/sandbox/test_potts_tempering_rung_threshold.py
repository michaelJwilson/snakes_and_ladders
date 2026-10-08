"""Per-rung tempering moves against single-site tempering, at equal site visits and equal wall time (issue #1158).

The measurement behind :func:`~sal.sample.potts_mcmc.rung_moves`' threshold
and the ``tempering-mixed`` row of `search.ground_state.METHODS`: each ladder
through :func:`~sal.search.potts_tempering.run_tempering` for 1,000 sweeps' site
visits at five seeds, the field scaled x1, x10 and x30; equal wall time gives
each ladder the visits its mean seconds per visit buys in single-site's
seconds. Measured on the 4-core host, mean energy (standard error) at equal
visits on `spatio_only/release` x1: single-site -9,900.1 (10.9), heat-bath
Swendsen-Wang on the three hottest rungs and single-site below -9,740.4
(13.2), heat-bath Swendsen-Wang then single-site on every rung -10,215.1
(19.1), `rung_moves` -10,215.9 (19.0); at equal wall time -9,900.1,
-9,702.7, -10,212.7, -10,215.3. On `spatio_tiling/release` x1 at equal
visits: -17,001.2 (5.4), -16,985.0 (17.5), -17,018.6 (0.6), -17,018.9 (0.6).
At x10 and x30 every ladder reaches one energy on `spatio_tiling` and within
0.8 on `spatio_only` (standard errors to 0.6). A finding recorded where it
was measured, not a referee: `test_potts_tempering_rung_moves.py` referees
the law.
"""

from __future__ import annotations

import time

import numpy as np
import pytest
from sal.cost import Cost
from sal.opt.budget import Budget
from sal.sample.potts_mcmc import PottsMove, RungMoves, rung_moves
from sal.sandbox.potts_tempering import run_tempering, tempering_ladder
from sal.search.ground_state import Problem
from sal.search.potts_starts import spatio_rung, tiling_rung
from sal.sim.fixtures import fixture

SEEDS = range(5)
SWEEPS = 1_000
#: Standard errors of the difference within which a strong field's ladders
#: count as one energy.
TIED = 3.0


def _problem(name: str, scale: float) -> Problem:
    params = fixture(name, "release").params
    rung = (
        tiling_rung(params, "release")
        if name == "spatio_tiling"
        else spatio_rung(params, "release")
    )
    return Problem(rung.graph, rung.field * scale, rung.n_states)


def _runs(problem: Problem, move: RungMoves, size: int) -> tuple[np.ndarray, float]:
    """Each seed's energy, and the mean seconds a run took."""
    energies, seconds = [], []
    for seed in SEEDS:
        started = time.perf_counter()
        run = run_tempering(
            problem,
            Budget(Cost.SITE_VISITS, size),
            np.random.default_rng(seed),
            move=move,
        )
        seconds.append(time.perf_counter() - started)
        energies.append(run.energy)
    return np.array(energies), float(np.mean(seconds))


def _standard_error(values: np.ndarray) -> float:
    return float(values.std(ddof=1) / np.sqrt(values.size))


@pytest.mark.experiment
@pytest.mark.release
@pytest.mark.parametrize("scale", [1.0, 10.0, 30.0])
@pytest.mark.parametrize("name", ["spatio_tiling", "spatio_only"])
def test_rung_moves_reaches_lower_than_single_site_or_ties_it(
    name: str, scale: float
) -> None:
    # At x1 the per-rung moves reach a lower mean energy at equal visits and
    # at equal wall time; at x10 and x30 the two are one energy within
    # three standard errors of the difference.
    problem = _problem(name, scale)
    size = SWEEPS * (problem.n_nodes + 2 * len(problem.graph.edges))
    mixed = rung_moves(problem.graph, problem.field, tempering_ladder())

    single, single_seconds = _runs(problem, PottsMove.SINGLE_SITE, size)
    visits, mixed_seconds = _runs(problem, mixed, size)
    wall, _ = _runs(problem, mixed, int(size * single_seconds / mixed_seconds))

    for paired in (visits, wall):
        if scale == 1.0:
            assert paired.mean() < single.mean()
        else:
            spread = np.hypot(_standard_error(paired), _standard_error(single))
            assert abs(paired.mean() - single.mean()) <= TIED * spread + 1e-9
