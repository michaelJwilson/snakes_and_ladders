"""Tempering on cluster-move replicas against single-site tempering, at equal sweeps (issue #1156).

The measurement behind leaving `tempering-*` cluster rows out of
`search.ground_state.METHODS`: each move tempered through
:func:`~sal.search.ground_state.run_tempering` for 1,000 sweeps' site visits
(six replicas, 166 steps each) at five seeds. Measured on the 4-core host,
mean energy (standard error) on `spatio_tiling/release` (71 x 71 triangular,
q = 10, J = 0.7): single-site -17,001.2 (5.4), heat-bath Swendsen-Wang
-16,914.1 (3.7), Swendsen-Wang -16,227.3 (32.1), heat-bath Wolff -2,320.6
(21.7), Wolff -1,956.4 (17.3). On `spatio_only/release`: single-site -9,900.1
(10.9) against -9,021.5 (11.7) for the nearest cluster move. A finding
recorded where it was measured, not a referee:
`tests/regression/sample/test_potts_tempering_cluster.py` referees the law.
"""

from __future__ import annotations

import numpy as np
import pytest
from sal.cost import Cost
from sal.opt.budget import Budget
from sal.sample.potts_mcmc import PottsMove
from sal.search.ground_state import run_tempering
from sal.search.potts_starts import tiling_rung
from sal.sim.fixtures import fixture

SEEDS = range(5)
SWEEPS = 1_000
CLUSTER_MOVES = (
    PottsMove.SWENDSEN_WANG,
    PottsMove.WOLFF,
    PottsMove.SWENDSEN_WANG_HEAT_BATH,
    PottsMove.WOLFF_HEAT_BATH,
)


@pytest.mark.experiment
@pytest.mark.release
@pytest.mark.parametrize("move", CLUSTER_MOVES, ids=str)
def test_single_site_tempering_reaches_lower_than_cluster_tempering(
    move: PottsMove,
) -> None:
    rung = tiling_rung(fixture("spatio_tiling", "release").params, "release")
    budget = Budget(Cost.SITE_VISITS, SWEEPS * rung.visits_per_sweep)

    def mean_energy(tempered: PottsMove) -> float:
        return float(
            np.mean(
                [
                    run_tempering(
                        rung, budget, np.random.default_rng(seed), move=tempered
                    ).energy
                    for seed in SEEDS
                ]
            )
        )

    assert mean_energy(PottsMove.SINGLE_SITE) < mean_energy(move)
