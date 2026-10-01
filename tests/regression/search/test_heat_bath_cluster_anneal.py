"""The heat-bath cluster moves against the uniform-proposal ones, annealed in a strong field (issue #1142).

The measurement behind their rows in `search.ground_state.METHODS`: on
`spatio_tiling/release` (71 x 71 triangular, q = 10, J = 0.7, a field of 0.1 J
to 10 J favouring one planted state per tile), each move annealed for 1,000
sweeps on the comparison's schedule at five seeds, through the METHODS entry
itself. Measured on the 4-core host, mean energy (standard error):
Swendsen-Wang -16,895.8 (7.9) against the heat bath's -16,934.1 (3.7); Wolff
-2,428.7 (24.6) against -3,902.5 (11.0). The merge after each lowers no
Swendsen-Wang run and every Wolff one. A finding recorded where it was
measured, not a referee: `test_potts_heat_bath_cluster.py` and
`test_potts_mcmc.py` referee the moves' law.
"""

from __future__ import annotations

import numpy as np
import pytest
from sal.cost import Cost
from sal.opt.budget import Budget
from sal.search.ground_state import METHODS
from sal.search.icm import merge_labels
from sal.search.potts_starts import tiling_rung
from sal.sim.fixtures import fixture

SEEDS = range(5)
SWEEPS = 1_000
PAIRS = (
    ("swendsen-wang", "swendsen-wang-heat-bath"),
    ("wolff", "wolff-heat-bath"),
)


@pytest.mark.experiment
@pytest.mark.release
@pytest.mark.parametrize(("uniform", "heat_bath"), PAIRS)
def test_the_heat_bath_anneals_lower_than_the_uniform_proposal(
    uniform: str, heat_bath: str
) -> None:
    rung = tiling_rung(fixture("spatio_tiling", "release").params, "release")
    budget = Budget(Cost.SITE_VISITS, SWEEPS * rung.visits_per_sweep)
    energies = {
        name: np.array(
            [
                METHODS[name](rung, budget, np.random.default_rng(seed)).energy
                for seed in SEEDS
            ]
        )
        for name in (uniform, heat_bath)
    }

    assert energies[heat_bath].mean() < energies[uniform].mean()


@pytest.mark.experiment
@pytest.mark.release
def test_the_merge_lowers_every_annealed_wolff_energy() -> None:
    rung = tiling_rung(fixture("spatio_tiling", "release").params, "release")
    budget = Budget(Cost.SITE_VISITS, SWEEPS * rung.visits_per_sweep)
    for name in ("wolff", "wolff-heat-bath"):
        for seed in SEEDS:
            run = METHODS[name](rung, budget, np.random.default_rng(seed))
            merged = merge_labels(rung.graph, rung.field, run.labelling)
            assert merged.energy < run.energy, (name, seed)
