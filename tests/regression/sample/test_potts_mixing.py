"""A frozen Potts chain says so on its result (issue #1316, Part A).

Declared before any run: :data:`~sal.sample.potts_mcmc.chains.ESS_FLOOR`
(50 effective draws, #1276's floor) and
:data:`~sal.sample.potts_mcmc.chains.RHAT_THRESHOLD` (1.01, Vehtari et al.
2021). #1314's instance --- 64 x 64 periodic, q = 3, a field N(0, 1) per
(site, label), 1.5 beta_c, Niedermayer --- froze from the ordered start; a
2 x 3 enumerable chain under the heat bath mixes.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from sal.opt.termination import Stop
from sal.parallel import Pool
from sal.sample.potts_mcmc import (
    PottsMove,
    sample_potts,
    sample_potts_starts,
)
from sal.sample.potts_mcmc.chains import RHAT_THRESHOLD, observables
from sal.sample.statistics import split_rhat
from sal.sim.graph import BoundaryCondition, lattice_graph
from sal.sim.potts import critical_coupling

from tests._chains import enumerated_law

SEED = 1316
#: #1314's temperature: 1.5 of the critical coupling, the coupling held at 1.
TEMPERATURE = 1.0 / (1.5 * critical_coupling(3))


@pytest.mark.oracle
def test_an_enumerable_heat_bath_chain_converges_on_its_exact_energy() -> None:
    # Declared: every chain CONVERGED, R-hat at most RHAT_THRESHOLD, and the
    # pooled mean energy within 4 standard errors of the enumerated one.
    graph = lattice_graph((2, 3), BoundaryCondition.OPEN, 1.0)
    field = np.random.default_rng(1314).normal(0.0, 1.0, (6, 3))
    starts = sample_potts_starts(
        graph,
        field,
        PottsMove.SINGLE_SITE,
        np.random.default_rng(SEED),
        4_000,
        100,
        temperature=TEMPERATURE,
    )
    index, probability = enumerated_law(graph, field, temperature=TEMPERATURE)
    configurations = np.array(list(index), dtype=np.int64)
    exact = float(probability @ observables(graph, field, configurations)[0])
    energy = np.concatenate(
        [observables(graph, field, chain.states)[0] for chain in starts.chains]
    )
    ess = sum(float(chain.ess[0]) for chain in starts.chains)

    assert starts.termination.reason is Stop.CONVERGED
    assert all(chain.termination.converged for chain in starts.chains)
    assert np.all(starts.rhat <= RHAT_THRESHOLD), starts.rhat
    assert abs(energy.mean() - exact) < 4.0 * energy.std() / math.sqrt(ess)


@pytest.mark.analytic
def test_a_forbidden_field_chain_reports_no_move_by_hand() -> None:
    # Labels 1 and 2 forbidden: every heat-bath step leaves the ordered start
    # where it is. By hand: acceptance 0, no cluster so share 1, a constant
    # energy so ess[0] = 0 and NOT_MIXING after (2 + 5) * 3 = 21 steps.
    graph = lattice_graph((2, 3), BoundaryCondition.OPEN, 1.0)
    field = np.array([0.0, -np.inf, -np.inf])
    chain = sample_potts(
        graph,
        field,
        PottsMove.SINGLE_SITE,
        np.random.default_rng(SEED),
        5,
        2,
        3,
        start=np.zeros(6, dtype=np.int64),
    )

    assert chain.acceptance == 0.0
    assert chain.largest_cluster_share == 1.0
    assert chain.mean_cluster_size == 6.0
    assert chain.ess[0] == 0.0
    assert np.all(np.isinf(chain.ess[1:]))
    assert chain.termination.reason is Stop.NOT_MIXING
    assert chain.termination.iterations == 21


@pytest.mark.analytic
def test_split_rhat_matches_its_formula_by_hand() -> None:
    # Two chains of four: halves (0, 1), (2, 3), (1, 1), (1, 3). Variances
    # 0.5, 0.5, 0, 2 give W = 0.75; half means 0.5, 2.5, 1, 2 have variance
    # 5/6, so var+ = 0.5 * 0.75 + 5/6 = 29/24 and R = sqrt(29/18).
    chains = np.array([[0.0, 1.0, 2.0, 3.0], [1.0, 1.0, 1.0, 3.0]])

    assert split_rhat(chains) == pytest.approx(math.sqrt(29.0 / 18.0), rel=1e-15)
    assert split_rhat(np.ones((3, 8))) == 1.0
    assert split_rhat(np.array([[0.0] * 4, [1.0] * 4])) == math.inf


#: The serial run and a thread pool of three, one worker per start.
POOLS: tuple[tuple[int, Pool], ...] = ((1, "serial"), (3, "threads"))
