"""The joint count-pair HMM rebuilds from its seed, and Baum-Welch climbs from its truth (issue #1416).

`count_pair_hmm` declares experiment 030's instance, which
`sal.qa.count_pair_hmm_anneal` built in place before; the file pins the draw
by the digest that construction drew, so the experiment's numbers stand on
the fixture.
"""

from __future__ import annotations

import numpy as np
import pytest
from sal.cost import Cost
from sal.fixtures import Scale
from sal.opt.budget import Budget
from sal.opt.hmm import EmissionHmmObjective
from sal.opt.starts import polish_by_baum_welch
from sal.ragged import Ragged
from sal.sim.count_pair_hmm import CountPairHmmParams, observations_digest
from sal.sim.fixtures import fixture

#: The truth's negative log-likelihood, as experiment 030 reports it (63,361)
#: and the QA script's in-place construction computed it.
AT_TRUTH = 63361.013106580765


@pytest.mark.analytic
def test_the_count_pair_hmm_rebuilds_from_its_seed_bitwise() -> None:
    # The digest pins the path and the pairs; the chain is stochastic with
    # stationary law `occupancy`, and a success count never exceeds its total.
    params: CountPairHmmParams = fixture("count_pair_hmm", Scale.CI).params
    simulated = params.instance()
    pairs = np.asarray(simulated.observations).reshape(-1, 2)

    assert observations_digest(simulated) == params.observations_digest
    np.testing.assert_allclose(params.transition.sum(axis=1), 1.0, rtol=0, atol=1e-15)
    np.testing.assert_allclose(
        params.occupancy @ params.transition, params.occupancy, rtol=0, atol=1e-15
    )
    assert np.all(pairs[:, 1] <= pairs[:, 0])


@pytest.mark.end2end
def test_baum_welch_from_the_generating_parameters_does_not_lose_likelihood() -> None:
    # The generating parameters are a point of the likelihood, not its
    # maximum: twenty Baum-Welch iterations from them climb.
    params: CountPairHmmParams = fixture("count_pair_hmm", Scale.CI).params
    simulated = params.instance()
    data = Ragged(
        np.asarray(simulated.observations).reshape(-1, 2), (params.n_positions,)
    )
    objective = EmissionHmmObjective(data, params.components)
    theta = objective.theta_from_truth(
        params.occupancy, params.transition, **params.components.named_parameters()
    )
    polished = polish_by_baum_welch(objective, theta, Budget(Cost.ITERATIONS, 20))

    assert float(objective(theta)) == pytest.approx(AT_TRUTH, rel=1e-12)
    assert float(objective(polished.theta)) <= float(objective(theta)) * (1 + 1e-12)
