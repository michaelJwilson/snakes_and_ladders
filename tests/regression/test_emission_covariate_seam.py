"""The covariate seam: accepted as ``None``, refused otherwise (issue #631).

Step 1 of the plan on that ticket adds ``covariate`` to the three data-facing
members of :class:`~snakes_and_ladders.emissions.EmissionFamily` and to all
eight implementers, and nothing else. What is asserted here is the seam's two
halves --- that ``None`` changes no behaviour, and that a covariate is
*refused* rather than dropped --- because a covariate silently ignored fits a
different likelihood than the caller asked for, and the failure would read as
a worse recovery rather than as an error.

The families that will condition on one, ``NegativeBinomialEmission`` and
``BetaBinomialEmission``, refuse it here too: the path exists before the
mathematics does, and accepting an argument that does nothing would be exactly
the silent change this guards against.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from snakes_and_ladders.emissions import (
    BetaBinomialEmission,
    BinomialEmission,
    CategoricalEmission,
    CountPairEmission,
    CovariateNotSupportedError,
    EmissionFamily,
    GaussianEmission,
    NegativeBinomialEmission,
    PoissonEmission,
)


def _families() -> list[EmissionFamily]:
    """One instance of every implementer, at two states."""
    return [
        CategoricalEmission(torch.tensor([[0.7, 0.3], [0.2, 0.8]])),
        GaussianEmission(
            torch.tensor([0.0, 3.0]), torch.tensor([1.0, 2.0]), variance_floor=1e-6
        ),
        NegativeBinomialEmission(torch.tensor([2.0, 5.0]), torch.tensor([1.0, 4.0])),
        PoissonEmission(torch.tensor([1.0, 4.0])),
        BinomialEmission(torch.tensor([10.0, 10.0]), torch.tensor([0.3, 0.7])),
        BetaBinomialEmission(
            torch.tensor([10.0, 10.0]),
            torch.tensor([2.0, 5.0]),
            torch.tensor([5.0, 2.0]),
        ),
    ]


IDS = [type(family).__name__ for family in _families()]


@pytest.mark.critical
@pytest.mark.structural
@pytest.mark.parametrize("family", _families(), ids=IDS)
def test_a_covariate_is_refused_and_names_the_families_that_will_take_one(
    family: EmissionFamily,
) -> None:
    # Refused on all three data-facing members, by every implementer. The two
    # count families that gain a covariate later refuse it now, so the step
    # that adds the mathematics is the step the behaviour changes in.
    covariate = torch.ones(2, 1)
    states = np.array([0, 1])
    rng = np.random.default_rng(20260916)
    observations = torch.zeros(2, dtype=family.observation_dtype)
    posterior = torch.full((1, 2, family.n_states), 1.0 / family.n_states)

    with pytest.raises(CovariateNotSupportedError, match="issue #631"):
        family.sample(states, rng, covariate=covariate.reshape(-1))
    with pytest.raises(CovariateNotSupportedError, match="issue #631"):
        family.log_density(observations, covariate=covariate)
    with pytest.raises(CovariateNotSupportedError, match="issue #631"):
        family.reestimate(
            observations.reshape(1, 2), posterior, covariate=covariate.reshape(1, 2)
        )


@pytest.mark.critical
@pytest.mark.structural
@pytest.mark.parametrize("family", _families(), ids=IDS)
def test_passing_none_scores_what_passing_nothing_scores(
    family: EmissionFamily,
) -> None:
    # The seam's other half, and the one a default argument can break quietly:
    # the new keyword left at its default must be the old call, bitwise.
    observations = torch.zeros(3, dtype=family.observation_dtype)

    assert torch.equal(
        family.log_density(observations), family.log_density(observations, None)
    )


@pytest.mark.critical
@pytest.mark.structural
def test_the_pair_family_refuses_one_too() -> None:
    # `CountPairEmission` takes a tuple observation, so it is built and scored
    # apart from the six above rather than left out of the assertion.
    family = CountPairEmission(
        torch.tensor([20.0, 20.0]),
        torch.tensor([2.0, 5.0]),
        torch.tensor([5.0, 2.0]),
        torch.tensor([3.0, 6.0]),
        joint=True,
    )
    observations = torch.zeros(2, 2, dtype=family.observation_dtype)

    with pytest.raises(CovariateNotSupportedError, match="CountPairEmission"):
        family.log_density(observations, covariate=torch.ones(2, 1))
    assert torch.equal(
        family.log_density(observations), family.log_density(observations, None)
    )
