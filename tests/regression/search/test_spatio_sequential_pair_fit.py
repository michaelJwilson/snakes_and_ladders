"""The coupled fit under a per-channel covariate (issue #670).

`m_step` re-derived the covariate's slice rather than calling
`likelihood.spatio_sequential.covariate_block`, and appended the trailing
singleton unconditionally. On an ``(S, V, 2)`` covariate that gives
``(n_m, S, 2, 1)``, whose last axis names no channel, so the pair family
refused it: the params admitted a covariate, the E step scored with it, and
the M step could not re-estimate under it.

What referees the fix is recovery and not shape. A planted pair of rates is
recovered by a fit told the exposure and missed by the same fit on the same
data not told it, by the factor the exposure averages to --- the referee #658
states for the families, raised here to the coupled fit.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch
from snakes_and_ladders.emissions import CovariateNotSupportedError
from snakes_and_ladders.likelihood.spatio_sequential import (
    ClassPosteriors,
    class_posteriors,
    covariate_block,
)
from snakes_and_ladders.search.spatio_sequential import fit_spatio_sequential, m_step
from snakes_and_ladders.sim.count_pairs import IndependentCountPair
from snakes_and_ladders.sim.fixtures import fixture
from snakes_and_ladders.sim.graph import BoundaryCondition, lattice_graph
from snakes_and_ladders.sim.spatio_sequential import (
    SpatioSequentialParams,
    simulate_spatio_sequential,
)

#: Four vertices and 400 positions: the smallest instance on which the
#: exposure's factor of 16 separates the two fits well past the sampling error.
N_POSITIONS = 400
LABELS = np.array([0, 0, 1, 1])

#: The exposure's range. `U(0.25, 4)` spans a factor of 16 and averages 2.125,
#: which is the factor a fit not told it must inflate the rate by.
LOW, HIGH = 0.25, 4.0


def _planted() -> tuple[SpatioSequentialParams, np.ndarray]:
    """A two-class count-pair model under a varying exposure, and its draw."""
    base = fixture("spatio_sequential_counts", "ci").params.model
    rng = np.random.default_rng(5)
    covariate = np.stack(
        [
            rng.uniform(LOW, HIGH, (N_POSITIONS, 4)),
            np.full((N_POSITIONS, 4), 40.0),
        ],
        axis=-1,
    )
    params = replace(
        base,
        graph=lattice_graph((2, 2), BoundaryCondition.OPEN, 1.0),
        n_positions=N_POSITIONS,
        covariate=covariate,
    )
    drawn = simulate_spatio_sequential(params, np.random.default_rng(1), labels=LABELS)
    return params, drawn.observations


def _posteriors(
    params: SpatioSequentialParams, observations: np.ndarray
) -> ClassPosteriors:
    """The E step at the planted labels, the input :func:`m_step` takes."""
    return class_posteriors(params, observations, LABELS)


def _rates(params: SpatioSequentialParams) -> list[np.ndarray]:
    """Every class's negative-binomial means."""
    rates = []
    for family in params.emissions:
        assert isinstance(family, IndependentCountPair)
        rates.append(family.total.mean.numpy())
    return rates


@pytest.mark.end2end
def test_a_fit_told_the_exposure_recovers_the_planted_rates() -> None:
    params, observations = _planted()
    fit = fit_spatio_sequential(
        params, observations, np.random.default_rng(3), labels=LABELS, n_blocks=8
    )

    for planted, fitted in zip(_rates(params), _rates(fit.params), strict=True):
        assert np.all(np.abs(fitted - planted) < planted * 0.05)


@pytest.mark.end2end
def test_the_same_fit_not_told_the_exposure_inflates_the_rates_by_its_mean() -> None:
    # The other half of the referee: the covariate must change the answer, and
    # change it by about the amount the model says. Without it the fit explains
    # the counts with a rate averaged over the exposures, so every planted rate
    # comes back scaled rather than recovered. The factor is near E[U(0.25, 4)]
    # = 2.125 and not equal to it --- the rate that best explains a mixture of
    # exposures is not the mixture's mean --- so the band is stated at 12%,
    # against the measured 1.994 and 2.107.
    params, observations = _planted()
    blind = fit_spatio_sequential(
        replace(params, covariate=None),
        observations,
        np.random.default_rng(3),
        labels=LABELS,
        n_blocks=8,
    )

    average = 0.5 * (LOW + HIGH)
    for planted, fitted in zip(_rates(params), _rates(blind.params), strict=True):
        factor = fitted / planted
        assert np.all(np.abs(factor - average) < average * 0.12)
        # And the miss is an order past the 5% the told fit meets, so the two
        # tests cannot both pass on a fit that ignores its covariate.
        assert np.all(np.abs(factor - 1.0) > 0.5)


@pytest.mark.smoke
def test_the_m_step_slices_the_covariate_the_likelihood_module_does() -> None:
    # The slice is `covariate_block`'s and not this module's, which is the
    # whole of the fix: two expressions for one pairing is how a fit comes to
    # condition on the wrong exposures and converge anyway.
    params, observations = _planted()
    members = np.flatnonzero(LABELS == 0)
    fitted = m_step(
        params,
        observations,
        LABELS,
        _posteriors(params, observations),
    )

    block = covariate_block(params, members)
    assert block is not None
    assert block.movedim(1, 0).shape == (members.size, N_POSITIONS, 2)
    assert isinstance(fitted.emissions[0], IndependentCountPair)


@pytest.mark.edge_case
def test_the_slice_the_m_step_used_to_build_is_refused_by_the_family() -> None:
    # The defect, reproduced: appending the singleton to a covariate that
    # already carries the channel axis gives a last axis of 1, which names no
    # channel. Kept as a test so the expression cannot come back.
    params, _ = _planted()
    members = np.flatnonzero(LABELS == 0)
    covariate = params.covariate
    assert covariate is not None
    stale = torch.as_tensor(np.moveaxis(covariate[:, members], 1, 0)[..., None])
    family = params.emissions[0]

    assert stale.shape[-1] == 1
    with pytest.raises(CovariateNotSupportedError, match="names no channel"):
        family.log_density(
            torch.as_tensor(np.zeros((members.size, N_POSITIONS, 2))),
            covariate=stale,
        )
