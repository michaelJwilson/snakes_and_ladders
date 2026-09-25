"""A covariate per channel on the pair families (issue #658, item 1).

#631 gave the single-channel families an exposure and a trial count and refused
both on a pair, because "the two channels would each need their own --- an
exposure for the total, a trial count for the successes --- and one tensor
cannot be both". It can, with the channel axis the observation already has:
`(..., 2)`, splitting where the observation splits.

The coupled model is declared over these families, so until this every fit over
count pairs was a fit that could not condition on anything.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from sal.emissions import (
    BetaBinomialEmission,
    CovariateNotSupportedError,
    NegativeBinomialEmission,
)
from sal.sim.count_pairs import IndependentCountPair

#: The declared trial count. It is also the *neutral* covariate for the
#: successes channel, which is the asymmetry this module exists to pin.
_TRIALS = 20.0


def _family() -> IndependentCountPair:
    return IndependentCountPair(
        NegativeBinomialEmission(
            torch.tensor([4.0, 6.0], dtype=torch.float64),
            torch.tensor([3.0, 9.0], dtype=torch.float64),
        ),
        BetaBinomialEmission(
            torch.tensor([_TRIALS, _TRIALS], dtype=torch.float64),
            torch.tensor([2.0, 5.0], dtype=torch.float64),
            torch.tensor([5.0, 2.0], dtype=torch.float64),
        ),
    )


def _observations() -> torch.Tensor:
    return torch.tensor([[[6.0, 3.0], [11.0, 7.0]]], dtype=torch.float64)


def _neutral(observations: torch.Tensor) -> torch.Tensor:
    """Exposure one for the total, the declared trials for the successes."""
    neutral = torch.ones_like(observations)
    neutral[..., 1] = _TRIALS
    return neutral


@pytest.mark.smoke
@pytest.mark.critical
def test_the_neutral_covariate_reproduces_the_uncovaried_score_bitwise() -> None:
    """The referee for admitting the shape: neutral changes nothing at all.

    Bitwise: an exposure of one and the declared trial count change nothing.
    """
    family, observations = _family(), _observations()

    assert torch.equal(
        family.log_density(observations),
        family.log_density(observations, covariate=_neutral(observations)),
    )


@pytest.mark.smoke
@pytest.mark.critical
def test_the_neutral_covariate_is_not_ones_on_both_channels() -> None:
    """The asymmetry a caller has to know, asserted rather than documented only.

    Ones on both asks for 3 and 7 successes out of 1 trial: `-inf`.
    """
    family, observations = _family(), _observations()

    scored = family.log_density(observations, covariate=torch.ones_like(observations))

    assert torch.isinf(scored).any()
    assert torch.isfinite(family.log_density(observations)).all()


@pytest.mark.analytic
def test_each_channel_moves_the_score_on_its_own() -> None:
    """Both covariates reach a family, and neither is the other's.

    Two movements from one neutral, so a covariate on the wrong channel fails.
    """
    family, observations = _family(), _observations()
    neutral = _neutral(observations)
    base = family.log_density(observations, covariate=neutral)

    exposed = neutral.clone()
    exposed[..., 0] = 2.5
    trialled = neutral.clone()
    trialled[..., 1] = 2.0 * _TRIALS

    assert (
        float((base - family.log_density(observations, covariate=exposed)).abs().max())
        > 1.0
    )
    assert (
        float((base - family.log_density(observations, covariate=trialled)).abs().max())
        > 1.0
    )


@pytest.mark.smoke
def test_a_covariate_without_the_channel_axis_is_refused() -> None:
    """The tensor #631 was right about: nothing says which channel it is.

    Checked by width alone; refused only where it cannot be the channels.
    """
    family, observations = _family(), _observations()

    with pytest.raises(CovariateNotSupportedError, match="one covariate per channel"):
        family.log_density(observations, covariate=torch.ones(1, 2, 1))
    with pytest.raises(CovariateNotSupportedError, match="one covariate per channel"):
        family.reestimate(
            observations,
            torch.full((1, 2, 2), 0.5, dtype=torch.float64),
            covariate=torch.ones(1, 2, 3),
        )


@pytest.mark.end2end
def test_a_draw_under_an_exposure_scales_with_it() -> None:
    """`sample` conditions too, which is what item 4 needs to plant an instance.

    A tenfold exposure is a tenfold mean, over enough draws to be the model's.
    """
    family = _family()
    states = np.zeros(4000, dtype=np.int64)
    low = torch.ones(4000, 2, dtype=torch.float64)
    low[:, 1] = _TRIALS
    high = low.clone()
    high[:, 0] = 10.0

    drawn_low = family.sample(states, np.random.default_rng(5), covariate=low)
    drawn_high = family.sample(states, np.random.default_rng(5), covariate=high)

    ratio = drawn_high[:, 0].mean() / drawn_low[:, 0].mean()
    assert 8.5 < ratio < 11.5, f"a tenfold exposure gave a {ratio:.2f}-fold mean"
