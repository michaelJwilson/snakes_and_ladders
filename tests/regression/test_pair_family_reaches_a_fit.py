"""A family whose observation has its own axes, fitted (issue #658).

`baum_welch_family` unpacked `n_sequences, length = data.shape`, which is the
whole shape and not its leading two, so it refused outright every family whose
observation carries axes of its own — a family over a pair of counts among
them. #658 made those families condition on a covariate, and until this nothing
could call one in a fit: the family could be told an exposure and no fit could
tell it one.

The claim here is the one that shape change is for: a pair family fits, and
told the exposure its draw was made under it recovers the rate that generated
it where a fit not told it does not.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from snakes_and_ladders.emissions import BetaBinomialEmission, NegativeBinomialEmission
from snakes_and_ladders.opt.hmm import baum_welch_family
from snakes_and_ladders.sim.count_pairs import IndependentCountPair

_TRIALS = 40.0
_RATE = np.array([3.0, 12.0])
_INITIAL = torch.log(torch.tensor([0.5, 0.5], dtype=torch.float64))
_TRANSITION = torch.log(torch.tensor([[0.94, 0.06], [0.06, 0.94]], dtype=torch.float64))


def _family(mean: np.ndarray) -> IndependentCountPair:
    return IndependentCountPair(
        NegativeBinomialEmission(
            torch.tensor([8.0, 8.0], dtype=torch.float64),
            torch.as_tensor(mean, dtype=torch.float64),
        ),
        BetaBinomialEmission(
            torch.tensor([_TRIALS, _TRIALS], dtype=torch.float64),
            torch.tensor([2.0, 6.0], dtype=torch.float64),
            torch.tensor([6.0, 2.0], dtype=torch.float64),
        ),
    )


def _draw(spread: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """A sticky two-state pair chain drawn under a per-sequence exposure."""
    rng = np.random.default_rng(seed)
    n_sequences, length = 6, 400
    states = np.empty((n_sequences, length), dtype=np.int64)
    states[:, 0] = rng.integers(0, 2, size=n_sequences)
    for t in range(1, length):
        flip = rng.random(n_sequences) > 0.94
        states[:, t] = np.where(flip, 1 - states[:, t - 1], states[:, t - 1])

    covariate = np.empty((n_sequences, length, 2))
    covariate[..., 0] = rng.uniform(1.0 / spread, spread, size=(n_sequences, 1))
    covariate[..., 1] = _TRIALS
    observations = _family(_RATE).sample(
        states, rng, covariate=torch.as_tensor(covariate)
    )
    return observations, covariate


def _fit(observations: np.ndarray, covariate: np.ndarray | None) -> np.ndarray:
    result = baum_welch_family(
        observations,
        _INITIAL,
        _TRANSITION,
        _family(np.array([2.0, 8.0])),
        max_iterations=60,
        covariate=covariate,
    )
    family = result.emissions
    assert isinstance(family, IndependentCountPair)
    return np.sort(family.total.mean.detach().numpy())


@pytest.mark.structural
@pytest.mark.critical
def test_a_family_with_its_own_axes_fits_at_all() -> None:
    """The shape claim, on its own: the unpacking no longer refuses the family.

    Separate from recovery because it fails differently --- a `ValueError` out
    of the shape unpacking, before any arithmetic --- and a reader looking at
    a recovery failure should not have to rule this out first.
    """
    observations, _ = _draw(spread=1.0, seed=2)

    assert observations.ndim == 3
    assert np.isfinite(_fit(observations, None)).all()


@pytest.mark.simulated_truth
def test_told_the_exposure_the_fit_recovers_the_planted_rate() -> None:
    """Over an exposure spanning a factor of 9, the fit reaches what drew it."""
    observations, covariate = _draw(spread=3.0, seed=2)

    np.testing.assert_allclose(_fit(observations, covariate), _RATE, rtol=0.15)


@pytest.mark.simulated_truth
def test_not_told_it_the_same_fit_misses() -> None:
    """And it is worth telling: withheld, the same fit on the same data misses.

    It converges to the rate averaged over the exposures it was not told
    about, which for a `U(1/3, 3)` exposure is a factor near its mean.
    """
    observations, _ = _draw(spread=3.0, seed=2)

    untold = _fit(observations, None)

    assert np.abs(untold / _RATE - 1.0).max() > 0.25
