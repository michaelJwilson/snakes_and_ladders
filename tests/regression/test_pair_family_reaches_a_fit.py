"""A family whose observation has its own axes, fitted (issue #658).

`baum_welch_family` unpacked the whole data shape as two axes, refusing every
family with observation axes of its own, such as a pair of counts. Claimed: a
pair family fits, and told its exposure it recovers the generating rate where
a fit not told it does not.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch
from sal.emissions import BetaBinomialEmission, NegativeBinomialEmission
from sal.opt.em import EM
from sal.opt.hmm import baum_welch_family
from sal.sim.count_pairs import IndependentCountPair

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
        covariate=covariate,
        config=replace(EM, max_iterations=60),
    )
    family = result.components
    assert isinstance(family, IndependentCountPair)
    return np.sort(family.total.mean.detach().numpy())


@pytest.mark.smoke
@pytest.mark.critical
def test_a_family_with_its_own_axes_fits_at_all() -> None:
    """The shape claim, on its own: the unpacking does not refuse the family.

    Separate from recovery: it fails as a `ValueError` before any arithmetic.
    """
    observations, _ = _draw(spread=1.0, seed=2)

    assert observations.ndim == 3
    assert np.isfinite(_fit(observations, None)).all()


@pytest.mark.end2end
def test_told_the_exposure_the_fit_recovers_the_planted_rate() -> None:
    """Over an exposure spanning a factor of 9, the fit reaches what drew it."""
    observations, covariate = _draw(spread=3.0, seed=2)

    np.testing.assert_allclose(_fit(observations, covariate), _RATE, rtol=0.15)


@pytest.mark.end2end
def test_not_told_it_the_same_fit_misses() -> None:
    """And it is worth telling: withheld, the same fit on the same data misses.

    It converges near the rate times the `U(1/3, 3)` exposure's mean.
    """
    observations, _ = _draw(spread=3.0, seed=2)

    untold = _fit(observations, None)

    assert np.abs(untold / _RATE - 1.0).max() > 0.25
