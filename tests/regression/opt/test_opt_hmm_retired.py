"""The six retired per-family HMM objectives: deprecated constructors of `EmissionHmmObjective` (issue #1189).

Referee: the `EmissionHmmObjective` at :func:`~sal.opt.hmm.family_start` each
name now builds, bitwise in ``theta``'s start, value and gradient. That the
package constructs none of the six is `tests/regression/test_retired_names.py`.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch
from sal.backend import Backend
from sal.emissions import (
    BetaBinomialEmission,
    BinomialEmission,
    CategoricalEmission,
    GaussianEmission,
    NegativeBinomialEmission,
    PoissonEmission,
)
from sal.opt.hmm import (
    BetaBinomialHmmObjective,
    BinomialHmmObjective,
    EmissionHmmObjective,
    GaussianHmmObjective,
    HmmObjective,
    NegativeBinomialHmmObjective,
    PoissonHmmObjective,
    family_start,
)
from sal.opt.objective import value_and_gradient


def _pairs() -> dict[str, tuple[object, EmissionHmmObjective]]:
    """Each retired call, unbuilt, beside the `EmissionHmmObjective` it stands for, on two sequences of 30."""
    rng = np.random.default_rng(1189)
    counts = rng.poisson(5.0, size=(2, 30))
    successes = rng.binomial(12, 0.3, size=(2, 30))
    reals = rng.normal(size=(2, 30))
    symbols = rng.integers(0, 3, size=(2, 30))
    exposure = rng.uniform(0.5, 2.0, size=(2, 30))
    trials = np.array([12, 12])
    return {
        "categorical": (
            lambda: HmmObjective(symbols, 2, 3),
            EmissionHmmObjective(
                symbols,
                family_start(CategoricalEmission, symbols, 2, n_symbols=3),
                backend=Backend.JAX,
            ),
        ),
        "gaussian": (
            lambda: GaussianHmmObjective(reals, 2),
            EmissionHmmObjective(
                reals, family_start(GaussianEmission, reals, 2), backend=Backend.JAX
            ),
        ),
        "poisson": (
            lambda: PoissonHmmObjective(counts, 2),
            EmissionHmmObjective(
                counts, family_start(PoissonEmission, counts, 2), backend=Backend.JAX
            ),
        ),
        "binomial": (
            lambda: BinomialHmmObjective(successes, 2, trials),
            EmissionHmmObjective(
                successes,
                family_start(BinomialEmission, successes, 2, trials=trials),
                backend=Backend.JAX,
            ),
        ),
        "beta_binomial": (
            lambda: BetaBinomialHmmObjective(successes, 2, trials),
            EmissionHmmObjective(
                successes,
                family_start(BetaBinomialEmission, successes, 2, trials=trials),
                backend=Backend.JAX,
            ),
        ),
        "negative_binomial_exposure": (
            lambda: NegativeBinomialHmmObjective(counts, 2, covariate=exposure),
            EmissionHmmObjective(
                counts,
                family_start(NegativeBinomialEmission, counts, 2),
                covariate=exposure[..., None],
                backend=Backend.JAX,
            ),
        ),
    }


@pytest.mark.smoke
@pytest.mark.warning
@pytest.mark.parametrize("name", list(_pairs()))
def test_a_retired_name_warns_and_builds_its_emission_hmm_objective(name: str) -> None:
    # Bitwise: the same start family through the same maps, and the same JAX
    # route the retired name defaulted to.
    build, expected = _pairs()[name]
    with pytest.warns(DeprecationWarning, match="EmissionHmmObjective"):
        retired = build()  # type: ignore[operator]
    assert isinstance(retired, EmissionHmmObjective)
    theta = expected.initial() + 0.1
    assert torch.equal(retired.initial(), expected.initial())
    assert dict(retired.blocks) == dict(expected.blocks)
    for got, want in zip(
        value_and_gradient(retired, theta),
        value_and_gradient(expected, theta),
        strict=True,
    ):
        assert torch.equal(got, want)


@pytest.mark.smoke
@pytest.mark.warning
def test_a_retired_truth_point_is_theta_from_truth_by_name() -> None:
    # The retired positional `theta_from_truth` against the keyword one.
    reals = np.random.default_rng(0).normal(size=(2, 30))
    initial, transition = np.array([0.4, 0.6]), np.array([[0.9, 0.1], [0.2, 0.8]])
    mean, scale = np.array([-1.0, 1.0]), np.array([0.5, 2.0])
    with pytest.warns(DeprecationWarning):
        retired = GaussianHmmObjective(reals, 2)
    expected = EmissionHmmObjective(reals, family_start(GaussianEmission, reals, 2))
    assert torch.equal(
        retired.theta_from_truth(initial, transition, mean, scale),
        expected.theta_from_truth(initial, transition, mean=mean, scale=scale),
    )
    assert retired.variance_floor == expected.start.variance_floor  # type: ignore[attr-defined]


@pytest.mark.smoke
def test_the_successor_raises_no_deprecation() -> None:
    reals = np.random.default_rng(1).normal(size=(2, 30))
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        EmissionHmmObjective(reals, family_start(GaussianEmission, reals, 2))
