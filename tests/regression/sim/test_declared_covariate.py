"""The continuous covariate a count-pair fixture declares (issue #1064).

`spatio_sequential_counts_covariate` declares a seeded log-normal exposure and
trial count per position, vertex and channel. Refereed by closed forms: the
log of a log-normal draw is normal with the declared median and sigma, and
under an exposure ``e`` the negative binomial total has mean ``e mu``, so the
ratio of summed counts to summed exposures recovers ``mu`` per class and
state. The refusals are asserted beside them.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from sal.fixtures import load_declared
from sal.sim.count_pairs import (
    SUCCESSES,
    TOTAL,
    IndependentCountPair,
    SpatioSequentialCountsParams,
)
from sal.sim.count_pairs.rust import fine_instance
from sal.sim.fixtures import fixture

PROBLEM = "spatio_sequential_counts_covariate"

#: Standard errors a sample moment may sit from its closed form. The
#: log-exposure's mean over 64,000 draws at sigma 0.5 has a standard error of
#: 0.0020; four of them bound a false failure below 1e-4.
_STANDARD_ERRORS = 4.0

#: Relative tolerance on a per-(class, state) ratio of summed counts to summed
#: exposures, about 16,000 counts each: the 3% the count-pair moment tests use
#: (`tests/regression/sim/test_pair_covariates.py`).
_MOMENT = 0.03


def _declared() -> SpatioSequentialCountsParams:
    """The ci declaration, its covariate drawn."""
    declared = fixture(PROBLEM, "ci").params
    assert isinstance(declared, SpatioSequentialCountsParams)
    return declared


def _raw() -> dict[str, Any]:
    """The ci file as parsed, for a refusal to edit."""
    return dict(load_declared(fixture(PROBLEM, "ci").path, ()))


@pytest.mark.oracle
def test_the_declared_covariate_is_the_declared_log_normal() -> None:
    # log(exposure) ~ N(log median, sigma^2) in closed form; the trial count is
    # the same law rounded, so it is checked for being whole and positive and
    # for its median, which rounding moves by at most one half.
    raw = _raw()["covariate"]
    covariate = _declared().model.covariate
    assert covariate is not None
    n = covariate[..., TOTAL].size

    log_exposure = np.log(covariate[..., TOTAL])
    sigma = float(raw["exposure"]["sigma"])
    assert np.unique(covariate[..., TOTAL]).size == n
    assert abs(log_exposure.mean() - np.log(raw["exposure"]["median"])) < (
        _STANDARD_ERRORS * sigma / np.sqrt(n)
    )
    assert abs(log_exposure.std() - sigma) < _STANDARD_ERRORS * sigma / np.sqrt(2 * n)
    trials = covariate[..., SUCCESSES]
    assert np.array_equal(trials, np.rint(trials))
    assert trials.min() >= 1.0
    assert abs(float(np.median(trials)) - float(raw["trials"]["median"])) <= 1.0


@pytest.mark.oracle
def test_the_counts_are_drawn_against_their_own_exposure() -> None:
    # E[x | e] = e mu, so sum(x) / sum(e) over one (class, state) estimates mu
    # whatever the exposures are; the successes never exceed their own trials.
    declared = _declared()
    instance = fine_instance(fixture(PROBLEM, "ci").path)
    covariate = instance.params.covariate
    assert covariate is not None
    assert declared.model.covariate is not None
    assert np.array_equal(covariate, declared.model.covariate)
    observations = np.asarray(instance.observations, dtype=np.float64)

    assert (observations[..., SUCCESSES] <= covariate[..., SUCCESSES]).all()
    for m, family in enumerate(declared.model.emissions):
        assert isinstance(family, IndependentCountPair)
        columns = np.flatnonzero(instance.labels == m)
        for state in range(declared.model.n_states):
            rows = np.flatnonzero(instance.states[m] == state)
            block = np.ix_(rows, columns)
            ratio = (
                observations[..., TOTAL][block].sum()
                / covariate[..., TOTAL][block].sum()
            )
            expected = float(family.total.mean[state])
            assert abs(ratio - expected) < expected * _MOMENT


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("edit", "match"),
    [
        ({"form": "uniform"}, "form"),
        ({"seed": None, "extra": 1}, "exactly"),
        ({"exposure": {"median": 1.0, "sigma": 0.0}}, "covariate.exposure"),
        ({"trials": {"median": -40.0, "sigma": 0.25}}, "covariate.trials"),
        ({"exposure": {"median": 1.0, "sigma": float("inf")}}, "covariate.exposure"),
    ],
)
def test_a_malformed_covariate_is_refused(
    edit: Mapping[str, Any], match: str, tmp_path: Path
) -> None:
    # Each refusal names the field at fault, since a fixture is written by hand.
    raw = _raw()
    block = {**raw["covariate"], **edit}
    if block.get("seed") is None:
        block.pop("seed")
    raw["covariate"] = block
    path = tmp_path / "ci.yaml"

    with pytest.raises(ValueError, match=match):
        SpatioSequentialCountsParams.from_declared(raw, path)
