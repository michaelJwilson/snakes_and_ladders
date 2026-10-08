"""`emissions.coded` against the dense route, a one-hot control and the NumPy pmfs (issue #1340).

Draws are simulated from each family at a seeded per-observation covariate,
with a share of each channel unobserved. A :class:`Coded` score is judged
bitwise against its :class:`Dense` form and against a one-hot
``scipy.sparse`` product over the same per-code table, and within 1e-13
``max(|f|, 1)`` of :mod:`sal.emissions.nb`'s and :mod:`sal.emissions.bb`'s
NumPy pmfs evaluated per observation (#1334, #1336).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import scipy.sparse
from sal.emissions import (
    BetaBinomialEmission,
    CountPairEmission,
    NegativeBinomialEmission,
    RateConcentrationBetaBinomialEmission,
)
from sal.emissions.bb import beta_binomial_log_pmf
from sal.emissions.coded import Coded, Dense, encode, log_emission
from sal.emissions.nb import negative_binomial_log_pmf
from scipy.stats import poisson as scipy_poisson

SEED = 20261008

#: Observations of the draws.
N = 6000

#: Against the NumPy pmfs, over ``max(|f|, 1)``: the stated bound of #1340
#: and ``test_emissions_dense``'s ``_DENSITY_TOLERANCE``.
_TOLERANCE = 1e-13

NB = NegativeBinomialEmission([4.0, 7.0, 12.0], [20.0, 80.0, 200.0])
BB = BetaBinomialEmission([40.0, 40.0, 40.0], [2.4, 7.0, 19.5], [9.6, 7.0, 10.5])
PAIR = CountPairEmission(
    NB.dispersion, NB.mean, BB.alpha, BB.beta, BB.trials, joint=False
)


def _draws() -> dict[str, np.ndarray]:
    """Counts at a log-normal exposure, successes out of a varying trial count; some unobserved."""
    rng = np.random.default_rng(SEED)
    states = rng.integers(0, 3, N)
    exposure = np.exp(0.5 * rng.standard_normal(N))
    exposure[rng.random(N) < 0.05] = 0.0
    trials = np.rint(40.0 * np.exp(0.25 * rng.standard_normal(N)))
    trials[rng.random(N) < 0.1] = 0.0
    totals = NB.sample(
        states, rng, covariate=np.where(exposure == 0.0, 1.0, exposure)[:, None]
    )
    successes = BB.sample(states, rng, covariate=trials[:, None])
    return {
        "totals": np.asarray(totals, dtype=np.float64).reshape(-1),
        "exposure": exposure,
        "successes": np.asarray(successes, dtype=np.float64).reshape(-1),
        "trials": trials,
    }


DRAWS = _draws()


def _cases() -> list[tuple[str, Any, np.ndarray, np.ndarray | None]]:
    """Every family, with and without its covariate."""
    pair_counts = np.stack([DRAWS["totals"], DRAWS["successes"]], axis=1)
    pair_cov = np.stack([DRAWS["exposure"], DRAWS["trials"]], axis=1)
    rc = RateConcentrationBetaBinomialEmission(
        [40.0] * 3, [0.2, 0.5, 0.65], [12.0, 14.0, 30.0]
    )
    return [
        ("nb", NB, DRAWS["totals"], None),
        ("nb-exposure", NB, DRAWS["totals"], DRAWS["exposure"]),
        ("bb", BB, DRAWS["successes"], None),
        ("bb-trials", BB, DRAWS["successes"], DRAWS["trials"]),
        ("rc-trials", rc, DRAWS["successes"], DRAWS["trials"]),
        ("pair", PAIR, pair_counts, None),
        ("pair-covariate", PAIR, pair_counts, pair_cov),
    ]


CASES = _cases()
IDS = [case[0] for case in CASES]


def _bits(values: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(values).view(np.uint64)


@pytest.mark.oracle
@pytest.mark.parametrize(("name", "family", "counts", "covariate"), CASES, ids=IDS)
def test_dense_equals_its_coded_form_bitwise(
    name: str,  # noqa: ARG001
    family: Any,
    counts: np.ndarray,
    covariate: np.ndarray | None,
) -> None:
    dense = log_emission(family, Dense(counts, covariate))
    coded = log_emission(family, encode(counts, covariate))
    assert coded.shape == (3, N)
    assert coded.flags.c_contiguous
    assert coded.dtype == np.float64
    assert np.array_equal(_bits(coded), _bits(dense))


@pytest.mark.oracle
def test_encode_is_a_numpy_unique_and_a_single_label_changes_nothing() -> None:
    counts, exposure = DRAWS["totals"], DRAWS["exposure"]
    coded = encode(counts, exposure)
    observed = exposure != 0.0
    distinct, inverse, weight = np.unique(
        counts[observed], return_inverse=True, return_counts=True
    )
    assert np.array_equal(coded.counts, distinct.astype(np.uint32))
    assert coded.inverse.dtype == np.int32
    assert np.array_equal(coded.inverse[observed], inverse)
    assert (coded.inverse[~observed] == -1).all()
    assert np.array_equal(coded.weight, weight)
    labelled = encode(counts, exposure, label=np.full(N, 3))
    assert (labelled.label == 3).all()
    assert np.array_equal(
        _bits(log_emission(NB, labelled)), _bits(log_emission(NB, coded))
    )


@pytest.mark.oracle
@pytest.mark.parametrize("name", ["nb", "bb", "pair"])
def test_decode_is_the_one_hot_product_bitwise(name: str) -> None:
    """Without a covariate a score is a table row: the gather equals ``onehot @ table``."""
    _, family, counts, _ = CASES[IDS.index(name)]
    coded = encode(counts)
    table = log_emission(
        family,
        Coded(
            coded.label,
            coded.counts,
            np.arange(coded.weight.size, dtype=np.int32),
            coded.weight,
            None,
        ),
    )
    onehot = scipy.sparse.csr_matrix(
        (np.ones(N), (np.arange(N), coded.inverse)), shape=(N, coded.weight.size)
    )
    control = np.ascontiguousarray((onehot @ table.T).T)
    assert np.array_equal(_bits(log_emission(family, coded)), _bits(control))


def _pmf(
    name: str, family: Any, counts: np.ndarray, covariate: np.ndarray | None
) -> np.ndarray:
    """The NumPy pmf per observation, ``(K, n)``, unobserved channels zero."""
    if name.startswith("nb"):
        c = np.ones(N) if covariate is None else covariate
        want = negative_binomial_log_pmf(
            counts[:, None], NB.dispersion.numpy(), c[:, None] * NB.mean.numpy()
        )
        return np.where(c[:, None] == 0.0, 0.0, want).T
    if name.startswith(("bb", "rc")):
        bb = family
        n = bb.trials.numpy() if covariate is None else covariate[:, None]
        want = beta_binomial_log_pmf(
            counts[:, None],
            n,
            bb.alpha.numpy(),
            bb.beta.numpy(),
        )
        return np.where(np.broadcast_to(n, want.shape) == 0.0, 0.0, want).T
    cov = (None, None) if covariate is None else (covariate[:, 0], covariate[:, 1])
    summed: np.ndarray = _pmf("nb", NB, counts[:, 0], cov[0]) + _pmf(
        "bb", BB, counts[:, 1], cov[1]
    )
    return summed


@pytest.mark.oracle
@pytest.mark.parametrize(("name", "family", "counts", "covariate"), CASES, ids=IDS)
def test_within_the_bound_of_the_numpy_pmfs(
    name: str, family: Any, counts: np.ndarray, covariate: np.ndarray | None
) -> None:
    got = log_emission(family, encode(counts, covariate))
    want = _pmf(name, family, counts, covariate)
    infinite = np.isinf(want)
    assert np.array_equal(got[infinite], want[infinite])
    worst = np.abs(got[~infinite] - want[~infinite]) / np.maximum(
        np.abs(want[~infinite]), 1.0
    )
    assert worst.max() <= _TOLERANCE


@pytest.mark.analytic
def test_the_limits() -> None:
    counts = np.array([0.0, 3.0, 11.0, 3.0])
    exposure = np.array([1.5, 0.0, 0.7, 2.0])
    # r = inf is the Poisson.
    poisson = NegativeBinomialEmission([np.inf], [6.0])
    got = log_emission(poisson, encode(counts, exposure))[0]
    rate = 6.0 * exposure
    want = np.where(exposure == 0.0, 0.0, scipy_poisson.logpmf(counts, rate))
    assert np.allclose(got, want, rtol=1e-13, atol=0.0)
    # Covariate zero scores 0, exactly.
    assert got[1] == 0.0
    # Successes past trials score -inf; a zero trial count scores 0.
    successes = np.array([2.0, 5.0, 4.0])
    trials = np.array([6.0, 4.0, 0.0])
    scored = log_emission(BB, encode(successes, trials))
    assert (scored[:, 1] == -np.inf).all()
    assert (scored[:, 2] == 0.0).all()
    # tau = inf: the family scores NaN on every route (log_density, dense,
    # coded); the binomial limit is an open item of #1340, pinned as it stands.
    binomial = RateConcentrationBetaBinomialEmission([10.0], [0.3], [np.inf])
    with np.errstate(invalid="ignore"):
        limit = log_emission(binomial, encode(successes, trials))
        dense = log_emission(binomial, Dense(successes, trials))
    assert np.array_equal(_bits(limit), _bits(dense))
    # An invalid parameter or covariate raises.
    with pytest.raises(ValueError, match="dispersion must be positive"):
        NegativeBinomialEmission([-1.0], [6.0])
    with pytest.raises(ValueError, match="finite and non-negative"):
        log_emission(NB, encode(counts, -exposure))


@pytest.mark.analytic
def test_what_stage_one_refuses() -> None:
    with pytest.raises(ValueError, match="stage 3"):
        encode(np.array([1.0, 2.0]), label=np.array([0, 1]))
    with pytest.raises(ValueError, match="non-negative integer"):
        encode(np.array([1.5]))
    with pytest.raises(TypeError, match="Dense or Coded"):
        log_emission(NB, np.array([1.0]))  # type: ignore[arg-type]
