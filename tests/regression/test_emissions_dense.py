"""The dense log-emission against each family's own ``log_density`` (issues #1132, #1340).

The dense route is :func:`sal.emissions.coded.log_emission` over a
:class:`~sal.emissions.coded.Dense` since #1340 folded
``emissions.dense.log_emission`` into it; :func:`log_emission` here flattens
the ``(T, N)`` draws into that form and restores their shape.

Draws are simulated from each family at a seeded per-observation covariate,
shaped ``(T, N)`` as a count-HMM over replicates holds them, with a share of
each channel unobserved. The beta-binomial and every density without a
covariate are judged bitwise. The negative binomial's exposure term is judged
within ``test_emissions_factored``'s bound on the summed terms.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from sal.emissions import (
    BetaBinomialEmission,
    CountPairEmission,
    NegativeBinomialEmission,
)
from sal.emissions.bb import beta_binomial_log_pmf, density_table
from sal.emissions.coded import Dense
from sal.emissions.coded import log_emission as coded_log_emission
from sal.sim.count_pairs import IndependentCountPair
from scipy.special import gammaln

SEED = 20260926

#: Positions and replicates of the draws.
SHAPE = (400, 12)

#: The negative binomial against ``log_density``, in ulp of the terms'
#: summed magnitudes (:func:`_nb_magnitude`), the scale #1064 judged the
#: tabulated order on (issue #1335). Both orders are now
#: :mod:`sal.emissions.nb`'s, whose ``S(r, y) - lgamma(y + 1)`` and
#: ``y log(lambda / (1 + q))`` cancel ``y log r`` at small ``r``, so the
#: family order's 16 ulp relative pin is restated at the tabulated order's
#: declared 4 ulp of magnitudes, not re-recorded. Measured 0.97 ulp.
_NB_ULPS = 4

EPS = float(np.finfo(np.float64).eps)

#: The beta-binomial's scaled rising factorials against the family's ``lgamma``
#: arithmetic, over ``max(|f|, 1)`` (issue #1332). Both routes form ``log C``
#: from three ``lgamma`` of size up to ``lgamma(41)``, each rounded to half an
#: ulp: 3.7e-14 per route, 7.3e-14 between two, 1e-13 with the rest of the
#: sum. Derived, not fitted. Measured 4.2e-14.
_DENSITY_TOLERANCE = 1e-13

NB = NegativeBinomialEmission([4.0, 7.0, 12.0], [20.0, 80.0, 200.0])
BB = BetaBinomialEmission([40.0, 40.0, 40.0], [2.4, 7.0, 19.5], [9.6, 7.0, 10.5])


def _draws() -> dict[str, np.ndarray]:
    """Counts at a log-normal exposure and successes out of a varying trial count.

    A twentieth of the exposures and a tenth of the trial counts are zero,
    which marks that channel unobserved.
    """
    rng = np.random.default_rng(SEED)
    states = rng.integers(0, 3, SHAPE)
    exposure = np.exp(0.5 * rng.standard_normal((*SHAPE, 1)))
    exposure[rng.random((*SHAPE, 1)) < 0.05] = 0.0
    trials = np.rint(40.0 * np.exp(0.25 * rng.standard_normal((*SHAPE, 1))))
    trials[rng.random((*SHAPE, 1)) < 0.1] = 0.0
    totals = NB.sample(states, rng, covariate=np.where(exposure == 0.0, 1.0, exposure))
    successes = BB.sample(states, rng, covariate=trials)
    return {
        "totals": totals,
        "exposure": exposure,
        "successes": successes,
        "trials": trials,
    }


DRAWS = _draws()


def log_emission(
    family: object, observations: np.ndarray, covariate: np.ndarray | None = None
) -> np.ndarray:
    """``(K, *batch)`` scores of ``(*batch,)`` or a pair's ``(*batch, 2)`` draws.

    A single channel's covariate carries a trailing singleton, as
    ``log_density`` takes it; both are flattened into a :class:`Dense`.
    """
    single = isinstance(family, NegativeBinomialEmission | BetaBinomialEmission)
    batch = observations.shape if single else observations.shape[:-1]
    width = () if single else (2,)
    dense = Dense(
        observations.reshape(-1, *width),
        None if covariate is None else covariate.reshape(-1, *width),
    )
    scores = coded_log_emission(family, dense)  # type: ignore[arg-type]
    return scores.reshape(scores.shape[0], *batch)


def _density(
    family: object, observations: np.ndarray, covariate: np.ndarray | None
) -> np.ndarray:
    """``log_density`` with its state axis first: what the entry point claims to be."""
    scored = family.log_density(  # type: ignore[attr-defined]
        torch.as_tensor(observations, dtype=torch.float64),
        None if covariate is None else torch.as_tensor(covariate),
    )
    return np.moveaxis(scored.numpy(), -1, 0)


def _worst_relative(got: np.ndarray, want: np.ndarray) -> float:
    """The largest relative difference, where an unobserved score of zero is matched exactly."""
    zero = want == 0.0
    assert np.array_equal(got[zero], want[zero])
    return float((np.abs(got - want)[~zero] / np.abs(want[~zero])).max())


def _nb_magnitude(counts: np.ndarray, exposure: np.ndarray) -> np.ndarray:
    """The summed magnitudes of every term either NB route forms, ``(*batch, K)``."""
    r, mu = NB.dispersion.numpy(), NB.mean.numpy()
    c = np.where(exposure == 0.0, 1.0, exposure)
    y = counts[..., None].astype(np.float64)
    rate = mu * c
    magnitude: np.ndarray = (
        np.abs(gammaln(y + r))
        + np.abs(gammaln(r))
        + np.abs(gammaln(y + 1.0))
        + np.abs(y * np.log(r))
        + np.abs(y * np.log(rate / (r + rate)))
        + np.abs(r * np.log1p(rate / r))
    )
    return magnitude


@pytest.mark.oracle
def test_the_negative_binomial_is_the_density_within_the_bound_of_its_terms() -> None:
    counts, exposure = DRAWS["totals"], DRAWS["exposure"]
    got = log_emission(NB, counts, exposure)
    want = _density(NB, counts, exposure)

    assert got.shape == (3, *SHAPE)
    assert got.flags.c_contiguous
    observed = exposure[..., 0] > 0.0
    assert np.array_equal(got[:, ~observed], want[:, ~observed])
    magnitude = _nb_magnitude(counts, exposure)
    error = np.moveaxis(np.abs(got - want), 0, -1)[observed] / magnitude[observed]
    assert float(error.max()) <= _NB_ULPS * EPS, f"{error.max() / EPS:.2f} ulp"


@pytest.mark.oracle
def test_the_beta_binomial_is_the_numpy_pmf_bitwise() -> None:
    # Four tabulated terms summed as `beta_binomial_log_pmf` sums them: the
    # same bits, the unobserved successes zero as the family scores them; the
    # family's `log_density` within _DENSITY_TOLERANCE (#1332).
    got = log_emission(BB, DRAWS["successes"], DRAWS["trials"])
    trials = DRAWS["trials"][..., 0]
    pmf = beta_binomial_log_pmf(
        DRAWS["successes"],
        trials,
        BB.alpha.numpy()[:, None, None],
        BB.beta.numpy()[:, None, None],
    )
    want = _density(BB, DRAWS["successes"], DRAWS["trials"])

    assert (trials == 0.0).any()
    assert np.array_equal(got, np.where(trials == 0.0, 0.0, pmf))
    assert np.array_equal(got[:, trials == 0.0], want[:, trials == 0.0])
    worst = float((np.abs(got - want) / np.maximum(np.abs(want), 1.0)).max())
    assert worst <= _DENSITY_TOLERANCE, worst


@pytest.mark.oracle
def test_without_a_covariate_each_channel_is_its_density_bitwise() -> None:
    # The table is `log_density` at every count, so reading it is bitwise;
    # successes past the declared 40 trials score `-inf` in both. The
    # beta-binomial's is `bb.density_table`, bitwise, and torch's within
    # _DENSITY_TOLERANCE (#1332; measured 3.1e-14).
    successes = DRAWS["successes"] + 2

    assert (successes > 40).any()
    assert np.array_equal(
        log_emission(NB, DRAWS["totals"]),
        _density(NB, DRAWS["totals"], None),
    )
    got = log_emission(BB, successes)
    want = _density(BB, successes, None)
    table = density_table(BB, int(successes.max()) + 1)
    assert np.array_equal(got, np.moveaxis(table[successes.astype(np.int64)], -1, 0))
    finite = np.isfinite(want)
    assert np.array_equal(np.isfinite(got), finite)
    scale = np.maximum(np.abs(want[finite]), 1.0)
    assert float((np.abs(got[finite] - want[finite]) / scale).max()) <= (
        _DENSITY_TOLERANCE
    )


def _pairs() -> list[object]:
    return [
        CountPairEmission(
            NB.dispersion, NB.mean, BB.alpha, BB.beta, BB.trials, joint=False
        ),
        IndependentCountPair(NB, BB),
    ]


@pytest.mark.oracle
@pytest.mark.parametrize(
    "pair", _pairs(), ids=["CountPairEmission", "IndependentCountPair"]
)
def test_a_pair_is_its_channels_summed_as_the_family_sums_them(pair: object) -> None:
    observations = np.stack([DRAWS["totals"], DRAWS["successes"]], axis=-1)
    covariate = np.concatenate([DRAWS["exposure"], DRAWS["trials"]], axis=-1)

    got = log_emission(pair, observations, covariate)
    channels = log_emission(NB, DRAWS["totals"], DRAWS["exposure"]) + log_emission(
        BB, DRAWS["successes"], DRAWS["trials"]
    )

    assert np.array_equal(got, channels)
    # The negative binomial's _NB_ULPS of its terms' magnitudes plus the
    # beta-binomial's _DENSITY_TOLERANCE over max(|f|, 1) (#1332, #1335).
    want = _density(pair, observations, covariate)
    bound = _NB_ULPS * EPS * np.moveaxis(
        _nb_magnitude(DRAWS["totals"], DRAWS["exposure"]), -1, 0
    ) + _DENSITY_TOLERANCE * np.maximum(np.abs(want), 1.0)
    assert (np.abs(got - want) <= bound).all()


@pytest.mark.smoke
def test_what_it_does_not_score_is_refused() -> None:
    joint = CountPairEmission(NB.dispersion, NB.mean, BB.alpha, BB.beta, joint=True)
    counts = DRAWS["totals"]

    with pytest.raises(TypeError, match="joint pair"):
        log_emission(joint, np.zeros((4, 2)))
    with pytest.raises(ValueError, match="non-negative integer"):
        log_emission(NB, counts + 0.5)
    with pytest.raises(ValueError, match="shaped as its counts"):
        coded_log_emission(NB, Dense(counts.reshape(-1), DRAWS["exposure"]))
    with pytest.raises(ValueError, match="finite and non-negative"):
        log_emission(NB, counts, -DRAWS["exposure"] - 1.0)


def _binding(
    family: object, observations: np.ndarray, covariate: np.ndarray | None
) -> np.ndarray:
    """``oxisal.dense_log_emission`` on tables built as #1132's caller built them, over every count."""
    from sal import oxisal
    from sal.emissions.bb import log_factorial, trial_tables
    from sal.emissions.dense import checked_counts
    from sal.emissions.nb import exposure_table

    def total(
        fam: NegativeBinomialEmission, y: np.ndarray, c: np.ndarray | None
    ) -> dict[str, np.ndarray]:
        counts = checked_counts(y, "count")
        extent = int(counts.max()) + 1
        if c is None:
            grid = torch.arange(extent, dtype=torch.float64)
            table = fam.log_density(grid).numpy()
            return {
                "totals": counts,
                "total_table": np.ascontiguousarray(table).reshape(-1),
            }
        return {
            "totals": counts,
            "total_table": np.ascontiguousarray(exposure_table(fam, extent)).reshape(
                -1
            ),
            "exposure": np.ascontiguousarray(c.reshape(-1)),
            "dispersion": np.ascontiguousarray(fam.dispersion.numpy()),
            "mean": np.ascontiguousarray(fam.mean.numpy()),
        }

    def success(
        fam: BetaBinomialEmission, z: np.ndarray, n: np.ndarray | None
    ) -> dict[str, np.ndarray]:
        counts = checked_counts(z, "success count")
        extent = int(counts.max()) + 1
        if n is None:
            table = density_table(fam, extent)
            return {
                "successes": counts,
                "success_table": np.ascontiguousarray(table).reshape(-1),
            }
        trials = checked_counts(n, "trial count")
        trials_extent = int(trials.max()) + 1
        tables = trial_tables(fam, extent, trials_extent)
        return {
            "successes": counts,
            "success_table": np.ascontiguousarray(tables.success).reshape(-1),
            "trials": trials,
            "failure_table": np.ascontiguousarray(tables.failure).reshape(-1),
            "trial_table": np.ascontiguousarray(tables.trial).reshape(-1),
            "log_factorial": log_factorial(trials_extent),
            "log_rate": np.ascontiguousarray(tables.log_rate).reshape(-1),
        }

    if isinstance(family, NegativeBinomialEmission):
        shape, arguments = observations.shape, total(family, observations, covariate)
    elif isinstance(family, BetaBinomialEmission):
        shape, arguments = observations.shape, success(family, observations, covariate)
    else:
        shape = observations.shape[:-1]
        arguments = {
            **total(NB, observations[..., 0], covariate[..., 0]),  # type: ignore[index]
            **success(BB, observations[..., 1], covariate[..., 1]),  # type: ignore[index]
        }
    k = family.n_states  # type: ignore[attr-defined]
    out = np.empty(k * int(np.prod(shape)))
    oxisal.dense_log_emission(k, True, out, **arguments)
    return out.reshape(k, *shape)


def _binding_cases() -> list[tuple[str, object, np.ndarray, np.ndarray | None]]:
    pair_obs = np.stack([DRAWS["totals"], DRAWS["successes"]], axis=-1)
    pair_cov = np.concatenate([DRAWS["exposure"], DRAWS["trials"]], axis=-1)
    pairs = _pairs()
    return [
        ("nb-exposure", NB, DRAWS["totals"], DRAWS["exposure"]),
        ("bb-trials", BB, DRAWS["successes"], DRAWS["trials"]),
        ("nb", NB, DRAWS["totals"], None),
        ("bb", BB, DRAWS["successes"] + 2, None),
        ("CountPairEmission", pairs[0], pair_obs, pair_cov),
        ("IndependentCountPair", pairs[1], pair_obs, pair_cov),
    ]


@pytest.mark.oracle
@pytest.mark.parametrize(
    ("name", "family", "observations", "covariate"),
    _binding_cases(),
    ids=[case[0] for case in _binding_cases()],
)
def test_the_deprecated_binding_is_the_coded_route_bitwise(
    name: str, family: object, observations: np.ndarray, covariate: np.ndarray | None
) -> None:
    # `oxisal.dense_log_emission`, kept for a caller passing its own tables
    # (#1340), over tables at every count against the coded route at the
    # distinct counts: one kernel, bitwise. Both were also pinned against the
    # pre-fold `dense.log_emission` outputs on these six cases, bitwise.
    del name  # the case's id
    got = _binding(family, observations, covariate)
    want = log_emission(family, observations, covariate)
    assert np.array_equal(got.view(np.uint64), want.view(np.uint64))
