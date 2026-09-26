"""`emissions.dense.log_emission` against each family's own ``log_density`` (issue #1132).

Draws are simulated from each family at a seeded per-observation covariate,
shaped ``(T, N)`` as a count-HMM over replicates holds them, with a share of
each channel unobserved. The beta-binomial and every density without a
covariate are judged bitwise. The negative binomial's exposure term is judged
per order: the family's within the rounding of its two logarithms, the
tabulated within ``test_emissions_factored``'s bound on the summed terms.
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
from sal.emissions.dense import Order, log_emission
from sal.emissions.nb import exposure_table
from sal.sim.count_pairs import IndependentCountPair

SEED = 20260926

#: Positions and replicates of the draws.
SHAPE = (400, 12)

#: The family order against ``log_density``, relative per score: the two
#: differ only in which ``log`` rounds ``r / t`` and ``mu c / t``. Measured
#: 6.7 ulp on these draws.
_FAMILY_ULPS = 16

#: The tabulated order, in ulp of the three terms' summed magnitudes, as
#: ``test_emissions_factored`` states it and for its reason: ``y ln mu`` in
#: the table cancels against ``-y ln t``.
_TABULATED_ULPS = 4

EPS = float(np.finfo(np.float64).eps)

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


@pytest.mark.oracle
def test_the_family_order_is_the_density_within_the_rounding_of_its_logarithms() -> (
    None
):
    got = log_emission(NB, DRAWS["totals"], DRAWS["exposure"], order=Order.FAMILY)
    want = _density(NB, DRAWS["totals"], DRAWS["exposure"])

    assert got.shape == (3, *SHAPE)
    assert got.flags.c_contiguous
    assert (want == 0.0).any()
    worst = _worst_relative(got, want)
    assert worst <= _FAMILY_ULPS * EPS, f"{worst / EPS:.1f} ulp"


@pytest.mark.oracle
def test_the_tabulated_order_is_the_density_within_the_factored_bound() -> None:
    counts, exposure = DRAWS["totals"], DRAWS["exposure"]
    got = log_emission(NB, counts, exposure, order=Order.TABULATED)
    want = _density(NB, counts, exposure)

    # The three terms the kernel sums, `B`, `y ln c` and `(y + r) ln t`, each
    # at its magnitude: the scale the rounding of their sum is judged on.
    table = exposure_table(NB, int(counts.max()) + 1).numpy()
    r, mu = NB.dispersion.numpy(), NB.mean.numpy()
    c = np.where(exposure == 0.0, 1.0, exposure)
    y = counts[..., None].astype(np.float64)
    magnitude = (
        np.abs(table[counts])
        + np.abs(y * np.log(c))
        + np.abs((y + r) * np.log(r + mu * c))
    )
    observed = exposure[..., 0] > 0.0
    error = np.moveaxis(np.abs(got - want), 0, -1)[observed] / magnitude[observed]
    assert float(error.max()) <= _TABULATED_ULPS * EPS, f"{error.max() / EPS:.2f} ulp"


@pytest.mark.oracle
@pytest.mark.parametrize("order", list(Order), ids=str)
def test_the_beta_binomial_is_its_density_bitwise(order: Order) -> None:
    # Nine tabulated terms summed in the family's order: the same bits,
    # including the unobserved successes and none past their trials.
    got = log_emission(BB, DRAWS["successes"], DRAWS["trials"], order=order)

    assert (DRAWS["trials"] == 0.0).any()
    assert np.array_equal(got, _density(BB, DRAWS["successes"], DRAWS["trials"]))


@pytest.mark.oracle
def test_without_a_covariate_each_channel_is_its_density_bitwise() -> None:
    # The table is `log_density` at every count, so reading it is bitwise;
    # successes past the declared 40 trials score `-inf` in both.
    successes = DRAWS["successes"] + 2

    assert (successes > 40).any()
    assert np.array_equal(
        log_emission(NB, DRAWS["totals"], order=Order.FAMILY),
        _density(NB, DRAWS["totals"], None),
    )
    assert np.array_equal(
        log_emission(BB, successes, order=Order.TABULATED),
        _density(BB, successes, None),
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
@pytest.mark.parametrize("order", list(Order), ids=str)
def test_a_pair_is_its_channels_summed_as_the_family_sums_them(
    pair: object, order: Order
) -> None:
    observations = np.stack([DRAWS["totals"], DRAWS["successes"]], axis=-1)
    covariate = np.concatenate([DRAWS["exposure"], DRAWS["trials"]], axis=-1)

    got = log_emission(pair, observations, covariate, order=order)  # type: ignore[arg-type]
    channels = log_emission(
        NB, DRAWS["totals"], DRAWS["exposure"], order=order
    ) + log_emission(BB, DRAWS["successes"], DRAWS["trials"], order=order)

    assert np.array_equal(got, channels)
    if order is Order.FAMILY:
        want = _density(pair, observations, covariate)
        worst = _worst_relative(got, want)
        assert worst <= _FAMILY_ULPS * EPS, f"{worst / EPS:.1f} ulp"


@pytest.mark.smoke
def test_what_it_does_not_score_is_refused() -> None:
    joint = CountPairEmission(NB.dispersion, NB.mean, BB.alpha, BB.beta, joint=True)
    counts = DRAWS["totals"]

    with pytest.raises(TypeError, match="joint pair"):
        log_emission(joint, np.zeros((4, 2)), order=Order.FAMILY)
    with pytest.raises(ValueError, match="non-negative integer"):
        log_emission(NB, counts + 0.5, order=Order.FAMILY)
    with pytest.raises(ValueError, match="trailing singleton"):
        log_emission(NB, counts, DRAWS["exposure"][..., 0], order=Order.FAMILY)
    with pytest.raises(ValueError, match="finite and non-negative"):
        log_emission(NB, counts, -DRAWS["exposure"] - 1.0, order=Order.FAMILY)
