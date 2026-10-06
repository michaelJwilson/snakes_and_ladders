"""`EmissionHmmObjective`'s JAX twin on count pairs and unequal segments, against the RUST route (issue #1206).

Referee: the same objective under ``Backend.RUST`` --- the compiled E step
and one torch backward pass, itself pinned to autograd through ``__call__``
(``test_opt_hmm_emission_objective.py``) --- at three points away from the
start, value and gradient within 1e-12 relative, the gradient relative to
its largest entry. The fixtures are drawn by
``sim.hmm.simulate_sequences`` from a sticky three-state chain.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose
from sal.backend import Backend
from sal.emissions import (
    BetaBinomialEmission,
    CountPairEmission,
    EmissionFamily,
    NegativeBinomialEmission,
    PoissonEmission,
    RateConcentrationBetaBinomialEmission,
)
from sal.opt.hmm import EmissionHmmObjective
from sal.opt.hmm import jax as hmm_jax
from sal.ragged import Ragged
from sal.sim.count_pairs import IndependentCountPair
from sal.sim.hmm import HmmParams, simulate_sequences

SEED = 1206

#: Unequal segments, the shortest two positions (`Ragged`'s floor): padded to 41.
UNEQUAL = (41, 2, 17, 30, 3, 25)
#: Six segments of one length: no mask.
EQUAL = (20,) * 6

#: The declared relative tolerance on value and gradient (issue #1206).
RTOL = 1e-12


def _pair(*, joint: bool) -> CountPairEmission:
    return CountPairEmission(
        [4.0, 9.0, 30.0],
        [6.0, 15.0, 40.0],
        [2.0, 5.0, 3.0],
        [3.0, 2.0, 6.0],
        None if joint else [45, 45, 45],
        joint=joint,
    )


def _independent(successes: BetaBinomialEmission) -> IndependentCountPair:
    return IndependentCountPair(
        NegativeBinomialEmission([4.0, 9.0, 30.0], [6.0, 15.0, 40.0]), successes
    )


#: Each start family, by the name a parameter reads.
FAMILIES: dict[str, Callable[[], EmissionFamily]] = {
    "joint": lambda: _pair(joint=True),
    "independent": lambda: _pair(joint=False),
    "joint rate-concentration": lambda: _pair(joint=True).rate_concentration(),
    "independent rate-concentration": lambda: _pair(joint=False).rate_concentration(),
    "sim independent": lambda: _independent(
        BetaBinomialEmission([45, 45, 45], [2.0, 5.0, 3.0], [3.0, 2.0, 6.0])
    ),
    "sim independent rate-concentration": lambda: _independent(
        RateConcentrationBetaBinomialEmission(
            [45, 45, 45], [0.4, 0.7, 0.33], [5.0, 7.0, 9.0]
        )
    ),
}


def _drawn(family: EmissionFamily, lengths: tuple[int, ...]) -> Ragged:
    """Segments drawn from a sticky three-state chain under ``family``."""
    transition = np.full((3, 3), 0.05) + np.eye(3) * 0.85
    params = HmmParams(
        n_states=3,
        lengths=lengths,
        initial=np.full(3, 1.0 / 3.0),
        transition=transition,
        emissions=family,
        seed=SEED,
        tolerance=0.0,
    )
    values = np.asarray(simulate_sequences(params).observations, dtype=np.float64)
    flat = values.reshape(sum(lengths), -1)
    return Ragged(flat[:, 0] if flat.shape[1] == 1 else flat, lengths)


def _covariate(total: int, *, scalar: str | None = None) -> np.ndarray:
    """An exposure and a trial count per position, each zero (unobserved) at two positions."""
    rng = np.random.default_rng(SEED + 1)
    exposure = rng.uniform(0.5, 2.0, total)
    trials = rng.integers(45, 60, total).astype(np.float64)
    exposure[[0, 7]] = 0.0
    trials[[3, 9]] = 0.0
    if scalar == "exposure":
        return exposure[:, None]
    if scalar == "trials":
        return trials[:, None]
    return np.stack([exposure, trials], axis=1)


def _assert_twin(objective: EmissionHmmObjective, oracle: EmissionHmmObjective) -> None:
    """Value and gradient of the JAX route equal the RUST route's, at three points off the start."""
    assert hmm_jax.twinned(objective)
    rng = np.random.default_rng(SEED)
    for _ in range(3):
        theta = objective.initial() + 0.2 * torch.as_tensor(
            rng.normal(size=objective.n_parameters)
        )
        value, gradient = objective.value_and_gradient(theta)
        want_value, want_gradient = oracle.value_and_gradient(theta)
        assert np.isfinite(float(value))
        assert_allclose(float(value), float(want_value), rtol=RTOL)
        # Relative to the gradient's scale: an entry near zero is a sum that
        # cancels, and its own relative error is not the route's.
        assert_allclose(
            gradient.numpy(),
            want_gradient.numpy(),
            rtol=RTOL,
            atol=RTOL * float(want_gradient.abs().max()),
        )


def _routes(
    data: Ragged, family: EmissionFamily, covariate: np.ndarray | None = None
) -> tuple[EmissionHmmObjective, EmissionHmmObjective]:
    return tuple(  # type: ignore[return-value]
        EmissionHmmObjective(data, family, covariate=covariate, backend=backend)
        for backend in (Backend.JAX, Backend.RUST)
    )


@pytest.mark.oracle
@pytest.mark.parametrize("lengths", [EQUAL, UNEQUAL], ids=["equal", "unequal"])
@pytest.mark.parametrize("name", list(FAMILIES))
def test_the_count_pair_twin_is_the_rust_route(
    name: str, lengths: tuple[int, ...]
) -> None:
    family = FAMILIES[name]()
    _assert_twin(*_routes(_drawn(family, lengths), family))


@pytest.mark.oracle
@pytest.mark.parametrize("lengths", [EQUAL, UNEQUAL], ids=["equal", "unequal"])
@pytest.mark.parametrize("name", ["independent", "sim independent rate-concentration"])
def test_the_count_pair_twin_takes_the_per_channel_covariate(
    name: str, lengths: tuple[int, ...]
) -> None:
    # An exposure for the total and a trial count for the successes, zero
    # at two positions each: those channels score log 1 (issue #933).
    family = FAMILIES[name]()
    data = _drawn(family, lengths)
    _assert_twin(*_routes(data, family, _covariate(sum(lengths))))


@pytest.mark.oracle
@pytest.mark.parametrize(
    ("family", "covariate"),
    [
        (NegativeBinomialEmission([4.0, 9.0, 30.0], [6.0, 15.0, 40.0]), "exposure"),
        (
            RateConcentrationBetaBinomialEmission(
                [45, 45, 45], [0.4, 0.7, 0.33], [5.0, 7.0, 9.0]
            ),
            "trials",
        ),
        (
            RateConcentrationBetaBinomialEmission(
                [45, 45, 45], [0.4, 0.7, 0.33], [5.0, 7.0, 9.0]
            ),
            None,
        ),
    ],
    ids=[
        "negative binomial exposure",
        "rate-concentration trials",
        "rate-concentration",
    ],
)
def test_a_scalar_twin_on_unequal_segments_is_the_rust_route(
    family: EmissionFamily, covariate: str | None
) -> None:
    data = _drawn(family, UNEQUAL)
    given = None if covariate is None else _covariate(sum(UNEQUAL), scalar=covariate)
    _assert_twin(*_routes(data, family, given))


@pytest.mark.oracle
def test_padding_is_the_identity() -> None:
    # Referee: the unmasked twin of each group of equal-length segments,
    # summed. The padded program scores the four at once; a padded step that
    # moved alpha, counted a pair or carried a posterior would move the sum.
    # The forward recursion is the family's alone to read, so the cheapest
    # family to compile stands for every one.
    family = PoissonEmission([2.0, 6.0, 15.0])
    lengths = (30, 12, 30, 12)
    data = _drawn(family, lengths)
    theta = EmissionHmmObjective(data, family).initial().numpy() + 0.1
    assert hmm_jax._prepared(EmissionHmmObjective(data, family))[0].masked
    whole = hmm_jax.value_and_grad(
        EmissionHmmObjective(data, family, backend=Backend.JAX)
    )(theta)
    starts = data.offsets[:-1]
    parts = []
    for length in (30, 12):
        members = [i for i, each in enumerate(lengths) if each == length]
        rows = np.concatenate(
            [data.values[starts[i] : starts[i] + length] for i in members]
        )
        group = EmissionHmmObjective(
            Ragged(rows, (length,) * len(members)), family, backend=Backend.JAX
        )
        assert not hmm_jax._prepared(group)[0].masked
        parts.append(hmm_jax.value_and_grad(group)(theta))
    gradient = parts[0][1] + parts[1][1]
    assert_allclose(whole[0], parts[0][0] + parts[1][0], rtol=RTOL)
    assert_allclose(
        whole[1], gradient, rtol=RTOL, atol=RTOL * float(np.abs(gradient).max())
    )


@pytest.mark.smoke
def test_the_joint_pair_with_a_covariate_has_no_twin() -> None:
    # The joint form refuses a covariate when it scores; the twin refuses it
    # where the backend is chosen.
    family = FAMILIES["joint"]()
    data = _drawn(family, EQUAL)
    covariate = _covariate(sum(EQUAL))
    assert not hmm_jax.twinned(EmissionHmmObjective(data, family, covariate=covariate))
    with pytest.raises(ValueError, match="no JAX twin"):
        EmissionHmmObjective(data, family, covariate=covariate, backend=Backend.JAX)
