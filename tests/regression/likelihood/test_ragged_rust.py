"""The compiled ragged forward-backward against its NumPy oracle.

Issue #666. The oracle is `forward_backward` run one segment at a time, so what
is checked is the segmentation and not a second batched recursion.
"""

from __future__ import annotations

import numpy as np
import pytest
from snakes_and_ladders.backend import Backend
from snakes_and_ladders.likelihood.device import CROSS_DEVICE_RTOL_FLOAT64
from snakes_and_ladders.likelihood.ragged_rust import posteriors, posteriors_oracle
from snakes_and_ladders.ragged import Ragged

from tests._rows import every_value

STATES = 3


def _instance(
    lengths: tuple[int, ...], seed: int
) -> tuple[Ragged, np.ndarray, np.ndarray]:
    """Scores and parameters that are not uniform, so a mistake shows."""
    rng = np.random.default_rng(seed)
    density = np.log(rng.random((sum(lengths), STATES)))
    return (
        Ragged(density, lengths),
        np.log(rng.dirichlet(np.ones(STATES))),
        np.log(rng.dirichlet(np.ones(STATES), size=STATES)),
    )


@pytest.mark.critical
@pytest.mark.oracle
def test_the_compiled_kernel_matches_the_oracle() -> None:
    """Marginals, transition counts and evidence, all three."""

    def check(lengths: tuple[int, ...]) -> None:
        density, initial, transition = _instance(lengths, seed=4)
        gamma, counts, evidence = posteriors(density, initial, transition)
        want_gamma, want_counts, want_evidence = posteriors_oracle(
            density, initial, transition
        )
        tolerance = CROSS_DEVICE_RTOL_FLOAT64
        np.testing.assert_allclose(evidence, want_evidence, rtol=tolerance)
        np.testing.assert_allclose(gamma, want_gamma, rtol=tolerance)
        np.testing.assert_allclose(counts, want_counts, rtol=tolerance)

    every_value([(5, 11, 3, 40), (2, 2), (400, 2, 7), (17,) * 6], check)


@pytest.mark.critical
@pytest.mark.analytic
def test_the_marginals_are_normalized_within_every_segment() -> None:
    """Each position's posterior sums to one, boundaries included."""
    density, initial, transition = _instance((6, 19, 3), seed=9)
    gamma, _, _ = posteriors(density, initial, transition)
    np.testing.assert_allclose(np.exp(gamma).sum(axis=1), 1.0, rtol=1e-12)


@pytest.mark.critical
@pytest.mark.analytic
def test_the_counts_hold_one_transition_fewer_than_the_positions() -> None:
    """The boundary pairs are absent, and the arithmetic says how many.

    A batch of `S` segments totalling `T` positions took `T - S` transitions,
    not `T - 1`: each boundary is a restart, not a step. Summing the expected
    counts recovers exactly that, which is the claim the mask exists for.
    """
    lengths = (6, 19, 3, 11)
    density, initial, transition = _instance(lengths, seed=13)
    _, counts, _ = posteriors(density, initial, transition)
    taken = float(np.exp(counts).sum())
    assert taken == pytest.approx(sum(lengths) - len(lengths), rel=1e-10)


@pytest.mark.backend
@pytest.mark.oracle
def test_the_enum_selects_the_kernel_or_its_oracle_bitwise() -> None:
    # #860: the choice between this kernel and the sibling it is pinned
    # against is named by `Backend` rather than by importing one of the two
    # functions. Each member is that function's own call, bit for bit --- the
    # tolerance above is between the two implementations, not across a door.
    density, initial, transition = _instance((5, 11, 3), seed=4)

    for chosen, expected in (
        (Backend.RUST, posteriors(density, initial, transition, backend=Backend.RUST)),
        (Backend.PYTHON, posteriors_oracle(density, initial, transition)),
    ):
        got = posteriors(density, initial, transition, backend=chosen)
        for one, want in zip(got, expected, strict=True):
            np.testing.assert_array_equal(one, want)

    with pytest.raises(ValueError, match="not numba"):
        posteriors(density, initial, transition, backend=Backend.NUMBA)
