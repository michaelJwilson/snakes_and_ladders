"""The compiled ragged forward-backward against its NumPy oracle.

Issue #666. The oracle is `forward_backward` run one segment at a time, so what
is checked is the segmentation and not a second batched recursion.
"""

from __future__ import annotations

import numpy as np
import pytest
from sal.backend import Backend
from sal.likelihood.device import CROSS_DEVICE_RTOL_FLOAT64
from sal.likelihood.ragged import posteriors, posteriors_oracle
from sal.likelihood.ragged import rust as ragged_rust
from sal.ragged import Ragged

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
    """Marginals, transition counts and evidence, all three, from the twin itself."""

    def check(lengths: tuple[int, ...]) -> None:
        density, initial, transition = _instance(lengths, seed=4)
        gamma, counts, evidence = ragged_rust.posteriors(density, initial, transition)
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

    `S` segments over `T` positions take `T - S` transitions: each boundary restarts.
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


#: Segment layouts the switched kernel is checked on: even, uneven, and the
#: shortest a segment may be.
SWITCHED_LAYOUTS = [(6, 6, 6), (2, 11, 3, 7), (2, 2)]


@pytest.mark.oracle
@pytest.mark.parametrize("lengths", SWITCHED_LAYOUTS, ids=str)
def test_a_switched_transition_matches_the_materialized_stack(
    lengths: tuple[int, ...],
) -> None:
    # Issue #1082 H1: the kernel builds `(1 - s) I + s A` per step from one
    # `A`; the oracle hands `forward_backward` the `(T - 1, K, K)` stack.
    density, initial, transition = _instance(lengths, seed=1082)
    switch = np.random.default_rng(len(lengths)).uniform(size=sum(lengths))

    compiled = posteriors(density, initial, transition, switch=switch)
    reference = posteriors_oracle(density, initial, transition, switch)

    for got, want in zip(compiled, reference, strict=True):
        np.testing.assert_allclose(got, want, rtol=CROSS_DEVICE_RTOL_FLOAT64)


@pytest.mark.oracle
@pytest.mark.parametrize("value", [0.0, 0.35, 1.0])
def test_a_constant_switch_is_the_mixed_kernel(value: float) -> None:
    # A constant `s` is one kernel, `(1 - s) I + s A`, passed as the plain
    # `(K, K)` transition; at `s = 1` that is `A` itself, and at `s = 0` the
    # chain never moves. Bitwise at 0 and 0.35; at 1 the kernel's `ln` of
    # `exp(log A)` and NumPy's differ by up to 5.4e-16 relative, inside the
    # twin's declared tolerance.
    density, initial, transition = _instance((5, 8), seed=7)
    switch = np.full(13, value)
    with np.errstate(divide="ignore"):
        mixed = np.log((1.0 - value) * np.eye(STATES) + value * np.exp(transition))

    switched = posteriors(density, initial, transition, switch=switch)
    plain = posteriors(density, initial, mixed)

    for got, want in zip(switched, plain, strict=True):
        np.testing.assert_allclose(got, want, rtol=CROSS_DEVICE_RTOL_FLOAT64)
    if value < 1.0:
        assert all(
            np.array_equal(got, want) for got, want in zip(switched, plain, strict=True)
        )


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("switch", "message"),
    [(np.full(4, 0.5), "one per position"), (np.full(13, 1.5), "in \\[0, 1\\]")],
)
def test_a_malformed_switch_is_refused(switch: np.ndarray, message: str) -> None:
    density, initial, transition = _instance((5, 8), seed=7)

    with pytest.raises(ValueError, match=message):
        posteriors(density, initial, transition, switch=switch)
