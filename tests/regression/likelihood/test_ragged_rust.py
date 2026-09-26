"""The compiled ragged forward-backward against its NumPy oracle.

Issue #666. The oracle is `forward_backward` run one segment at a time, so what
is checked is the segmentation and not a second batched recursion.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from sal import oxisal
from sal.backend import Backend
from sal.likelihood.device import CROSS_DEVICE_RTOL_FLOAT64
from sal.likelihood.ragged import (
    SwitchKind,
    posteriors,
    posteriors_oracle,
    step_transitions,
)
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


#: The two Kronecker kinds (issue #1133).
KRONECKER_KINDS = [SwitchKind.KRONECKER, SwitchKind.KRONECKER_DIAGONAL]

#: Slow states under the fast binary layer.
SLOW = 3


def _layered(
    lengths: tuple[int, ...], seed: int
) -> tuple[Ragged, np.ndarray, np.ndarray, np.ndarray]:
    """Scores over ``2 K`` states, a ``2 K`` prior, a ``K x K`` slow chain and a switch."""
    rng = np.random.default_rng(seed)
    total = sum(lengths)
    return (
        Ragged(np.log(rng.random((total, 2 * SLOW))), lengths),
        np.log(rng.dirichlet(np.ones(2 * SLOW))),
        np.log(rng.dirichlet(np.ones(SLOW), size=SLOW)),
        rng.uniform(size=total),
    )


@pytest.mark.oracle
@pytest.mark.parametrize("kind", KRONECKER_KINDS, ids=str)
@pytest.mark.parametrize("lengths", SWITCHED_LAYOUTS, ids=str)
def test_a_kronecker_switch_matches_the_materialized_stack(
    lengths: tuple[int, ...], kind: SwitchKind
) -> None:
    # The kernel takes each step in its factors; the oracle hands
    # `forward_backward` the `(T - 1, 2K, 2K)` stack of `np.kron(A, S_t)`.
    density, initial, slow, switch = _layered(lengths, seed=1133)

    compiled = posteriors(density, initial, slow, switch=switch, switch_kind=kind)
    reference = posteriors_oracle(density, initial, slow, switch, kind)

    for got, want in zip(compiled, reference, strict=True):
        np.testing.assert_allclose(got, want, rtol=CROSS_DEVICE_RTOL_FLOAT64)


@pytest.mark.oracle
@pytest.mark.parametrize("kind", KRONECKER_KINDS, ids=str)
def test_the_kronecker_evidence_and_marginals_are_the_sum_over_paths(
    kind: SwitchKind,
) -> None:
    # Every one of the 6^5 paths through one segment of five positions,
    # scored from the explicit per-step matrices and summed.
    density, initial, slow, switch = _layered((5,), seed=7)
    steps = step_transitions(slow, switch[1:], kind)
    n = 2 * SLOW
    paths = np.array(list(itertools.product(range(n), repeat=5)))
    score = initial[paths[:, 0]] + density.values[0, paths[:, 0]]
    for t in range(1, 5):
        score = score + steps[t - 1, paths[:, t - 1], paths[:, t]]
        score = score + density.values[t, paths[:, t]]
    evidence = np.logaddexp.reduce(score)
    marginals = np.array(
        [
            [np.logaddexp.reduce(score[paths[:, t] == state]) for state in range(n)]
            for t in range(5)
        ]
    )

    gamma, _, got = posteriors(density, initial, slow, switch=switch, switch_kind=kind)

    np.testing.assert_allclose(got, [evidence], rtol=1e-12)
    np.testing.assert_allclose(gamma, marginals - evidence, rtol=1e-11, atol=1e-12)


@pytest.mark.oracle
def test_under_an_identity_slow_chain_the_kronecker_switch_is_stay_or_move() -> None:
    # `I ⊗ S_t = (1 - s) I + s (I ⊗ J)`: the one case the stay-or-move form
    # covers, reached by a different arithmetic.
    density, initial, _, switch = _layered((6, 9, 2), seed=11)
    with np.errstate(divide="ignore"):
        identity = np.log(np.eye(SLOW))
        flipped = np.log(np.kron(np.eye(SLOW), 1.0 - np.eye(2)))

    kronecker = posteriors(
        density, initial, identity, switch=switch, switch_kind=SwitchKind.KRONECKER
    )
    moved = posteriors(density, initial, flipped, switch=switch)

    for got, want in zip(kronecker, moved, strict=True):
        np.testing.assert_allclose(got, want, rtol=CROSS_DEVICE_RTOL_FLOAT64)


@pytest.mark.analytic
@pytest.mark.parametrize("kind", list(SwitchKind), ids=str)
def test_every_step_transition_is_stochastic(kind: SwitchKind) -> None:
    _, _, slow, switch = _layered((4,), seed=3)

    rows = np.exp(step_transitions(slow, switch, kind)).sum(axis=-1)

    np.testing.assert_allclose(rows, 1.0, rtol=1e-14)


@pytest.mark.smoke
def test_a_malformed_kronecker_call_is_refused() -> None:
    density, initial, slow, switch = _layered((5, 8), seed=7)
    odd = Ragged(density.values[:, :5], density.lengths)

    with pytest.raises(ValueError, match="switch per position"):
        posteriors(density, initial, slow, switch_kind=SwitchKind.KRONECKER)
    with pytest.raises(ValueError, match="2 K states"):
        posteriors(
            odd, initial[:5], slow, switch=switch, switch_kind=SwitchKind.KRONECKER
        )
    with pytest.raises(ValueError, match="not a valid SwitchKind"):
        ragged_rust.posteriors(density, initial, slow, switch, "sideways")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="switch_kind"):
        oxisal.ragged_posteriors(
            density.values,
            np.asarray(density.lengths, dtype=np.int64),
            initial,
            slow,
            np.empty_like(density.values),
            np.empty((2 * SLOW, 2 * SLOW)),
            np.empty(2),
            switch,
            "sideways",
        )
