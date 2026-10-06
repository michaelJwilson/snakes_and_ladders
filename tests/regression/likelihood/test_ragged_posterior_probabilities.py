"""`oxisal.ragged_posterior_probabilities`, the scaled E step, against two oracles (issue #1253).

The kernel runs the streamed core of `src/hmm_stream.rs` and writes the
posteriors and pair counts as probabilities. It is pinned to the NumPy oracle
`posteriors_oracle`, run in logs one segment at a time, on short segments; and
to an 80-bit long-double forward-backward on segments of thousands, where the
log-space recursions' own rounding reaches 4e-11 and the NumPy oracle stops
being the finer referee.

The same 80-bit referee pins the log-space `oxisal.ragged_posteriors`, with
and without a switch, on a segment of 3,000 (issue #1262).
"""

from __future__ import annotations

import numpy as np
import pytest
from sal import oxisal
from sal.likelihood.device import CROSS_DEVICE_RTOL_FLOAT64
from sal.likelihood.ragged import SwitchKind, posteriors_oracle, step_transitions
from sal.ragged import Ragged

from tests._rows import every_value

STATES = 4

#: Long-double reference against the kernel, measured at 4.6e-15 (marginals,
#: absolute), 2.8e-15 (counts, relative) and 4.6e-16 (evidence, relative) on
#: `LONG`; the log-space kernel there reads 3.7e-11, 1.1e-11 and 8.8e-15.
LONG_TOLERANCE = 1e-13

#: Three segments of thousands of positions.
LONG = (3000, 2500, 1200)

#: The log-space kernel against the long-double reference, the declared
#: `CROSS_DEVICE_RTOL_FLOAT64` (issue #1262): absolute on the marginals,
#: relative on counts and evidence.
LOG_SPACE_TOLERANCE = CROSS_DEVICE_RTOL_FLOAT64

#: One position, a short segment and a long one (issue #1262).
MIXED = (1, 60, 3000)


def _instance(
    lengths: tuple[int, ...], seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gaussian log-densities from a sampled chain and non-uniform parameters, seeded."""
    rng = np.random.default_rng(seed)
    means = np.arange(STATES, dtype=np.float64)
    values = means[rng.integers(0, STATES, sum(lengths))] + rng.normal(
        size=sum(lengths)
    )
    density = -0.5 * (values[:, None] - means[None, :]) ** 2 - 0.5 * np.log(2 * np.pi)
    return (
        np.ascontiguousarray(density),
        np.log(rng.dirichlet(np.ones(STATES))),
        np.log(rng.dirichlet(3.0 * np.ones(STATES), size=STATES)),
    )


def _kernel(
    density: np.ndarray,
    lengths: tuple[int, ...],
    initial: np.ndarray,
    transition: np.ndarray,
    threads: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The kernel's posterior, pair counts and per-segment log evidence."""
    posterior = np.empty_like(density)
    pairs = np.empty((STATES, STATES))
    evidence = np.empty(len(lengths))
    oxisal.ragged_posterior_probabilities(
        density,
        np.asarray(lengths, dtype=np.int64),
        initial,
        transition,
        posterior,
        pairs,
        evidence,
        threads,
    )
    return posterior, pairs, evidence


def _long_double(
    density: np.ndarray,
    lengths: tuple[int, ...],
    initial: np.ndarray,
    transition: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scaled forward-backward in `np.longdouble`, one segment at a time, written independently of the kernel.

    `transition` is one log matrix, or a `(total, n, n)` stack whose row `t`
    is the step into position `t`, as `step_transitions` writes it.
    """
    wide = np.longdouble
    n_states = density.shape[1]
    stack = np.exp(transition.astype(wide))
    if stack.ndim == 2:
        stack = np.broadcast_to(stack, (density.shape[0], n_states, n_states))
    prior = np.exp(initial.astype(wide))
    posterior = np.empty(density.shape, dtype=wide)
    pairs = np.zeros((n_states, n_states), dtype=wide)
    evidence = []
    start = 0
    for n in lengths:
        x = density[start : start + n].astype(wide)
        steps = stack[start : start + n]
        high = x.max(axis=1, keepdims=True)
        b = np.exp(x - high)
        alpha = np.empty((n, n_states), dtype=wide)
        scale = np.empty(n, dtype=wide)
        step = prior * b[0]
        scale[0] = step.sum()
        alpha[0] = step / scale[0]
        for t in range(1, n):
            step = (alpha[t - 1] @ steps[t]) * b[t]
            scale[t] = step.sum()
            alpha[t] = step / scale[t]
        beta = np.ones(n_states, dtype=wide)
        posterior[start + n - 1] = alpha[n - 1]
        for t in range(n - 1, 0, -1):
            onward = b[t] * beta / scale[t]
            pairs += alpha[t - 1][:, None] * steps[t] * onward[None, :]
            beta = steps[t] @ onward
            posterior[start + t - 1] = alpha[t - 1] * beta
        evidence.append(np.log(scale).sum() + high.sum())
        start += n
    return posterior, pairs, np.array(evidence)


@pytest.mark.critical
@pytest.mark.oracle
def test_the_scaled_kernel_matches_the_log_space_oracle() -> None:
    """Marginals, counts and evidence against `posteriors_oracle`, exponentiated."""

    def check(lengths: tuple[int, ...]) -> None:
        density, initial, transition = _instance(lengths, seed=1253)
        posterior, pairs, evidence = _kernel(density, lengths, initial, transition)
        gamma, counts, want = posteriors_oracle(
            Ragged(density, lengths), initial, transition
        )
        tolerance = CROSS_DEVICE_RTOL_FLOAT64
        np.testing.assert_allclose(evidence, want, rtol=tolerance)
        np.testing.assert_allclose(posterior, np.exp(gamma), rtol=tolerance)
        np.testing.assert_allclose(pairs, np.exp(counts), rtol=tolerance)

    every_value([(1, 2, 7, 60), (5, 11, 3, 40), (1,), (17,) * 6], check)


@pytest.mark.oracle
def test_long_segments_match_a_long_double_recursion() -> None:
    # Where the log-space oracle's own rounding is 4e-11, the referee is the
    # same recursion in 80-bit arithmetic.
    density, initial, transition = _instance(LONG, seed=1253)
    got = _kernel(density, LONG, initial, transition)
    posterior, pairs, evidence = _long_double(density, LONG, initial, transition)

    np.testing.assert_allclose(
        got[0], posterior.astype(np.float64), rtol=0, atol=LONG_TOLERANCE
    )
    np.testing.assert_allclose(got[1], pairs.astype(np.float64), rtol=LONG_TOLERANCE)
    np.testing.assert_allclose(got[2], evidence.astype(np.float64), rtol=LONG_TOLERANCE)


def _switched(
    kind: SwitchKind | None, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
    """`MIXED` under `kind`: density, initial, transition, switch and the step stack.

    The Kronecker kinds read the `STATES / 2` slow chain and a binary layer.
    """
    density, initial, transition = _instance(MIXED, seed=seed)
    if kind is None:
        return density, initial, transition, None, transition
    rng = np.random.default_rng(seed + 1)
    switch = rng.uniform(size=density.shape[0])
    if kind is not SwitchKind.STAY_OR_MOVE:
        transition = np.log(rng.dirichlet(3.0 * np.ones(STATES // 2), size=STATES // 2))
    return (
        density,
        initial,
        transition,
        switch,
        step_transitions(transition, switch, kind),
    )


@pytest.mark.oracle
@pytest.mark.parametrize(
    "kind", [None, *SwitchKind], ids=lambda kind: str(kind or "unswitched")
)
def test_the_log_space_kernel_matches_a_long_double_recursion(
    kind: SwitchKind | None,
) -> None:
    # Issue #1262: at 3,000 positions the log-space kernel read 3.7e-11 on
    # the marginals against this referee, outside the declared 1e-11.
    density, initial, transition, switch, steps = _switched(kind, seed=1262)
    gamma = np.empty_like(density)
    counts = np.empty((STATES, STATES))
    evidence = np.empty(len(MIXED))
    oxisal.ragged_posteriors(
        density,
        np.asarray(MIXED, dtype=np.int64),
        initial,
        transition,
        gamma,
        counts,
        evidence,
        switch,
        str(kind or SwitchKind.STAY_OR_MOVE),
    )
    posterior, pairs, want = _long_double(density, MIXED, initial, steps)

    np.testing.assert_allclose(
        np.exp(gamma), posterior.astype(np.float64), rtol=0, atol=LOG_SPACE_TOLERANCE
    )
    np.testing.assert_allclose(
        np.exp(counts), pairs.astype(np.float64), rtol=LOG_SPACE_TOLERANCE
    )
    np.testing.assert_allclose(
        evidence, want.astype(np.float64), rtol=LOG_SPACE_TOLERANCE
    )


@pytest.mark.oracle
def test_every_thread_count_writes_the_same_bits() -> None:
    # The blocks are cut by the lengths alone and the counts merged in block
    # order, so the pool size changes no bit.
    lengths = tuple(int(n) for n in np.random.default_rng(3).integers(1, 90, 40))
    density, initial, transition = _instance(lengths, seed=7)
    one = _kernel(density, lengths, initial, transition, threads=1)
    for threads in (2, 4):
        for got, want in zip(
            _kernel(density, lengths, initial, transition, threads=threads),
            one,
            strict=True,
        ):
            np.testing.assert_array_equal(got, want)


@pytest.mark.smoke
def test_misshapen_buffers_and_empty_segments_are_refused() -> None:
    density, initial, transition = _instance((3, 4), seed=1)
    with pytest.raises(ValueError, match="length 0"):
        _kernel(np.ascontiguousarray(density), (7, 0), initial, transition)
    with pytest.raises(ValueError, match="evidence"):
        oxisal.ragged_posterior_probabilities(
            density,
            np.array([3, 4]),
            initial,
            transition,
            np.empty_like(density),
            np.empty((STATES, STATES)),
            np.empty(3),
        )
