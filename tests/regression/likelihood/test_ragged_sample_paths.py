"""The segmented forward-filter backward-sample against enumeration and its oracle.

Issue #1170. Three referees: every path of every segment enumerated, the
sampled frequencies within 4 sigma of the enumerated posterior under every
`SwitchKind`, and `hmm_paths`' enumeration on a categorical chain; the NumPy
oracle, which the compiled kernel matches path for path from the same
uniforms; and `forward_backward.sample_path`, which a batch reproduces segment
by segment from the same generator. On a segment too long to enumerate, an
80-bit filter places every uniform just inside a CDF step (issue #1266).
"""

from __future__ import annotations

from itertools import product

import numpy as np
import pytest
from sal.backend import Backend
from sal.likelihood.forward_backward import draw_path, inverse_cdf, sample_path
from sal.likelihood.hmm_paths import (
    emission_log_density,
    enumerate_hidden_paths,
    path_log_probability,
)
from sal.likelihood.ragged import (
    SampledPaths,
    SwitchKind,
    posteriors,
    sample_paths,
    sample_paths_oracle,
    sampled_posteriors,
    step_transitions,
)
from sal.likelihood.ragged import rust as ragged_rust
from sal.ragged import Ragged

from tests._long_double import long_double_filter
from tests.regression.likelihood.conftest import random_hmm

#: Rust against NumPy on the log-joint and the evidence: the kernel sums
#: `log_add` in a fold and takes a Kronecker entry as `ln A + ln S`, where
#: the oracle takes `logsumexp` and `ln(A S)`. Realized 1.1e-14.
KERNEL_RTOL = 1e-12

#: Sampled frequency against the enumerated probability, in binomial
#: standard deviations: a cell outside it has probability 6.3e-5 under the
#: normal approximation, which the lumping below keeps admissible.
SIGMAS = 4.0

#: Expected count a cell is not cut below, `test_hmm_paths.py`'s floor: under
#: it the binomial is too skewed for a standard deviation to bound it.
CELL_FLOOR = 25.0

#: Draws per segment. The batch repeats each segment this many times, so one
#: call makes every draw.
DRAWS = 3000

KINDS = list(SwitchKind)

#: How far inside a CDF step of the 80-bit law each uniform is placed, in
#: probability. With each filter row shifted by its maximum (issue #1266) no
#: draw of 3,000 lands on the wrong side at 1e-14 under any kind or backend;
#: unshifted, 19 to 47 did at this margin and 532 to 728 at 1e-14.
STRADDLE = 1e-13


def _case(
    kind: SwitchKind, lengths: tuple[int, ...], seed: int
) -> tuple[Ragged, np.ndarray, np.ndarray, np.ndarray]:
    """Random scores and parameters, three states or `2 K` with `K = 2`."""
    rng = np.random.default_rng(seed)
    slow = 3 if kind is SwitchKind.STAY_OR_MOVE else 2
    n_states = slow if kind is SwitchKind.STAY_OR_MOVE else 2 * slow
    density = np.log(rng.random((sum(lengths), n_states)))
    initial = np.log(rng.dirichlet(np.ones(n_states)))
    transition = np.log(rng.dirichlet(np.ones(slow), size=slow))
    switch = rng.uniform(0.05, 0.95, size=sum(lengths))
    return Ragged(density, lengths), initial, transition, switch


def _law(
    segment: np.ndarray,
    initial: np.ndarray,
    steps: np.ndarray,
) -> tuple[list[tuple[int, ...]], np.ndarray, float]:
    """Every path of one segment, its log-joint, and the log evidence."""
    n_states, length = segment.shape[1], segment.shape[0]
    paths = list(product(range(n_states), repeat=length))
    joints = np.empty(len(paths))
    for index, path in enumerate(paths):
        score = initial[path[0]] + segment[0, path[0]]
        for t in range(1, length):
            score += steps[t - 1, path[t - 1], path[t]] + segment[t, path[t]]
        joints[index] = score
    top = joints.max()
    return paths, joints, float(top + np.log(np.exp(joints - top).sum()))


def _within(counts: np.ndarray, law: np.ndarray, n_draws: int) -> float:
    """The largest deviation of the frequencies from the law, in standard deviations.

    Cells are taken in descending probability and lumped wherever one expects
    fewer than :data:`CELL_FLOOR` draws, the remainder joining the last.
    """
    cells: list[list[int]] = []
    current: list[int] = []
    for index in np.argsort(-law, kind="stable"):
        current.append(int(index))
        if law[current].sum() * n_draws >= CELL_FLOOR:
            cells.append(current)
            current = []
    if current:
        cells[-1].extend(current)
    counts = np.array([counts[cell].sum() for cell in cells])
    law = np.array([law[cell].sum() for cell in cells])
    spread = np.sqrt(np.maximum(law * (1.0 - law), 1e-300) / n_draws)
    return float(np.max(np.abs(counts / n_draws - law) / spread))


@pytest.mark.critical
@pytest.mark.oracle
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("backend", [Backend.RUST, Backend.PYTHON])
def test_sampled_paths_follow_the_enumerated_posterior(
    kind: SwitchKind, backend: Backend
) -> None:
    """Segments of 2, 3 and 4 positions, 3,000 draws each: every path within 4 sigma.

    Every path of every segment is a cell, 3 + 9 + 27 = 39 stay-or-move and
    16 + 64 + 256 = 336 Kronecker; each draw's log-joint is the
    enumeration's for its path, and its evidence the enumeration's sum.
    """
    lengths = (2, 3, 4)
    density, initial, transition, switch = _case(kind, lengths, seed=1170)
    batch = Ragged(np.tile(density.values, (DRAWS, 1)), lengths * DRAWS)
    drawn = sample_paths(
        batch,
        initial,
        transition,
        np.random.default_rng(len(kind)),
        switch=np.tile(switch, DRAWS),
        switch_kind=kind,
        backend=backend,
    )
    assert isinstance(drawn, SampledPaths)
    at = 0
    for index, segment in enumerate(density.segments()):
        length = len(segment)
        steps = step_transitions(transition, switch[at + 1 : at + length], kind)
        paths, joints, evidence = _law(segment, initial, steps)
        lookup = {path: position for position, path in enumerate(paths)}
        # Draw `d` of this segment is segment `d * 3 + index` of the batch.
        starts = np.cumsum((0, *lengths))[index] + sum(lengths) * np.arange(DRAWS)
        rows = drawn.path[starts[:, None] + np.arange(length)]
        cells = np.array([lookup[tuple(int(s) for s in row)] for row in rows])
        counts = np.bincount(cells, minlength=len(paths))
        assert _within(counts, np.exp(joints - evidence), DRAWS) < SIGMAS
        np.testing.assert_allclose(
            drawn.log_joint[index :: len(lengths)], joints[cells], rtol=1e-12
        )
        np.testing.assert_allclose(
            drawn.log_evidence[index :: len(lengths)], evidence, rtol=1e-12
        )
        at += length


@pytest.mark.critical
@pytest.mark.oracle
@pytest.mark.parametrize("backend", [Backend.RUST, Backend.PYTHON])
def test_sampled_marginals_are_the_path_enumeration(backend: Backend) -> None:
    """A categorical chain of 5 over 3 states, 6,000 draws: every marginal within 4 sigma.

    The referee is `hmm_paths.enumerate_hidden_paths`, which shares no
    recursion with the filter; the evidence is its, to 1e-12, and each
    draw's log-joint is `path_log_probability` of the path drawn.
    """
    n_draws, length = 6000, 5
    params = random_hmm(3, 4, length, 17)
    observations = np.random.default_rng(17).integers(0, 4, size=length)
    enumerated = enumerate_hidden_paths(params, observations)
    log_density = emission_log_density(params, observations)
    drawn = sample_paths(
        Ragged(np.tile(log_density, (n_draws, 1)), (length,) * n_draws),
        np.log(params.initial),
        np.log(params.transition),
        np.random.default_rng(1170),
        backend=backend,
    )
    rows = drawn.path.reshape(n_draws, length)
    for position in range(length):
        counts = np.bincount(rows[:, position], minlength=3)
        assert _within(counts, enumerated.posterior[position], n_draws) < SIGMAS
    np.testing.assert_allclose(drawn.log_evidence, enumerated.log_evidence, rtol=1e-12)
    for row, joint in zip(rows[:50], drawn.log_joint[:50], strict=True):
        assert joint == pytest.approx(
            path_log_probability(params, row, observations), rel=1e-12
        )


@pytest.mark.critical
@pytest.mark.oracle
@pytest.mark.backend
@pytest.mark.parametrize("kind", KINDS)
def test_the_compiled_kernel_matches_the_oracle(kind: SwitchKind) -> None:
    """Uneven lengths at ten states (`K = 5` Kronecker): path bitwise, log-joint to 1e-12.

    Both read the uniforms `rng.random(total)` from one seed and invert by
    one rule, so a path differs only where the two filters' rounding straddles
    a uniform; at 1,416 positions per call none does.
    """
    rng = np.random.default_rng(23)
    lengths = (2, 300, 7, 41, 1000, 2, 64)
    slow = 10 if kind is SwitchKind.STAY_OR_MOVE else 5
    n_states = slow if kind is SwitchKind.STAY_OR_MOVE else 2 * slow
    density = Ragged(np.log(rng.random((sum(lengths), n_states))), lengths)
    initial = np.log(rng.dirichlet(np.ones(n_states)))
    transition = np.log(rng.dirichlet(np.ones(slow), size=slow))
    for switch in (
        rng.uniform(size=sum(lengths)),
        None if kind is SwitchKind.STAY_OR_MOVE else np.full(sum(lengths), 0.5),
    ):
        ours = sample_paths(
            density,
            initial,
            transition,
            np.random.default_rng(5),
            switch=switch,
            switch_kind=kind,
        )
        uniforms = np.random.default_rng(5).random(sum(lengths))
        theirs = sample_paths_oracle(
            density, initial, transition, uniforms, switch, kind
        )
        np.testing.assert_array_equal(ours.path, theirs.path)
        np.testing.assert_allclose(ours.log_joint, theirs.log_joint, rtol=KERNEL_RTOL)
        np.testing.assert_allclose(
            ours.log_evidence, theirs.log_evidence, rtol=KERNEL_RTOL
        )


def _straddled(
    segment: np.ndarray, initial: np.ndarray, steps: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Uniforms each `STRADDLE` inside a CDF step of the 80-bit law, and the path they select.

    `steps` is `(T, n, n)` in log space, row `t` the step into position `t`.
    The draws run last to first, as `sample_paths` reads the uniforms: each
    targets a random state of the 80-bit weights given the state after it,
    from just above or just below, so a filter off by more than `STRADDLE`
    there selects a neighbour.
    """
    rng = np.random.default_rng(seed)
    length = segment.shape[0]
    alpha = long_double_filter(segment, initial, steps)
    kernels = np.exp(steps.astype(np.longdouble))
    uniforms, path = np.empty(length), np.empty(length, dtype=np.int64)
    for k in range(length):
        t = length - 1 - k
        weights = alpha[t] if k == 0 else alpha[t] * kernels[t + 1][:, path[t + 1]]
        upper = np.cumsum(weights / weights.sum())
        lower = np.concatenate([[0.0], upper[:-1]])
        state = int(rng.choice(np.flatnonzero(upper - lower > 4 * STRADDLE)))
        below = rng.random() < 0.5
        uniforms[k] = float(
            upper[state] - STRADDLE if below else lower[state] + STRADDLE
        )
        path[t] = state
    return uniforms, path


@pytest.mark.critical
@pytest.mark.oracle
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("backend", [Backend.RUST, Backend.PYTHON])
def test_a_long_segment_draws_on_the_80_bit_side_of_every_step(
    kind: SwitchKind, backend: Backend
) -> None:
    """3,000 positions: every draw selects the state the 80-bit filter puts its uniform in."""
    length = 3000
    density, initial, transition, switch = _case(kind, (length,), seed=1266)
    steps = step_transitions(transition, switch[1:], kind)
    stack = np.concatenate([np.zeros((1, *steps.shape[1:])), steps])
    uniforms, want = _straddled(density.values, initial, stack, seed=3)
    if backend is Backend.RUST:
        got = ragged_rust.sample_paths(
            density, initial, transition, uniforms, switch, kind
        )
    else:
        got = sample_paths_oracle(density, initial, transition, uniforms, switch, kind)
    np.testing.assert_array_equal(got.path, want)


@pytest.mark.critical
@pytest.mark.oracle
@pytest.mark.backend
def test_a_batch_is_sample_path_segment_by_segment() -> None:
    """One generator, the batch against `sample_path` per segment in order: bitwise.

    `rng.random(total)` is the per-segment `rng.random(length)` calls laid end
    to end, so the oracle is `sample_path` on each segment, exactly.
    """
    rng = np.random.default_rng(31)
    lengths = (2, 9, 40, 3)
    density = Ragged(np.log(rng.random((sum(lengths), 4))), lengths)
    initial = np.log(rng.dirichlet(np.ones(4)))
    transition = np.log(rng.dirichlet(np.ones(4), size=4))
    batch = sample_paths(
        density, initial, transition, np.random.default_rng(8), backend=Backend.PYTHON
    )
    one = np.random.default_rng(8)
    each = np.concatenate(
        [
            sample_path(segment, initial, transition, one)
            for segment in density.segments()
        ]
    )
    np.testing.assert_array_equal(batch.path, each)


@pytest.mark.critical
@pytest.mark.analytic
def test_the_inverse_cdf_never_draws_a_zero_weight_state() -> None:
    """Every uniform at a boundary or within rounding of one selects a state of positive weight."""
    with np.errstate(divide="ignore"):
        weights = np.log(np.array([0.0, 0.25, 0.0, 0.75, 0.0]))
        frozen = np.log(np.eye(2))
    picks = {
        inverse_cdf(weights, uniform)
        for uniform in (0.0, 0.25, 0.2499999999999999, 0.5, 1.0 - 2.0**-53)
    }
    assert picks == {1, 3}
    # The first uniform reads the last position: a frozen chain draws its
    # initial state everywhere, chosen by `uniforms[0]` alone.
    initial = np.log(np.array([0.25, 0.75]))
    for first, state in ((0.1, 0), (0.9, 1)):
        drawn = draw_path(
            np.zeros((3, 2)), initial, frozen, np.array([first, 0.5, 0.5])
        )
        np.testing.assert_array_equal(drawn.path, [state] * 3)


@pytest.mark.critical
@pytest.mark.analytic
def test_a_kronecker_kind_without_a_switch_or_uniforms_short_is_refused() -> None:
    density, initial, transition, _ = _case(SwitchKind.KRONECKER, (3, 2), seed=1)
    rng = np.random.default_rng(0)
    for backend in (Backend.RUST, Backend.PYTHON):
        with pytest.raises(ValueError, match="switch"):
            sample_paths(
                density,
                initial,
                transition,
                rng,
                switch_kind=SwitchKind.KRONECKER,
                backend=backend,
            )
    with pytest.raises(ValueError, match="one per position"):
        sample_paths_oracle(density, initial, transition, np.zeros(4))


@pytest.mark.critical
@pytest.mark.oracle
@pytest.mark.parametrize("kind", KINDS)
def test_sampled_posteriors_estimate_the_exact_posteriors(kind: SwitchKind) -> None:
    """400 draws on uneven segments: gamma and counts within 4 sigma of `posteriors`.

    The occupancy's spread is the binomial's; the counts' is read off the
    400 single draws, floored at the Poisson's, which the averaged call reproduces from one seed, since
    it reads the generator as the 400 calls do. The evidence is the exact
    one, to 1e-12.
    """
    n_draws, lengths = 400, (2, 30, 7, 61)
    density, initial, transition, switch = _case(kind, lengths, seed=7)
    exact = posteriors(density, initial, transition, switch=switch, switch_kind=kind)
    estimate = sampled_posteriors(
        density,
        initial,
        transition,
        np.random.default_rng(9),
        n_paths=n_draws,
        switch=switch,
        switch_kind=kind,
    )
    one = np.random.default_rng(9)
    singles = np.stack(
        [
            np.exp(
                sampled_posteriors(
                    density,
                    initial,
                    transition,
                    one,
                    n_paths=1,
                    switch=switch,
                    switch_kind=kind,
                ).log_counts
            )
            for _ in range(n_draws)
        ]
    )
    np.testing.assert_allclose(
        np.exp(estimate.log_counts), singles.mean(axis=0), rtol=1e-12
    )
    probability = np.exp(exact.log_posterior)
    # `1 / n` beside the variance keeps a cell the law puts near zero from
    # failing on one draw, where the binomial is too skewed for its sigma.
    spread = np.sqrt((probability * (1 - probability) + 1 / n_draws) / n_draws)
    deviation = np.abs(np.exp(estimate.log_posterior) - probability) / spread
    assert float(deviation.max()) < SIGMAS
    expected = np.exp(exact.log_counts)
    # A rare pair's count is near Poisson, its variance near its mean; the
    # sample variance of 400 draws can miss it entirely.
    variance = np.maximum(singles.var(axis=0, ddof=1), expected) + 1 / n_draws
    standard_error = np.sqrt(variance / n_draws)
    deviation = np.abs(np.exp(estimate.log_counts) - expected)
    assert float((deviation / standard_error).max()) < SIGMAS
    np.testing.assert_allclose(estimate.log_evidence, exact.log_evidence, rtol=1e-12)
