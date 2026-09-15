"""Decoding a CSS code: the criterion first, then the oracle, then the decoder.

The order is the point. A success criterion that is wrong makes every number
after it meaningless, so :func:`decode_succeeds` is pinned from both sides
before a decoder is called: a correct answer plus a nonzero stabilizer must
read as a success, and a correct answer plus a logical operator must read as a
failure. An implementation that always answers one way passes one of those and
fails the other (``likelihood/CLAUDE.md``: a criterion's test is paired).

The oracle is exact degenerate maximum likelihood by enumeration, which is a
*different decoder* from the one returning the likeliest single error --- a sum
over a coset against a maximum over its members --- and the fixture is declared
where the two differ, so the gap is measured rather than asserted to exist.
Belief propagation is then reported against that floor, with its failures split
into the two defects a single rate hides.
"""

from __future__ import annotations

import numpy as np
import pytest
from snakes_and_ladders.likelihood.css import (
    decode_succeeds,
    decode_syndrome,
    error_cosets,
    measure_logical_error_rate,
)
from snakes_and_ladders.sim.css import CssCode, sample_x_error
from snakes_and_ladders.sim.fixtures import fixture
from snakes_and_ladders.sim.ldpc import BinarySymmetricChannel

#: The declared rate at each tier, and what the enumeration and the decoder
#: give there. ``ci`` carries the oracle; ``stress`` is past ``2 ** n`` and
#: carries a measured rate with nothing exact behind it.
CI_FLIP = 0.05
#: Exact degenerate maximum likelihood at the `ci` instance and `CI_FLIP`, and
#: the decoder that maximizes one error's probability instead. Both are sums
#: over an enumeration, so they are exact and pinned to the digit.
CI_DEGENERATE_RATE = 0.07029669
CI_SINGLE_ERROR_RATE = 0.07273500


def _ci_code() -> CssCode:
    """The CSS code the registry declares at the CI tier."""
    code: CssCode = fixture("bicycle_css", "ci").params.code()
    return code


def _stress_code() -> CssCode:
    """The CSS code the registry declares at the stress tier."""
    code: CssCode = fixture("bicycle_css", "stress").params.code()
    return code


def _code(tier: str) -> CssCode:
    """The CSS code of one tier, named so a parameterized test reads either."""
    return {"ci": _ci_code, "stress": _stress_code}[tier]()


def _stabilizers(code: CssCode) -> np.ndarray:
    """Every nonzero stabilizer: the span of the rows of ``H``."""
    dense = code.checks.dense()
    coefficients = (
        np.arange(1, 2**code.checks.n_checks)[:, None]
        >> np.arange(code.checks.n_checks)
    ) & 1
    span = (coefficients @ dense.astype(np.int64)) & 1
    return np.asarray(np.unique(span, axis=0), dtype=np.uint8)


# --- the criterion, before any decoder runs --------------------------------------


@pytest.mark.critical
@pytest.mark.mathematical
def test_a_residual_in_the_row_space_is_a_success() -> None:
    """Adding any nonzero stabilizer to the right answer still decodes.

    This is the whole of what degeneracy means, and the half a criterion
    written as `e == ê` gets wrong: 127 of the 128 stabilizers of the declared
    instance are nonzero, so 127 corrections that differ from the error are
    each a success.
    """
    code = _code("ci")
    error = np.zeros(code.n_qubits, dtype=np.uint8)
    error[[0, 5]] = 1
    stabilizers = _stabilizers(code)
    assert stabilizers.shape[0] == 2**code.stabilizer_rank - 1

    outcomes = [decode_succeeds(code, error, error ^ row) for row in stabilizers]

    assert all(outcomes)
    assert any((row != 0).any() and (error ^ row != error).any() for row in stabilizers)
    assert decode_succeeds(code, error, error)


@pytest.mark.critical
@pytest.mark.mathematical
def test_a_residual_outside_the_row_space_is_a_failure() -> None:
    """A logical operator added to the right answer fails, as does a wrong syndrome.

    The second needs no branch of its own: `rowspace(H)` sits inside `ker H`,
    so a residual that is not even in the kernel is in neither.
    """
    code = _code("ci")
    error = np.zeros(code.n_qubits, dtype=np.uint8)
    error[[0, 5]] = 1

    logical_failures = [
        decode_succeeds(code, error, error ^ row) for row in code.logical_x
    ]

    assert not any(logical_failures)
    assert not any(
        decode_succeeds(code, error, error ^ row ^ stabilizer)
        for row in code.logical_x
        for stabilizer in _stabilizers(code)[:8]
    )
    wrong_syndrome = error.copy()
    wrong_syndrome[1] ^= 1
    assert not decode_succeeds(code, error, wrong_syndrome)


# --- the oracle: the partition, then the two maximum-likelihood decoders ----------


@pytest.mark.mathematical
def test_the_cosets_partition_every_error_by_syndrome_and_by_quotient() -> None:
    """`2 ** k` cosets per reachable syndrome, and the probabilities sum to one.

    A mis-built quotient does not break loudly (``search/CLAUDE.md``): it
    merges or splits cosets, and both show here as a count that is not
    `2 ** k` and as probability that has gone missing.
    """
    code = _code("ci")

    cosets = error_cosets(code, CI_FLIP)

    assert int(cosets.reachable.sum()) == 2**code.stabilizer_rank
    occupied = cosets.cosets_per_syndrome[cosets.reachable]
    np.testing.assert_array_equal(occupied, np.full(occupied.size, 2**code.n_logical))
    assert not cosets.cosets_per_syndrome[~cosets.reachable].any()
    np.testing.assert_allclose(cosets.coset_probability.sum(), 1.0, atol=1e-12)


@pytest.mark.oracle
def test_a_coset_probability_is_the_sum_over_its_members() -> None:
    """The partition against a direct sum over the errors of one coset.

    Recomputed from the errors rather than from the partition: the cell the
    enumeration reports for each `(syndrome, label)` is the total probability
    of the errors carrying that pair, which is what makes degenerate maximum
    likelihood a sum and not a maximum.
    """
    code = _code("ci")
    n_qubits = code.n_qubits
    errors = ((np.arange(2**n_qubits)[:, None] >> np.arange(n_qubits)) & 1).astype(
        np.uint8
    )
    weight = errors.sum(axis=1)
    probability = CI_FLIP**weight * (1.0 - CI_FLIP) ** (n_qubits - weight)
    syndrome = ((errors.astype(np.int64) @ code.checks.dense().T) & 1) @ (
        1 << np.arange(code.checks.n_checks)
    )
    label = code.logical_label(errors).astype(np.int64) @ (
        1 << np.arange(code.n_logical)
    )

    cosets = error_cosets(code, CI_FLIP)

    for cell in ((0, 0), (1, 1), (17, 2), (63, 3)):
        members = (syndrome == cell[0]) & (label == cell[1])
        np.testing.assert_allclose(
            cosets.coset_probability[cell], probability[members].sum(), atol=1e-15
        )
    for row in np.flatnonzero(cosets.reachable)[:16]:
        np.testing.assert_allclose(
            cosets.coset_probability[row].sum(),
            probability[syndrome == row].sum(),
            atol=1e-15,
        )


@pytest.mark.oracle
def test_summing_a_coset_beats_maximizing_over_one_error() -> None:
    """Degenerate ML is strictly better here, and the margin is pinned.

    The two decoders read the same partition and differ in what they read: the
    coset of greatest total probability against the coset holding the single
    likeliest error. A fixture where they agreed would measure nothing, which
    is why the instance is declared at a seed where they do not
    (``sim/CLAUDE.md``: pin the margin, not only the answer). No syndrome ties
    at this rate, so no tie-break is being measured either.
    """
    code = _code("ci")

    cosets = error_cosets(code, CI_FLIP)

    assert cosets.ties == 0
    assert cosets.disagreements == 24
    np.testing.assert_allclose(
        cosets.degenerate_logical_error_rate, CI_DEGENERATE_RATE, atol=5e-9
    )
    np.testing.assert_allclose(
        cosets.single_error_logical_error_rate, CI_SINGLE_ERROR_RATE, atol=5e-9
    )
    margin = cosets.single_error_logical_error_rate - (
        cosets.degenerate_logical_error_rate
    )
    np.testing.assert_allclose(margin, 2.43832e-03, atol=5e-9)


@pytest.mark.mathematical
def test_no_other_coset_choice_beats_the_degenerate_decoder() -> None:
    """The returned coset carries the greatest probability at every syndrome.

    That is what makes the rate a floor rather than one decoder's score: any
    decoder reading only the syndrome chooses one coset per syndrome, so its
    success probability is a sum of cells this maximum dominates term by term.
    """
    code = _code("ci")

    cosets = error_cosets(code, CI_FLIP)

    rows = np.flatnonzero(cosets.reachable)
    chosen = cosets.coset_probability[rows, cosets.degenerate_label[rows]]
    np.testing.assert_array_equal(chosen, cosets.coset_probability[rows].max(axis=1))
    assert cosets.degenerate_logical_error_rate <= (
        cosets.single_error_logical_error_rate
    )


@pytest.mark.edge_case
def test_an_unreachable_rate_and_an_unreachable_size_are_refused() -> None:
    """A rate outside `(0, 0.5)` and an enumeration past the policy's limit."""
    code = _code("ci")

    with pytest.raises(ValueError, match="flip_probability must be in"):
        error_cosets(code, 0.5)
    with pytest.raises(ValueError, match="error patterns"):
        error_cosets(_code("stress"), CI_FLIP)


# --- belief propagation, scored by the criterion above ---------------------------


@pytest.mark.mathematical
def test_the_correction_depends_on_the_error_only_through_its_syndrome() -> None:
    """Two errors of one syndrome give one correction, which is what makes it a decoder.

    A syndrome decoder sees `H e` and nothing else. This decoder is given the
    ratios of an error, so the claim it rests on is the coset symmetry
    ``sec:ldpc`` proves for the all-zero word: adding a codeword to the error
    adds it to the estimate and leaves the correction fixed. It is checked here
    rather than taken on the argument.
    """
    code = _code("ci")
    channel = BinarySymmetricChannel(CI_FLIP)
    rng = np.random.default_rng(361)
    magnitude = np.log((1.0 - CI_FLIP) / CI_FLIP)
    kernel = np.vstack([code.checks.dense(), code.logical_x])

    for _ in range(25):
        error, llr = sample_x_error(code, channel, rng)
        shifted = error ^ kernel[rng.integers(kernel.shape[0])]

        first = decode_syndrome(code, error, llr)
        second = decode_syndrome(code, shifted, (1.0 - 2.0 * shifted) * magnitude)

        np.testing.assert_array_equal(
            code.checks.syndrome(first.correction),
            code.checks.syndrome(second.correction),
        )
        np.testing.assert_array_equal(first.correction, second.correction)


@pytest.mark.oracle
def test_belief_propagation_is_measured_against_the_degenerate_floor() -> None:
    """BP's decode at `ci`, against the coset exact degenerate ML returns.

    Two comparisons on the same 2,000 seeded draws, and they are different
    questions. Which coset: of the 1,732 decodes that converged, 1,674 return
    the coset Equation~`eq:coset-ml` does and 58 do not. Whether it worked:
    BP leaves the codespace fixed on 1,659 of the 2,000, a logical error rate
    of 0.1705 against the exact floor of 0.070297. The two counts do not
    agree, because the optimal decoder is not a correct one -- on 15 of these
    draws BP returns the optimum's coset and fails with it.

    The 30 four-cycles of the Tanner graph are the structural reason the gap
    exists at all: `H H^T = 0` forces even row overlaps, and an overlap of two
    is a four-cycle, so the condition making the code quantum puts short cycles
    in the graph BP needs free of them.
    """
    code = _ci_code()
    cosets = error_cosets(code, CI_FLIP)
    channel = BinarySymmetricChannel(CI_FLIP)
    rng = np.random.default_rng(2026)
    check_weights = 1 << np.arange(code.checks.n_checks)
    label_weights = 1 << np.arange(code.n_logical)
    agreed = converged = succeeded = 0

    for _ in range(2000):
        error, llr = sample_x_error(code, channel, rng)
        result = decode_syndrome(code, error, llr)
        succeeded += result.succeeded
        if not result.converged:
            continue
        converged += 1
        syndrome = int(code.checks.syndrome(result.correction) @ check_weights)
        label = int(code.logical_label(result.correction) @ label_weights)
        agreed += label == int(cosets.degenerate_label[syndrome])

    assert code.four_cycles() == 30
    assert (converged, agreed, succeeded) == (1732, 1674, 1659)
    rate = (2000 - succeeded) / 2000
    assert rate == 0.1705
    assert (
        cosets.degenerate_logical_error_rate
        < rate
        < 3.0 * (cosets.degenerate_logical_error_rate)
    )


@pytest.mark.structural
def test_the_two_decoding_failures_are_counted_apart() -> None:
    """Converging to a logical coset and reaching the cap are different defects.

    At `ci` and the declared rate the 341 failures are 73 of the first and 268
    of the second, so a single block-error count would report one number for
    two things and hide that BP's dominant failure here is that it does not
    converge at all.
    """
    code = _code("ci")

    measured = measure_logical_error_rate(
        code, BinarySymmetricChannel(CI_FLIP), np.random.default_rng(2026), 2000
    )

    assert (measured.logical_failures, measured.undecoded) == (73, 268)
    assert measured.failures == 341
    assert measured.rate == measured.failures / measured.trials


@pytest.mark.structural
def test_a_decode_that_did_not_converge_returns_no_usable_correction() -> None:
    """An unconverged residual satisfies no syndrome, so it is a failure by the test.

    Counted rather than assumed: over the seeded run every decode that reached
    the cap left a residual outside `ker H`, which is what makes
    :func:`decode_succeeds` refuse it without a second branch.
    """
    code = _code("ci")
    channel = BinarySymmetricChannel(CI_FLIP)
    rng = np.random.default_rng(2026)
    unconverged = 0

    for _ in range(200):
        error, llr = sample_x_error(code, channel, rng)
        result = decode_syndrome(code, error, llr)
        if not result.converged:
            unconverged += 1
            assert not result.succeeded
            assert code.checks.syndrome(result.residual).any()
        else:
            assert not code.checks.syndrome(result.residual).any()

    assert unconverged > 0


@pytest.mark.edge_case
def test_a_run_of_no_trials_is_refused() -> None:
    """A rate over zero draws is not an estimate of anything."""
    code = _code("ci")

    with pytest.raises(ValueError, match="trials must be at least 1"):
        measure_logical_error_rate(
            code, BinarySymmetricChannel(CI_FLIP), np.random.default_rng(1), 0
        )


@pytest.mark.stress
@pytest.mark.structural
def test_the_decoder_runs_at_the_stress_length_with_no_oracle_behind_it() -> None:
    """At 96 qubits `2 ** n` does not enumerate, so the rate is reported alone.

    `k = 16` and the graph carries 160 four-cycles. Over 1,000 draws at the
    declared rate the decoder fails 280: 168 converged to a logical coset and
    112 reached the cap. Nothing here is a gap to an optimum, and the fixture
    states that by declaring no oracle.
    """
    code = _code("stress")
    params = fixture("bicycle_css", "stress").params

    measured = measure_logical_error_rate(
        code, params.channel(), np.random.default_rng(2026), 1000
    )

    assert (code.n_logical, code.four_cycles()) == (16, 160)
    assert (measured.logical_failures, measured.undecoded) == (168, 112)
    assert measured.rate == 0.28
