"""Decoding a CSS code: the criterion first, then the oracle, then the decoder.

:func:`decode_succeeds` is pinned from both sides first: the right answer plus
a nonzero stabilizer succeeds, plus a logical operator fails
(``likelihood/CLAUDE.md``: a criterion's test is paired). The oracle is exact
degenerate ML by enumeration, a sum over a coset rather than the likeliest
single error, on a fixture declared where the two differ. Belief propagation
is reported against that floor, its failures split into two defects.
"""

from __future__ import annotations

import numpy as np
import pytest
from sal.likelihood.css import (
    decode_succeeds,
    decode_syndrome,
    error_cosets,
    measure_logical_error_rate,
)
from sal.likelihood.ldpc import (
    DecodingAlgorithm,
    decode,
    enumerate_codewords,
)
from sal.sim.css import CssCode, sample_x_error
from sal.sim.fixtures import fixture
from sal.sim.ldpc import BinarySymmetricChannel

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
@pytest.mark.analytic
def test_a_residual_in_the_row_space_is_a_success() -> None:
    """Adding any nonzero stabilizer to the right answer still decodes.

    127 of the declared instance's 128 stabilizers are nonzero: each a success.
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
@pytest.mark.analytic
def test_a_residual_outside_the_row_space_is_a_failure() -> None:
    """A logical operator added to the right answer fails, as does a wrong syndrome.

    `rowspace(H)` sits inside `ker H`, so no second branch is needed.
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


@pytest.mark.analytic
def test_the_cosets_partition_every_error_by_syndrome_and_by_quotient() -> None:
    """`2 ** k` cosets per reachable syndrome, and the probabilities sum to one.

    A mis-built quotient merges or splits cosets (``search/CLAUDE.md``).
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

    Recomputed from the errors: degenerate ML is a sum, not a maximum.
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

    Declared at a seed where the decoders differ; no syndrome ties (``sim/CLAUDE.md``).
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


@pytest.mark.analytic
def test_no_other_coset_choice_beats_the_degenerate_decoder() -> None:
    """The returned coset carries the greatest probability at every syndrome.

    So the rate is a floor for any decoder reading only the syndrome.
    """
    code = _code("ci")

    cosets = error_cosets(code, CI_FLIP)

    rows = np.flatnonzero(cosets.reachable)
    chosen = cosets.coset_probability[rows, cosets.degenerate_label[rows]]
    np.testing.assert_array_equal(chosen, cosets.coset_probability[rows].max(axis=1))
    assert cosets.degenerate_logical_error_rate <= (
        cosets.single_error_logical_error_rate
    )


@pytest.mark.smoke
def test_an_unreachable_rate_and_an_unreachable_size_are_refused() -> None:
    """A rate outside `(0, 0.5)` and an enumeration past the policy's limit."""
    code = _code("ci")

    with pytest.raises(ValueError, match="flip_probability must be in"):
        error_cosets(code, 0.5)
    with pytest.raises(ValueError, match="error patterns"):
        error_cosets(_code("stress"), CI_FLIP)


# --- belief propagation, scored by the criterion above ---------------------------


@pytest.mark.analytic
def test_the_correction_depends_on_the_error_only_through_its_syndrome() -> None:
    """Two errors of one syndrome give one correction, which is what makes it a decoder.

    ``sec:ldpc``'s coset symmetry, checked rather than taken on the argument.
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


@pytest.mark.critical
@pytest.mark.oracle
def test_the_syndrome_decode_is_belief_propagation_on_the_component_code() -> None:
    """The CSS decoder against `ldpc.decode` on the same Tanner graph.

    The rung below (issue #734), an identity: over 50 draws and both check
    updates the residual is `decode(code.checks, llr).bits` bitwise. All 512
    words of `ker H` added to two errors give one 16-bit correction on all
    1,024. `H ê == H e` on the 71 of 100 decodes that converged; of the 29
    capped, 25 differ. Sum-product converged on 40 of 50, min-sum on 31.
    """
    code = _ci_code()
    channel = BinarySymmetricChannel(CI_FLIP)
    magnitude = np.log((1.0 - CI_FLIP) / CI_FLIP)
    rng = np.random.default_rng(734)
    converged = {algorithm: 0 for algorithm in DecodingAlgorithm}
    reproduced = departed = 0

    for _ in range(50):
        error, llr = sample_x_error(code, channel, rng)
        for algorithm in DecodingAlgorithm:
            result = decode_syndrome(code, error, llr, algorithm=algorithm)
            decoding = decode(code.checks, llr, algorithm=algorithm)

            np.testing.assert_array_equal(result.residual, decoding.bits)
            np.testing.assert_array_equal(result.correction, error ^ decoding.bits)
            assert result.converged == decoding.decoded
            assert result.iterations == decoding.iterations

            same = np.array_equal(
                code.checks.syndrome(result.correction), code.checks.syndrome(error)
            )
            converged[algorithm] += result.converged
            if result.converged:
                assert same
                reproduced += 1
            else:
                departed += int(not same)

    assert converged == {
        DecodingAlgorithm.SUM_PRODUCT: 40,
        DecodingAlgorithm.MIN_SUM: 31,
    }
    assert (reproduced, departed) == (71, 25)

    words = enumerate_codewords(code.checks)
    assert words.shape == (512, code.checks.n_bits)
    rng = np.random.default_rng(11)
    cosets: dict[bool, tuple[np.ndarray, np.ndarray]] = {}
    while len(cosets) < 2:
        error, llr = sample_x_error(code, channel, rng)
        cosets.setdefault(decode_syndrome(code, error, llr).converged, (error, llr))

    for error, llr in cosets.values():
        base = decode_syndrome(code, error, llr)
        for word in words:
            shifted = np.asarray(error ^ word, dtype=np.uint8)
            result = decode_syndrome(code, shifted, (1.0 - 2.0 * shifted) * magnitude)
            np.testing.assert_array_equal(result.correction, base.correction)
            assert result.converged == base.converged


@pytest.mark.oracle
def test_belief_propagation_is_measured_against_the_degenerate_floor() -> None:
    """BP's decode at `ci`, against the coset exact degenerate ML returns.

    Over 2,000 draws: of 1,732 converged, 1,674 return eq:coset-ml's coset and
    58 do not; BP fixes the codespace on 1,659, a logical error rate of 0.1705
    against the floor 0.070297; on 15 draws it fails with the optimum's coset.
    `H H^T = 0` forces even overlaps, hence the graph's 30 four-cycles.
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


@pytest.mark.smoke
def test_the_two_decoding_failures_are_counted_apart() -> None:
    """Converging to a logical coset and reaching the cap are different defects.

    At `ci`: 341 failures, 73 converged to a logical coset and 268 capped.
    """
    code = _code("ci")

    measured = measure_logical_error_rate(
        code, BinarySymmetricChannel(CI_FLIP), np.random.default_rng(2026), 2000
    )

    assert (measured.logical_failures, measured.undecoded) == (73, 268)
    assert measured.failures == 341
    assert measured.rate == measured.failures / measured.trials


@pytest.mark.smoke
def test_a_decode_that_did_not_converge_returns_no_usable_correction() -> None:
    """An unconverged residual satisfies no syndrome, so it is a failure by the test.

    Counted: every capped decode left a residual outside `ker H`.
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


@pytest.mark.smoke
def test_a_run_of_no_trials_is_refused() -> None:
    """A rate over zero draws is not an estimate of anything."""
    code = _code("ci")

    with pytest.raises(ValueError, match="trials must be at least 1"):
        measure_logical_error_rate(
            code, BinarySymmetricChannel(CI_FLIP), np.random.default_rng(1), 0
        )


@pytest.mark.stress
@pytest.mark.smoke
def test_the_decoder_runs_at_the_stress_length_with_no_oracle_behind_it() -> None:
    """At 96 qubits `2 ** n` does not enumerate, so the rate is reported alone.

    `k = 16`, 160 four-cycles; 280 of 1,000 fail (168 logical, 112 capped).
    """
    code = _code("stress")
    params = fixture("bicycle_css", "stress").params

    measured = measure_logical_error_rate(
        code, params.channel(), np.random.default_rng(2026), 1000
    )

    assert (code.n_logical, code.four_cycles()) == (16, 160)
    assert (measured.logical_failures, measured.undecoded) == (168, 112)
    assert measured.rate == 0.28


# --- end to end: a planted error, its syndrome, and the rate on record ----------

#: The seeds and the budget `docs/experiments/015-css-decoding-under-degeneracy.md`
#: declares, and the run `STATUS.md` reports from them. 12,000 decodes are
#: 10.5 s, over the 10 s a per-pull-request test may take, so the run is
#: `release`: a shorter one would report a rate the record does not carry.
EXPERIMENT_SEEDS = (11, 12, 13)
EXPERIMENT_TRIALS = 4000

#: The logical error rate that run records, its spread over the three seeds,
#: and the two failures counted apart: 403 decodes that converged on a logical
#: coset and 1,742 that reached the iteration cap.
RECORDED_LOGICAL_RATE = 0.178750
RECORDED_SPREAD = 0.005788
RECORDED_FAILURES = (403, 1742)

#: What a figure recorded to six decimals is recomputed to (`DEV.md`): half of
#: its last digit. A rate over seeded draws reproduces exactly, so the
#: recomputation is held to the precision the record has and no looser.
RECORDED_PRECISION = 5e-7


@pytest.mark.release
@pytest.mark.end2end
def test_a_planted_error_is_corrected_at_the_logical_rate_on_record() -> None:
    """Realized 0.178750 over 12,000 planted errors, against the 0.178750
    `STATUS.md` records, with the 403 and 1,742 failures split as recorded and
    the spread over the three seeds 0.005788; 2.54 times the exact degenerate
    floor 0.070297 the enumeration returns on the same code and rate.

    No message is transmitted: the truth is the planted `X` error
    (`sim.css.sample_x_error`), the decoder sees its syndrome, and the
    judgement is :func:`decode_succeeds`. The record says where this decoder
    lands (three seeds of 4,000); `error_cosets` over `2 ** 16` errors says
    where the optimum is. The 2.54x gap is the cost of 30 four-cycles.
    """
    code = _ci_code()
    channel = BinarySymmetricChannel(CI_FLIP)
    rates = []
    logical = undecoded = 0

    for seed in EXPERIMENT_SEEDS:
        measured = measure_logical_error_rate(
            code, channel, np.random.default_rng(seed), EXPERIMENT_TRIALS
        )
        assert measured.trials == EXPERIMENT_TRIALS
        rates.append(measured.rate)
        logical += measured.logical_failures
        undecoded += measured.undecoded

    realized = float(np.mean(rates))
    assert (logical, undecoded) == RECORDED_FAILURES
    assert abs(realized - RECORDED_LOGICAL_RATE) < RECORDED_PRECISION
    assert abs(float(np.std(rates)) - RECORDED_SPREAD) < RECORDED_PRECISION

    floor = error_cosets(code, CI_FLIP).degenerate_logical_error_rate
    assert floor == pytest.approx(CI_DEGENERATE_RATE, abs=RECORDED_PRECISION)
    assert realized == pytest.approx(2.54 * floor, rel=0.01)
