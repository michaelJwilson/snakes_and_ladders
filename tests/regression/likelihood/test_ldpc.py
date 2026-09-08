"""The specialised decoder, held to the general sum-product and to enumeration.

Three oracles, none sharing code with the decoder: on a cycle-free code the
tree schedule of `message_passing.sum_product` is exact and so is
enumeration, and the decoder must equal both; on a loopy code both the
decoder and the general flooding are the Bethe approximation, so the
assertion is that they are the *same* approximation; and the erasure
threshold the size measurement brackets is a closed form of density
evolution (``app:density-evolution``), pinned to its published value before
the code at size is held to it.

Codes are drawn here at several block lengths and seeds rather than read from
the LDPC fixture: the claim is about the (3,6) *ensemble*, and an ensemble
reaches structures one declared member does not (``sim/CLAUDE.md``). The
declared members are what the notebook and the figures run on.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import pytest
from snakes_and_ladders.likelihood.ldpc import (
    Decoding,
    DecodingAlgorithm,
    MapEstimate,
    decode,
    enumerate_codewords,
    erasure_density_evolution,
    erasure_threshold,
    exact_decoding,
)
from snakes_and_ladders.likelihood.message_passing import (
    MessageSchedule,
    max_product,
    sum_product,
)
from snakes_and_ladders.sim.factor_graph import from_parity_check
from snakes_and_ladders.sim.ldpc import (
    LLR_CAP,
    BinaryErasureChannel,
    BinaryInputGaussianChannel,
    BinarySymmetricChannel,
    Channel,
    ParityCheck,
    all_zero_transmission,
    encode,
    gallager_code,
    generator_matrix,
)

#: Agreement between the decoder and the general implementation at a shared
#: fixed point: same arithmetic in a different order, both run to a message
#: residual of 1e-12. Realized 4.6e-11 on the worst of the six loopy fixtures.
LOOPY_TOLERANCE = 1e-9

#: On a cycle-free code both are exact and agree to rounding in log-odds
#: wherever the exact value is inside the message cap; realized 1.9e-13.
TREE_TOLERANCE = 1e-12

#: Where the exact log-odds pass the cap the decoder's do not, and the gap in
#: probability is bounded by `exp(-LLR_CAP) = 9.36e-14`; realized 9.37e-14 on
#: the erasure channel, the excess being rounding.
CAPPED_TOLERANCE = 2e-13

#: The (3,6) erasure threshold, Richardson and Urbanke (2008), Table 3.xx --
#: the closed form the decoder at size is bracketed against.
PUBLISHED_THRESHOLD = {(3, 6): 0.4294, (4, 8): 0.3834, (3, 5): 0.5176}

CHANNELS: list[Channel] = [
    BinarySymmetricChannel(0.1),
    BinaryErasureChannel(0.3),
    BinaryInputGaussianChannel(1.0),
]


def _caterpillar(n_checks: int, degree: int) -> ParityCheck:
    """A cycle-free code: consecutive checks share exactly one bit."""
    n_bits = n_checks * (degree - 1) + 1
    dense = np.zeros((n_checks, n_bits), dtype=np.uint8)
    for j in range(n_checks):
        dense[j, j * (degree - 1) : j * (degree - 1) + degree] = 1
    return ParityCheck.from_dense(dense)


def _assert_exact(realized: np.ndarray, exact: np.ndarray) -> None:
    """Equal in probability everywhere; in log-odds too, on a fixture the cap
    never touches -- a capped message upstream moves a log-odds of 29 by 0.4
    and its probability by 1e-13, so past the cap only the second is exact."""
    probability = 1.0 / (1.0 + np.exp(-realized))
    np.testing.assert_allclose(
        probability, 1.0 / (1.0 + np.exp(-exact)), rtol=0, atol=CAPPED_TOLERANCE
    )
    if np.abs(exact).max() < LLR_CAP / 2:
        np.testing.assert_allclose(realized, exact, rtol=0, atol=TREE_TOLERANCE)


def _general_llr(result: Mapping[str, np.ndarray], n_bits: int) -> np.ndarray:
    """The general implementation's marginals as log-odds, the decoder's unit."""
    return np.array(
        [np.log(result[f"x{i}"][0]) - np.log(result[f"x{i}"][1]) for i in range(n_bits)]
    )


# --- exact on a tree -----------------------------------------------------------


@pytest.mark.oracle
@pytest.mark.parametrize("channel", CHANNELS, ids=lambda c: type(c).__name__)
def test_sum_product_is_exact_on_a_cycle_free_code(channel: Channel) -> None:
    """On a 22-bit caterpillar code the decoder equals the tree schedule and the
    enumeration over 32,768 codewords: 1e-12 in log-odds on the symmetric and
    Gaussian channels, 2e-13 in probability on the erasure channel, whose
    certain bits sit at the message cap."""
    code = _caterpillar(7, 4)
    graph = from_parity_check(
        code, llr := all_zero_transmission(code, channel, np.random.default_rng(1))
    )
    assert graph.is_tree()

    decoded = decode(code, llr, early_stop=False, max_iterations=20, tolerance=1e-14)
    general = sum_product(graph, schedule=MessageSchedule.TREE)
    exact = exact_decoding(code, llr)

    assert exact.n_codewords == 2**15
    assert decoded.estimate is MapEstimate.BITWISE
    _assert_exact(decoded.posterior_llr, exact.posterior_llr)
    _assert_exact(_general_llr(general.variable, code.n_bits), exact.posterior_llr)
    np.testing.assert_array_equal(decoded.bits, exact.posterior_llr < 0)


@pytest.mark.oracle
@pytest.mark.parametrize("seed", [2, 3, 4])
def test_min_sum_is_max_product_and_the_ml_codeword_on_a_cycle_free_code(
    seed: int,
) -> None:
    """On the caterpillar code min-sum's max-marginals equal the tree-schedule
    max-product's and its decision is the enumerated ML codeword, whose margin
    over the runner-up is pinned above 0.1 nats so no tie is being broken."""
    code = _caterpillar(7, 4)
    llr = all_zero_transmission(
        code, BinaryInputGaussianChannel(1.0), np.random.default_rng(seed)
    )

    decoded = decode(
        code,
        llr,
        algorithm=DecodingAlgorithm.MIN_SUM,
        early_stop=False,
        max_iterations=20,
        tolerance=1e-14,
    )
    assignment, general = max_product(
        from_parity_check(code, llr), schedule=MessageSchedule.TREE
    )
    exact = exact_decoding(code, llr)

    scores = np.sort(-(enumerate_codewords(code).astype(float) @ llr))
    assert scores[-1] - scores[-2] > 0.1
    assert decoded.estimate is MapEstimate.BLOCKWISE
    np.testing.assert_allclose(
        decoded.posterior_llr,
        _general_llr(general.variable, code.n_bits),
        rtol=0,
        atol=TREE_TOLERANCE,
    )
    np.testing.assert_array_equal(decoded.bits, exact.ml_codeword)
    np.testing.assert_array_equal(
        decoded.bits, [assignment[f"x{i}"] for i in range(code.n_bits)]
    )


@pytest.mark.oracle
def test_the_enumeration_oracle_on_a_single_parity_check_by_hand() -> None:
    """On `H = [1 1 1]` the four codewords give posteriors a reader can write down."""
    code = ParityCheck.from_dense(np.array([[1, 1, 1]]))
    llr = np.array([1.0, -0.5, 2.0])

    exact = exact_decoding(code, llr)

    words = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]
    weights = [
        math.exp(-sum(b * v for b, v in zip(w, llr, strict=True))) for w in words
    ]
    for i in range(3):
        zero = sum(wt for w, wt in zip(words, weights, strict=True) if w[i] == 0)
        one = sum(wt for w, wt in zip(words, weights, strict=True) if w[i] == 1)
        assert math.isclose(exact.posterior_llr[i], math.log(zero / one), rel_tol=1e-13)
    assert math.isclose(exact.log_evidence, math.log(sum(weights)), rel_tol=1e-13)
    np.testing.assert_array_equal(exact.ml_codeword, words[int(np.argmax(weights))])


# --- the same approximation on a loopy code ------------------------------------


@pytest.mark.oracle
@pytest.mark.parametrize(
    ("n_bits", "channel", "seed"),
    [
        (12, BinarySymmetricChannel(0.15), 1),
        (12, BinaryInputGaussianChannel(1.0), 5),
        (12, BinaryErasureChannel(0.4), 6),
        (18, BinarySymmetricChannel(0.15), 7),
        (18, BinaryInputGaussianChannel(1.2), 8),
        (12, BinarySymmetricChannel(0.2), 10),
    ],
    ids=lambda v: v if isinstance(v, int) else type(v).__name__,
)
def test_flooding_reaches_the_general_fixed_point_on_a_loopy_code(
    n_bits: int, channel: Channel, seed: int
) -> None:
    """On six loopy (3,6) codes the decoder's posteriors equal the general
    damped flooding's to 1e-9 -- the same Bethe fixed point from two codes."""
    rng = np.random.default_rng(seed)
    code = gallager_code(n_bits, 3, 6, rng)
    llr = all_zero_transmission(code, channel, rng)
    graph = from_parity_check(code, llr)
    assert not graph.is_tree()

    decoded = decode(code, llr, early_stop=False, max_iterations=3000, tolerance=1e-12)
    general = sum_product(
        graph, schedule=MessageSchedule.FLOODING, tolerance=1e-12, max_iterations=3000
    )

    assert decoded.residual <= 1e-12
    np.testing.assert_allclose(
        decoded.posterior_llr,
        _general_llr(general.variable, n_bits),
        rtol=0,
        atol=LOOPY_TOLERANCE,
    )


# --- the sign symmetry the all-zero shortcut rests on --------------------------


@pytest.mark.mathematical
@pytest.mark.parametrize("channel", CHANNELS, ids=lambda c: type(c).__name__)
@pytest.mark.parametrize("algorithm", list(DecodingAlgorithm))
def test_negating_the_ratios_at_a_codeword_moves_the_decoding_with_it(
    channel: Channel, algorithm: DecodingAlgorithm
) -> None:
    """Negating `L` where `c_i = 1` negates every posterior exactly and XORs `c`
    into every decided bit, for both check updates and all three channels."""
    code = gallager_code(96, 3, 6, np.random.default_rng(30))
    codeword = encode(
        code,
        np.random.default_rng(31).integers(
            0, 2, size=int(generator_matrix(code).shape[0])
        ),
    )
    sign = 1.0 - 2.0 * codeword
    on_zero = all_zero_transmission(code, channel, np.random.default_rng(32))

    reference = decode(code, on_zero, algorithm=algorithm)
    moved = decode(code, sign * on_zero, algorithm=algorithm)

    assert moved.iterations == reference.iterations
    np.testing.assert_array_equal(moved.posterior_llr, sign * reference.posterior_llr)
    decided = reference.posterior_llr != 0
    np.testing.assert_array_equal(
        moved.bits[decided], (reference.bits ^ codeword)[decided]
    )


# --- the mid-size code, per pull request ---------------------------------------


def _error_rates(
    code: ParityCheck, channel: Channel, seeds: range
) -> tuple[float, float]:
    """Bit and block error rates of the zero word over `seeds`."""
    bit_errors = block_errors = 0
    for seed in seeds:
        decoded = decode(
            code, all_zero_transmission(code, channel, np.random.default_rng(seed))
        )
        bit_errors += int(decoded.bits.sum())
        block_errors += int(not decoded.decoded)
    return bit_errors / (len(seeds) * code.n_bits), block_errors / len(seeds)


@pytest.mark.simulated_truth
def test_the_996_bit_code_corrects_the_bsc_below_its_threshold_and_not_above() -> None:
    """Over 20 shared seeds the 996-bit (3,6) code has zero bit errors at
    `p = 0.05` and a bit error rate above 0.02 at `p = 0.09`, either side of
    the (3,6) BSC threshold `p* = 0.084`; realized 0 and 0.049."""
    code = gallager_code(996, 3, 6, np.random.default_rng(3))
    seeds = range(100, 120)

    below, below_blocks = _error_rates(code, BinarySymmetricChannel(0.05), seeds)
    above, above_blocks = _error_rates(code, BinarySymmetricChannel(0.09), seeds)

    assert below == 0.0
    assert below_blocks == 0.0
    assert above > 0.02
    assert above_blocks > 0.3


@pytest.mark.simulated_truth
def test_the_996_bit_code_on_the_erasure_channel_either_side_of_the_threshold() -> None:
    """Over 20 seeds no erasure survives at `epsilon = 0.35` and more than
    half of them survive at `epsilon = 0.5`; realized 0 and 0.433 of 0.5."""
    code = gallager_code(996, 3, 6, np.random.default_rng(3))

    residual = {}
    for epsilon in (0.35, 0.5):
        left = []
        for seed in range(100, 120):
            llr = all_zero_transmission(
                code, BinaryErasureChannel(epsilon), np.random.default_rng(seed)
            )
            left.append(
                float(np.mean(decode(code, llr, max_iterations=300).posterior_llr == 0))
            )
        residual[epsilon] = float(np.mean(left))

    assert residual[0.35] == 0.0
    assert residual[0.5] > 0.25


# --- density evolution, and the code at size ------------------------------------


@pytest.mark.mathematical
@pytest.mark.parametrize(("degrees", "published"), sorted(PUBLISHED_THRESHOLD.items()))
def test_the_erasure_threshold_matches_its_published_value(
    degrees: tuple[int, int], published: float
) -> None:
    """Bisection on `eq:density-evolution` reproduces the published (3,6), (4,8)
    and (3,5) thresholds to 5e-4."""
    assert (
        abs(erasure_threshold(*degrees, iterations=2000, precision=1e-4) - published)
        < 5e-4
    )


@pytest.mark.mathematical
def test_density_evolution_is_monotone_and_fixed_at_zero() -> None:
    """The recursion is non-increasing below the threshold, stays at its
    fixed point above it, and zero is a fixed point at every `epsilon`."""
    below = erasure_density_evolution(0.40, 3, 6, 500)
    above = erasure_density_evolution(0.50, 3, 6, 500)

    assert np.all(np.diff(below) <= 0)
    assert below[-1] < 1e-6
    assert above[-1] > 0.4
    assert abs(above[-1] - above[-2]) < 1e-12
    assert erasure_density_evolution(0.0, 3, 6, 3)[-1] == 0.0


@pytest.mark.release
@pytest.mark.simulated_truth
def test_the_20000_bit_code_brackets_the_erasure_threshold() -> None:
    """At `n = 19,998` (the multiple of six nearest the ticket's 20,000) the
    flooding decoder resolves every erasure at `epsilon = 0.42` and leaves
    more than a fifth of the bits erased at `0.44`, bracketing the density
    evolution threshold 0.4294 on three seeds each; realized 0 and 0.25-0.30."""
    code = gallager_code(19_998, 3, 6, np.random.default_rng(340))
    assert code.n_checks == 9_999

    left = {}
    for epsilon in (0.42, 0.44):
        fractions = []
        for seed in range(200, 203):
            llr = all_zero_transmission(
                code, BinaryErasureChannel(epsilon), np.random.default_rng(seed)
            )
            decoded = decode(code, llr, max_iterations=300)
            assert not decoded.bits.any()  # an erasure channel never flips a bit
            fractions.append(float(np.mean(decoded.posterior_llr == 0)))
        left[epsilon] = fractions

    assert left[0.42] == [0.0, 0.0, 0.0]
    assert min(left[0.44]) > 0.2


# --- stopping, and the shape of a result ----------------------------------------


@pytest.mark.mathematical
def test_the_syndrome_stop_returns_at_the_first_codeword() -> None:
    """With the early stop the loop ends at the first decided codeword, before
    the cap; without it the same ratios run to the cap, sit at an exact fixed
    point there (every message at the cap) and decode the same word."""
    code = gallager_code(96, 3, 6, np.random.default_rng(40))
    llr = all_zero_transmission(
        code, BinarySymmetricChannel(0.05), np.random.default_rng(41)
    )

    stopped = decode(code, llr, max_iterations=50)
    unstopped = decode(code, llr, max_iterations=50, early_stop=False)

    assert stopped.decoded
    assert stopped.iterations < 50
    assert not np.any(code.syndrome(stopped.bits))
    assert np.all(stopped.posterior_llr != 0)
    assert unstopped.iterations == 50
    assert unstopped.residual == 0.0
    np.testing.assert_array_equal(stopped.bits, unstopped.bits)


@pytest.mark.edge_case
def test_an_erased_bit_is_not_a_decision_for_zero() -> None:
    """The zero word through an erasure channel satisfies every check at once,
    and must not be reported decoded while a posterior is still zero."""
    code = gallager_code(96, 3, 6, np.random.default_rng(42))
    llr = all_zero_transmission(
        code, BinaryErasureChannel(0.3), np.random.default_rng(43)
    )

    first = decode(code, llr, max_iterations=1)
    finished = decode(code, llr, max_iterations=100)

    assert not np.any(code.syndrome(first.bits))
    assert not first.decoded
    assert np.any(first.posterior_llr == 0)
    assert finished.decoded
    assert finished.iterations > 1
    everything_erased: Decoding = decode(code, np.zeros(code.n_bits), max_iterations=5)
    assert not everything_erased.decoded
    assert np.all(everything_erased.posterior_llr == 0)


@pytest.mark.edge_case
def test_the_decoder_and_the_enumeration_refuse_what_they_cannot_do() -> None:
    code = gallager_code(48, 3, 6, np.random.default_rng(44))
    with pytest.raises(ValueError, match="shape"):
        decode(code, np.zeros(47))
    with pytest.raises(ValueError, match="max_iterations"):
        decode(code, np.zeros(48), max_iterations=0)
    with pytest.raises(ValueError, match="refusing to enumerate"):
        enumerate_codewords(code)
