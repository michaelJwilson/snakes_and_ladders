"""BCJR and Viterbi on one trellis, against enumeration and against sum-product.

Enumerating all ``2 ** K`` messages says the posterior is right; message
passing on :func:`snakes_and_ladders.sim.factor_graph.from_trellis` says it is
the same computation as every other problem class (the pairing of #340).
Tolerances, absolute on a per-bit log-odds ratio: ``1e-9`` against
enumeration (measured ``2.8e-14``) and against the tree schedule (``2.7e-15``).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from snakes_and_ladders.backend import Backend
from snakes_and_ladders.likelihood.convolutional import (
    bcjr,
    enumerate_messages,
    exact_bitwise_posterior,
    viterbi,
)
from snakes_and_ladders.likelihood.message_passing import (
    MessageScheduleName,
    max_product,
    sum_product,
)
from snakes_and_ladders.sim.convolutional import (
    IMPOSSIBLE_EDGE,
    Trellis,
    encode_stream,
    recursive_systematic_trellis,
    terminate,
)
from snakes_and_ladders.sim.factor_graph import Factor, FactorGraph, from_trellis
from snakes_and_ladders.sim.ldpc import BinaryInputGaussianChannel

from tests._rows import every_value

#: Agreement with an exact answer on a log-odds ratio. Measured at 2.8e-14
#: against enumeration and 2.7e-15 against the tree schedule at the sizes
#: below; five orders of margin, so a real disagreement fails and rounding
#: does not.
EXACT_TOLERANCE = 1e-9

FEEDBACK, FEEDFORWARD, MEMORY = 0o7, 0o5, 2


def _received(
    trellis: Trellis, message: np.ndarray, sigma: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Terminate, encode and transmit; return the two streams' channel ratios."""
    inputs = terminate(trellis, message)
    parity, _ = encode_stream(trellis, inputs)
    ratios = BinaryInputGaussianChannel(sigma).log_likelihood_ratios(
        np.concatenate([inputs, parity]), rng
    )
    return ratios[: inputs.size], ratios[inputs.size :]


# --- against enumeration --------------------------------------------------------


@pytest.mark.oracle
@pytest.mark.parametrize("backend", [Backend.PYTHON, Backend.RUST])
def test_bcjr_posteriors_are_the_exact_bitwise_map(backend: Backend) -> None:
    """The forward-backward ratio equals the sum over all `2 ** K` messages.

    A terminated trellis is a chain, so exact; both backends against enumeration.
    """

    def check(message_length: int) -> None:
        trellis = recursive_systematic_trellis(FEEDBACK, FEEDFORWARD, MEMORY)
        rng = np.random.default_rng(233)

        for _ in range(5):
            message = rng.integers(0, 2, message_length).astype(np.uint8)
            systematic, parity = _received(trellis, message, 0.9, rng)

            decoding = bcjr(trellis, systematic, parity, backend=backend)
            exact = exact_bitwise_posterior(trellis, systematic, parity, message_length)

            np.testing.assert_allclose(
                decoding.posterior_llr[:message_length],
                exact.posterior_llr,
                atol=EXACT_TOLERANCE,
            )
            assert decoding.log_evidence == pytest.approx(
                exact.log_evidence, abs=EXACT_TOLERANCE
            )

    every_value([6, 10], check)


@pytest.mark.oracle
def test_viterbi_returns_the_maximum_likelihood_message_with_a_pinned_margin() -> None:
    """The best path is the enumerated maximum, and the runner-up is not tied to it.

    A positive margin per draw (``sim/CLAUDE.md``: a degenerate fixture proves nothing).
    """
    trellis = recursive_systematic_trellis(FEEDBACK, FEEDFORWARD, MEMORY)
    rng = np.random.default_rng(11)

    for _ in range(10):
        message = rng.integers(0, 2, 10).astype(np.uint8)
        systematic, parity = _received(trellis, message, 1.0, rng)
        exact = exact_bitwise_posterior(trellis, systematic, parity, 10)

        assert exact.margin > 1e-6
        np.testing.assert_array_equal(
            viterbi(trellis, systematic, parity)[:10], exact.ml_message
        )


@pytest.mark.oracle
def test_the_two_decodings_of_one_trellis_are_different_answers() -> None:
    """A received word on which the bitwise and blockwise MAP disagree.

    ``likelihood/CLAUDE.md``: two decodings are different answers; seed found by search.
    """
    trellis = recursive_systematic_trellis(FEEDBACK, FEEDFORWARD, MEMORY)
    rng = np.random.default_rng(2)
    message = rng.integers(0, 2, 8).astype(np.uint8)
    systematic, parity = _received(trellis, message, 1.6, rng)
    exact = exact_bitwise_posterior(trellis, systematic, parity, 8)

    bitwise = (exact.posterior_llr < 0.0).astype(np.uint8)

    assert not np.array_equal(bitwise, exact.ml_message)
    np.testing.assert_array_equal(
        viterbi(trellis, systematic, parity)[:8], exact.ml_message
    )
    np.testing.assert_array_equal(
        (bcjr(trellis, systematic, parity).posterior_llr[:8] < 0.0).astype(np.uint8),
        bitwise,
    )


# --- against the general message passing ----------------------------------------


@pytest.mark.oracle
def test_the_trellis_graph_is_a_chain_and_sum_product_gives_bcjr() -> None:
    """The tree schedule's beliefs and `log Z` are BCJR's, edge for edge.

    The input bit is read off the pairwise belief: the one extra arithmetic step.
    """
    trellis = recursive_systematic_trellis(FEEDBACK, FEEDFORWARD, MEMORY)
    rng = np.random.default_rng(19)
    message = rng.integers(0, 2, 8).astype(np.uint8)
    systematic, parity = _received(trellis, message, 1.0, rng)
    graph = from_trellis(trellis, systematic, parity)

    assert graph.is_tree()
    marginals = sum_product(graph, schedule=MessageScheduleName.TREE)
    decoding = bcjr(trellis, systematic, parity)

    assert marginals.exact
    assert marginals.log_partition == pytest.approx(
        decoding.log_evidence, abs=EXACT_TOLERANCE
    )
    for step in range(systematic.size):
        belief = marginals.factor[f"T{step}"]
        states = np.arange(trellis.n_states)
        zero = belief[states, trellis.next_state[:, 0]].sum()
        one = belief[states, trellis.next_state[:, 1]].sum()
        assert np.log(zero / one) == pytest.approx(
            decoding.posterior_llr[step], abs=EXACT_TOLERANCE
        )


@pytest.mark.oracle
def test_max_product_on_the_trellis_graph_gives_the_viterbi_path() -> None:
    """The general max-product's assignment is the specialized Viterbi's, per draw."""
    trellis = recursive_systematic_trellis(FEEDBACK, FEEDFORWARD, MEMORY)
    rng = np.random.default_rng(21)

    for _ in range(5):
        message = rng.integers(0, 2, 8).astype(np.uint8)
        systematic, parity = _received(trellis, message, 1.0, rng)
        graph = from_trellis(trellis, systematic, parity)

        assignment, _ = max_product(graph, schedule=MessageScheduleName.TREE)
        path = [assignment[f"s{t}"] for t in range(systematic.size + 1)]
        from_graph = np.array(
            [
                0 if trellis.next_state[path[t], 0] == path[t + 1] else 1
                for t in range(systematic.size)
            ],
            dtype=np.uint8,
        )

        np.testing.assert_array_equal(viterbi(trellis, systematic, parity), from_graph)


# --- the a priori input and the boundary conditions ------------------------------


@pytest.mark.analytic
def test_an_a_priori_ratio_enters_the_posterior_exactly_once() -> None:
    """`posterior = channel + a priori + extrinsic`, the decomposition eq:extrinsic.

    A turbo iteration rests on it; a leaked a priori term fails here.
    """
    trellis = recursive_systematic_trellis(FEEDBACK, FEEDFORWARD, MEMORY)
    rng = np.random.default_rng(31)
    message = rng.integers(0, 2, 10).astype(np.uint8)
    systematic, parity = _received(trellis, message, 1.0, rng)
    apriori = rng.normal(size=systematic.size)

    decoding = bcjr(trellis, systematic, parity, apriori)

    np.testing.assert_allclose(
        decoding.posterior_llr,
        systematic + apriori + decoding.extrinsic_llr,
        atol=1e-12,
    )


@pytest.mark.oracle
def test_an_a_priori_ratio_is_the_same_evidence_as_a_second_channel() -> None:
    """A priori `L_a` equals enumerating with `L_a` added to the systematic stream.

    Evidence about the bit from elsewhere, refereed by enumeration.
    """
    trellis = recursive_systematic_trellis(FEEDBACK, FEEDFORWARD, MEMORY)
    rng = np.random.default_rng(37)
    message = rng.integers(0, 2, 10).astype(np.uint8)
    systematic, parity = _received(trellis, message, 1.0, rng)
    apriori = rng.normal(size=systematic.size)

    decoding = bcjr(trellis, systematic, parity, apriori)
    exact = exact_bitwise_posterior(trellis, systematic + apriori, parity, 10)

    np.testing.assert_allclose(
        decoding.posterior_llr[:10], exact.posterior_llr, atol=EXACT_TOLERANCE
    )


@pytest.mark.analytic
def test_an_unterminated_decoder_computes_a_different_posterior() -> None:
    """Telling the decoder the register was not emptied changes the answer.

    An unterminated backward recursion starts uniform, admitting other paths.
    """
    trellis = recursive_systematic_trellis(FEEDBACK, FEEDFORWARD, MEMORY)
    rng = np.random.default_rng(41)
    message = rng.integers(0, 2, 10).astype(np.uint8)
    systematic, parity = _received(trellis, message, 1.0, rng)

    terminated = bcjr(trellis, systematic, parity, terminated=True)
    free = bcjr(trellis, systematic, parity, terminated=False)

    assert free.log_evidence > terminated.log_evidence
    assert np.abs(free.posterior_llr - terminated.posterior_llr).max() > 1e-3


@pytest.mark.smoke
def test_streams_of_disagreeing_length_are_refused() -> None:
    trellis = recursive_systematic_trellis(FEEDBACK, FEEDFORWARD, MEMORY)

    with pytest.raises(ValueError, match="one-dimensional"):
        bcjr(trellis, np.zeros(6), np.zeros(5))
    with pytest.raises(ValueError, match="one-dimensional"):
        from_trellis(trellis, np.zeros(6), np.zeros(5))


@pytest.mark.smoke
def test_an_impossible_edge_is_a_finite_floor_and_not_minus_infinity() -> None:
    """A hard zero would make the general sum-product return `nan`, not a number.

    An all-`-inf` row becomes `-inf - (-inf)`; rebuilt with `-inf`, it does.
    """
    trellis = recursive_systematic_trellis(FEEDBACK, FEEDFORWARD, MEMORY)
    rng = np.random.default_rng(43)
    message = rng.integers(0, 2, 8).astype(np.uint8)
    systematic, parity = _received(trellis, message, 1.0, rng)
    graph = from_trellis(trellis, systematic, parity)

    assert np.isfinite(
        sum_product(graph, schedule=MessageScheduleName.TREE).log_partition
    )

    hard = FactorGraph(
        graph.variables,
        [
            Factor(
                factor.name,
                factor.variables,
                np.where(
                    factor.log_table <= IMPOSSIBLE_EDGE, -np.inf, factor.log_table
                ),
            )
            for factor in graph.factors
        ],
    )

    assert not np.isfinite(
        sum_product(hard, schedule=MessageScheduleName.TREE).log_partition
    )


# --- end to end: a planted message through the Gaussian channel -----------------

#: The message length the weight spectrum is enumerated at: 1,024 terminated
#: inputs, each run through the register once, is 0.1 s.
END2END_LENGTH = 10

#: The two noise scales. At 0.5 the union bound over the spectrum is 5e-5, so
#: the run is inside what the code guarantees and exact recovery is asserted;
#: at 0.8 the bound is 8.1e-2 and a rate is measurable.
CLEAN_SIGMA, NOISY_SIGMA = 0.5, 0.8

#: Draws at each noise scale. The clean point asserts recovery where the union
#: bound is 4.9e-5, so 500 draws expect 0.02 failures and one would refute it;
#: the noisy point carries the rate, where 2,000 draws give a standard error of
#: 5.3e-3 against a bracket 7.8e-2 wide.
CLEAN_DRAWS, NOISY_DRAWS = 500, 2000

#: The free distance of the `(7, 5)` memory-2 register, which the terminated
#: code's enumerated spectrum must show as its minimum weight.
FREE_DISTANCE = 5


def _q_function(argument: float) -> float:
    """`Q(x)`, the Gaussian tail, through `erfc` rather than a table."""
    return 0.5 * math.erfc(argument / math.sqrt(2.0))


@pytest.mark.critical
@pytest.mark.end2end
def test_the_trellis_returns_the_planted_message_and_fails_inside_the_union_bound() -> (
    None
):
    """At `sigma = 0.5` both decoders return all 500 planted messages exactly;
    at `sigma = 0.8` Viterbi's block error rate over 2,000 draws is 0.0600 and
    BCJR's 0.0630, inside `[2.6e-3, 8.1e-2]` --- the nearest-codeword
    probability and the union bound over the enumerated weight spectrum.

    The path: a seeded message, `terminate` and `encode_stream` through the
    `(7, 5)` register, `sim.ldpc`'s Gaussian channel, `viterbi` and `bcjr`'s
    hard decision, judged on the first `END2END_LENGTH` inputs. The 1,024
    codeword weights are enumerated; below is `Q(sqrt(d_free) / sigma)`,
    above `sum_w A_w Q(sqrt(w) / sigma)`, bracketing ML (Viterbi); BCJR is
    reported beside it. Not slack: 4,000 draws read 0.0510, so the bound is
    1.6x the rate and a decoder losing a factor of two (0.102) fails.
    """
    trellis = recursive_systematic_trellis(FEEDBACK, FEEDFORWARD, MEMORY)
    weights = np.array(
        [
            int(inputs.sum() + encode_stream(trellis, inputs).parity.sum())
            for inputs in (
                terminate(trellis, message)
                for message in enumerate_messages(END2END_LENGTH)
            )
        ]
    )
    spectrum = weights[weights > 0]
    assert weights.size == 2**END2END_LENGTH
    assert int(spectrum.min()) == FREE_DISTANCE

    rates: dict[float, tuple[float, float]] = {}
    for sigma, draws in ((CLEAN_SIGMA, CLEAN_DRAWS), (NOISY_SIGMA, NOISY_DRAWS)):
        rng = np.random.default_rng(729)
        blockwise = bitwise = 0
        for _ in range(draws):
            message = rng.integers(0, 2, size=END2END_LENGTH).astype(np.uint8)
            systematic, parity = _received(trellis, message, sigma, rng)
            path = viterbi(trellis, systematic, parity)
            decisions = (bcjr(trellis, systematic, parity).posterior_llr < 0.0).astype(
                np.uint8
            )
            blockwise += int(not np.array_equal(path[:END2END_LENGTH], message))
            bitwise += int(not np.array_equal(decisions[:END2END_LENGTH], message))
        rates[sigma] = (blockwise / draws, bitwise / draws)

    assert rates[CLEAN_SIGMA] == (0.0, 0.0)

    union = sum(_q_function(math.sqrt(weight) / NOISY_SIGMA) for weight in spectrum)
    nearest = _q_function(math.sqrt(FREE_DISTANCE) / NOISY_SIGMA)
    noisy_blockwise, noisy_bitwise = rates[NOISY_SIGMA]
    assert (noisy_blockwise, noisy_bitwise) == (0.06, 0.063)
    assert nearest < noisy_blockwise < union
    assert nearest < noisy_bitwise < union
