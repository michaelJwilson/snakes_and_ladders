"""The CSS code a bicycle matrix defines, and the quotient it is scored in.

Every claim here is exact over GF(2) and needs no tolerance. The gate comes
first, because everything downstream is vacuous without it: ``H H^T = 0``, so
the matrix defines a code at all; ``k = n - 2 rank(H) > 0``, so the code
encodes something and a decode of it can fail; and a logical basis, so there
is a quotient to score in. The classical fixtures fail the second of those and
are checked to be refused rather than quietly decoded.

The two ways of deciding a coset --- the rank test and the label matrix --- are
held to each other over every one of the 65,536 vectors of the declared
instance. They share no computation, which is what makes the agreement mean
something: a mis-built quotient does not break loudly (``search/CLAUDE.md``).
"""

from __future__ import annotations

import numpy as np
import pytest
from snakes_and_ladders.sim.css import CssCode, sample_x_error
from snakes_and_ladders.sim.fixtures import fixture
from snakes_and_ladders.sim.ldpc import (
    BinarySymmetricChannel,
    bicycle_code,
    gallager_code,
    gf2_rank,
)

#: The instance each tier declares, and what the construction gives on it: the
#: logical qubits and the four-cycles of the Tanner graph. Both are exact
#: counts read off the matrix, so they are pinned rather than bounded.
DECLARED = {"ci": (16, 7, 2, 30), "stress": (96, 40, 16, 160)}


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


def _all_vectors(n_bits: int) -> np.ndarray:
    """Every word of ``n_bits`` bits, as rows."""
    return ((np.arange(2**n_bits)[:, None] >> np.arange(n_bits)) & 1).astype(np.uint8)


# --- the gate: the condition, the dimension, the basis ---------------------------


@pytest.mark.critical
@pytest.mark.mathematical
@pytest.mark.parametrize("tier", sorted(DECLARED))
def test_the_declared_instance_satisfies_the_css_condition(tier: str) -> None:
    """`H H^T = 0` over GF(2), which is what makes `Hx = Hz = H` a code."""
    dense = _code(tier).checks.dense().astype(np.int64)

    product = (dense @ dense.T) & 1

    np.testing.assert_array_equal(product, np.zeros_like(product))


@pytest.mark.critical
@pytest.mark.mathematical
@pytest.mark.parametrize("tier", sorted(DECLARED))
def test_the_declared_instance_encodes_the_logical_qubits_it_claims(tier: str) -> None:
    """`k = n - 2 rank(H)`, positive and equal to the count the fixture is for."""
    n_bits, n_checks, n_logical, _ = DECLARED[tier]
    code = _code(tier)

    rank = gf2_rank(code.checks.dense())

    assert (code.n_qubits, code.checks.n_checks) == (n_bits, n_checks)
    assert code.stabilizer_rank == rank == n_checks
    assert code.n_logical == n_bits - 2 * rank == n_logical > 0


@pytest.mark.mathematical
@pytest.mark.parametrize("n_bits", [12, 96])
def test_the_classical_fixture_lengths_encode_nothing_at_half_the_rows(
    n_bits: int,
) -> None:
    """At `m = n / 2` the rank is `n / 2`, so `k = 0` and the code is refused.

    This is the measurement that forced a second set of fixtures: the bicycle
    fixtures of ``sec:ldpc:bicycle`` keep every row of the circulant, and a
    decode of the code they define succeeds because there is nothing to fail
    at rather than because the decoder worked.
    """
    checks = bicycle_code(n_bits, n_bits // 2, 3, np.random.default_rng(361))

    assert gf2_rank(checks.dense()) == n_bits // 2

    with pytest.raises(ValueError, match="encodes no logical qubit"):
        CssCode.from_parity_check(checks)


@pytest.mark.edge_case
def test_a_matrix_that_is_not_self_orthogonal_is_refused() -> None:
    """A Gallager draw defines no CSS code, and says so rather than decoding."""
    checks = gallager_code(12, 3, 6, np.random.default_rng(4))

    with pytest.raises(ValueError, match="H H\\^T is nonzero"):
        CssCode.from_parity_check(checks)


@pytest.mark.critical
@pytest.mark.mathematical
@pytest.mark.parametrize("tier", sorted(DECLARED))
def test_the_logical_basis_lies_in_the_kernel_and_outside_the_row_space(
    tier: str,
) -> None:
    """`k` representatives of `ker(H) / rowspace(H)`, independent modulo it.

    Independence is the rank of the rows of `H` and the basis together: it is
    `rank + k` exactly when no combination of the representatives, and no
    combination with a stabilizer, is a stabilizer. Checking the rows one at a
    time would pass on a basis whose *sum* is a stabilizer.
    """
    code = _code(tier)
    dense = code.checks.dense()

    for basis in (code.logical_x, code.logical_z):
        assert basis.shape == (code.n_logical, code.n_qubits)
        syndromes = (basis.astype(np.int64) @ dense.T) & 1
        np.testing.assert_array_equal(syndromes, np.zeros_like(syndromes))
        assert not any(code.is_stabilizer(row) for row in basis)
        assert gf2_rank(np.vstack([dense, basis])) == code.stabilizer_rank + (
            code.n_logical
        )


@pytest.mark.critical
@pytest.mark.mathematical
@pytest.mark.parametrize("tier", sorted(DECLARED))
def test_the_two_logical_bases_pair_to_the_identity(tier: str) -> None:
    """`logical_z logical_x^T = I`, which is what makes a label read a coset."""
    code = _code(tier)

    pairing = (code.logical_z.astype(np.int64) @ code.logical_x.T.astype(np.int64)) & 1

    np.testing.assert_array_equal(pairing, np.eye(code.n_logical, dtype=np.int64))


# --- the quotient, two ways ------------------------------------------------------


@pytest.mark.mathematical
def test_the_label_and_the_rank_test_agree_on_every_kernel_word() -> None:
    """A stabilizer is a kernel word of zero label, over all 512 kernel words.

    The two share no computation --- an elimination on an `(m + 1) x n` matrix
    against a product with a `k x n` matrix --- so this is the check that the
    label map's kernel really is the row space, and not a rederivation of it.
    The kernel is where the distinction is subtle and it is walked in full;
    outside it every word fails both tests, which 1,000 seeded draws cover
    rather than the remaining 65,024, the rank test costing an elimination
    each.
    """
    code = _code("ci")
    words = _all_vectors(code.n_qubits)
    in_kernel = ~((words.astype(np.int64) @ code.checks.dense().T) & 1).any(axis=1)
    kernel = words[in_kernel]
    assert kernel.shape[0] == 2 ** (code.stabilizer_rank + code.n_logical)

    by_label = ~code.logical_label(kernel).any(axis=1)

    by_rank = np.array([code.is_stabilizer(word) for word in kernel])
    np.testing.assert_array_equal(by_label, by_rank)
    assert int(by_rank.sum()) == 2**code.stabilizer_rank
    outside = words[~in_kernel][
        np.random.default_rng(361).choice(int((~in_kernel).sum()), 1000, replace=False)
    ]
    assert not any(code.is_stabilizer(word) for word in outside)


@pytest.mark.mathematical
def test_every_coset_of_the_quotient_is_reached_exactly_once_per_label() -> None:
    """The `2 ** k` labels partition `ker H` into cosets of equal size.

    `dim ker H = rank + k`, so each of the `2 ** k` labels carries
    `2 ** rank` kernel words: one coset of the row space per label, which is
    what "the quotient has dimension `k`" means as a count.
    """
    code = _code("ci")
    words = _all_vectors(code.n_qubits)
    kernel = words[~((words.astype(np.int64) @ code.checks.dense().T) & 1).any(axis=1)]

    labels = code.logical_label(kernel).astype(np.int64) @ (
        1 << np.arange(code.n_logical)
    )

    counts = np.bincount(labels, minlength=2**code.n_logical)
    assert counts.size == 2**code.n_logical
    np.testing.assert_array_equal(counts, np.full(counts.size, 2**code.stabilizer_rank))


@pytest.mark.edge_case
def test_a_word_of_the_wrong_length_is_refused_by_both_routes() -> None:
    """A residual that is not `n` bits is a caller's error, not a failed decode."""
    code = _code("ci")

    with pytest.raises(ValueError, match="expected 16 bits"):
        code.is_stabilizer(np.zeros(15, dtype=np.uint8))
    with pytest.raises(ValueError, match="a label needs 16 bits"):
        code.logical_label(np.zeros(15, dtype=np.uint8))


# --- what the CSS condition costs the graph --------------------------------------


@pytest.mark.mathematical
@pytest.mark.parametrize("tier", sorted(DECLARED))
def test_self_orthogonality_makes_every_row_overlap_even(tier: str) -> None:
    """`H H^T = 0` over GF(2) says each pair of checks meets an even number of bits.

    That is the structural claim under the four-cycle count: a pair that meets
    at all meets at least twice, and two checks sharing two bits *are* a
    four-cycle. The count is pinned beside it, exactly.
    """
    _, _, _, four_cycles = DECLARED[tier]
    code = _code(tier)
    dense = code.checks.dense().astype(np.int64)

    overlaps = np.triu(dense @ dense.T, k=1)

    assert not (overlaps % 2).any()
    assert overlaps.max() >= 2
    assert code.four_cycles() == four_cycles


@pytest.mark.mathematical
def test_the_four_cycle_count_is_the_pairs_of_bits_two_checks_share() -> None:
    """Counted a second way: over pairs of bits, the checks covering both.

    A four-cycle is two checks and two bits, so counting it from the bits must
    give what counting it from the checks gave. The transpose shares no code
    path with :meth:`CssCode.four_cycles`, which reduces over the checks.
    """
    code = _code("ci")
    dense = code.checks.dense().astype(np.int64)

    shared = np.triu(dense.T @ dense, k=1)

    assert int((shared * (shared - 1) // 2).sum()) == code.four_cycles()


# --- the error the sector draws ---------------------------------------------------


@pytest.mark.simulated_truth
def test_the_drawn_error_has_the_rate_the_channel_declares() -> None:
    """The flip rate over 400 draws recovers `p`, and the ratios match the error.

    The error is read back from the ratios rather than drawn a second time, so
    what is checked is that the two agree per qubit and that the draw is the
    channel's: the count of flips over 6,400 qubits at `p = 0.05` is
    `Binomial(6400, 0.05)`, whose four-sigma band is 253 to 387.
    """
    code = _code("ci")
    channel = BinarySymmetricChannel(0.05)
    rng = np.random.default_rng(361)
    magnitude = np.log(0.95 / 0.05)

    errors = []
    for _ in range(400):
        error, llr = sample_x_error(code, channel, rng)
        np.testing.assert_allclose(np.abs(llr), magnitude)
        np.testing.assert_array_equal(error, (llr < 0.0).astype(np.uint8))
        errors.append(error)

    flips = int(np.stack(errors).sum())
    assert 253 <= flips <= 387
