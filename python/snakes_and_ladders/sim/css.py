"""The CSS code a self-orthogonal parity-check matrix defines.

A bicycle matrix satisfies ``H H^T = 0`` over GF(2) (``eq:bicycle-css``),
which is the condition a Calderbank--Shor--Steane code puts on its two
parity-check matrices, so ``Hx = Hz = H`` is a quantum code with
``k = n - 2 rank(H)`` logical qubits (``sec:ldpc:css``). Nothing here
performs inference: this module builds the code, states its dimension, and
gives the quotient every decoding claim about it is scored in.

**The quotient, and why it is the whole point.** The stabilizer group is the
row space of ``H``; the operators that commute with it are ``ker H``; and
``rowspace(H)`` sits inside ``ker H`` precisely because ``H H^T = 0``. A
vector in the row space acts as the identity on the codespace, so two errors
differing by one are the *same* error as far as the encoded state is
concerned. What a decoder must get right is therefore the coset in
``ker(H) / rowspace(H)``, of dimension ``k``, and not the error.

**Two ways to decide a coset, kept apart on purpose.**
:meth:`CssCode.is_stabilizer` runs the rank test ``rank([H; r]) == rank(H)``,
one vector at a time. :meth:`CssCode.logical_label` multiplies by a fixed
``k x n`` matrix, which is what makes an enumeration over ``2 ** n`` errors
affordable. They are built from different computations -- an elimination on
an ``(m + 1) x n`` matrix against a matrix product -- and the suite holds
them to each other over every vector of a small code, because a mis-built
quotient does not break loudly (``search/CLAUDE.md``).

**One sector.** ``Hx = Hz = H`` describes ``X`` errors against ``Z`` checks,
which is what the construction gives without a symplectic representation.
Correlated ``X`` and ``Z`` errors and non-CSS codes are named in
``sec:ldpc:extensions`` and not attempted.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from snakes_and_ladders.fixtures import load_declared
from snakes_and_ladders.sim.ldpc import (
    BinarySymmetricChannel,
    ParityCheck,
    bicycle_code,
    gf2_inverse,
    gf2_rank,
    null_space,
)


@dataclass(frozen=True)
class CssCode:
    """A CSS code ``Hx = Hz = H``, with a basis of its logical operators.

    Built by :meth:`from_parity_check`, which refuses a matrix that is not
    self-orthogonal and a code that encodes nothing.

    Parameters
    ----------
    checks : ParityCheck
        ``H``, serving as both ``Hx`` and ``Hz``.
    stabilizer_rank : int
        ``rank(H)``: the independent stabilizer generators, which is the row
        count only when no row is dependent on the others.
    logical_x : np.ndarray
        ``(k, n)`` ``uint8``. Representatives of ``ker(H) / rowspace(H)``:
        each is in ``ker H``, none is in ``rowspace(H)``, and no sum of them
        is either.
    logical_z : np.ndarray
        ``(k, n)`` ``uint8``, the partner basis: ``logical_z logical_x^T = I``
        over GF(2), which is what makes :meth:`logical_label` read off a
        coset. Its rows are combinations of ``logical_x``'s and so are
        logical operators too; with ``Hx = Hz`` the two bases live in the same
        quotient and the pairing is the only thing separating them.
    """

    checks: ParityCheck
    stabilizer_rank: int
    logical_x: np.ndarray
    logical_z: np.ndarray

    @property
    def n_qubits(self) -> int:
        """The block length ``n``: one qubit per column of ``H``."""
        return self.checks.n_bits

    @property
    def n_logical(self) -> int:
        """``k = n - 2 rank(H)``, the encoded qubits."""
        return self.n_qubits - 2 * self.stabilizer_rank

    @classmethod
    def from_parity_check(cls, checks: ParityCheck) -> CssCode:
        """The code ``Hx = Hz = checks``, with its logical basis built.

        The logical basis is a greedy extension: the rows of ``H`` are
        reduced to a basis of the row space, and a basis of ``ker H`` is then
        walked, each vector kept when it raises the rank. Exactly ``k`` are
        kept, since ``dim ker H = n - rank`` and ``rowspace(H)`` occupies
        ``rank`` of it. The partner basis is ``G^-1 logical_x`` for the Gram
        matrix ``G = logical_x logical_x^T``: the pairing induced on the
        quotient is non-degenerate -- its radical is
        ``ker H ∩ rowspace(H) = rowspace(H)``, which is what the quotient
        divides out -- so ``G`` is invertible, and a singular one is a
        construction error rather than a code without a partner basis.

        Parameters
        ----------
        checks : ParityCheck
            ``H``. Small enough for a dense array: the elimination is dense.

        Raises
        ------
        ValueError
            If ``H H^T != 0``, so the matrix defines no CSS code; or if
            ``k <= 0``, so the code encodes nothing and every decode of it
            succeeds vacuously (``search/CLAUDE.md``: a fixture whose answer
            is trivial measures nothing).
        """
        dense = checks.dense()
        if np.any((dense.astype(np.int64) @ dense.T.astype(np.int64)) & 1):
            msg = (
                "H H^T is nonzero over GF(2), so H defines no CSS code; "
                "eq:bicycle-css is what the bicycle construction supplies"
            )
            raise ValueError(msg)
        rank = gf2_rank(dense)
        n_logical = checks.n_bits - 2 * rank
        if n_logical <= 0:
            msg = (
                f"k = n - 2 rank(H) = {checks.n_bits} - 2 * {rank} = "
                f"{n_logical}: the code encodes no logical qubit, and every "
                f"decode of it succeeds for that reason alone. Delete rows "
                f"to fewer than n / 2 = {checks.n_bits // 2}"
            )
            raise ValueError(msg)
        logical_x = _quotient_representatives(dense, rank, n_logical)
        gram = (logical_x.astype(np.int64) @ logical_x.T.astype(np.int64)) & 1
        logical_z = (
            gf2_inverse(gram.astype(np.uint8)).astype(np.int64)
            @ logical_x.astype(np.int64)
        ) & 1
        return cls(checks, rank, logical_x, np.asarray(logical_z, dtype=np.uint8))

    def is_stabilizer(self, vector: np.ndarray) -> bool:
        """Whether ``vector`` lies in the row space of ``H``.

        The rank test: ``r`` is a combination of the rows exactly when
        appending it raises the rank by nothing. A stabilizer acts as the
        identity on the codespace, so this is the question "does this residual
        matter", answered without reference to any basis of the quotient --
        which is why the suite can hold :meth:`logical_label` to it.

        Parameters
        ----------
        vector : np.ndarray
            ``n`` bits.

        Raises
        ------
        ValueError
            If ``vector`` is not ``n`` bits long.
        """
        bits = _as_bits(vector, self.n_qubits)
        stacked = np.vstack([self.checks.dense(), bits[np.newaxis, :]])
        return gf2_rank(stacked) == self.stabilizer_rank

    def logical_label(self, vectors: np.ndarray) -> np.ndarray:
        """``logical_z v`` over GF(2), per row of ``vectors``.

        For ``v`` in ``ker H`` this labels the coset: it is zero on
        ``rowspace(H)`` -- the rows of ``logical_z`` are in ``ker H``, so
        ``logical_z H^T = 0`` -- and it is ``e_j`` on ``logical_x``'s row
        ``j``, so it is onto ``GF(2)^k`` with ``rowspace(H)`` as its kernel.

        For a ``v`` outside ``ker H`` the value is still what separates the
        cosets of one syndrome, and that is what
        :func:`snakes_and_ladders.likelihood.css.error_cosets` uses it for:
        two errors of the same syndrome differ by an element of ``ker H``, so
        their labels agree exactly when that difference is a stabilizer. The
        label is comparable within a syndrome and meaningless across
        syndromes.

        Parameters
        ----------
        vectors : np.ndarray
            ``(n,)`` or ``(rows, n)`` of 0/1.

        Returns
        -------
        np.ndarray
            ``(k,)`` or ``(rows, k)`` ``uint8``.
        """
        bits = np.asarray(vectors, dtype=np.uint8)
        if bits.shape[-1] != self.n_qubits:
            msg = f"a label needs {self.n_qubits} bits, got {bits.shape[-1]}"
            raise ValueError(msg)
        product = bits.astype(np.int64) @ self.logical_z.T.astype(np.int64)
        return np.asarray(product & 1, dtype=np.uint8)

    def four_cycles(self) -> int:
        """The four-cycles of the Tanner graph of ``H``.

        A four-cycle is two checks and the two bits both cover, so the count
        is ``sum_{a < b} C(|row_a ∩ row_b|, 2)`` over the pairwise overlaps
        the dense product ``H H^T`` reports as integers. This is the structure
        the CSS condition forces: ``H H^T = 0`` over GF(2) says every overlap
        is even, so a pair of rows that meet at all meet at least twice and
        contribute at least one four-cycle. Belief propagation is exact on a
        graph without cycles, so the count is reported beside every decoding
        result here rather than cited.
        """
        dense = self.checks.dense().astype(np.int64)
        overlaps = dense @ dense.T
        pairs = np.triu(overlaps, k=1)
        return int((pairs * (pairs - 1) // 2).sum())


def _as_bits(vector: np.ndarray, n_bits: int) -> np.ndarray:
    """``vector`` as ``n_bits`` of 0/1, refusing anything else."""
    bits = np.asarray(vector, dtype=np.uint8)
    if bits.shape != (n_bits,):
        msg = f"expected {n_bits} bits, got shape {bits.shape}"
        raise ValueError(msg)
    return bits


def _quotient_representatives(
    dense: np.ndarray, rank: int, n_logical: int
) -> np.ndarray:
    """``n_logical`` vectors of ``ker H`` independent modulo ``rowspace(H)``.

    The rows of ``H`` are the starting basis and each kernel vector is kept
    when appending it raises the rank. Greedy suffices: a kernel vector that
    raises the rank is outside the span of everything kept, which is the row
    space together with the representatives already taken.

    Raises
    ------
    RuntimeError
        If fewer than ``n_logical`` are found, which the dimension count
        forbids and so means the elimination has gone wrong.
    """
    kept: list[np.ndarray] = []
    stack = dense
    for candidate in null_space(dense):
        if gf2_rank(np.vstack([stack, candidate[np.newaxis, :]])) > rank + len(kept):
            kept.append(candidate)
            stack = np.vstack([stack, candidate[np.newaxis, :]])
            if len(kept) == n_logical:
                break
    if len(kept) != n_logical:
        msg = (
            f"found {len(kept)} of {n_logical} logical representatives; "
            f"dim ker H - rank H = {dense.shape[1] - 2 * rank} says there are "
            f"{n_logical}"
        )
        raise RuntimeError(msg)
    return np.asarray(np.vstack(kept), dtype=np.uint8)


def sample_x_error(
    code: CssCode, channel: BinarySymmetricChannel, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """An ``X`` error on every qubit independently, and the ratios it gives.

    The CSS sector's error model is one independent bit flip per qubit, which
    is :class:`~snakes_and_ladders.sim.ldpc.BinarySymmetricChannel` and no
    other: the erasure and Gaussian channels of ``sec:ldpc`` describe a
    received symbol, and a qubit is not measured. The draw is the channel's
    own, on the zero word, so the classical and quantum sectors meet the same
    noise from the same code; the channel's ratios take one of two values, so
    the error that produced them is read back from their signs rather than
    drawn a second time.

    Parameters
    ----------
    code : CssCode
    channel : BinarySymmetricChannel
        Its ``flip_probability`` is the per-qubit ``X`` error rate.
    rng : np.random.Generator

    Returns
    -------
    error : np.ndarray
        ``(n,)`` ``uint8``, one where the qubit carries ``X``.
    llr : np.ndarray
        ``(n,)``, the ratios of ``eq:ldpc-llr`` a decoder is given.
    """
    llr = channel.log_likelihood_ratios(np.zeros(code.n_qubits, dtype=np.uint8), rng)
    return (llr < 0.0).astype(np.uint8), llr


_CSS_FIELDS = frozenset(
    {"seed", "n_bits", "n_checks", "circulant_weight", "flip_probability"}
)


@dataclass(frozen=True)
class CssBicycleParams:
    """Fully-specified truth for a CSS code over a bicycle matrix.

    Five fields and no more. The classical fixtures declare three channels
    because a received symbol can be erased or carry a real value; a qubit is
    not measured, so the sector here has one parameter -- the per-qubit ``X``
    rate -- and the file states no erasure or noise scale it would never read.

    Parameters
    ----------
    n_bits : int
        Physical qubits ``n``.
    n_checks : int
        Rows kept after deletion, strictly below ``n / 2``: at ``n / 2`` the
        code encodes ``k = 0`` logical qubits and measures nothing.
    circulant_weight : int
        Ones in the circulant's first row.
    seed : int
        Seed the circulant and the deletion tie-breaks are drawn under.
    flip_probability : float
        The per-qubit ``X`` error rate.
    """

    n_bits: int
    n_checks: int
    circulant_weight: int
    seed: int
    flip_probability: float

    def code(self) -> CssCode:
        """Draw the CSS code this fixture declares.

        Returns
        -------
        CssCode
            :meth:`CssCode.from_parity_check` over
            :func:`~snakes_and_ladders.sim.ldpc.bicycle_code` under this
            fixture's seed, so the instance is the seed's rather than a matrix
            written out, as the classical fixtures are. The ``k > 0`` and
            ``H H^T = 0`` gates are that constructor's, run when the code is
            drawn, so one statement of them serves every caller.
        """
        return CssCode.from_parity_check(
            bicycle_code(
                self.n_bits,
                self.n_checks,
                self.circulant_weight,
                np.random.default_rng(self.seed),
            )
        )

    def channel(self) -> BinarySymmetricChannel:
        """The declared ``X`` error channel."""
        return BinarySymmetricChannel(self.flip_probability)


def load_css_params(path: Path) -> CssBicycleParams:
    """Load and validate a CSS bicycle fixture yaml.

    Parameters
    ----------
    path : Path
        Path to the yaml file.

    Returns
    -------
    CssBicycleParams
        The parsed truth. The shape and weight checks are
        :func:`~snakes_and_ladders.sim.ldpc.bicycle_code`'s and the ``k > 0``
        check :meth:`CssCode.from_parity_check`'s, both run when the code is
        drawn.
    """
    raw = load_declared(path, _CSS_FIELDS)
    return CssBicycleParams(
        n_bits=int(raw["n_bits"]),
        n_checks=int(raw["n_checks"]),
        circulant_weight=int(raw["circulant_weight"]),
        seed=int(raw["seed"]),
        flip_probability=float(raw["flip_probability"]),
    )
