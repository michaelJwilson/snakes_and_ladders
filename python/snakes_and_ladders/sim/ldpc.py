"""Low-density parity-check codes: the ensemble, the channels, and an encoder.

A binary linear code is the null space of a parity-check matrix ``H`` over
GF(2); a codeword ``c`` satisfies ``H c = 0``. Gallager's (1962) regular
ensemble makes ``H`` sparse -- every column carries ``column_weight`` ones
and every row ``row_weight`` -- and the decoding problem is then inference
on the factor graph with one parity factor per row (``sec:ldpc`` in the
textbook, ``eq:ldpc-code``). The ticket's instance is the ``10,000 x
20,000`` (3,6) code: 60,000 nonzeros, held here as offsets into one edge
array in both orientations, the layout root ``CLAUDE.md`` asks for. No
dense ``H`` exists at that size; :meth:`ParityCheck.dense` is for the small
codes an oracle can reach.

**Channels.** Every channel returns log-likelihood ratios in one
convention, ``eq:ldpc-llr``: ``L_i = log p(y_i | x_i = 0) - log p(y_i | x_i
= 1)``, positive when the received symbol favours zero. The binary erasure
channel's certain bits carry ``+-LLR_CAP`` rather than an infinity, because
the decoder forms leave-one-out sums by subtraction, and an infinity minus
itself is not a number.

**The all-zero codeword.** A binary-input channel whose outputs are
symmetric under negating the input, and a decoder whose message maps
commute with that negation, give an error probability that does not depend
on the codeword sent; :func:`all_zero_transmission` states the argument and
the suite checks it on a real codeword. At ``n <= MAX_ENCODABLE_BITS`` a
real encoder exists by Gaussian elimination, so ``H c = 0`` is asserted on
non-trivial codewords too.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from snakes_and_ladders.fixtures import load_declared

#: The magnitude that stands for certainty. ``tanh(LLR_CAP / 2)`` is below
#: one in ``float64`` by ``1.9e-13``, so the decoder's ``tanh`` rule never
#: sees a saturated argument, and a posterior formed under the cap differs
#: from the uncapped one by less than ``exp(-LLR_CAP)``.
LLR_CAP = 30.0

#: Past this the dense Gaussian elimination the encoder runs is no longer
#: the cheap part: it is cubic, and a linear-time encoder is its own ticket.
MAX_ENCODABLE_BITS = 512


@dataclass(frozen=True)
class ParityCheck:
    """A sparse parity-check matrix as offsets into one edge array.

    An edge is one nonzero of ``H``, a ``(variable, check)`` pair. The edges
    are stored sorted by variable, so ``edge_check[variable_offsets[i]:
    variable_offsets[i + 1]]`` are the checks of bit ``i``; ``check_order``
    is the permutation that sorts the same edges by check, so
    ``edge_variable[check_order][check_offsets[j]:check_offsets[j + 1]]``
    are the bits of check ``j``. A per-edge array in variable order is read
    in check order through the one permutation, which is what lets the
    decoder keep one message per edge and no second copy.

    Parameters
    ----------
    n_bits, n_checks : int
        The shape of ``H``: ``n_checks`` rows over ``n_bits`` columns.
    edge_variable, edge_check : np.ndarray
        The nonzeros, sorted by variable then by check.
    variable_offsets, check_offsets : np.ndarray
        Segment boundaries, ``n_bits + 1`` and ``n_checks + 1`` long.
    check_order : np.ndarray
        The permutation of the edges into check order.

    Raises
    ------
    ValueError
        If an index is out of range, an edge repeats, or a bit sits in no
        check or a check holds no bit -- a segmented reduction over an
        empty segment returns its neighbour's value, so the layout refuses
        the case rather than computing on it.
    """

    n_bits: int
    n_checks: int
    edge_variable: np.ndarray
    edge_check: np.ndarray
    variable_offsets: np.ndarray
    check_offsets: np.ndarray
    check_order: np.ndarray

    @classmethod
    def from_edges(
        cls, n_bits: int, n_checks: int, variables: np.ndarray, checks: np.ndarray
    ) -> ParityCheck:
        """Build from the nonzeros ``(variables[e], checks[e])`` in any order."""
        variables = np.asarray(variables, dtype=np.int64)
        checks = np.asarray(checks, dtype=np.int64)
        if variables.shape != checks.shape or variables.ndim != 1:
            msg = "variables and checks must be one-dimensional and the same length"
            raise ValueError(msg)
        if variables.size and (
            variables.min() < 0
            or variables.max() >= n_bits
            or checks.min() < 0
            or checks.max() >= n_checks
        ):
            msg = f"an edge lies outside the {n_checks} x {n_bits} matrix"
            raise ValueError(msg)
        order = np.lexsort((checks, variables))
        edge_variable, edge_check = variables[order], checks[order]
        keys = edge_variable * n_checks + edge_check
        if np.any(keys[1:] == keys[:-1]):
            msg = "an edge (variable, check) is listed twice"
            raise ValueError(msg)
        variable_offsets = np.searchsorted(edge_variable, np.arange(n_bits + 1))
        check_order = np.argsort(edge_check, kind="stable")
        check_offsets = np.searchsorted(
            edge_check[check_order], np.arange(n_checks + 1)
        )
        if np.any(np.diff(variable_offsets) == 0):
            msg = "every bit must sit in at least one check"
            raise ValueError(msg)
        if np.any(np.diff(check_offsets) == 0):
            msg = "every check must hold at least one bit"
            raise ValueError(msg)
        return cls(
            n_bits,
            n_checks,
            edge_variable,
            edge_check,
            variable_offsets,
            check_offsets,
            check_order,
        )

    @classmethod
    def from_dense(cls, matrix: np.ndarray) -> ParityCheck:
        """Build from a dense ``(n_checks, n_bits)`` 0/1 array."""
        dense = np.asarray(matrix)
        checks, variables = np.nonzero(dense)
        return cls.from_edges(dense.shape[1], dense.shape[0], variables, checks)

    @property
    def n_edges(self) -> int:
        """The nonzeros of ``H``."""
        return int(self.edge_variable.size)

    @property
    def column_weights(self) -> np.ndarray:
        """Ones per column: the degree of each bit."""
        return np.diff(self.variable_offsets)

    @property
    def row_weights(self) -> np.ndarray:
        """Ones per row: the degree of each check."""
        return np.diff(self.check_offsets)

    def dense(self) -> np.ndarray:
        """``H`` as a ``(n_checks, n_bits)`` ``uint8`` array, for small codes."""
        matrix = np.zeros((self.n_checks, self.n_bits), dtype=np.uint8)
        matrix[self.edge_check, self.edge_variable] = 1
        return matrix

    def syndrome(self, bits: np.ndarray) -> np.ndarray:
        """``H c`` over GF(2): zero on every check exactly when ``bits`` is a codeword."""
        values = np.asarray(bits, dtype=np.uint8)[self.edge_variable[self.check_order]]
        return np.asarray(np.bitwise_xor.reduceat(values, self.check_offsets[:-1]))


def gallager_code(
    n_bits: int, column_weight: int, row_weight: int, rng: np.random.Generator
) -> ParityCheck:
    """One member of Gallager's regular ``(column_weight, row_weight)`` ensemble.

    The construction is Gallager's (1962, §2.2): ``H`` is ``column_weight``
    bands of ``n_bits / row_weight`` rows each; the first band's row ``i``
    covers bits ``i * row_weight`` to ``(i + 1) * row_weight - 1``, and every
    other band is the first under a uniformly random permutation of the
    columns drawn from ``rng``. Each bit sits once in every band, so the
    column weight is exact, and each row of a band is a block of the
    permutation, so the row weight is; both are asserted after the draw. The
    rows of one band sum to the all-ones vector, so ``H`` has at least
    ``column_weight - 1`` dependent rows and the code's dimension is at
    least ``n_bits - n_checks + column_weight - 1``. Short cycles are not
    avoided: that is a different construction, out of scope here.

    Parameters
    ----------
    n_bits : int
        The block length ``n``; a multiple of ``row_weight``.
    column_weight, row_weight : int
        Ones per column and per row, ``2 <= column_weight < row_weight`` so
        the design rate ``1 - column_weight / row_weight`` is positive.
    rng : np.random.Generator
        The permutations are drawn from it.

    Raises
    ------
    ValueError
        If the degrees do not divide the length or are out of order.
    """
    if not 2 <= column_weight < row_weight:
        msg = (
            f"need 2 <= column_weight < row_weight, got ({column_weight}, {row_weight})"
        )
        raise ValueError(msg)
    if n_bits <= 0 or n_bits % row_weight:
        msg = (
            f"n_bits = {n_bits} is not a positive multiple of row_weight = {row_weight}"
        )
        raise ValueError(msg)
    rows_per_band = n_bits // row_weight
    n_checks = rows_per_band * column_weight
    identity = np.arange(n_bits)
    variables = np.concatenate(
        [identity] + [rng.permutation(n_bits) for _ in range(column_weight - 1)]
    )
    checks = np.repeat(np.arange(n_checks), row_weight)
    code = ParityCheck.from_edges(n_bits, n_checks, variables, checks)
    if not (
        np.all(code.column_weights == column_weight)
        and np.all(code.row_weights == row_weight)
    ):
        msg = "the drawn matrix does not have the requested degrees"
        raise ValueError(msg)
    return code


# --- channels -------------------------------------------------------------------


class Channel(Protocol):
    """A memoryless binary-input channel, seen only through its LLRs."""

    def log_likelihood_ratios(
        self, codeword: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """Transmit ``codeword`` and return ``eq:ldpc-llr`` per bit."""


def _bits(codeword: np.ndarray) -> np.ndarray:
    bits = np.asarray(codeword, dtype=np.uint8)
    if bits.ndim != 1 or np.any(bits > 1):
        msg = "a codeword is a one-dimensional 0/1 array"
        raise ValueError(msg)
    return bits


@dataclass(frozen=True)
class BinarySymmetricChannel:
    """Each bit is flipped independently with probability ``flip_probability``.

    ``L_i = (1 - 2 y_i) log((1 - p) / p)``: the received bit, at the one
    magnitude the channel allows.
    """

    flip_probability: float

    def __post_init__(self) -> None:
        if not 0.0 < self.flip_probability < 0.5:
            msg = f"flip_probability must be in (0, 0.5), got {self.flip_probability}"
            raise ValueError(msg)

    def log_likelihood_ratios(
        self, codeword: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        bits = _bits(codeword)
        received = bits ^ (rng.random(bits.size) < self.flip_probability)
        magnitude = np.log((1.0 - self.flip_probability) / self.flip_probability)
        return np.asarray((1.0 - 2.0 * received) * magnitude)


@dataclass(frozen=True)
class BinaryErasureChannel:
    """Each bit is erased independently with probability ``erasure_probability``.

    An erased bit says nothing, ``L_i = 0``; a delivered bit is certain, and
    carries ``+-LLR_CAP`` for the reason the module docstring gives.
    """

    erasure_probability: float

    def __post_init__(self) -> None:
        if not 0.0 < self.erasure_probability < 1.0:
            msg = (
                f"erasure_probability must be in (0, 1), got {self.erasure_probability}"
            )
            raise ValueError(msg)

    def log_likelihood_ratios(
        self, codeword: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        bits = _bits(codeword)
        erased = rng.random(bits.size) < self.erasure_probability
        return np.where(erased, 0.0, (1.0 - 2.0 * bits) * LLR_CAP)


@dataclass(frozen=True)
class BinaryInputGaussianChannel:
    """Antipodal signalling ``x = 1 - 2c`` plus ``N(0, sigma^2)`` noise.

    ``L_i = 2 y_i / sigma^2``, the log-ratio of two Gaussian densities with
    means ``+-1``.
    """

    sigma: float

    def __post_init__(self) -> None:
        if not self.sigma > 0.0:
            msg = f"sigma must be positive, got {self.sigma}"
            raise ValueError(msg)

    def log_likelihood_ratios(
        self, codeword: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        bits = _bits(codeword)
        received = (1.0 - 2.0 * bits) + self.sigma * rng.standard_normal(bits.size)
        return 2.0 * received / self.sigma**2


def all_zero_transmission(
    code: ParityCheck, channel: Channel, rng: np.random.Generator
) -> np.ndarray:
    """The LLRs of the all-zero codeword sent through ``channel``.

    The zero word is a codeword of every linear code, and for the channels
    here it is the only one a simulation needs. Each channel is *output
    symmetric*: ``p(y | x = 1) = p(-y | x = 0)`` in the LLR domain, so
    sending ``c`` instead of ``0`` negates ``L_i`` exactly at the bits where
    ``c_i = 1``. The decoder's two check updates are odd in each argument
    and its variable update is linear, so the same negation propagates
    through every message, and the hard decision at bit ``i`` flips exactly
    when ``c_i = 1``: the decoded error pattern ``c XOR c_hat`` under ``c``
    equals the decoded word under ``0`` on the same noise. Bit and block
    error rates measured on the zero word are therefore the rates on any
    codeword (Richardson and Urbanke 2008, the conditional-independence
    lemma of §4.1; Gallager 1962, §4). The suite checks the identity per
    realization on a codeword from :func:`encode`.

    Parameters
    ----------
    code : ParityCheck
        Only its length is used.
    channel : Channel
    rng : np.random.Generator
    """
    return channel.log_likelihood_ratios(np.zeros(code.n_bits, dtype=np.uint8), rng)


# --- encoding: GF(2) elimination at small sizes --------------------------------


def _reduced_row_echelon(matrix: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Gauss--Jordan over GF(2): the reduced matrix and its pivot columns."""
    reduced = np.array(matrix, dtype=np.uint8)
    pivots: list[int] = []
    row = 0
    for column in range(reduced.shape[1]):
        if row == reduced.shape[0]:
            break
        candidates = np.nonzero(reduced[row:, column])[0]
        if candidates.size == 0:
            continue
        pivot = row + int(candidates[0])
        if pivot != row:
            reduced[[row, pivot]] = reduced[[pivot, row]]
        others = np.nonzero(reduced[:, column])[0]
        others = others[others != row]
        reduced[others] ^= reduced[row]
        pivots.append(column)
        row += 1
    return reduced, pivots


def generator_matrix(code: ParityCheck) -> np.ndarray:
    """A ``(k, n)`` basis of the code, systematic in the non-pivot columns.

    Row ``r`` is the codeword with a one at the ``r``-th free column, zeros
    at the other free columns, and the pivot bits that ``H c = 0`` forces.
    ``k = n - rank H`` is the code's dimension, read off the elimination.

    Raises
    ------
    ValueError
        Past ``MAX_ENCODABLE_BITS``, where the dense elimination is refused.
    """
    if code.n_bits > MAX_ENCODABLE_BITS:
        msg = (
            f"the dense GF(2) elimination is refused at n = {code.n_bits} > "
            f"{MAX_ENCODABLE_BITS}; a linear-time encoder is issue #340's later work"
        )
        raise ValueError(msg)
    reduced, pivots = _reduced_row_echelon(code.dense())
    free = [column for column in range(code.n_bits) if column not in set(pivots)]
    generator = np.zeros((len(free), code.n_bits), dtype=np.uint8)
    for r, column in enumerate(free):
        generator[r, column] = 1
        generator[r, pivots] = reduced[: len(pivots), column]
    return generator


def encode(code: ParityCheck, message: np.ndarray) -> np.ndarray:
    """The codeword carrying ``message`` in the code's free positions.

    ``c = u G`` over GF(2); ``H c = 0`` is asserted on the result, so a
    wrong elimination raises here rather than passing a corrupt word to a
    channel.

    Parameters
    ----------
    code : ParityCheck
        At most ``MAX_ENCODABLE_BITS`` long.
    message : np.ndarray
        ``k`` bits, ``k`` the dimension :func:`generator_matrix` reports.

    Raises
    ------
    ValueError
        If ``message`` is not ``k`` bits long, or the code is too long.
    RuntimeError
        If the elimination produced a word with a nonzero syndrome.
    """
    generator = generator_matrix(code)
    bits = _bits(message)
    if bits.size != generator.shape[0]:
        msg = f"the code carries {generator.shape[0]} message bits, got {bits.size}"
        raise ValueError(msg)
    codeword = np.asarray((bits.astype(np.int64) @ generator) & 1, dtype=np.uint8)
    if np.any(code.syndrome(codeword)):
        msg = "the encoder produced a word outside the code"
        raise RuntimeError(msg)
    return codeword


_REQUIRED_FIELDS = frozenset(
    {
        "seed",
        "n_bits",
        "column_weight",
        "row_weight",
        "flip_probability",
        "erasure_probability",
        "noise_scale",
    }
)


@dataclass(frozen=True)
class LdpcParams:
    """Fully-specified truth for a Gallager-ensemble fixture.

    The ensemble member is the seed's, not the file's: an ensemble is
    reproduced by drawing it again rather than by writing ``H`` out, which at
    the sizes the decoder is run on would be a file no reader could check.

    Parameters
    ----------
    n_bits : int
        Block length ``n``.
    column_weight, row_weight : int
        The regular degrees ``(j, k)`` of the ensemble.
    seed : int
        Seed the ensemble member is drawn under.
    flip_probability : float
        Crossover of the binary symmetric channel this instance is decoded on.
    erasure_probability : float
        Erasure rate of the binary erasure channel.
    noise_scale : float
        Standard deviation of the binary-input Gaussian channel.
    """

    n_bits: int
    column_weight: int
    row_weight: int
    seed: int
    flip_probability: float
    erasure_probability: float
    noise_scale: float

    def code(self) -> ParityCheck:
        """Draw the ensemble member this fixture declares.

        Returns
        -------
        ParityCheck
            :func:`gallager_code` under this fixture's seed.
        """
        return gallager_code(
            self.n_bits,
            self.column_weight,
            self.row_weight,
            np.random.default_rng(self.seed),
        )

    def symmetric_channel(self) -> BinarySymmetricChannel:
        """The declared binary symmetric channel."""
        return BinarySymmetricChannel(self.flip_probability)

    def erasure_channel(self) -> BinaryErasureChannel:
        """The declared binary erasure channel."""
        return BinaryErasureChannel(self.erasure_probability)

    def gaussian_channel(self) -> BinaryInputGaussianChannel:
        """The declared binary-input Gaussian channel."""
        return BinaryInputGaussianChannel(self.noise_scale)


def load_ldpc_params(path: Path) -> LdpcParams:
    """Load and validate an LDPC fixture yaml.

    Parameters
    ----------
    path : Path
        Path to the yaml file.

    Returns
    -------
    LdpcParams
        The parsed truth. The degree checks are :func:`gallager_code`'s, run
        when the code is drawn, so one statement of them serves both callers.
    """
    raw = load_declared(path, _REQUIRED_FIELDS)
    return LdpcParams(
        n_bits=int(raw["n_bits"]),
        column_weight=int(raw["column_weight"]),
        row_weight=int(raw["row_weight"]),
        seed=int(raw["seed"]),
        flip_probability=float(raw["flip_probability"]),
        erasure_probability=float(raw["erasure_probability"]),
        noise_scale=float(raw["noise_scale"]),
    )
