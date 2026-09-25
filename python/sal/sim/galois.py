"""``GF(2^m)`` arithmetic: the field Reed--Solomon needs and nothing here had.

Issue #594. Every code in this repository until now lives over ``GF(2)``,
where addition is exclusive-or and there is nothing else to say. Reed--Solomon
is defined over an extension field, and its decoder --- syndromes, the
Berlekamp--Massey recursion, the Chien search, Forney's formula --- is
arithmetic in that field throughout. This module is that arithmetic, and it is
the real cost of the Reed--Solomon ticket: a later BCH code reuses it
unchanged.

**Elements are integers and the representation is a choice, stated once.** An
element of ``GF(2^m)`` is a polynomial of degree below ``m`` over ``GF(2)``,
and the integer whose bits are its coefficients names it. Addition is then
exclusive-or --- there is no carry in characteristic two --- and multiplication
is polynomial multiplication reduced by a **primitive** polynomial, chosen so
that ``x`` generates every nonzero element and a discrete logarithm exists.

**Two tables, and everything else reads them.** ``alpha^i`` for every ``i``
and its inverse map turn multiplication into an addition of logarithms, which
is what makes a decoder over this field affordable. They are built once per
field and the field is cached, because a decoder constructs the same field on
every call and rebuilding a 255-entry table per call is the recompute the
root ``CLAUDE.md`` rule names.

**Zero has no logarithm, and this refuses rather than returns a sentinel.**
The usual trick is to store ``log(0) = -1`` and let it propagate; that turns a
caller's mistake into a plausible wrong element deep inside a decode, which is
exactly the silent failure `likelihood/CLAUDE.md` forbids.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import numpy as np

#: Primitive polynomials, as the integer whose bits are the coefficients
#: including the leading ``x^m``. These are the conventional choices --- the
#: sparsest primitive polynomial of each degree --- so the tables here agree
#: with any reference implementation a reader compares against.
PRIMITIVE: dict[int, int] = {
    2: 0b111,
    3: 0b1011,
    4: 0b10011,
    5: 0b100101,
    6: 0b1000011,
    7: 0b10000011,
    8: 0b100011101,
}


@dataclass(frozen=True)
class BinaryField:
    """``GF(2^m)``, as its exponential and logarithm tables.

    Parameters
    ----------
    m : int
        The extension degree, so the field has ``2 ** m`` elements.
    exponential : np.ndarray
        ``alpha ** i`` for ``i`` in ``[0, 2 ** m - 1)``, twice over, so a
        product's exponent needs no modulus at the read.
    logarithm : np.ndarray
        The inverse map on the nonzero elements; entry ``0`` is unused and set
        to ``-1`` so a wrong read is a loud index rather than a plausible one.
    """

    m: int
    exponential: np.ndarray
    logarithm: np.ndarray

    @property
    def order(self) -> int:
        """The number of elements, ``2 ** m``."""
        return 1 << self.m

    @property
    def nonzero(self) -> int:
        """The multiplicative group's order, ``2 ** m - 1``."""
        return self.order - 1

    def multiply(self, left: int, right: int) -> int:
        """``left * right``, through the logarithm tables."""
        if left == 0 or right == 0:
            return 0
        return int(
            self.exponential[int(self.logarithm[left]) + int(self.logarithm[right])]
        )

    def inverse(self, value: int) -> int:
        """``value ** -1``.

        Raises
        ------
        ZeroDivisionError
            For ``0``, which has no inverse. Raised rather than returning
            ``0``, because a decoder that divides by a zero magnitude has
            found a contradiction and should say so.
        """
        if value == 0:
            msg = "zero has no multiplicative inverse in GF(2^m)"
            raise ZeroDivisionError(msg)
        return int(self.exponential[self.nonzero - int(self.logarithm[value])])

    def divide(self, left: int, right: int) -> int:
        """``left / right``."""
        if left == 0:
            return 0
        return self.multiply(left, self.inverse(right))

    def power(self, value: int, exponent: int) -> int:
        """``value ** exponent`` for any integer exponent, including negative."""
        if value == 0:
            if exponent == 0:
                return 1
            return 0
        return int(
            self.exponential[(int(self.logarithm[value]) * exponent) % self.nonzero]
        )

    def alpha(self, exponent: int) -> int:
        """``alpha ** exponent``, the generator raised to a power."""
        return int(self.exponential[exponent % self.nonzero])


@cache
def field(m: int) -> BinaryField:
    """``GF(2 ** m)``, built once and cached.

    Parameters
    ----------
    m : int
        Extension degree, between 2 and 8 --- the range
        :data:`PRIMITIVE` carries, which covers every Reed--Solomon length
        this repository has a use for.

    Returns
    -------
    BinaryField
        The field and its tables.

    Raises
    ------
    KeyError
        If no primitive polynomial is recorded for ``m``. Refused rather than
        searched for one: a silently chosen polynomial changes every element's
        name and would make two runs of the same code disagree.
    """
    if m not in PRIMITIVE:
        msg = (
            f"no primitive polynomial recorded for GF(2^{m}); have {sorted(PRIMITIVE)}"
        )
        raise KeyError(msg)
    polynomial = PRIMITIVE[m]
    size = 1 << m
    exponential = np.zeros(2 * size, dtype=np.int64)
    logarithm = np.full(size, -1, dtype=np.int64)
    value = 1
    for power in range(size - 1):
        exponential[power] = value
        logarithm[value] = power
        value <<= 1
        if value & size:
            value ^= polynomial
    # The second copy, so `exponential[log a + log b]` never needs a modulus.
    exponential[size - 1 : 2 * (size - 1)] = exponential[: size - 1]
    return BinaryField(m=m, exponential=exponential, logarithm=logarithm)


def polynomial_multiply(
    left: np.ndarray, right: np.ndarray, gf: BinaryField
) -> np.ndarray:
    """Multiply two polynomials over the field, coefficients low degree first."""
    product = np.zeros(len(left) + len(right) - 1, dtype=np.int64)
    for index, coefficient in enumerate(left):
        if coefficient == 0:
            continue
        for offset, other in enumerate(right):
            if other == 0:
                continue
            product[index + offset] ^= gf.multiply(int(coefficient), int(other))
    return product


def polynomial_evaluate(coefficients: np.ndarray, point: int, gf: BinaryField) -> int:
    """Horner's rule over the field, coefficients low degree first."""
    total = 0
    for coefficient in reversed(coefficients):
        total = gf.multiply(total, point) ^ int(coefficient)
    return int(total)


def polynomial_remainder(
    dividend: np.ndarray, divisor: np.ndarray, gf: BinaryField
) -> np.ndarray:
    """``dividend mod divisor`` over the field, coefficients low degree first."""
    remainder = np.asarray(dividend, dtype=np.int64).copy()
    divisor = np.asarray(divisor, dtype=np.int64)
    lead = int(divisor[-1])
    degree = len(divisor) - 1
    for position in range(len(remainder) - 1, degree - 1, -1):
        factor = int(remainder[position])
        if factor == 0:
            continue
        scale = gf.divide(factor, lead)
        for offset in range(degree + 1):
            remainder[position - degree + offset] ^= gf.multiply(
                scale, int(divisor[offset])
            )
    return remainder[:degree]


def bits_from_symbols(symbols: np.ndarray, m: int) -> np.ndarray:
    """Each symbol as ``m`` bits, most significant first, concatenated.

    The route from a symbol code to a *binary* channel: a symbol of
    ``GF(2^m)`` is sent as ``m`` bits, so a bit-flipping channel corrupts a
    symbol when it flips any of them, and a symbol error rate is
    ``1 - (1 - p)^m`` rather than ``p``. Most significant first so the
    mapping is the integer's own binary expansion and a reader can check a
    row by eye.

    Raises
    ------
    ValueError
        If a symbol is outside ``[0, 2^m)``, which would silently lose its
        high bits.
    """
    values = np.asarray(symbols, dtype=np.int64)
    if values.size and (values.min() < 0 or values.max() >= 1 << m):
        msg = f"symbols must lie in [0, {1 << m}), got [{values.min()}, {values.max()}]"
        raise ValueError(msg)
    shifts = np.arange(m - 1, -1, -1, dtype=np.int64)
    return np.asarray(((values[:, np.newaxis] >> shifts) & 1).ravel())


def symbols_from_bits(bits: np.ndarray, m: int) -> np.ndarray:
    """The inverse of :func:`bits_from_symbols`.

    Raises
    ------
    ValueError
        If the length is not a multiple of ``m``, where the last symbol
        would be read from a partial group.
    """
    values = np.asarray(bits, dtype=np.int64) % 2
    if values.size % m:
        msg = f"{values.size} bits do not divide into symbols of {m}"
        raise ValueError(msg)
    weights = 1 << np.arange(m - 1, -1, -1, dtype=np.int64)
    return np.asarray((values.reshape(-1, m) * weights).sum(axis=1))
