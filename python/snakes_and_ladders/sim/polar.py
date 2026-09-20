"""Polar codes: the transform, the frozen set it earns, and the Reed-Muller rule.

Arikan's (2009) construction takes one channel and makes ``N = 2^n`` synthetic
ones out of it. The kernel ``F = [[1, 0], [1, 1]]`` combines two uses into a
worse channel and a better one; applied ``n`` times --- ``G = F^{(x) n}``, the
``n``-fold Kronecker power --- it *polarises*, and in the limit each synthetic
channel is either useless or perfect. The message rides the good ones and the
rest are **frozen** to zero, so a decoder that works through the bits in order
knows the frozen ones already (``sec:polar`` in the textbook).

**It is here as a declined route (issue #593).** The construction and its two
decoders answer the question the ticket posed --- the gap successive
cancellation leaves against maximum likelihood at ``N = 16``, and the list that
closes part of it --- and no search, learner or figure calls them; the fourth
step, CRC-aided list decoding, is not taken. The three modules are conserved
rather than deleted with the tests that referee them (``sandbox/CLAUDE.md``):
:mod:`snakes_and_ladders.likelihood.polar` carries the decoders and
:mod:`snakes_and_ladders.sandbox.polar_reference` the naive recursion they are
pinned to bitwise. Only ``tests/`` imports them, and the declared instance is
the file beside those tests rather than a row of the fixture registry.

**The reliability, and where it is exact.** On the erasure channel the
Bhattacharyya parameter obeys a closed recursion, ``eq:polar-bec``: ``Z(W^-) =
2Z - Z^2`` and ``Z(W^+) = Z^2``. That is an oracle, not an approximation, and
:func:`bec_reliability` is it. Off the erasure channel there is no closed form
and :func:`gaussian_reliability` is the usual Gaussian approximation, which is
refereed against Monte Carlo rather than trusted.

**Capacity is conserved, and polarisation is not the same claim.** The
transform moves capacity between the synthetic channels without creating or
destroying any: ``sum_i I(W_i) = N (1 - eps)`` exactly, at every ``N``, which
the suite asserts to ``1e-12``. That the capacities *separate* is asymptotic
and shows slowly --- 75 per cent of the channels are still undecided at ``N =
8`` and 6 per cent at ``N = 16,384`` --- so it is reported as a curve
(``docs/experiments/019``) and not asserted at a fixture size.

**Reed-Muller is the same transform.** ``RM(r, m)`` keeps the rows of ``G``
whose Hamming weight is at least ``2^(m - r)``; polar keeps the most reliable.
Two rules over one transform, so :func:`reed_muller_information_set` sits here
beside :func:`polar_information_set` rather than in a module of its own. At
``N = 8, k = 4`` the two rules **agree** --- both give ``{3, 5, 6, 7}``, the
extended Hamming code --- and at ``N = 16`` they part.

**A polar code is a linear code**, so :func:`parity_check` hands it to the
machinery that already exists: `snakes_and_ladders.likelihood.ldpc`'s
``exact_decoding`` and ``enumerate_codewords`` take it unchanged, which is
what gives this section a maximum-likelihood oracle at the ``ci`` size without
writing one.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Self

import numpy as np

from snakes_and_ladders.sim.ldpc import ChannelParams, ParityCheck, null_space

#: Arikan's kernel. Every polar construction is this matrix and nothing else.
KERNEL = np.array([[1, 0], [1, 1]], dtype=np.int64)

#: Past this the dense generator is no longer the cheap part: it is ``N^2``
#: entries, and a decoder that needs one at ``N = 2^20`` wants the recursive
#: transform instead. The constructions below refuse rather than allocate it.
MAX_DENSE_LENGTH = 4096


def polar_transform(n: int) -> np.ndarray:
    """``F^{(x) n}``: the ``2^n x 2^n`` generator over GF(2).

    Bit-reversal is **not** applied. The two conventions differ by a
    permutation of the bit-channel indices and give the same code; this one
    keeps row ``i`` of the result equal to the ``i``-th Kronecker row, which
    is what makes :func:`reed_muller_information_set`'s weight rule read
    directly off it.

    Raises
    ------
    ValueError
        If ``n`` is negative, or ``2^n`` exceeds :data:`MAX_DENSE_LENGTH`.
    """
    if n < 0:
        msg = f"the transform is defined for n >= 0, got {n}"
        raise ValueError(msg)
    if 2**n > MAX_DENSE_LENGTH:
        msg = (
            f"a dense {2**n} x {2**n} transform is past MAX_DENSE_LENGTH "
            f"({MAX_DENSE_LENGTH}); the recursive decoder does not need one"
        )
        raise ValueError(msg)
    generator = np.array([[1]], dtype=np.int64)
    for _ in range(n):
        generator = np.kron(generator, KERNEL)
    return generator % 2


def bec_reliability(n: int, erasure_probability: float) -> np.ndarray:
    """Bhattacharyya ``Z`` per synthetic channel on the erasure channel: exact.

    Arikan's recursion, ``eq:polar-bec``. Each application of the kernel turns
    one channel of parameter ``z`` into ``2z - z^2`` (the worse) and ``z^2``
    (the better); ``n`` applications give ``2^n`` of them in the order the
    transform's rows are indexed. On this channel ``I(W_i) = 1 - Z(W_i)``
    exactly, which is what makes the capacity sum checkable.

    Smaller is better: :func:`polar_information_set` takes the smallest.

    Raises
    ------
    ValueError
        If ``erasure_probability`` is outside ``[0, 1]`` or ``n`` is negative.
    """
    if not 0.0 <= erasure_probability <= 1.0:
        msg = f"an erasure probability lies in [0, 1], got {erasure_probability}"
        raise ValueError(msg)
    if n < 0:
        msg = f"the recursion runs for n >= 0, got {n}"
        raise ValueError(msg)
    z = np.array([erasure_probability], dtype=float)
    for _ in range(n):
        z = np.concatenate([2 * z - z**2, z**2])
    return z


def gaussian_reliability(n: int, noise_scale: float) -> np.ndarray:
    """Bhattacharyya ``Z`` per synthetic channel on the Gaussian channel: approximate.

    There is no closed recursion off the erasure channel. This is the standard
    Gaussian approximation: the all-zero codeword's log-likelihood ratio is
    taken Gaussian with mean ``m`` and variance ``2m`` at every stage, so one
    number per channel propagates, and ``Z = exp(-m / 4)``. The check node
    ``m^- = phi^{-1}(1 - (1 - phi(m))^2)`` uses the usual closed form for
    ``phi`` and the variable node adds.

    Approximate, and labelled so: the suite pins the *ordering* it induces
    against Monte Carlo rather than the values, because the frozen set depends
    only on the order.

    Raises
    ------
    ValueError
        If ``noise_scale`` is not positive, or ``n`` is negative.
    """
    if noise_scale <= 0.0:
        msg = f"a noise scale is positive, got {noise_scale}"
        raise ValueError(msg)
    if n < 0:
        msg = f"the recursion runs for n >= 0, got {n}"
        raise ValueError(msg)
    means = np.array([2.0 / noise_scale**2], dtype=float)
    for _ in range(n):
        worse = _phi_inverse(1.0 - (1.0 - _phi(means)) ** 2)
        means = np.concatenate([worse, 2 * means])
    return np.asarray(np.exp(-means / 4.0))


def _phi(mean: np.ndarray) -> np.ndarray:
    """The Gaussian-approximation ``phi``, in its usual piecewise closed form."""
    out = np.empty_like(mean)
    small = mean < 10.0
    out[small] = np.exp(-0.4527 * mean[small] ** 0.859 + 0.0218)
    large = ~small
    out[large] = (
        np.sqrt(np.pi / mean[large])
        * np.exp(-mean[large] / 4.0)
        * (1.0 - 10.0 / (7.0 * mean[large]))
    )
    return out


def _phi_inverse(value: np.ndarray) -> np.ndarray:
    """``phi^{-1}`` by bisection, which is exact enough and cannot diverge.

    A Newton iteration on ``phi`` is the usual route and needs a derivative
    that is itself piecewise; ``phi`` is monotone decreasing on ``(0, inf)``,
    so fifty bisections bracket the root to ``1e-12`` relative with no
    derivative and no failure mode.
    """
    low = np.full_like(value, 1e-10)
    high = np.full_like(value, 1e4)
    for _ in range(200):
        middle = 0.5 * (low + high)
        smaller = _phi(middle) > value
        high = np.where(smaller, high, middle)
        low = np.where(smaller, middle, low)
    return np.asarray(0.5 * (low + high))


def polar_information_set(reliability: np.ndarray, n_info: int) -> np.ndarray:
    """The ``n_info`` most reliable bit-channels, ascending.

    Ties break on the index, so the set is a function of the reliabilities and
    not of the sort's internals; ``kind="stable"`` states it.

    Raises
    ------
    ValueError
        If ``n_info`` is outside ``[0, len(reliability)]``.
    """
    if not 0 <= n_info <= reliability.size:
        msg = f"an information set of {n_info} does not fit {reliability.size} channels"
        raise ValueError(msg)
    order = np.argsort(reliability, kind="stable")
    return np.sort(order[:n_info])


def reed_muller_information_set(n: int, order: int) -> np.ndarray:
    """``RM(order, n)``'s information set: the rows of weight ``>= 2^(n - order)``.

    The same transform as :func:`polar_information_set` reads, under a
    different rule --- weight rather than reliability --- which is the whole
    difference between the two families.

    Raises
    ------
    ValueError
        If ``order`` is outside ``[0, n]``.
    """
    if not 0 <= order <= n:
        msg = f"RM(r, m) needs 0 <= r <= m, got r = {order}, m = {n}"
        raise ValueError(msg)
    weights = polar_transform(n).sum(axis=1)
    return np.flatnonzero(weights >= 2 ** (n - order))


@dataclass(frozen=True)
class PolarCode:
    """``N = 2^n`` bits, ``k`` of them carrying the message.

    Parameters
    ----------
    n_stages : int
        ``n``, so the block length is ``2^n``.
    information : np.ndarray
        The bit-channels carrying the message, ascending. Its complement is
        the frozen set, held to zero.

    Raises
    ------
    ValueError
        If an index is out of range, repeats, or is not ascending --- the
        decoder walks the indices in order and a violation would silently
        decode a different code.
    """

    n_stages: int
    information: np.ndarray

    def __post_init__(self) -> None:
        indices = np.asarray(self.information, dtype=np.int64)
        if indices.ndim != 1:
            msg = "the information set is one-dimensional"
            raise ValueError(msg)
        if indices.size and (indices.min() < 0 or indices.max() >= self.n_bits):
            msg = f"an information index lies outside [0, {self.n_bits})"
            raise ValueError(msg)
        if np.any(np.diff(indices) <= 0):
            msg = "the information set is strictly ascending and without repeats"
            raise ValueError(msg)

    @property
    def n_bits(self) -> int:
        """The block length ``N = 2^n``."""
        return int(2**self.n_stages)

    @property
    def n_info(self) -> int:
        """The message length ``k``."""
        return int(self.information.size)

    @property
    def rate(self) -> float:
        """``k / N``."""
        return self.n_info / self.n_bits

    @property
    def frozen(self) -> np.ndarray:
        """The bit-channels held to zero, ascending."""
        return np.setdiff1d(np.arange(self.n_bits), self.information)

    def encode(self, message: np.ndarray) -> np.ndarray:
        """``u G``, with the message on the information channels and zeros elsewhere.

        Raises
        ------
        ValueError
            If ``message`` is not ``k`` bits.
        """
        bits = np.asarray(message, dtype=np.int64).reshape(-1)
        if bits.size != self.n_info:
            msg = f"the message carries {bits.size} bits, the code takes {self.n_info}"
            raise ValueError(msg)
        source = np.zeros(self.n_bits, dtype=np.int64)
        source[self.information] = bits % 2
        return np.asarray((source @ polar_transform(self.n_stages)) % 2, dtype=np.uint8)

    def generator(self) -> np.ndarray:
        """The ``k x N`` generator: the transform's information rows."""
        return np.asarray(polar_transform(self.n_stages)[self.information, :])


def parity_check(code: PolarCode) -> ParityCheck:
    """The code as a :class:`~snakes_and_ladders.sim.ldpc.ParityCheck`.

    A polar code is linear, so its dual is the null space of its generator and
    everything written against `ParityCheck` applies --- in particular
    ``likelihood.ldpc.exact_decoding``, which is the maximum-likelihood oracle
    this section's decoders are measured against. Dense, so it is for the
    sizes an enumeration reaches and not for a real block length.

    Raises
    ------
    ValueError
        If the code has no frozen bits, since then the dual is empty and there
        is nothing for a parity check to check.
    """
    if code.n_info == code.n_bits:
        msg = "a rate-one code has an empty dual; there is no parity check to build"
        raise ValueError(msg)
    dual = null_space(np.ascontiguousarray(code.generator().astype(np.uint8)))
    return ParityCheck.from_dense(dual)


def bec_polar_code(n_stages: int, n_info: int, erasure_probability: float) -> PolarCode:
    """The code the exact erasure recursion picks at this length and rate."""
    reliability = bec_reliability(n_stages, erasure_probability)
    return PolarCode(n_stages, polar_information_set(reliability, n_info))


def gaussian_polar_code(n_stages: int, n_info: int, noise_scale: float) -> PolarCode:
    """The code the Gaussian approximation picks at this length and rate."""
    reliability = gaussian_reliability(n_stages, noise_scale)
    return PolarCode(n_stages, polar_information_set(reliability, n_info))


def reed_muller_code(n_stages: int, order: int) -> PolarCode:
    """``RM(order, n_stages)`` as a :class:`PolarCode`: the same transform, the weight rule."""
    return PolarCode(n_stages, reed_muller_information_set(n_stages, order))


def capacity(reliability: np.ndarray) -> np.ndarray:
    """``I(W_i) = 1 - Z(W_i)``, which holds **exactly on the erasure channel only**.

    Elsewhere ``1 - Z`` is a lower bound on the capacity rather than equal to
    it, so this is named for what it computes and the suite uses it where the
    identity holds.
    """
    return np.asarray(1.0 - reliability)


_REQUIRED_FIELDS = frozenset(
    {
        "n_stages",
        "n_info",
        "construction",
        "flip_probability",
        "erasure_probability",
        "noise_scale",
    }
)


@dataclass(frozen=True)
class PolarParams(ChannelParams):
    """Fully-specified truth for a polar fixture.

    Parameters
    ----------
    n_stages : int
        ``n``, so the block length is ``2^n``.
    n_info : int
        The message length ``k``.
    construction : str
        ``"bec"`` for the exact erasure recursion, ``"gaussian"`` for the
        approximation, ``"reed-muller"`` for the weight rule. Named in the
        fixture because the frozen set is the construction's and a reader
        comparing two runs needs to know which rule chose it.
    """

    n_stages: int
    n_info: int
    construction: str

    def code(self) -> PolarCode:
        """Build the code this fixture declares.

        Raises
        ------
        ValueError
            If ``construction`` is not one of the three, or is
            ``reed-muller`` at a ``n_info`` no order of ``RM`` produces --- the
            weight rule fixes ``k``, so a fixture cannot ask for another.
        """
        if self.construction == "bec":
            return bec_polar_code(self.n_stages, self.n_info, self.erasure_probability)
        if self.construction == "gaussian":
            return gaussian_polar_code(self.n_stages, self.n_info, self.noise_scale)
        if self.construction == "reed-muller":
            for order in range(self.n_stages + 1):
                candidate = reed_muller_code(self.n_stages, order)
                if candidate.n_info == self.n_info:
                    return candidate
            sizes = [
                reed_muller_code(self.n_stages, order).n_info
                for order in range(self.n_stages + 1)
            ]
            msg = (
                f"no RM(r, {self.n_stages}) has k = {self.n_info}; "
                f"the weight rule gives {sizes}"
            )
            raise ValueError(msg)
        msg = (
            f"construction {self.construction!r} is not one of "
            "'bec', 'gaussian', 'reed-muller'"
        )
        raise ValueError(msg)

    #: The fields :func:`snakes_and_ladders.fixtures.load_params` checks are present before
    #: calling :meth:`from_declared`.
    required_fields: ClassVar[frozenset[str]] = _REQUIRED_FIELDS

    @classmethod
    def from_declared(cls, declared: Mapping[str, Any], _path: Path, /) -> Self:
        """Read a polar fixture.

        ``declared`` is the mapping
        :func:`snakes_and_ladders.fixtures.load_params` read from ``path``
        with :attr:`required_fields` present; ``path`` names the file in
        every error.

        The parsed truth. Which construction the fields describe is checked
        by :meth:`PolarParams.code`, so one statement of that serves both
        callers.
        """
        return cls(
            n_stages=int(declared["n_stages"]),
            n_info=int(declared["n_info"]),
            construction=str(declared["construction"]),
            flip_probability=float(declared["flip_probability"]),
            erasure_probability=float(declared["erasure_probability"]),
            noise_scale=float(declared["noise_scale"]),
        )


__all__ = [
    "KERNEL",
    "MAX_DENSE_LENGTH",
    "PolarCode",
    "PolarParams",
    "bec_polar_code",
    "bec_reliability",
    "capacity",
    "gaussian_polar_code",
    "gaussian_reliability",
    "parity_check",
    "polar_information_set",
    "polar_transform",
    "reed_muller_code",
    "reed_muller_information_set",
]
