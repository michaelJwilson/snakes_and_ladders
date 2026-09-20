"""Successive cancellation on a polar code, and the list that fixes it.

Successive cancellation decodes bit ``i`` assuming bits ``0..i-1`` were right.
It is what makes the polarisation argument work --- the synthetic channel
``W_i`` is *defined* as the one a genie-aided SC decoder sees --- and it is
**not** maximum likelihood: having committed to a bit it cannot revisit it, so
one early mistake propagates through everything after. That gap is the whole
reason list decoding exists, and at the fixture sizes here it is measured
against `snakes_and_ladders.likelihood.ldpc.exact_decoding` rather than cited
(``sec:polar`` in the textbook).

**Two operations, applied over a butterfly.** At each stage the left child
sees ``f(a, b)``, the soft combination of two observations of the same
information, and the right child sees ``g(a, b, u)``, the second observation
corrected by what the left child decided. ``f`` is written in the exact
``atanh`` form rather than min-sum; min-sum is the hardware approximation and
this module is the reference the approximations would be pinned against.

**The list.** :func:`decode_scl` carries ``L`` candidate prefixes, each with a
path metric --- the log-likelihood of the prefix, exactly, through
``log(1 + exp(-(1 - 2u) L))``. At an information bit every path forks and the
best ``L`` of the ``2L`` survive. The paths share the recursion's *shape* and
differ only in their values, so the whole list decodes as one array with a
path axis rather than as ``L`` independent traversals.

**Two identities make the list checkable.** ``L = 1`` must reproduce
:func:`decode_sc` bit for bit, and ``L = 2^k`` cannot prune anything, so it is
an exhaustive search and must equal maximum likelihood. The suite asserts
both, which pins the fork-and-prune machinery at each end.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from snakes_and_ladders.sim.elementary_codes import crc_remainder
from snakes_and_ladders.sim.ldpc import LLR_CAP
from snakes_and_ladders.sim.polar import PolarCode, polar_transform


@dataclass(frozen=True)
class PolarDecoding:
    """What a polar decoder returns.

    Parameters
    ----------
    message : np.ndarray
        The ``k`` information bits, in information-set order.
    codeword : np.ndarray
        The ``N`` transmitted bits the decoder settled on.
    source : np.ndarray
        All ``N`` pre-transform bits, frozen positions included, so a caller
        can see what the decoder believed everywhere and not only where the
        message was.
    metric : float
        The path metric of the returned candidate: the negative
        log-likelihood of its source vector, so smaller is better. Comparable
        across candidates of one code and not across codes.
    list_size : int
        The list the decoder ran with; ``1`` for plain successive cancellation.
    crc_passed : bool | None
        Under CRC-aided decoding, whether the returned candidate's check
        passes --- ``False`` means no survivor passed and the best metric was
        returned as plain list decoding would; ``None`` without a CRC.
    """

    message: np.ndarray
    codeword: np.ndarray
    source: np.ndarray
    metric: float
    list_size: int
    crc_passed: bool | None = None


def _f(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The check-node combination, in the exact form.

    ``2 atanh(tanh(a/2) tanh(b/2))``, written through ``LLR_CAP`` so a certain
    input --- which the erasure channel produces --- does not saturate
    ``tanh`` to exactly one and make the ``atanh`` infinite.
    """
    capped_a = np.clip(a, -LLR_CAP, LLR_CAP)
    capped_b = np.clip(b, -LLR_CAP, LLR_CAP)
    return np.asarray(
        2.0 * np.arctanh(np.tanh(capped_a / 2.0) * np.tanh(capped_b / 2.0))
    )


def _g(a: np.ndarray, b: np.ndarray, left: np.ndarray) -> np.ndarray:
    """The variable-node combination: ``b + (1 - 2 u) a`` given the left decision."""
    return np.asarray(b + (1.0 - 2.0 * left) * a)


def _penalty(llr: np.ndarray, bit: np.ndarray | int) -> np.ndarray:
    """``log(1 + exp(-(1 - 2 bit) llr))``: the exact cost of deciding ``bit``.

    Through :func:`numpy.logaddexp`, which is the stable spelling: the naive
    ``log1p(exp(x))`` overflows for a confident message, which is exactly the
    case a decoder spends most of its time in.
    """
    return np.asarray(np.logaddexp(0.0, -(1.0 - 2.0 * bit) * llr))


def _recurse(
    llr: np.ndarray, frozen: np.ndarray, metric: np.ndarray, limit: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Decode one subtree for every live path at once.

    Parameters
    ----------
    llr : np.ndarray
        ``(paths, m)`` --- the observations this subtree sees, per path.
    frozen : np.ndarray
        ``(m,)`` boolean; the frozen positions of this subtree, shared by
        every path because the code is.
    metric : np.ndarray
        ``(paths,)`` path metrics so far.
    limit : int
        The list size. Pruning happens at the leaves, where the forks are.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ``source`` and ``codeword`` for the subtree, both ``(paths', m)``; the
        updated metrics ``(paths',)``; and ``survivors`` ``(paths',)``, which
        maps each surviving path to the incoming path it came from. The caller
        gathers its own arrays through ``survivors``, which is what lets a
        prune deep in the tree reorder the whole list.
    """
    paths, m = llr.shape
    if m == 1:
        if frozen[0]:
            # Frozen: no fork and no choice, but the metric still pays for the
            # evidence against zero. Skipping that is the usual bug --- it
            # makes an unlikely prefix look as good as a likely one.
            bits = np.zeros((paths, 1), dtype=np.int64)
            return bits, bits.copy(), metric + _penalty(llr[:, 0], 0), np.arange(paths)
        stacked = np.concatenate(
            [metric + _penalty(llr[:, 0], 0), metric + _penalty(llr[:, 0], 1)]
        )
        bits = np.concatenate(
            [np.zeros(paths, dtype=np.int64), np.ones(paths, dtype=np.int64)]
        )
        came_from = np.concatenate([np.arange(paths), np.arange(paths)])
        keep = np.argsort(stacked, kind="stable")[:limit]
        chosen = bits[keep].reshape(-1, 1)
        return chosen, chosen.copy(), stacked[keep], came_from[keep]

    half = m // 2
    a, b = llr[:, :half], llr[:, half:]

    left_source, left_word, metric, survivors = _recurse(
        _f(a, b), frozen[:half], metric, limit
    )
    a, b = a[survivors], b[survivors]

    right_source, right_word, metric, right_survivors = _recurse(
        _g(a, b, left_word), frozen[half:], metric, limit
    )
    left_source, left_word = left_source[right_survivors], left_word[right_survivors]

    return (
        np.concatenate([left_source, right_source], axis=1),
        np.concatenate([(left_word + right_word) % 2, right_word], axis=1),
        metric,
        survivors[right_survivors],
    )


def decode_scl(
    code: PolarCode,
    llr: np.ndarray,
    list_size: int = 1,
    *,
    crc: np.ndarray | None = None,
) -> PolarDecoding:
    """Successive-cancellation list decoding; ``list_size = 1`` is plain SC.

    Parameters
    ----------
    code : PolarCode
        The code, which fixes the frozen set the decoder relies on knowing.
    llr : np.ndarray
        ``N`` channel log-likelihood ratios in the convention
        ``snakes_and_ladders.sim.ldpc`` states: positive favours zero.
    list_size : int
        Candidate prefixes carried. Past ``2^k`` nothing more can be carried,
        and at ``2^k`` the search is exhaustive and the answer is maximum
        likelihood.
    crc : np.ndarray | None
        A CRC generator, most significant bit first, whose remainder the
        message's last ``r`` bits carry (:func:`crc_encode`). Given, the
        survivor with the best metric *whose check passes* is returned, and
        the best metric of all when none passes (Tal & Vardy 2015): the list
        holds ``list_size`` candidates the metric alone cannot rank, and the
        check is the outer code that ranks them. The payload is the first
        ``k - r`` message bits.

    Returns
    -------
    PolarDecoding

    Raises
    ------
    ValueError
        If ``llr`` is not ``N`` long or ``list_size`` is not positive.
    """
    values = np.asarray(llr, dtype=float).reshape(-1)
    if values.size != code.n_bits:
        msg = f"the code takes {code.n_bits} ratios, got {values.size}"
        raise ValueError(msg)
    if list_size < 1:
        msg = f"a list carries at least one path, got {list_size}"
        raise ValueError(msg)

    frozen = np.ones(code.n_bits, dtype=bool)
    frozen[code.information] = False
    source, codeword, metric, _ = _recurse(
        values.reshape(1, -1), frozen, np.zeros(1), list_size
    )
    order = np.argsort(metric, kind="stable")
    best = int(order[0])
    passed: bool | None = None
    if crc is not None:
        passed = False
        for candidate in order:
            if crc_checks(source[candidate][code.information], crc):
                best, passed = int(candidate), True
                break
    return PolarDecoding(
        message=np.asarray(source[best][code.information], dtype=np.uint8),
        codeword=np.asarray(codeword[best], dtype=np.uint8),
        source=np.asarray(source[best], dtype=np.uint8),
        metric=float(metric[best]),
        list_size=list_size,
        crc_passed=passed,
    )


def crc_encode(payload: np.ndarray, crc: np.ndarray) -> np.ndarray:
    """The ``k`` message bits a CRC-aided code carries: the payload, then its remainder.

    ``crc`` is the generator :func:`snakes_and_ladders.sim.elementary_codes.crc_remainder`
    divides by, so the receiver's check is that function on the payload
    against the tail. The message is what :meth:`PolarCode.encode` takes.
    """
    bits = np.asarray(payload, dtype=np.int64).reshape(-1) % 2
    return np.concatenate([bits, crc_remainder(bits, crc)]).astype(np.uint8)


def crc_checks(message: np.ndarray, crc: np.ndarray) -> bool:
    """Whether ``message``'s last ``r`` bits are the remainder of its first ``k - r``."""
    bits = np.asarray(message, dtype=np.int64).reshape(-1) % 2
    degree = np.asarray(crc).size - 1
    if bits.size <= degree:
        msg = f"a message of {bits.size} bits cannot carry a {degree}-bit check"
        raise ValueError(msg)
    return bool(np.array_equal(crc_remainder(bits[:-degree], crc), bits[-degree:]))


def decode_sc(code: PolarCode, llr: np.ndarray) -> PolarDecoding:
    """Plain successive cancellation: :func:`decode_scl` at ``list_size = 1``.

    Written as the special case rather than as a second implementation, so the
    identity the suite asserts --- ``SC == SCL(1)`` --- is true by
    construction and the test is checking that the list machinery has not
    broken it rather than that two codebases agree.
    """
    return decode_scl(code, llr, list_size=1)


def transmitted(code: PolarCode, message: np.ndarray) -> np.ndarray:
    """The codeword a message produces, for a test that needs both ends."""
    return code.encode(message)


def is_codeword(code: PolarCode, bits: np.ndarray) -> bool:
    """Whether ``bits`` lies in the code: its frozen source positions are zero.

    The transform is its **own inverse** over GF(2) --- ``F F = I`` on the
    kernel and the Kronecker power of an involution is one --- so the source
    vector is recovered by applying the same matrix rather than by inverting
    a float copy of it and rounding, which is what this did until the
    involution was asserted (``test_polar.py``).
    """
    word = np.asarray(bits, dtype=np.int64).reshape(-1)
    source = (word @ polar_transform(code.n_stages)) % 2
    return bool(np.all(source[code.frozen] == 0))


__all__ = [
    "PolarDecoding",
    "crc_checks",
    "crc_encode",
    "decode_sc",
    "decode_scl",
    "is_codeword",
    "transmitted",
]
