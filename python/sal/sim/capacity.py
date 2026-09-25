"""Channel capacity, and the largest noise a rate can carry.

Issue #594. A decoder's error rate says what one code does; capacity says
what **no** code of that rate does, so it is the only referee in this
repository that bounds a result from above without running anything. Every
curve in `docs/nb/classical_codes.ipynb` is read against it.

Three channels, three capacities in bits per transmitted symbol
(``sec:ldpc:capacity``):

* **erasure** --- ``C = 1 - eps``, exactly, since an unerased symbol arrives
  intact and an erased one carries nothing;
* **binary symmetric** --- ``C = 1 - H(p)`` for the binary entropy ``H`` in
  bits, the one place a logarithm's base is load-bearing here;
* **binary-input Gaussian** --- an integral with no closed form,
  ``C = 1 - E log2(1 + exp(-2 y / sigma^2))`` over ``y ~ N(1, sigma^2)``,
  taken by Gauss--Hermite quadrature on 100 nodes.

**The quadrature carries a published pin.** At ``sigma = 0.9787``, the
rate-1/2 binary-input limit (Richardson and Urbanke 2008, §4.10), it returns
0.49999606 --- 3.9e-6 from one half. Fifty nodes and one hundred agree to
3.3e-10, so the node count is not what the number rests on; 400 nodes is
where `numpy.polynomial.hermite.hermgauss` overflows its own weights and
returns ``nan``, which is why the default is not simply raised.

**What the limit is *not* is a threshold.** Capacity is a converse: above it
no code of that rate is reliable at any length. A code's own threshold sits
below it and the distance is the code's, not the channel's --- the (3,6)
ensemble's erasure threshold is 0.4294 against a limit of 0.5 at rate 1/2,
leaving 14.1% of the channel unused, and that gap survives to infinite
length. `sim/elementary_codes.py` and `sim/reed_solomon.py` carry the codes
whose distance from the limit the notebook measures.

A channel outside the three raises: capacity is a property of a declared
channel, and guessing one from an interface that exposes only log-likelihood
ratios would return a number for a channel nobody described.

See ``sec:ldpc:capacity`` of ``docs/tex/textbook.tex`` (Shannon 1948;
Cover and Thomas ch. 7; Richardson and Urbanke 2008 §4.10).
"""

from __future__ import annotations

import math

import numpy as np

from sal.sim.ldpc import (
    BinaryErasureChannel,
    BinaryInputGaussianChannel,
    BinarySymmetricChannel,
    Channel,
)

# Gauss--Hermite nodes for the Gaussian channel's integral. 100 rather than
# more: it agrees with 50 to 3.3e-10 and with 200 to 3.1e-11, and at 400
# `hermgauss` overflows its own weight recursion and returns `nan`.
DEFAULT_NODES = 100
# Each parameter's search interval for `noise_at_capacity`. The symmetric
# channel stops at 1/2 because capacity rises again past it -- a flip
# probability of 0.9 is a good channel with the labels swapped, and bisecting
# over [0, 1] would return that branch. The Gaussian interval's ends carry
# capacity 1.0 -- indistinguishable from it in double precision -- and
# 0.00180, which bracket every rate in (0, 1).
ERASURE_INTERVAL = (0.0, 1.0)
CROSSOVER_INTERVAL = (0.0, 0.5)
SIGMA_INTERVAL = (1e-6, 20.0)

ChannelFamily = (
    type[BinaryErasureChannel]
    | type[BinarySymmetricChannel]
    | type[BinaryInputGaussianChannel]
)


def binary_entropy(probability: float) -> float:
    """``H(p) = -p log2 p - (1-p) log2 (1-p)``, in bits.

    Zero at both ends by the ``0 log 0 = 0`` convention, which is a limit and
    not a special case: the entropy of a certain symbol is zero.

    Raises
    ------
    ValueError
        If ``probability`` is outside ``[0, 1]``.
    """
    if not 0.0 <= probability <= 1.0:
        msg = f"a probability lies in [0, 1], got {probability}"
        raise ValueError(msg)
    if probability in (0.0, 1.0):
        return 0.0
    complement = 1.0 - probability
    return float(
        -probability * math.log2(probability) - complement * math.log2(complement)
    )


def erasure_capacity(erasure: float) -> float:
    """``1 - eps``: the fraction of symbols that arrive."""
    if not 0.0 <= erasure <= 1.0:
        msg = f"an erasure probability lies in [0, 1], got {erasure}"
        raise ValueError(msg)
    return 1.0 - erasure


def symmetric_capacity(crossover: float) -> float:
    """``1 - H(p)``, zero at ``p = 1/2`` and one at both certainties."""
    return 1.0 - binary_entropy(crossover)


def gaussian_capacity(sigma: float, *, nodes: int = DEFAULT_NODES) -> float:
    """The binary-input Gaussian channel's capacity, by quadrature.

    ``C = 1 - E log2(1 + exp(-2 y / sigma^2))`` with ``y ~ N(1, sigma^2)``,
    the expectation taken on ``nodes`` Gauss--Hermite points. The integrand is
    written through :func:`numpy.logaddexp` rather than ``log(1 + exp(-t))``,
    which overflows at ``t`` a few hundred and is reached here at
    ``sigma = 0.1``.

    Parameters
    ----------
    sigma : float
        The noise scale, positive.
    nodes : int
        Quadrature nodes. See ``DEFAULT_NODES`` for why the default is 100.

    Returns
    -------
    float
        Capacity in bits per channel use, in ``(0, 1)``.

    Raises
    ------
    ValueError
        If ``sigma`` is not positive, or ``nodes`` is under two.
    """
    if sigma <= 0.0:
        msg = f"a noise scale is positive, got {sigma}"
        raise ValueError(msg)
    if nodes < 2:
        msg = f"quadrature needs at least two nodes, got {nodes}"
        raise ValueError(msg)
    points, weights = np.polynomial.hermite.hermgauss(nodes)
    received = 1.0 + sigma * np.sqrt(2.0) * points
    integrand = np.logaddexp(0.0, -2.0 * received / sigma**2) / np.log(2.0)
    return 1.0 - float((weights * integrand).sum() / np.sqrt(np.pi))


def capacity(channel: Channel, *, nodes: int = DEFAULT_NODES) -> float:
    """The capacity of one of the three declared channels.

    Parameters
    ----------
    channel : Channel
        A channel of ``sim/ldpc.py``.
    nodes : int
        Passed to :func:`gaussian_capacity`; ignored by the other two, whose
        capacity is closed form.

    Returns
    -------
    float
        Bits per channel use.

    Raises
    ------
    NotImplementedError
        For any other channel. The ``Channel`` protocol exposes
        log-likelihood ratios and nothing else, so there is no capacity to
        derive from it -- and returning one anyway would attribute a number
        to a channel this module has never seen.
    """
    if isinstance(channel, BinaryErasureChannel):
        return erasure_capacity(channel.erasure_probability)
    if isinstance(channel, BinarySymmetricChannel):
        return symmetric_capacity(channel.flip_probability)
    if isinstance(channel, BinaryInputGaussianChannel):
        return gaussian_capacity(channel.sigma, nodes=nodes)
    msg = (
        f"no capacity is declared for {type(channel).__name__}: the three in "
        "sim/ldpc.py are the erasure, symmetric and binary-input Gaussian "
        "channels"
    )
    raise NotImplementedError(msg)


def noise_at_capacity(
    rate: float,
    family: ChannelFamily,
    *,
    precision: float = 1e-9,
    nodes: int = DEFAULT_NODES,
) -> float:
    """The noise level whose capacity equals ``rate``: the Shannon limit.

    Bisection, as :func:`sal.likelihood.ldpc.erasure_threshold`
    is, and on the same reasoning: capacity decreases monotonically in each
    channel's one parameter over the interval this module declares for it, so
    the crossing is unique and a bisection cannot land on the wrong branch.
    No code of rate ``rate`` is reliable at any length past the number
    returned, and the distance from it to a code's own threshold is what the
    code costs.

    Parameters
    ----------
    rate : float
        ``k / n``, in ``(0, 1)``. At 0 and 1 the limit is the interval's own
        end rather than a crossing, so both are refused.
    family : ChannelFamily
        One of the three channel *classes*, not an instance: the parameter is
        what is being solved for.
    precision : float
        Bisection stops when the interval is this wide.
    nodes : int
        Quadrature nodes for the Gaussian channel.

    Returns
    -------
    float
        ``eps*``, ``p*`` or ``sigma*``, by family.

    Raises
    ------
    ValueError
        If ``rate`` is outside ``(0, 1)``.
    NotImplementedError
        For a family outside the three.
    """
    if not 0.0 < rate < 1.0:
        msg = f"a rate lies strictly in (0, 1), got {rate}"
        raise ValueError(msg)
    if family is BinaryErasureChannel:
        low, high = ERASURE_INTERVAL
    elif family is BinarySymmetricChannel:
        low, high = CROSSOVER_INTERVAL
    elif family is BinaryInputGaussianChannel:
        low, high = SIGMA_INTERVAL
    else:
        msg = f"no capacity is declared for {family.__name__}"
        raise NotImplementedError(msg)

    while high - low > precision:
        middle = 0.5 * (low + high)
        if capacity(family(middle), nodes=nodes) > rate:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def eb_n0_decibels(sigma: float, rate: float) -> float:
    """``E_b/N_0`` in dB at a noise scale and a rate.

    Antipodal signalling puts one unit of energy in each transmitted symbol,
    so ``E_s = 1``, ``E_b = 1 / R`` and ``N_0 = 2 sigma^2``: this is
    ``1 / (2 R sigma^2)`` in dB. It is the inverse of
    :func:`sal.likelihood.turbo.noise_scale`, which maps the
    other way and is where a waterfall's points are set; the two are pinned to
    each other in the suite rather than restated, since a pair of inverses
    that drift apart moves every curve read through them.

    Raises
    ------
    ValueError
        If ``sigma`` is not positive or ``rate`` is outside ``(0, 1]``.
    """
    if sigma <= 0.0:
        msg = f"a noise scale is positive, got {sigma}"
        raise ValueError(msg)
    if not 0.0 < rate <= 1.0:
        msg = f"rate must be in (0, 1], got {rate}"
        raise ValueError(msg)
    return float(10.0 * math.log10(1.0 / (2.0 * rate * sigma**2)))


def decibels_at_capacity(
    rate: float, *, nodes: int = DEFAULT_NODES, precision: float = 1e-9
) -> float:
    """The binary-input Gaussian limit at ``rate``, in dB.

    The number a waterfall is read against: **+0.187 dB** at rate 1/2, which
    is the figure every coding text quotes, and it falls with the rate ---
    -0.508 dB at the rate 0.3299 the turbo fixture transmits at once its tail
    bits are charged. Below the value returned no code of that rate is
    reliable at any length, however it is decoded.
    """
    sigma = noise_at_capacity(
        rate, BinaryInputGaussianChannel, precision=precision, nodes=nodes
    )
    return eb_n0_decibels(sigma, rate)
