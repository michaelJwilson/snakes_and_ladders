"""The negative binomial from scaled rising factorials, one construction on every route (issues #1064, #1335).

Under a rate ``lambda = mu c`` --- the mean times a per-observation exposure
``c``, ``c = 1`` without one --- the negative binomial's log pmf is

    ``log p(y | r, lambda) = ((T(r, y) + y log(lambda / (1 + q))) - D)``,
    ``T(r, y) = S(r, y) - lgamma(y + 1)``, ``q = lambda / r``,
    ``D = r log1p(q)``, and ``D = lambda`` where ``q`` is ``0``,

summed in that order, with ``S(r, y) = lgamma(r + y) - lgamma(r) - y log r``
the scaled log rising factorial,
:func:`~sal.emissions.rising.scaled_rising_array`. It is the textbook
``lgamma(y + r) - lgamma(r) - lgamma(y + 1) + r log(r / t) + y log(lambda /
t)``, ``t = r + lambda``, with the rising factorial's ``y log r`` collected
into the ``log`` of one rounded quotient, ``r lambda / t``, and ``r log(r /
t)`` written as ``-r log1p(q)``: no term of size ``r log r`` or ``y log r``
is formed and cancelled, so ``S`` goes to ``0`` as ``r`` grows and the pmf to
the Poisson's, which it is at ``r = inf``. ``T`` is a function of the count
alone, so a caller scoring many exposures tabulates it once by count
(:func:`count_log_factor`, :func:`exposure_table`) and completes the rest per
observation: one ``log``, one ``log1p`` and two divisions a score, in the same
order in ``src/coupled.rs`` and ``src/dense_emission.rs``. The tables summed in
that order are :func:`negative_binomial_log_pmf` bit for bit. The torch
:meth:`~sal.emissions.counts.NegativeBinomialEmission.log_density` keeps its
own arithmetic and is held to a declared tolerance.
"""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray
from scipy.special import gammaln

from sal.emissions.counts import NegativeBinomialEmission
from sal.emissions.rising import scaled_rising_array


def count_log_factor(
    family: NegativeBinomialEmission, observations: ArrayLike | torch.Tensor
) -> NDArray[np.float64]:
    """``T_k(y) = S(r_k, y) - lgamma(y + 1)``, the part of the pmf that no exposure reaches.

    Parameters
    ----------
    family : NegativeBinomialEmission
        The family whose dispersions ``r`` enter.
    observations : ArrayLike | torch.Tensor
        Counts, any shape.

    Returns
    -------
    np.ndarray
        Shape ``(..., n_states)``.
    """
    counts = np.asarray(observations, dtype=np.float64)[..., None]
    dispersion = family.dispersion.detach().numpy()
    out: NDArray[np.float64] = scaled_rising_array(dispersion, counts) - gammaln(
        counts + 1.0
    )
    return out


def exposure_table(
    family: NegativeBinomialEmission, extent: int
) -> NDArray[np.float64]:
    """:func:`count_log_factor` at every count ``y < extent``, ``(extent, n_states)``.

    The table the Rust kernels complete per observation in the module's
    order (``src/coupled.rs``, ``src/dense_emission.rs``).
    """
    return count_log_factor(family, np.arange(extent, dtype=np.float64))


def negative_binomial_log_pmf(
    counts: ArrayLike, dispersion: ArrayLike, rate: ArrayLike
) -> NDArray[np.float64]:
    """The negative binomial's log pmf, summed in the module's order; broadcasts.

    The negative binomial on NumPy arrays (issue #1335), mean ``rate`` and
    variance ``rate + rate**2 / dispersion``; ``dispersion = inf`` is the
    Poisson. A count that is not a non-negative integer is not checked.

    Parameters
    ----------
    counts : ArrayLike
        ``y``, non-negative integers.
    dispersion : ArrayLike
        ``r``, positive, ``inf`` allowed.
    rate : ArrayLike
        ``lambda = mu c``, non-negative.

    Returns
    -------
    np.ndarray
        The broadcast shape of the three inputs.
    """
    y, r, rate_ = np.broadcast_arrays(
        *(np.asarray(v, dtype=np.float64) for v in (counts, dispersion, rate))
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        q = rate_ / r
        decay = np.where(q == 0.0, rate_, r * np.log1p(q))
        rated = np.where(y == 0.0, 0.0, y * np.log(rate_ / (1.0 + q)))
    table = scaled_rising_array(r, y) - gammaln(y + 1.0)
    out: NDArray[np.float64] = (table + rated) - decay
    return out
