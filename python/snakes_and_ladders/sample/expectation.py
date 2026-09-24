"""Expectations of given operators along a chain, by a Kalman filter, without the chain (issue #988).

A chain's draws are correlated, so the mean of ``f(theta_t)`` over them is a
fine estimate of ``E f`` and its white-noise standard error is not: at a
lag-one autocorrelation of 0.9 it is 4.4 times too small. :class:`KalmanMean`
estimates the mean and a standard error that accounts for the correlation,
per output coordinate, from O(1) state per coordinate, so a sampler run with
the chain discarded still returns what the chain was for.

**The model.** Observation ``t`` is ``y_t = mu + e_t``, the noise AR(1):
``e_t = phi e_(t-1) + w_t``, ``w_t ~ N(0, q)``. Whitening by ``phi``,
``z_t = y_t - phi y_(t-1) = (1 - phi) mu + w_t`` for ``t >= 2``, a static
state observed through ``h = 1 - phi`` in white noise of variance ``q``: the
Kalman filter for it, in information form, adds ``h^2 / q`` to the precision
and ``h z_t / q`` to the information vector per observation, and its
posterior is ``mu = sum z / ((n - 1) h)`` with variance
``q / ((n - 1) h^2)``, the AR(1) long-run variance over ``n - 1``. That is the
conditional generalized least squares estimator of ``mu``, which is what the
oracle test pins it to.

**What is stored.** The filter's sums ``sum z`` and ``(n - 1)`` depend on
``phi``, which is estimated from the same stream: the lag-zero and lag-one
moments give ``phi = gamma_1 / gamma_0`` and ``q = gamma_0 (1 - phi^2)``. So
the state is the stream's sufficient statistics --- ``n``, ``sum y``,
``sum y^2``, ``sum y_t y_(t-1)``, the first and the last ``y`` --- and the
filter is evaluated on them at the current ``phi`` when :meth:`KalmanMean.estimate`
is called, exactly what the recursion at that ``phi`` would give. ``phi`` is
held inside ``(-PHI_BOUND, PHI_BOUND)`` so the variance stays finite on a
chain that has not moved.

**NumPy in, NumPy out.** The filter takes no derivative, so it holds no
autodiff type (issue #1011): :meth:`KalmanMean.update` and
:meth:`KalmanMean.update_block` take array-likes, and :class:`Expectation`
holds ``np.ndarray``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

#: The largest ``|phi|`` the estimate uses: at 0.999 the long-run variance is
#: 1,999 times the marginal one, beyond what a chain of the lengths run here
#: identifies.
PHI_BOUND = 0.999


@dataclass(frozen=True)
class Expectation:
    """``E f`` estimated from a stream of ``f(theta_t)``, per output coordinate."""

    #: The posterior mean of ``mu``.
    mean: np.ndarray
    #: Its posterior standard deviation under the AR(1) noise model.
    standard_error: np.ndarray
    #: The lag-one autocorrelation the noise model used.
    phi: np.ndarray
    #: Observations taken.
    n: int


def _running(total: np.ndarray, terms: np.ndarray) -> np.ndarray:
    """``total + terms[0] + terms[1] + ...``, added one row at a time, left to right.

    The order :meth:`KalmanMean.update` adds in, so a block is summed
    bitwise as its rows one at a time would be: ``cumsum`` accumulates
    sequentially, where ``sum`` over an axis may reorder (pairwise, or
    torch's cascade before issue #1011).
    """
    running = np.cumsum(np.concatenate((total[None], terms)), axis=0)
    last: np.ndarray = running[-1]
    return last


class KalmanMean:
    """A running estimate of a stream's mean, and its standard error under AR(1) noise.

    Feed it one observation at a time with :meth:`update`; read it with
    :meth:`estimate`. It holds six numbers per coordinate whatever the
    stream's length.
    """

    def __init__(self) -> None:
        self.n = 0
        self._sum: np.ndarray | None = None
        self._squares: np.ndarray | None = None
        self._lagged: np.ndarray | None = None
        self._first: np.ndarray | None = None
        self._last: np.ndarray | None = None

    @classmethod
    def from_statistics(
        cls,
        n: int,
        sums: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> KalmanMean:
        """A filter holding statistics kept elsewhere: ``n``, the sums of ``y``, ``y^2``, ``y_t y_(t-1)``, and the first and last ``y``.

        What a compiled chain hands back instead of its draws (issue #1006).
        The arrays are held, not copied.
        """
        kalman = cls()
        if n:
            total, squares, lagged, first, last = (np.asarray(v) for v in sums)
            kalman.n = n
            kalman._sum, kalman._squares, kalman._lagged = total, squares, lagged
            kalman._first, kalman._last = first, last
        return kalman

    def update(self, value: ArrayLike) -> None:
        """Take one observation, of any fixed shape."""
        y = np.asarray(value, dtype=np.float64).reshape(-1)
        if self._last is None:
            self._sum = y.copy()
            self._squares = y * y
            self._lagged = np.zeros_like(y)
            self._first = y.copy()
        else:
            assert self._sum is not None
            assert self._squares is not None
            assert self._lagged is not None
            self._sum += y
            self._squares += y * y
            self._lagged += y * self._last
        self._last = y.copy()
        self.n += 1

    def update_block(self, values: ArrayLike) -> None:
        """Take ``values.shape[0]`` observations at once, in order (issue #1006).

        The sums :meth:`update` keeps, added in its order, so a compiled
        chain handing back draws in blocks is observed bitwise as it would
        be one draw at a time (issue #1011).
        """
        y = np.asarray(values, dtype=np.float64)
        y = y.reshape(y.shape[0], -1)
        if y.shape[0] == 0:
            return
        if self._last is None:
            self.update(y[0])
            y = y[1:]
            if y.shape[0] == 0:
                return
        assert self._sum is not None
        assert self._squares is not None
        assert self._lagged is not None
        assert self._last is not None
        previous = np.concatenate((self._last[None], y[:-1]))
        self._sum = _running(self._sum, y)
        self._squares = _running(self._squares, y * y)
        self._lagged = _running(self._lagged, y * previous)
        self._last = y[-1].copy()
        self.n += int(y.shape[0])

    def estimate(self) -> Expectation:
        """The filter's posterior at the ``phi`` the stream so far gives.

        Raises
        ------
        ValueError
            Before two observations, where no lag-one moment exists.
        """
        if self.n < 2:
            msg = f"an AR(1) estimate needs two observations, got {self.n}"
            raise ValueError(msg)
        assert self._sum is not None
        assert self._squares is not None
        assert self._lagged is not None
        assert self._first is not None
        assert self._last is not None
        n = self.n
        mean = self._sum / n
        gamma0 = self._squares / n - mean * mean
        # The lag-one moment about the full mean: sum of (y_t - m)(y_(t-1) - m)
        # over t >= 2, expanded so it needs the sums alone.
        head = self._sum - self._last
        tail = self._sum - self._first
        gamma1 = (self._lagged - mean * (head + tail) + (n - 1) * mean * mean) / (n - 1)
        moving = gamma0 > 0.0
        phi = np.where(
            moving,
            np.clip(gamma1 / np.where(moving, gamma0, 1.0), -PHI_BOUND, PHI_BOUND),
            np.zeros_like(gamma0),
        )
        gain = 1.0 - phi
        noise = np.maximum(gamma0 * (1.0 - phi * phi), 0.0)
        whitened = tail - phi * head
        mu = whitened / ((n - 1) * gain)
        variance = noise / ((n - 1) * gain * gain)
        return Expectation(mean=mu, standard_error=np.sqrt(variance), phi=phi, n=n)
