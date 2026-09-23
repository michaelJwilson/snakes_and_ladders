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
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

#: The largest ``|phi|`` the estimate uses: at 0.999 the long-run variance is
#: 1,999 times the marginal one, beyond what a chain of the lengths run here
#: identifies.
PHI_BOUND = 0.999


@dataclass(frozen=True)
class Expectation:
    """``E f`` estimated from a stream of ``f(theta_t)``, per output coordinate."""

    #: The posterior mean of ``mu``.
    mean: torch.Tensor
    #: Its posterior standard deviation under the AR(1) noise model.
    standard_error: torch.Tensor
    #: The lag-one autocorrelation the noise model used.
    phi: torch.Tensor
    #: Observations taken.
    n: int


class KalmanMean:
    """A running estimate of a stream's mean, and its standard error under AR(1) noise.

    Feed it one observation at a time with :meth:`update`; read it with
    :meth:`estimate`. It holds six numbers per coordinate whatever the
    stream's length.
    """

    def __init__(self) -> None:
        self.n = 0
        self._sum: torch.Tensor | None = None
        self._squares: torch.Tensor | None = None
        self._lagged: torch.Tensor | None = None
        self._first: torch.Tensor | None = None
        self._last: torch.Tensor | None = None

    def update(self, value: torch.Tensor) -> None:
        """Take one observation, of any fixed shape."""
        y = torch.as_tensor(value, dtype=torch.float64).detach().reshape(-1)
        if self._last is None:
            self._sum = y.clone()
            self._squares = y * y
            self._lagged = torch.zeros_like(y)
            self._first = y.clone()
        else:
            assert self._sum is not None
            assert self._squares is not None
            assert self._lagged is not None
            self._sum += y
            self._squares += y * y
            self._lagged += y * self._last
        self._last = y.clone()
        self.n += 1

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
        phi = torch.where(
            moving,
            (gamma1 / torch.where(moving, gamma0, 1.0)).clamp(-PHI_BOUND, PHI_BOUND),
            torch.zeros_like(gamma0),
        )
        gain = 1.0 - phi
        noise = (gamma0 * (1.0 - phi * phi)).clamp(min=0.0)
        whitened = tail - phi * head
        mu = whitened / ((n - 1) * gain)
        variance = noise / ((n - 1) * gain * gain)
        return Expectation(mean=mu, standard_error=variance.sqrt(), phi=phi, n=n)
