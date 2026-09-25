"""The categorical and Gaussian families, and the variance floor a Gaussian fit refuses at.

A Gaussian emission's likelihood has no maximum (the package docstring states
why), so :class:`GaussianEmission` carries an explicit floor and
:func:`pooled_variance_floor` derives one from the data.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch
from numpy.typing import ArrayLike

from sal.emissions.base import (
    EmissionFamily,
    ParameterDomainError,
    Reestimate,
    Values,
    as_tensor,
    refuse_covariate,
)
from sal.numerics import sample_rows

#: Nearest-neighbour spacing, in units of the pooled standard deviation, that
#: :func:`pooled_variance_floor` treats as the smallest scale a state can
#: occupy. ``n`` observations spread over a pooled standard deviation ``s``
#: sit at a typical spacing ``s / n``, so a state explaining a *neighbourhood*
#: has variance of at least that order and one collapsed onto a *single*
#: observation has variance heading to zero. Derived from the data rather than
#: fixed, so it transfers across fixture sizes.
COLLAPSE_EXPONENT = 2


class CategoricalEmission(EmissionFamily):
    """A symbol drawn from a per-state distribution over a fixed alphabet.

    Parameters
    ----------
    matrix : Values
        Row-stochastic emission matrix, shape ``(n_states, n_symbols)``.
    """

    def __init__(self, matrix: Values) -> None:
        values = torch.as_tensor(matrix, dtype=torch.float64)
        if values.ndim != 2 or values.shape[1] < 1:
            msg = f"emission matrix must be 2-D, got shape {tuple(values.shape)}"
            raise ValueError(msg)
        # Both forms are stored rather than one derived on demand: a round
        # trip through ``exp(log(p))`` moves the last bits of a probability,
        # enough to move an inverse-CDF draw at a cell boundary and so change
        # a pinned simulated sequence.
        self._matrix = values
        self._log_matrix = torch.log(values)

    @classmethod
    def from_log(cls, log_matrix: torch.Tensor) -> CategoricalEmission:
        """Build from log-probabilities, the form the recursions carry.

        Parameters
        ----------
        log_matrix : torch.Tensor
            Log emission matrix, shape ``(n_states, n_symbols)``.

        Returns
        -------
        CategoricalEmission
            A family holding exactly these log-probabilities, without a round
            trip through ``exp`` and ``log``.
        """
        family = cls.__new__(cls)
        family._log_matrix = log_matrix
        family._matrix = torch.exp(log_matrix)
        return family

    @property
    def n_states(self) -> int:
        """Hidden states this family emits from."""
        return int(self._log_matrix.shape[0])

    @property
    def n_symbols(self) -> int:
        """Size of the emission alphabet."""
        return int(self._log_matrix.shape[1])

    @property
    def is_discrete(self) -> bool:
        """True: the alphabet is finite, so the evidence is a probability."""
        return True

    @property
    def observation_dtype(self) -> torch.dtype:
        """Symbols are indices into the alphabet."""
        return torch.long

    @property
    def log_matrix(self) -> torch.Tensor:
        """Log emission matrix, shape ``(n_states, n_symbols)``."""
        return self._log_matrix

    @property
    def matrix(self) -> torch.Tensor:
        """Emission matrix as probabilities, shape ``(n_states, n_symbols)``."""
        return self._matrix

    def sample(
        self,
        states: np.ndarray,
        rng: np.random.Generator,
        covariate: ArrayLike | None = None,
    ) -> np.ndarray:
        """Draw one symbol per entry of ``states`` by inverse-CDF sampling."""
        refuse_covariate(self, covariate)
        return sample_rows(rng, self._matrix.numpy(), states)

    def log_density(
        self, observations: torch.Tensor, covariate: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Gather ``log B[:, symbol]`` for every observation."""
        refuse_covariate(self, covariate)
        return self._log_matrix.t()[observations]

    def bregman_divergence(self, observations: torch.Tensor) -> torch.Tensor:
        """``-log P(symbol | state)``: the best member puts all its mass on the symbol.

        The categorical's mean parameter is its probability vector, and the
        member matched to an observation is the point mass on that symbol,
        which scores ``0``. The divergence is the negative log probability
        itself --- the one family where the two rules agree.

        Returns
        -------
        torch.Tensor
            Shape ``(..., n_states)``.
        """
        return -self.log_density(observations)

    def validate(self, observations: np.ndarray) -> None:
        """Raise if a symbol lies outside the alphabet."""
        low, high = int(observations.min()), int(observations.max())
        if low < 0 or high >= self.n_symbols:
            msg = f"observations must lie in [0, {self.n_symbols}), got [{low}, {high}]"
            raise ValueError(msg)

    def reestimate(
        self,
        observations: ArrayLike,
        posterior: ArrayLike,
        covariate: ArrayLike | None = None,
    ) -> Reestimate[CategoricalEmission]:
        """Normalized expected symbol counts, in log space.

        Closed form, so the record it returns carries no convergence to
        report: this is the case the seam was designed against.
        """
        refuse_covariate(self, covariate)
        posterior = as_tensor(posterior)
        mask = torch.nn.functional.one_hot(
            as_tensor(observations, self.observation_dtype).reshape(-1).to(torch.long),
            self.n_symbols,
        ).to(posterior.dtype)
        weights = posterior.reshape(-1, self.n_states)
        counts = torch.log(weights.t() @ mask + torch.finfo(posterior.dtype).tiny)
        return Reestimate(
            CategoricalEmission.from_log(
                counts - torch.logsumexp(counts, dim=1, keepdim=True)
            )
        )

    def alignment_key(self) -> torch.Tensor:
        """The emission rows themselves, as probabilities."""
        return self.matrix

    def named_parameters(self) -> Mapping[str, torch.Tensor]:
        """``log_emission``, the form the forward recursion consumes."""
        return {"log_emission": self._log_matrix}


class GaussianEmission(EmissionFamily):
    """A real observation drawn from ``Normal(mean[state], scale[state])``.

    The family whose likelihood has **no maximum**. With ``mean[s]`` on a
    single observation and ``scale[s] -> 0`` the density diverges, so a
    Gaussian-emission fit that converges was stopped by its initialization or
    by a floor, and which one has to be knowable: the floor is explicit,
    derived from the data, and reaching it is a refusal rather than a clamp.

    **One state may emit several channels at once.** Give ``mean`` and
    ``scale`` shape ``(n_states, n_channels)`` and an observation carries a
    trailing channel axis, as :class:`CountPairEmission`'s does; the channels
    are independent given the state, so the log-densities add and every M
    step is the one-channel M step run per channel. A 1-D ``mean`` is the
    single-channel family, unchanged in every value it returns --- the
    channel axis is a widening, not a reparameterization (issue #548).

    Parameters
    ----------
    mean : Values
        Per-state mean, shape ``(n_states,)`` or ``(n_states, n_channels)``.
    scale : Values
        Per-state standard deviation, of ``mean``'s shape, strictly positive.
    variance_floor : float
        Variance at or below which :meth:`reestimate` refuses. Derive it from
        the data with :func:`pooled_variance_floor` rather than choosing a
        constant, so it scales with the fixture.

    Raises
    ------
    ValueError
        If the shapes disagree, either is neither 1- nor 2-D, a scale is not
        positive, or the floor is not positive.
    """

    def __init__(
        self,
        mean: Values,
        scale: Values,
        variance_floor: float,
    ) -> None:
        self._mean = _one_axis_at_least(torch.as_tensor(mean, dtype=torch.float64))
        self._scale = _one_axis_at_least(torch.as_tensor(scale, dtype=torch.float64))
        if self._mean.ndim > 2:
            msg = (
                f"mean is per state, or per state and channel; got shape "
                f"{tuple(self._mean.shape)}"
            )
            raise ValueError(msg)
        if self._mean.shape != self._scale.shape:
            msg = (
                f"mean and scale must have the same shape, got "
                f"{tuple(self._mean.shape)} and {tuple(self._scale.shape)}"
            )
            raise ValueError(msg)
        if bool((self._scale <= 0.0).any()):
            msg = f"every scale must be positive, got {self._scale.tolist()}"
            raise ParameterDomainError(msg)
        if variance_floor <= 0.0:
            msg = f"variance_floor must be positive, got {variance_floor}"
            raise ValueError(msg)
        self._variance_floor = variance_floor

    @property
    def n_states(self) -> int:
        """Hidden states this family emits from."""
        return int(self._mean.shape[0])

    @property
    def n_channels(self) -> int:
        """Entries an observation carries; ``1`` for the single-channel family."""
        return 1 if self._mean.ndim == 1 else int(self._mean.shape[1])

    @property
    def is_discrete(self) -> bool:
        """False: the support is the real line, so the evidence is a density."""
        return False

    @property
    def observation_dtype(self) -> torch.dtype:
        """Observations are real."""
        return torch.float64

    @property
    def mean(self) -> torch.Tensor:
        """Per-state mean, of the shape it was given."""
        return self._mean

    @property
    def scale(self) -> torch.Tensor:
        """Per-state standard deviation, of :attr:`mean`'s shape."""
        return self._scale

    @property
    def variance_floor(self) -> float:
        """Variance at or below which a re-estimate is refused."""
        return self._variance_floor

    def sample(
        self,
        states: np.ndarray,
        rng: np.random.Generator,
        covariate: ArrayLike | None = None,
    ) -> np.ndarray:
        """Draw one real observation per entry of ``states``.

        Returns
        -------
        np.ndarray
            Shape ``(n_draws,)`` for the single-channel family, and
            ``(n_draws, n_channels)`` otherwise --- the per-state row is
            indexed whole, so every channel of one draw is drawn together.
        """
        refuse_covariate(self, covariate)
        return np.asarray(
            rng.normal(
                loc=self._mean.numpy()[states], scale=self._scale.numpy()[states]
            )
        )

    def log_density(
        self, observations: torch.Tensor, covariate: torch.Tensor | None = None
    ) -> torch.Tensor:
        """The Normal log-density of every observation under every state.

        Unbounded above: as a scale shrinks with its mean on an observation,
        the value there grows without bound. That is the model, not a defect,
        and a test exhibits it.

        Parameters
        ----------
        observations : torch.Tensor
            Shape ``(...)`` for the single-channel family, ``(..., n_channels)``
            otherwise.

        Returns
        -------
        torch.Tensor
            Shape ``(..., n_states)``. The channels are independent given the
            state, so a multi-channel score is the sum over them.
        """
        refuse_covariate(self, covariate)
        values = observations.to(self._mean.dtype)
        if self._mean.ndim == 1:
            return _normal_log_density(values.unsqueeze(-1) - self._mean, self._scale)
        # Channel by channel, accumulating into one (..., n_states) array
        # rather than the (..., n_states, n_channels) one a single expression
        # would build: at the key rung that is 1.6 GB against 3.2 GB, and the
        # sum is over a length the loop unrolls anyway.
        scored = _normal_log_density(
            values[..., 0].unsqueeze(-1) - self._mean[:, 0], self._scale[:, 0]
        )
        for channel in range(1, self.n_channels):
            scored = scored + _normal_log_density(
                values[..., channel].unsqueeze(-1) - self._mean[:, channel],
                self._scale[:, channel],
            )
        return scored

    def bregman_divergence(self, observations: torch.Tensor) -> torch.Tensor:
        """``((y - mu) / scale) ** 2 / 2``, summed over the channels.

        A Gaussian of known scale has ``phi`` the squared norm over twice the
        variance, so its divergence is the squared Euclidean distance up to
        that positive factor, and D-squared sampling under it *is*
        :func:`sal.opt.mixture.kmeans_plus_plus`: normalizing
        cancels a factor shared by every candidate. The identity is what pins
        the rule exactly rather than approximately (issue #560).

        Parameters
        ----------
        observations : torch.Tensor
            Shape ``(...)`` for the single-channel family, ``(..., n_channels)``
            otherwise.

        Returns
        -------
        torch.Tensor
            Shape ``(..., n_states)``.
        """
        values = observations.to(self._mean.dtype)
        if self._mean.ndim == 1:
            return ((values.unsqueeze(-1) - self._mean) / self._scale) ** 2 / 2.0
        divergence = _half_squared_z(
            values[..., 0].unsqueeze(-1) - self._mean[:, 0], self._scale[:, 0]
        )
        for channel in range(1, self.n_channels):
            divergence = divergence + _half_squared_z(
                values[..., channel].unsqueeze(-1) - self._mean[:, channel],
                self._scale[:, channel],
            )
        return divergence

    def validate(self, observations: np.ndarray) -> None:
        """Raise if an observation is not finite, or carries the wrong channels.

        Raises
        ------
        ValueError
            If an observation is not finite, or the trailing axis is not the
            declared channel count.
        """
        if self._mean.ndim == 2 and (
            observations.ndim < 1 or observations.shape[-1] != self.n_channels
        ):
            msg = (
                f"an observation carries {self.n_channels} channels; got a "
                f"trailing axis of shape {tuple(observations.shape)}"
            )
            raise ValueError(msg)
        if not bool(np.isfinite(observations).all()):
            msg = "observations must be finite"
            raise ValueError(msg)

    def reestimate(
        self,
        observations: ArrayLike,
        posterior: ArrayLike,
        covariate: ArrayLike | None = None,
    ) -> Reestimate[GaussianEmission]:
        """Posterior-weighted mean and variance, in closed form.

        Raises
        ------
        ValueError
            If a state's re-estimated variance reaches
            :attr:`variance_floor`. Refusal rather than clamping: the
            likelihood is unbounded in that direction, so a clamped fit would
            report a point estimate at a degenerate optimum (issue #122).
        """
        refuse_covariate(self, covariate)
        posterior = as_tensor(posterior)
        weights = posterior.reshape(-1, self.n_states)
        mass = weights.sum(dim=0)
        values = (
            as_tensor(observations, self.observation_dtype)
            .reshape(-1, self.n_channels)
            .to(posterior.dtype)
        )
        # Channel by channel, so a channel's statistics never materialize the
        # (n_samples, n_states, n_channels) array their product would: at the
        # key rung that array is 3.2 GB against the 1.6 GB of one channel's.
        located = [
            _weighted_moments(weights, values[:, channel].unsqueeze(-1), mass)
            for channel in range(self.n_channels)
        ]
        if self._mean.ndim == 1:
            mean, variance = located[0]
        else:
            mean = torch.stack([moments[0] for moments in located], dim=1)
            variance = torch.stack([moments[1] for moments in located], dim=1)
        refuse_collapsed(variance, self._variance_floor)
        return Reestimate(
            GaussianEmission(mean, torch.sqrt(variance), self._variance_floor)
        )

    def alignment_key(self) -> torch.Tensor:
        """The per-state means, one row per state and one column per channel."""
        return self._mean.reshape(self.n_states, -1)

    def named_parameters(self) -> Mapping[str, torch.Tensor]:
        """``mean`` and ``scale``, the parameters the model is stated in."""
        return {"mean": self._mean, "scale": self._scale}


def refuse_collapsed(variance: torch.Tensor, floor: float) -> None:
    """Raise if a re-estimated variance is at or below ``floor``.

    Shared by :meth:`GaussianEmission.reestimate` and the streamed M step of
    :func:`sal.opt.hmm.baum_welch_family` (issue #997), so the
    two routes refuse the same fits in the same words.

    Raises
    ------
    ValueError
        If any entry of ``variance``, per state or per state and channel, is
        at or below ``floor``.
    """
    collapsed = variance <= floor
    if bool(collapsed.any()):
        states = (
            torch.nonzero(collapsed.reshape(variance.shape[0], -1).any(dim=1))
            .reshape(-1)
            .tolist()
        )
        msg = (
            f"state(s) {states} re-estimated to variance "
            f"{variance[collapsed].tolist()}, at or below the floor "
            f"{floor:.6g}: the Gaussian likelihood is "
            f"unbounded as a variance goes to zero, so this fit is "
            f"heading to a degenerate optimum rather than converging"
        )
        raise ValueError(msg)


def _one_axis_at_least(values: torch.Tensor) -> torch.Tensor:
    """``values`` with a state axis: a scalar becomes the one-state family."""
    return values.reshape(1) if values.ndim == 0 else values


def _normal_log_density(centred: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """``log N(y; mu, s)`` given ``y - mu`` and ``s``, broadcast over the states."""
    return (
        -0.5 * torch.log(torch.tensor(2.0 * torch.pi, dtype=scale.dtype))
        - torch.log(scale)
        - 0.5 * (centred / scale) ** 2
    )


def _half_squared_z(centred: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """``((y - mu) / s) ** 2 / 2``, the Gaussian's divergence in one channel."""
    return (centred / scale) ** 2 / 2.0


def _weighted_moments(
    weights: torch.Tensor, column: torch.Tensor, mass: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """One channel's posterior-weighted mean and variance, shape ``(n_states,)`` each."""
    mean = (weights * column).sum(dim=0) / mass
    return mean, (weights * (column - mean) ** 2).sum(dim=0) / mass


def pooled_variance_floor(observations: np.ndarray) -> float:
    """The variance below which a Gaussian state has collapsed onto a point.

    ``n`` observations spread over a pooled standard deviation ``s`` sit at a
    typical nearest-neighbour spacing ``s / n``, so a state explaining a
    *neighbourhood* of the data carries variance of at least that order, while
    a state collapsed onto a *single* observation carries variance heading to
    zero. The floor is ``s**2 / n**2``: the boundary between the two, derived
    from the data rather than chosen, so it transfers across fixture sizes
    instead of being retuned per fixture.

    Parameters
    ----------
    observations : np.ndarray
        Every observation the fit will see, of any shape.

    Returns
    -------
    float
        The floor, strictly positive.

    Raises
    ------
    ValueError
        If fewer than two observations are supplied, or they are all equal ---
        in both cases there is no scale to derive a floor from.
    """
    values = np.asarray(observations, dtype=np.float64).reshape(-1)
    if values.size < 2:
        msg = f"need at least 2 observations to derive a floor, got {values.size}"
        raise ValueError(msg)
    pooled = float(values.var(ddof=1))
    if pooled <= 0.0:
        msg = "observations have zero spread, so no variance floor follows"
        raise ValueError(msg)
    return pooled / float(values.size) ** COLLAPSE_EXPONENT
