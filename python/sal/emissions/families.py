"""The categorical and Gaussian families, and the variance floor a Gaussian state collapses at.

A Gaussian emission's likelihood has no maximum (the package docstring states
why), so :class:`GaussianEmission` carries an explicit floor,
:func:`pooled_variance_floor` derives one from the data, and :class:`Collapse`
names what a re-estimate does with a state that reaches it.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum

import numpy as np
import torch
from numpy.typing import ArrayLike

from sal.emissions.base import (
    COLLAPSED_MASS,
    Domain,
    EmissionFamily,
    ParameterDomainError,
    Reestimate,
    Values,
    as_tensor,
    marked_states,
    refuse_covariate,
    require_parameter_names,
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

#: The variance floor, relative to ``max(1, |mean|)**2``, that
#: :func:`pooled_variance_floor` returns where the observations have zero
#: spread and ``s**2 / n**2`` would be zero (issue #1234). Such data carry no
#: scale, so the floor takes the location's: a scale of ``3.2%`` of the value,
#: or of one unit near zero, the ``1e-3`` the warm-up's regularized variance
#: adds for a flat coordinate (issue #1207). Positive, so the density at the
#: constant is finite, and wide against ``float64``'s relative spacing of
#: ``2.2e-16``, so a mean that rounds off the constant does not move it.
ZERO_SPREAD_FLOOR = 1e-3


class Collapse(StrEnum):
    """What a Gaussian re-estimate does with a collapsed state (issue #1160).

    A state has collapsed when its re-estimated variance is at or below the
    family's ``variance_floor``, in any channel, or when the E step left it
    less than :data:`~sal.emissions.base.COLLAPSED_MASS` posterior mass to
    estimate from. Every mode lists the state in
    :attr:`~sal.emissions.base.Reestimate.frozen`, so the fit says it is not
    a clean optimum whichever mode ran.
    """

    HOLD = "hold"
    """Keep the state's last mean and scale, as the count families hold an emptied state (issue #1136)."""

    CLAMP = "clamp"
    """Set the collapsed variances to the floor and re-estimate the mean; an emptied state's mean, undefined, is held."""

    REFUSE = "refuse"
    """Raise :class:`ValueError`, as every Gaussian fit did from issue #122 to #1160."""


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

    def parameter_domains(self) -> Mapping[str, Domain]:
        """``log_emission`` is a row of log-probabilities per state."""
        return {"log_emission": Domain.LOG_SIMPLEX}

    def with_parameters(self, named: Mapping[str, torch.Tensor]) -> CategoricalEmission:
        """The family at ``log_emission``, built by :meth:`from_log`.

        Handed this family's own log matrix, it keeps the probabilities stored
        beside it, so a draw is the original's bit for bit; ``exp`` of the log
        matrix moves the last bits a draw at a cell boundary reads.
        """
        require_parameter_names(self, named, ("log_emission",))
        log_matrix = named["log_emission"]
        if log_matrix is self._log_matrix:
            family = CategoricalEmission.__new__(CategoricalEmission)
            family._log_matrix, family._matrix = self._log_matrix, self._matrix
            return family
        return CategoricalEmission.from_log(log_matrix)


class GaussianEmission(EmissionFamily):
    """A real observation drawn from ``Normal(mean[state], scale[state])``.

    The family whose likelihood has **no maximum**. With ``mean[s]`` on a
    single observation and ``scale[s] -> 0`` the density diverges, so a
    Gaussian-emission fit that converges was stopped by its initialization or
    by a floor, and which one has to be knowable: the floor is explicit,
    derived from the data, and a state that reaches it is held, clamped or
    refused as ``on_collapse`` says, and in every case named in the
    re-estimate's ``frozen`` (issue #1160).

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
        Variance at or below which :meth:`reestimate` treats a state as
        collapsed. Derive it from the data with :func:`pooled_variance_floor`
        rather than choosing a constant, so it scales with the fixture.
    on_collapse : Collapse
        What :meth:`reestimate` does with a collapsed state;
        :attr:`Collapse.HOLD` by default. An attribute of the family rather
        than an argument of :meth:`reestimate`, as ``variance_floor`` is,
        because every EM driver calls ``reestimate`` through the
        :class:`~sal.emissions.base.EmissionFamily` protocol, and the compiled
        steps read it from the family they are handed.
    flat : tuple[int, ...]
        Channels whose observations have zero spread, as
        :func:`flat_channels` names them (issue #1234). Every state's variance
        there is zero, which is the data rather than a collapse: a re-estimate
        puts the scale at the floor and does not count the channel towards
        ``frozen``. Empty by default, which leaves every value unchanged.

    Raises
    ------
    ValueError
        If the shapes disagree, either is neither 1- nor 2-D, a scale is not
        positive, the floor is not positive, or a flat channel is out of range.
    """

    def __init__(
        self,
        mean: Values,
        scale: Values,
        variance_floor: float,
        *,
        on_collapse: Collapse = Collapse.HOLD,
        flat: tuple[int, ...] = (),
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
        self._on_collapse = Collapse(on_collapse)
        self._flat = tuple(int(channel) for channel in flat)
        if any(not 0 <= channel < self.n_channels for channel in self._flat):
            msg = f"flat channels {self._flat} out of range for {self.n_channels} channel(s)"
            raise ValueError(msg)

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
        """Variance at or below which a re-estimate treats a state as collapsed."""
        return self._variance_floor

    @property
    def on_collapse(self) -> Collapse:
        """What :meth:`reestimate` does with a collapsed state."""
        return self._on_collapse

    @property
    def flat(self) -> tuple[int, ...]:
        """Channels whose observations have zero spread, settled at the floor."""
        return self._flat

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

        A collapsed state --- variance at or below :attr:`variance_floor`, or
        posterior mass below :data:`~sal.emissions.base.COLLAPSED_MASS` --- is
        settled by :func:`settle_collapse` as :attr:`on_collapse` says and
        listed in ``frozen``.

        Raises
        ------
        ValueError
            Under :attr:`Collapse.REFUSE`, if a state collapsed: the
            likelihood is unbounded in that direction (issue #122).
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
        return settle_collapse(self, mass, mean, variance)

    def alignment_key(self) -> torch.Tensor:
        """The per-state means, one row per state and one column per channel."""
        return self._mean.reshape(self.n_states, -1)

    def named_parameters(self) -> Mapping[str, torch.Tensor]:
        """``mean`` and ``scale``, the parameters the model is stated in."""
        return {"mean": self._mean, "scale": self._scale}

    def parameter_domains(self) -> Mapping[str, Domain]:
        """``mean`` is real and ``scale`` positive."""
        return {"mean": Domain.REAL, "scale": Domain.POSITIVE}

    def with_parameters(self, named: Mapping[str, torch.Tensor]) -> GaussianEmission:
        """The family at ``mean`` and ``scale``, its variance floor, :attr:`on_collapse` and :attr:`flat` kept."""
        require_parameter_names(self, named, ("mean", "scale"))
        return GaussianEmission(
            named["mean"],
            named["scale"],
            self._variance_floor,
            on_collapse=self._on_collapse,
            flat=self._flat,
        )


def settle_collapse(
    family: GaussianEmission,
    mass: torch.Tensor,
    mean: torch.Tensor,
    variance: torch.Tensor,
) -> Reestimate[GaussianEmission]:
    """The re-estimate at ``mean`` and ``variance``, its collapsed states settled (issue #1160).

    Shared by :meth:`GaussianEmission.reestimate` and the compiled steps of
    :func:`sal.opt.mixture.expectation_maximization` and
    :func:`sal.opt.hmm.baum_welch_family`, so every route holds, clamps or
    refuses the same states to the same values.

    Parameters
    ----------
    family : GaussianEmission
        The family re-estimated from: its parameters are the ones a held
        state keeps, and its floor and ``on_collapse`` decide the rest.
    mass : torch.Tensor
        Per-state posterior mass, shape ``(n_states,)``.
    mean, variance : torch.Tensor
        The posterior-weighted moments, of ``family.mean``'s shape.

    A channel in ``family.flat`` is not a collapse: its variance is put at
    the floor and the rest of the rule runs on the other channels (issue
    #1234).

    Returns
    -------
    Reestimate[GaussianEmission]
        With ``frozen`` the collapsed states. With none, the family at
        ``mean`` and ``sqrt(variance)`` exactly as before issue #1160.

    Raises
    ------
    ValueError
        Under :attr:`Collapse.REFUSE`, if a state collapsed.
    """
    floor = family.variance_floor
    if family.flat:
        variance = _floored_flat(variance, family.flat, floor)
    narrow = variance <= floor
    if family.flat:
        narrow = _floored_flat(narrow, family.flat, False)
    emptied = mass < COLLAPSED_MASS
    collapsed = emptied | narrow.reshape(family.n_states, -1).any(dim=1)
    rebuilt = {"on_collapse": family.on_collapse}
    if not bool(collapsed.any()):
        return Reestimate(
            GaussianEmission(
                mean, torch.sqrt(variance), floor, **rebuilt, flat=family.flat
            )
        )
    if family.on_collapse is Collapse.REFUSE:
        refuse_collapsed(
            torch.where(narrow, variance, torch.inf) if family.flat else variance,
            floor,
            emptied=emptied,
        )
    row = collapsed if mean.ndim == 1 else collapsed.unsqueeze(-1)
    if family.on_collapse is Collapse.HOLD:
        held_mean = torch.where(row, family.mean, mean)
        held_scale = torch.where(row, family.scale, torch.sqrt(variance))
    else:
        undefined = emptied if mean.ndim == 1 else emptied.unsqueeze(-1)
        held_mean = torch.where(undefined, family.mean, mean)
        at_floor = narrow | undefined
        held_scale = torch.where(
            at_floor,
            torch.sqrt(torch.tensor(floor, dtype=variance.dtype)),
            torch.sqrt(variance),
        )
    return Reestimate(
        GaussianEmission(held_mean, held_scale, floor, **rebuilt, flat=family.flat),
        frozen=marked_states(collapsed),
    )


def _floored_flat(
    values: torch.Tensor, flat: tuple[int, ...], fill: float | bool
) -> torch.Tensor:
    """``values`` with the ``flat`` channels' entries set to ``fill``; one channel is the last axis or none."""
    filled = values.clone()
    if filled.ndim == 1:
        filled[:] = fill
    else:
        filled[:, list(flat)] = fill
    return filled


def refuse_collapsed(
    variance: torch.Tensor,
    floor: float,
    *,
    emptied: torch.Tensor | None = None,
) -> None:
    """Raise if a re-estimated variance is at or below ``floor``, or a state was emptied.

    The refusal :attr:`Collapse.REFUSE` asks for, in the words every route
    has raised since issue #122.

    Parameters
    ----------
    variance : torch.Tensor
        Shape ``(n_states,)`` or ``(n_states, n_channels)``.
    floor : float
        The family's variance floor.
    emptied : torch.Tensor | None
        Per-state flags, shape ``(n_states,)``: states left less than
        :data:`~sal.emissions.base.COLLAPSED_MASS` posterior mass.

    Raises
    ------
    ValueError
        If any entry of ``variance``, per state or per state and channel, is
        at or below ``floor``, or any state is ``emptied``.
    """
    narrow = variance <= floor
    per_state = narrow.reshape(variance.shape[0], -1).any(dim=1)
    if emptied is not None:
        per_state = per_state | emptied
    if bool(per_state.any()):
        reasons = []
        if bool(narrow.any()):
            reasons.append(
                f"state(s) {list(marked_states(narrow.reshape(variance.shape[0], -1).any(dim=1)))} "
                f"re-estimated to variance {variance[narrow].tolist()}, at or "
                f"below the floor {floor:.6g}"
            )
        if emptied is not None and bool(emptied.any()):
            reasons.append(
                f"state(s) {list(marked_states(emptied))} left posterior mass "
                f"below {COLLAPSED_MASS:g}"
            )
        msg = (
            f"{'; '.join(reasons)}: the Gaussian likelihood is "
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

    Observations that are all equal have no spread to derive it from, and the
    floor is :data:`ZERO_SPREAD_FLOOR` ``* max(1, |mean|)**2`` instead (issue
    #1234): reported through :func:`flat_channels` rather than refused, as
    the warm-up reports a flat coordinate (issue #1207).

    Raises
    ------
    ValueError
        If fewer than two observations are supplied: there is no sample to
        derive a floor from.
    """
    values = np.asarray(observations, dtype=np.float64).reshape(-1)
    if values.size < 2:
        msg = f"need at least 2 observations to derive a floor, got {values.size}"
        raise ValueError(msg)
    pooled = float(values.var(ddof=1))
    # Equal values by comparison, not by the variance: a mean that rounds off
    # the constant leaves a variance of order 1e-31 that is not a spread.
    if pooled <= 0.0 or values.max() == values.min():
        return ZERO_SPREAD_FLOOR * max(1.0, abs(float(values[0]))) ** 2
    return pooled / float(values.size) ** COLLAPSE_EXPONENT


def flat_channels(observations: np.ndarray) -> tuple[int, ...]:
    """The channels whose observations are all equal (issue #1234).

    A Gaussian fit's variance there is zero whatever the parameters, so a
    family is told them (:attr:`GaussianEmission.flat`) and settles them at
    the floor rather than calling them collapsed.

    Parameters
    ----------
    observations : np.ndarray
        Shape ``(n_samples,)``, one channel, or ``(n_samples, n_channels)``.

    Returns
    -------
    tuple[int, ...]
        Indices into the channel axis, ascending; ``(0,)`` or ``()`` for one
        channel.
    """
    values = np.asarray(observations, dtype=np.float64)
    columns = values.reshape(values.shape[0] if values.ndim else 1, -1)
    if columns.shape[0] == 0:
        return ()
    return tuple(
        int(channel)
        for channel in np.flatnonzero(columns.max(axis=0) == columns.min(axis=0))
    )
