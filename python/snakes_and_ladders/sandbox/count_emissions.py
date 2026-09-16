"""The count families as they stood before they took a covariate (issue #631).

Conserved, not replaced. :class:`~snakes_and_ladders.emissions.NegativeBinomialEmission`
and :class:`~snakes_and_ladders.emissions.BetaBinomialEmission` gain a
per-observation exposure and trial count, and #631 asks the new families to
reproduce these **bit for bit** where the covariate is constant --- an exposure
of one, a trial count that does not vary. Against a replaced implementation
that is a claim about a memory; against a conserved one it is two objects
scored on the same observations in the same test, and it keeps working as the
live families change, which is when it is worth having.

**The solvers and the bounds come too, and that is the point.**
``_solve_dispersion`` takes a Python ``float`` mean here and takes a tensor in
the live module once the exposure lands; ``identifiable_dispersion_bound`` is
derived at one mean and has no per-state value under a varying one. A copy that
imported either would track the very edit it exists to referee, so **27
definitions** are carried: everything the two families reach, closed
transitively over the live module's own top-level names.

Only :class:`~snakes_and_ladders.emissions.Reestimate` is imported rather than
carried. It is the container an M step reports in and has no mathematics to
drift.

**One edit was made and it is the only one.** Both classes drop
``EmissionFamily`` and ``CountEmissionFamily`` from their bases. A frozen copy
inheriting the live protocol would break every time the protocol moved --- which
it just did, to add the covariate --- and "unedited" would last one pull
request. ``tests/regression/sandbox/test_conserved_count_emissions.py`` hashes
the 27 definitions, so a second edit fails rather than being noticed.

Under ``CLAUDE.md``'s capability clause in this directory: superseded in what
it can express, not on a hot path. Nothing on the package's live path imports
from here, and ``tests/regression/test_sandbox.py`` asserts it.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import torch

from snakes_and_ladders.emissions import Reestimate

#: What a family accepts for a parameter vector. A plain list is admitted
#: because a fixture states its truth in one; the constructor converts, and
#: validates what it converted.
type Values = np.ndarray | torch.Tensor | Sequence[float] | Sequence[int]


#: How far below :func:`identifiable_dispersion_bound` the dispersion solve's
#: bracket reaches. Nine decades: the score diverges to ``+inf`` as ``r -> 0``,
#: so the lower end only has to put the root above it, and bisection pays for
#: the width in ``log2`` -- three extra iterations per three decades.
_DISPERSION_BRACKET_RATIO = 1e-9


#: How far a re-estimated binomial probability is held from 0 and 1. A
#: weighted mean of exactly 0 or exactly n makes ``log p`` or ``log(1 - p)``
#: infinite, and a state that took no weight produces one; the margin is
#: below any probability a fixture of a realistic size can resolve.
_PROBABILITY_MARGIN = 1e-12


#: How far below :func:`identifiable_concentration_bound` the beta-binomial
#: solve's bracket reaches, and the iteration cap each bisection runs under.
#: Nine decades halved 60 times leaves a bracket far below any tolerance a
#: caller states.
_CONCENTRATION_BRACKET_RATIO = 1e-9


_MAX_BISECTIONS = 60


class NegativeBinomialEmission:
    """A count drawn from ``NegativeBinomial(dispersion[state], mean[state])``.

    Counts are a third data shape, and modelling them as a symbol or a real
    loses the mean-variance relation that defines them: ``Var = mu + mu**2 /
    r``, with ``r`` setting how far above Poisson the dispersion sits.

    Carried as ``(r, mu)`` rather than the textbook ``(r, p)``: the mean
    profiles out of the M step in closed form there, and it is the
    parameterization the mean-variance relation is stated in.
    :meth:`from_probability` accepts the textbook form.

    **The family whose M step is an optimization.** ``r``'s posterior-weighted
    score involves ``digamma`` and has no analytic root, so the M step is a
    solve --- the case an interface validated only against formulas has to
    carry.

    **Its identifiability hazard is a *flat* likelihood, not an unbounded
    one.** Where the data are not measurably overdispersed the likelihood is
    nearly flat in ``r`` as ``r -> inf`` (the Poisson limit), so the maximum
    runs to the boundary. That is the opposite failure from
    :class:`GaussianEmission`'s and needs a ceiling rather than a floor;
    :func:`identifiable_dispersion_bound` derives it.

    Parameters
    ----------
    dispersion : Values
        Per-state ``r``, shape ``(n_states,)``, strictly positive. Larger is
        closer to Poisson.
    mean : Values
        Per-state ``mu``, shape ``(n_states,)``, strictly positive.

    Raises
    ------
    ValueError
        If the shapes disagree or a parameter is not positive.
    """

    def __init__(
        self,
        dispersion: Values,
        mean: Values,
    ) -> None:
        self._dispersion = torch.as_tensor(dispersion, dtype=torch.float64).reshape(-1)
        self._mean = torch.as_tensor(mean, dtype=torch.float64).reshape(-1)
        if self._dispersion.shape != self._mean.shape:
            msg = (
                f"dispersion and mean must have the same shape, got "
                f"{tuple(self._dispersion.shape)} and {tuple(self._mean.shape)}"
            )
            raise ValueError(msg)
        for name, values in (
            ("dispersion", self._dispersion),
            ("mean", self._mean),
        ):
            if bool((values <= 0.0).any()):
                msg = f"every {name} must be positive, got {values.tolist()}"
                raise ValueError(msg)

    @classmethod
    def from_probability(
        cls,
        dispersion: Values,
        probability: Values,
    ) -> NegativeBinomialEmission:
        """Build from the textbook ``(r, p)``, where ``mu = r (1 - p) / p``.

        Parameters
        ----------
        dispersion : Values
            Per-state ``r``, shape ``(n_states,)``.
        probability : Values
            Per-state success probability ``p`` in ``(0, 1]``, shape
            ``(n_states,)``.

        Returns
        -------
        NegativeBinomialEmission
            The same family in the ``(r, mu)`` parameterization.
        """
        r = torch.as_tensor(dispersion, dtype=torch.float64).reshape(-1)
        p = torch.as_tensor(probability, dtype=torch.float64).reshape(-1)
        return cls(r, r * (1.0 - p) / p)

    @property
    def n_states(self) -> int:
        """Hidden states this family emits from."""
        return int(self._mean.shape[0])

    @property
    def is_discrete(self) -> bool:
        """True: the support is the non-negative integers.

        Pinned beside :class:`GaussianEmission`, since the bound it restores
        --- ``log P(observations) <= 0`` --- is the one the Gaussian case gave
        up.
        """
        return True

    @property
    def observation_dtype(self) -> torch.dtype:
        """Floating point, though the support is integral.

        ``lgamma(y + r)`` needs ``y`` in the same type as ``r``, so counts are
        carried as floats and :meth:`validate` enforces their integrality
        rather than the dtype doing it.
        """
        return torch.float64

    @property
    def dispersion(self) -> torch.Tensor:
        """Per-state ``r``, shape ``(n_states,)``."""
        return self._dispersion

    @property
    def mean(self) -> torch.Tensor:
        """Per-state ``mu``, shape ``(n_states,)``."""
        return self._mean

    @property
    def variance(self) -> torch.Tensor:
        """``mu + mu**2 / r``, the relation that makes this a count model."""
        return self._mean + self._mean**2 / self._dispersion

    def sample(self, states: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Draw one count per entry of ``states``."""
        r = self._dispersion.numpy()[states]
        mu = self._mean.numpy()[states]
        return np.asarray(rng.negative_binomial(r, r / (r + mu)))

    def log_density(self, observations: torch.Tensor) -> torch.Tensor:
        """The negative binomial log-probability of every count under every state."""
        counts = observations.unsqueeze(-1).to(self._mean.dtype)
        total = self._dispersion + self._mean
        return (
            torch.lgamma(counts + self._dispersion)
            - torch.lgamma(self._dispersion)
            - torch.lgamma(counts + 1.0)
            + self._dispersion * torch.log(self._dispersion / total)
            + counts * torch.log(self._mean / total)
        )

    def bregman_divergence(self, observations: torch.Tensor) -> torch.Tensor:
        """``r log((r + mu) / (r + y)) + y log(y (r + mu) / (mu (r + y)))``.

        At fixed dispersion the negative binomial is an exponential family in
        its mean, and the ``lgamma`` terms of :meth:`log_density` are exactly
        the ``log b_phi(y)`` the divergence drops: they depend on the count
        and not on the state, so they cancel between the count's own member
        and the state's. Zero at ``mu = y``, and ``r log((r + mu) / r)`` at
        ``y = 0``.

        Returns
        -------
        torch.Tensor
            Shape ``(..., n_states)``.
        """
        counts = observations.unsqueeze(-1).to(self._mean.dtype)
        total = self._dispersion + self._mean
        at_count = self._dispersion + counts
        # `xlogy` rather than a product, so a count of zero contributes zero
        # rather than the `0 * -inf` a bare `log` gives there.
        return self._dispersion * torch.log(total / at_count) + torch.xlogy(
            counts, counts * total / (self._mean * at_count)
        )

    def validate(self, observations: np.ndarray) -> None:
        """Raise if an observation is not a non-negative integer."""
        _validate_counts(observations)

    def reestimate(
        self, observations: torch.Tensor, posterior: torch.Tensor
    ) -> Reestimate[NegativeBinomialEmission]:
        """The M step: a closed form for the mean, and a solve for the dispersion.

        With the mean profiled at its posterior-weighted value the score in
        ``r`` reduces to ``sum_t gamma[t] (digamma(y_t + r) - digamma(r)) + W
        log(r / (r + mu))``, which has no analytic root. Solved by **bisection
        on** ``log r``, not by Newton: from the method-of-moments start Newton
        overshoots and underflows to zero on a near-Poisson sample, arriving
        as a domain error. Bisection cannot leave its bracket, whose upper end
        is where "not identified" is detected rather than approached.

        Returns
        -------
        Reestimate
            The re-estimated family, whether a state's dispersion reached the
            bound the data can identify it over, the iterations taken, and the
            weighted score at the answer.
        """
        values = observations.reshape(-1).to(posterior.dtype)
        weights = posterior.reshape(-1, self.n_states)
        mass = weights.sum(dim=0)
        mean = (weights * values.unsqueeze(-1)).sum(dim=0) / mass

        dispersion = torch.empty_like(mean)
        boundary = False
        iterations = 0
        residual = 0.0
        for state in range(self.n_states):
            solved = _solve_dispersion(values, weights[:, state], float(mean[state]))
            dispersion[state] = solved.value
            boundary = boundary or solved.at_boundary
            iterations = max(iterations, solved.iterations)
            residual = max(residual, solved.residual)
        return Reestimate(
            NegativeBinomialEmission(dispersion, mean),
            at_boundary=boundary,
            iterations=iterations,
            residual=residual,
        )

    def alignment_key(self) -> torch.Tensor:
        """The per-state mean **and variance**, both in observation units.

        Two moments rather than :class:`GaussianEmission`'s one. A Gaussian
        scale is a nuisance parameter; the negative binomial's dispersion is
        the parameter the family exists for, so two states sharing a mean and
        differing in dispersion are different states and a mean-only signature
        would tie them. Both entries are in observation units, so summing
        their absolute differences is dimensionally consistent --- which
        ``(mu, r)`` would not be, ``r`` being a shape.
        """
        return torch.stack([self._mean, self.variance], dim=1)

    def named_parameters(self) -> Mapping[str, torch.Tensor]:
        """``dispersion`` and ``mean``, the parameters the model is stated in."""
        return {"dispersion": self._dispersion, "mean": self._mean}


class BetaBinomialEmission:
    """A count of successes in ``trials[state]`` attempts with a random rate.

    ``p`` drawn from ``Beta(a, b)`` per observation, then a binomial count:
    the **over**-dispersed counterpart of :class:`BinomialEmission` on the same
    bounded support, with
    ``Var = n p (1 - p) (n + a + b) / (1 + a + b)`` where ``p = a / (a + b)``.

    **The second family whose M step is an optimization, and a different
    one.** The negative binomial's is a one-dimensional root find; this is
    two-dimensional in ``(a, b)``, so a seam that took the first by accident
    does not take this one by accident.

    **Its hazard is the negative binomial's, one family over.** As
    ``a + b -> inf`` at fixed ``a / (a + b)`` the family approaches
    ``Binomial(n, p)``, so the concentration stops being identified and the
    maximum runs to the boundary. :func:`identifiable_concentration_bound`
    says where.

    Parameters
    ----------
    trials : Values
        Per-state ``n``, shape ``(n_states,)``, positive integers.
    alpha, beta : Values
        Per-state Beta parameters, shape ``(n_states,)``, strictly positive.

    Raises
    ------
    ValueError
        If the shapes disagree or a parameter is out of range.
    """

    def __init__(
        self,
        trials: Values,
        alpha: Values,
        beta: Values,
    ) -> None:
        self._trials = torch.as_tensor(trials, dtype=torch.float64).reshape(-1)
        self._alpha = torch.as_tensor(alpha, dtype=torch.float64).reshape(-1)
        self._beta = torch.as_tensor(beta, dtype=torch.float64).reshape(-1)
        if not self._trials.shape == self._alpha.shape == self._beta.shape:
            msg = (
                f"trials, alpha and beta must have the same shape, got "
                f"{tuple(self._trials.shape)}, {tuple(self._alpha.shape)} and "
                f"{tuple(self._beta.shape)}"
            )
            raise ValueError(msg)
        if bool(((self._trials < 1) | (self._trials != self._trials.floor())).any()):
            msg = f"every trial count must be a positive integer, got {self._trials.tolist()}"
            raise ValueError(msg)
        for name, values in (("alpha", self._alpha), ("beta", self._beta)):
            if bool((values <= 0.0).any()):
                msg = f"every {name} must be positive, got {values.tolist()}"
                raise ValueError(msg)

    @property
    def n_states(self) -> int:
        """Hidden states this family emits from."""
        return int(self._trials.shape[0])

    @property
    def is_discrete(self) -> bool:
        """True: the support is ``{0, ..., n}``."""
        return True

    @property
    def observation_dtype(self) -> torch.dtype:
        """Floating point, since ``lgamma`` consumes the counts."""
        return torch.float64

    @property
    def trials(self) -> torch.Tensor:
        """Per-state ``n``, shape ``(n_states,)``."""
        return self._trials

    @property
    def alpha(self) -> torch.Tensor:
        """Per-state ``a``, shape ``(n_states,)``."""
        return self._alpha

    @property
    def beta(self) -> torch.Tensor:
        """Per-state ``b``, shape ``(n_states,)``."""
        return self._beta

    @property
    def concentration(self) -> torch.Tensor:
        """``a + b``: how close this is to a binomial. Larger is closer."""
        return self._alpha + self._beta

    @property
    def mean(self) -> torch.Tensor:
        """``n a / (a + b)``."""
        return self._trials * self._alpha / self.concentration

    @property
    def variance(self) -> torch.Tensor:
        """``n p (1 - p) (n + a + b) / (1 + a + b)``, above the binomial's."""
        total = self.concentration
        rate = self._alpha / total
        return (
            self._trials * rate * (1.0 - rate) * (self._trials + total) / (1.0 + total)
        )

    def sample(self, states: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Draw a rate from the Beta, then a binomial count at that rate."""
        rate = rng.beta(self._alpha.numpy()[states], self._beta.numpy()[states])
        return np.asarray(
            rng.binomial(self._trials.numpy()[states].astype(np.int64), rate)
        )

    def log_density(self, observations: torch.Tensor) -> torch.Tensor:
        """``log C(n, y) + log B(y + a, n - y + b) - log B(a, b)``."""
        counts = observations.unsqueeze(-1).to(self._trials.dtype)
        return _beta_binomial_log_density(counts, self._trials, self._alpha, self._beta)

    def bregman_divergence(self, observations: torch.Tensor) -> torch.Tensor:
        """The log-density gap to the best rate at this concentration.

        :func:`_beta_binomial_saturated` says why this is a deviance rather
        than the divergence of a log-partition, and why it is non-negative.

        Returns
        -------
        torch.Tensor
            Shape ``(..., n_states)``.
        """
        counts = observations.unsqueeze(-1).to(self._trials.dtype)
        return _clamped_deviance(
            _beta_binomial_saturated(counts, self._trials, self.concentration),
            _beta_binomial_log_density(counts, self._trials, self._alpha, self._beta),
        )

    def validate(self, observations: np.ndarray) -> None:
        """Raise if an observation is not an integer in ``[0, max(trials)]``."""
        values = _validate_counts(observations)
        largest = float(self._trials.max())
        if bool((values > largest).any()):
            msg = (
                f"observations must not exceed the largest trial count "
                f"{largest:.0f}, got {values.max()}"
            )
            raise ValueError(msg)

    def reestimate(
        self, observations: torch.Tensor, posterior: torch.Tensor
    ) -> Reestimate[BetaBinomialEmission]:
        """Alternating bisection for ``(a, b)``; see :func:`_solve_beta_binomial`.

        Returns
        -------
        Reestimate
            The re-estimated family, whether a state's concentration reached
            the bound this data identifies it over, the iterations taken, and
            the relative change at the last step.
        """
        values = observations.reshape(-1).to(posterior.dtype)
        weights = posterior.reshape(-1, self.n_states)

        alpha = torch.empty(self.n_states, dtype=torch.float64)
        beta = torch.empty(self.n_states, dtype=torch.float64)
        boundary = False
        converged = True
        iterations = 0
        residual = 0.0
        for state in range(self.n_states):
            total = float(self._alpha[state] + self._beta[state])
            solved = _solve_beta_binomial(
                values,
                weights[:, state],
                float(self._trials[state]),
                float(self._alpha[state]) / total,
                total,
            )
            alpha[state] = solved.alpha
            beta[state] = solved.beta
            boundary = boundary or solved.at_boundary
            converged = converged and solved.converged
            iterations = max(iterations, solved.iterations)
            residual = max(residual, solved.residual)
        return Reestimate(
            BetaBinomialEmission(self._trials, alpha, beta),
            at_boundary=boundary,
            converged=converged,
            iterations=iterations,
            residual=residual,
        )

    def alignment_key(self) -> torch.Tensor:
        """The per-state mean and variance, both in observation units."""
        return torch.stack([self.mean, self.variance], dim=1)

    def named_parameters(self) -> Mapping[str, torch.Tensor]:
        """``alpha`` and ``beta``. The trial count is a constant."""
        return {"alpha": self._alpha, "beta": self._beta}


def _beta_binomial_log_density(
    counts: torch.Tensor,
    trials: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    """``log C(n, y) + log B(y + a, n - y + b) - log B(a, b)``, broadcast.

    Two families evaluate it: :class:`BetaBinomialEmission` with a trial
    count per state, :class:`CountPairEmission`'s joint form with one per
    observation. Every argument broadcasts, so the caller decides which.
    """
    total = alpha + beta
    return (
        torch.lgamma(trials + 1.0)
        - torch.lgamma(counts + 1.0)
        - torch.lgamma(trials - counts + 1.0)
        + torch.lgamma(counts + alpha)
        + torch.lgamma(trials - counts + beta)
        - torch.lgamma(trials + total)
        + torch.lgamma(total)
        - torch.lgamma(alpha)
        - torch.lgamma(beta)
    )


def _clamped_deviance(saturated: torch.Tensor, scored: torch.Tensor) -> torch.Tensor:
    """``saturated - scored``, floored at zero.

    The gap is non-negative by construction --- ``saturated`` maximizes over a
    family ``scored`` is a member of --- so the floor catches only the
    bisection's own rounding at a state that is itself the best member, where
    the difference of two equal numbers can land a few ulps below zero. A
    negative score is not a small error to D-squared sampling: it is a
    negative probability.
    """
    return torch.clamp(saturated - scored, min=0.0)


#: How far inside the unit interval the saturated rate is held: one
#: ``float64`` step, so that ``1 - rate`` is the step itself rather than zero.
_RATE_FLOOR = 2.0**-52


#: Bisection steps taken for the saturated rate. The bracket is the unit
#: interval, so this halves it past ``float64``'s resolution on a rate and the
#: root is found to the precision the search is carried in.
_SATURATION_STEPS = 60


def _rate_rise(shape: torch.Tensor, count: torch.Tensor) -> torch.Tensor:
    """``psi(count + shape) - psi(shape)``, falling in ``shape`` for a positive count."""
    return torch.digamma(count + shape) - torch.digamma(shape)


def _beta_binomial_saturated(
    counts: torch.Tensor, trials: torch.Tensor, concentration: torch.Tensor
) -> torch.Tensor:
    """The largest beta-binomial log-probability of ``counts``, at this concentration.

    ``log b_phi(y)`` for a family that has no log-partition: a beta-binomial
    at fixed concentration is a *compound* distribution and not an exponential
    family in its success count, so the divergence it seeds under is the
    **unit deviance** the Bregman divergence generalizes to --- the log-density
    gap to the best member of the family at the observation, which for an
    exponential family is the divergence of the log-partition exactly
    (McCullagh & Nelder, 1989, ch. 2). Non-negativity, which D-squared
    sampling needs, is by construction rather than by argument.

    The rate attaining it solves ``psi(y + c p) - psi(c p) = psi(n - y + c (1 -
    p)) - psi(c (1 - p))``, whose left side falls and whose right side rises in
    ``p``, so the difference has one sign change and bisection converges on it.
    At ``y = 0`` and ``y = n`` the root runs to the boundary, where the member
    puts all its mass on the observation and the value is ``0``. The rate is
    held one ``float64`` step inside the unit interval so that limit is
    reached rather than stepped past: at a rate of exactly ``1`` the beta
    parameter is ``0``, whose ``lgamma`` is infinite, and the difference of
    two infinities is ``nan``.

    Parameters
    ----------
    counts, trials, concentration : torch.Tensor
        The successes, the trials they came from, and ``a + b``. Every
        argument broadcasts, as :func:`_beta_binomial_log_density`'s do.

    Returns
    -------
    torch.Tensor
        The broadcast shape of the arguments.
    """
    # The broadcast shape, as an expression rather than as a shape: every
    # argument enters the bisection anyway, so multiplying by zero is the
    # cheapest way to give both brackets that shape.
    low = 0.0 * counts * trials * concentration
    high = low + 1.0
    for _ in range(_SATURATION_STEPS):
        rate = 0.5 * (low + high)
        # Strictly inside the bracket at every step, so neither `digamma`
        # argument reaches the pole at zero.
        rising = _rate_rise(concentration * rate, counts) - _rate_rise(
            concentration * (1.0 - rate), trials - counts
        )
        low = torch.where(rising > 0.0, rate, low)
        high = torch.where(rising > 0.0, high, rate)
    rate = torch.clamp(0.5 * (low + high), min=_RATE_FLOOR, max=1.0 - _RATE_FLOOR)
    return _beta_binomial_log_density(
        counts, trials, concentration * rate, concentration * (1.0 - rate)
    )


@dataclass(frozen=True)
class _SolvedBetaBinomial:
    """One state's ``(a, b)`` solve."""

    alpha: float
    beta: float
    at_boundary: bool
    converged: bool
    iterations: int
    residual: float


def _beta_binomial_rate_score(
    values: torch.Tensor,
    weights: torch.Tensor,
    trials: float | torch.Tensor,
    rate: float,
    concentration: float,
) -> float:
    """Score in the mean rate ``p`` at fixed concentration, divided by ``M``.

    ``trials`` is a scalar where the family fixes it and a tensor of the same
    shape as ``values`` where the observation supplies it
    (:class:`CountPairEmission`'s joint form); every term broadcasts either
    way.
    """
    alpha = torch.tensor(rate * concentration, dtype=values.dtype)
    beta = torch.tensor((1.0 - rate) * concentration, dtype=values.dtype)
    return float(
        (
            weights
            * (
                torch.digamma(values + alpha)
                - torch.digamma(alpha)
                - torch.digamma(trials - values + beta)
                + torch.digamma(beta)
            )
        ).sum()
    )


def _beta_binomial_concentration_score(
    values: torch.Tensor,
    weights: torch.Tensor,
    trials: float | torch.Tensor,
    rate: float,
    concentration: float,
) -> float:
    """Score in the concentration ``M = a + b`` at fixed mean rate."""
    total = torch.tensor(concentration, dtype=values.dtype)
    alpha = torch.tensor(rate * concentration, dtype=values.dtype)
    beta = torch.tensor((1.0 - rate) * concentration, dtype=values.dtype)
    return float(
        (
            weights
            * (
                rate * (torch.digamma(values + alpha) - torch.digamma(alpha))
                + (1.0 - rate)
                * (torch.digamma(trials - values + beta) - torch.digamma(beta))
                - (
                    torch.digamma(torch.as_tensor(trials, dtype=values.dtype) + total)
                    - torch.digamma(total)
                )
            )
        ).sum()
    )


def _rate_score_at(
    values: torch.Tensor,
    weights: torch.Tensor,
    trials: float | torch.Tensor,
    concentration: float,
) -> Callable[[float], float]:
    """The rate score as a function of the rate alone, at a held concentration."""

    def score(rate: float) -> float:
        return _beta_binomial_rate_score(values, weights, trials, rate, concentration)

    return score


def _concentration_score_at(
    values: torch.Tensor,
    weights: torch.Tensor,
    trials: float | torch.Tensor,
    rate: float,
) -> Callable[[float], float]:
    """The concentration score as a function of ``log M`` alone, at a held rate."""

    def score(log_concentration: float) -> float:
        return _beta_binomial_concentration_score(
            values, weights, trials, rate, math.exp(log_concentration)
        )

    return score


def _bisect(
    score: Callable[[float], float], low: float, high: float, tolerance: float
) -> float:
    """Root of a decreasing ``score`` bracketed by ``[low, high]``."""
    for _ in range(_MAX_BISECTIONS):
        if high - low <= tolerance:
            break
        middle = 0.5 * (low + high)
        if score(middle) > 0.0:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def _solve_beta_binomial(
    values: torch.Tensor,
    weights: torch.Tensor,
    trials: float | torch.Tensor,
    rate: float,
    concentration: float,
    *,
    tolerance: float = 1e-10,
    max_iterations: int = 60,
) -> _SolvedBetaBinomial:
    """Maximize the weighted likelihood in ``(p, M)`` by alternating bisection.

    Minka's fixed point for the Polya distribution was tried first and
    rejected on measurement: monotone but linearly convergent, at a
    concentration of 120 it was still moving in the third decimal place after
    500 iterations, returning an unconverged answer that looked like an
    estimate. Alternating bisection brackets the root instead of stepping
    toward it, so the failure mode becomes "the root is outside the bracket"
    --- a fact worth reporting.

    Each coordinate's score is decreasing in that coordinate over its bracket,
    with the mean rate bracketed by ``(0, 1)`` and the concentration by
    ``(0, bound]``, so each inner solve is unconditional.
    """
    bound = identifiable_concentration_bound(
        _effective_trials(trials, weights), float(weights.sum())
    )
    concentration = min(concentration, bound)
    at_boundary = False
    residual = float("inf")
    iterations = 0
    while iterations < max_iterations:
        iterations += 1
        previous = (rate, concentration)
        rate = _bisect(
            _rate_score_at(values, weights, trials, concentration),
            _PROBABILITY_MARGIN,
            1.0 - _PROBABILITY_MARGIN,
            tolerance,
        )
        if (
            _beta_binomial_concentration_score(values, weights, trials, rate, bound)
            > 0.0
        ):
            concentration, at_boundary = bound, True
        else:
            at_boundary = False
            low, high = (
                math.log(bound) + math.log(_CONCENTRATION_BRACKET_RATIO),
                math.log(bound),
            )
            concentration = math.exp(
                _bisect(
                    _concentration_score_at(values, weights, trials, rate),
                    low,
                    high,
                    tolerance,
                )
            )
        residual = max(
            abs(rate - previous[0]),
            abs(concentration - previous[1]) / concentration,
        )
        if residual <= tolerance:
            break
    return _SolvedBetaBinomial(
        alpha=rate * concentration,
        beta=(1.0 - rate) * concentration,
        at_boundary=at_boundary,
        converged=residual <= tolerance,
        iterations=iterations,
        residual=residual,
    )


def _effective_trials(trials: float | torch.Tensor, weights: torch.Tensor) -> float:
    """The one trial count :func:`identifiable_concentration_bound` is read at.

    A fixed trial count is itself. A per-observation one has no single value,
    so the bound is taken at the *posterior-weighted mean* depth: the bound
    scales as ``n - 1``, and those same weights set each observation's
    contribution to the concentration's score.
    """
    if isinstance(trials, torch.Tensor):
        return float((weights * trials).sum() / weights.sum())
    return trials


def identifiable_concentration_bound(trials: float, weight: float) -> float:
    """The concentration above which this much data cannot tell ``a + b`` from infinity.

    A beta-binomial exceeds its binomial variance by a factor
    ``(n + a + b) / (1 + a + b)``, which is ``1 + (n - 1) / (a + b)`` to
    leading order in large ``a + b``. The sampling noise on a variance from
    ``W`` observations is a relative ``sqrt(2 / W)``. Setting the excess equal
    to the noise gives ``a + b = (n - 1) sqrt(W / 2)``: the same construction
    as :func:`identifiable_dispersion_bound`, one family over.

    Parameters
    ----------
    trials : float
        The state's trial count, at least 2 --- at ``n = 1`` a beta-binomial
        *is* a Bernoulli whatever its concentration, so nothing identifies it.
    weight : float
        Total posterior weight on the state, its effective sample size.

    Returns
    -------
    float
        The bound, strictly positive.

    Raises
    ------
    ValueError
        If the trial count is below 2 or the weight is not positive.
    """
    if trials < 2.0 or weight <= 0.0:
        msg = f"trials must be >= 2 and weight positive, got {trials} and {weight}"
        raise ValueError(msg)
    return (trials - 1.0) * math.sqrt(weight / 2.0)


def _validate_counts(observations: np.ndarray) -> np.ndarray:
    """Raise unless every observation is a non-negative integer."""
    values = np.asarray(observations)
    if bool((values < 0).any()) or bool((values != np.floor(values)).any()):
        msg = (
            f"observations must be non-negative integers, got a minimum of "
            f"{values.min()} and "
            f"{int((values != np.floor(values)).sum())} non-integral value(s)"
        )
        raise ValueError(msg)
    return values


@dataclass(frozen=True)
class _SolvedDispersion:
    """One state's dispersion solve."""

    value: float
    at_boundary: bool
    iterations: int
    residual: float


def _weighted_dispersion_score(
    values: torch.Tensor, weights: torch.Tensor, dispersion: float, mean: float
) -> float:
    """The posterior-weighted score in ``r``, with the mean profiled out.

    ``sum_t w_t (digamma(y_t + r) - digamma(r)) + W log(r / (r + mu))``. The
    term in ``(mu - y_t) / (r + mu)`` that appears in the full derivative
    vanishes because ``mu`` is the weighted mean of ``y``, which is what
    "profiled out" buys.
    """
    r = torch.tensor(dispersion, dtype=values.dtype)
    return float(
        (weights * (torch.digamma(values + r) - torch.digamma(r))).sum()
        + weights.sum() * math.log(dispersion / (dispersion + mean))
    )


def _solve_dispersion(
    values: torch.Tensor,
    weights: torch.Tensor,
    mean: float,
    *,
    tolerance: float = 1e-12,
) -> _SolvedDispersion:
    """Maximize the weighted likelihood in ``r`` by bisection on ``log r``."""
    upper = identifiable_dispersion_bound(mean, float(weights.sum()))
    lower = upper * _DISPERSION_BRACKET_RATIO
    if _weighted_dispersion_score(values, weights, upper, mean) > 0.0:
        return _SolvedDispersion(upper, at_boundary=True, iterations=0, residual=0.0)
    if _weighted_dispersion_score(values, weights, lower, mean) < 0.0:
        return _SolvedDispersion(lower, at_boundary=True, iterations=0, residual=0.0)

    low, high = math.log(lower), math.log(upper)
    iterations = 0
    while high - low > tolerance:
        middle = 0.5 * (low + high)
        if _weighted_dispersion_score(values, weights, math.exp(middle), mean) > 0.0:
            low = middle
        else:
            high = middle
        iterations += 1
    dispersion = math.exp(0.5 * (low + high))
    return _SolvedDispersion(
        dispersion,
        at_boundary=False,
        iterations=iterations,
        residual=abs(_weighted_dispersion_score(values, weights, dispersion, mean))
        / float(weights.sum()),
    )


def identifiable_dispersion_bound(mean: float, weight: float) -> float:
    """The dispersion above which this much data cannot tell ``r`` from infinity.

    A negative binomial exceeds its Poisson variance by ``mu**2 / r``. The
    sampling noise on a variance estimated from ``W`` observations is about
    ``Var sqrt(2 / W)``, which near the Poisson limit is ``mu sqrt(2 / W)``.
    Setting the excess equal to the noise gives ``r = mu sqrt(W / 2)``: beyond
    it the overdispersion the model is *for* is smaller than the error on
    measuring it, and a maximum reported there is a bound rather than an
    estimate. Derived rather than fixed, so a constant cap cannot flag an
    identified fixture at one size and miss an unidentified one at another.

    Parameters
    ----------
    mean : float
        The state's posterior-weighted mean count.
    weight : float
        Total posterior weight on the state, its effective sample size.

    Returns
    -------
    float
        The bound, strictly positive.

    Raises
    ------
    ValueError
        If the mean or the weight is not positive.
    """
    if mean <= 0.0 or weight <= 0.0:
        msg = f"mean and weight must be positive, got {mean} and {weight}"
        raise ValueError(msg)
    return mean * math.sqrt(weight / 2.0)
