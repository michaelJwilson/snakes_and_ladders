"""The count families: negative binomial, Poisson, binomial, beta-binomial and the two-channel count pair.

Each scores and draws here; the M-step solves that re-estimate a dispersion or
a concentration are in :mod:`snakes_and_ladders.emissions.mstep`, and the
route they run on is :data:`snakes_and_ladders.emissions.mstep.M_STEP_BACKEND`,
read at call time so that setting it takes effect.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch
from numpy.typing import ArrayLike

from snakes_and_ladders.backend import Backend
from snakes_and_ladders.emissions import mstep
from snakes_and_ladders.emissions.base import (
    CountEmissionFamily,
    EmissionFamily,
    ParameterDomainError,
    Reestimate,
    Values,
    as_tensor,
    exposure,
    refuse_covariate,
    trial_count,
    validated_exposure,
    validated_trials,
)
from snakes_and_ladders.emissions.mstep import (
    PROBABILITY_MARGIN,
    NoTails,
    solve_beta_binomial_m_step,
    solve_beta_binomial_tied,
    solve_dispersion,
    solve_dispersion_exposed_rust,
    solve_dispersion_m_step,
    solve_dispersion_tied,
)


def _check_tied(tied: bool, name: str, values: torch.Tensor, rtol: float = 0.0) -> None:
    """Refuse a tied family whose shared parameter differs across states.

    ``rtol`` admits the rounding of a parameter the family stores in parts:
    ``p M + (1 - p) M`` is ``M`` to a few units in the last place.
    """
    if tied and not bool(((values - values[0]).abs() <= rtol * values[0]).all()):
        msg = f"a tied {name} holds one value across states, got {values.tolist()}"
        raise ValueError(msg)


class NegativeBinomialEmission(EmissionFamily, CountEmissionFamily):
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
    tied : bool
        One dispersion shared by every state (issue #933): the M step solves
        the score summed over the states, and ``dispersion`` must hold one
        value. Keyword-only and off by default: it is a constraint on the
        model, not a solver setting.

    Raises
    ------
    ValueError
        If the shapes disagree, a parameter is not positive, or a tied
        dispersion differs across states.
    """

    def __init__(
        self,
        dispersion: Values,
        mean: Values,
        *,
        tied: bool = False,
    ) -> None:
        self._dispersion = torch.as_tensor(dispersion, dtype=torch.float64).reshape(-1)
        self._mean = torch.as_tensor(mean, dtype=torch.float64).reshape(-1)
        self._tied = tied
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
                raise ParameterDomainError(msg)
        _check_tied(tied, "dispersion", self._dispersion)

    @property
    def tied(self) -> bool:
        """Whether one dispersion is shared by every state."""
        return self._tied

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
        """Per-state ``mu``, shape ``(n_states,)``.

        **The state's own rate, and what recovery targets.** Under an exposure
        the *emission's* mean at observation ``i`` in state ``k`` is
        ``e_i mu_k``; ``mu_k`` is the per-state association that survives it,
        and is the parameter fitted and recovered. So this property keeps a
        value under a varying exposure rather than losing one (issue #631):
        the exposure is conditioned on, never fitted, and an exposure of one
        is the reference the family declares itself at.
        """
        return self._mean

    @property
    def variance(self) -> torch.Tensor:
        """``mu + mu**2 / r``, the relation that makes this a count model.

        At unit exposure, for the reason :attr:`mean` gives. At exposure
        ``e_i`` the emission's variance is ``e_i mu + (e_i mu)**2 / r``, which
        the relation still holds for, one rate over.
        """
        return self._mean + self._mean**2 / self._dispersion

    def sample(
        self,
        states: np.ndarray,
        rng: np.random.Generator,
        covariate: ArrayLike | None = None,
    ) -> np.ndarray:
        """Draw one count per entry of ``states``.

        ``covariate`` is the exposure per draw, carrying ``states``' own shape
        and a trailing singleton; the rate is ``e_i mu_k``. Without it every
        draw is at unit exposure.
        """
        r = self._dispersion.numpy()[states]
        rate = self._mean.numpy()[states]
        if covariate is not None:
            # Reshaped to the states, not flattened: `sample` draws one value
            # per entry of `states`, which a caller may hold in any shape ---
            # `(n_sequences, length)` for a fit over sequences. Flattening
            # assumed one dimension and broadcast against nothing at two
            # (issue #658).
            offsets = exposure(as_tensor(covariate), self._mean)
            rate = rate * offsets.reshape(states.shape).numpy()
        return np.asarray(rng.negative_binomial(r, r / (r + rate)))

    def log_density(
        self, observations: torch.Tensor, covariate: torch.Tensor | None = None
    ) -> torch.Tensor:
        """The negative binomial log-probability of every count under every state.

        ``covariate`` is the exposure per observation, shape ``(..., 1)`` so it
        broadcasts along the state axis; the rate scored at is ``e_i mu_k``.
        Where the exposure is constant the two models are the same model,
        absorbing it as ``log mu - log(c)``, which is what makes the conserved
        family a referee for this one (issue #631). A zero exposure marks the
        count unobserved: it scores log 1 under every state (issue #933).
        """
        counts = observations.unsqueeze(-1).to(self._mean.dtype)
        rate = self._mean
        unobserved = None
        if covariate is not None:
            offsets = exposure(covariate, self._mean)
            unobserved = offsets == 0.0
            if bool(unobserved.any()):
                # Scored at a unit exposure and then replaced, so neither
                # branch of the `where` holds the `0 * log 0` a zero rate
                # gives, which would reach a gradient as `nan`.
                offsets = torch.where(unobserved, 1.0, offsets)
            else:
                unobserved = None
            rate = offsets * self._mean
        total = self._dispersion + rate
        scores = (
            lgamma_shifted(counts, self._dispersion)
            - torch.lgamma(self._dispersion)
            - torch.lgamma(counts + 1.0)
            + self._dispersion * torch.log(self._dispersion / total)
            + counts * torch.log(rate / total)
        )
        if unobserved is None:
            return scores
        return torch.where(unobserved, torch.zeros_like(scores), scores)

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
        self,
        observations: ArrayLike,
        posterior: ArrayLike,
        covariate: ArrayLike | None = None,
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
        posterior = as_tensor(posterior)
        values = (
            as_tensor(observations, self.observation_dtype)
            .reshape(-1)
            .to(posterior.dtype)
        )
        weights = posterior.reshape(-1, self.n_states)
        offsets = (
            None
            if covariate is None
            else validated_exposure(as_tensor(covariate), self._mean).reshape(-1)
        )
        if offsets is not None and bool((offsets == 0.0).any()):
            # An unobserved total carries no information about the state:
            # the fit is the fit without it (issue #933).
            observed = offsets > 0.0
            values, weights, offsets = (
                values[observed],
                weights[observed],
                offsets[observed],
            )
        # Both sums are matmuls: the same numbers as
        # `(weights * v.unsqueeze(-1)).sum(0)` with no `(n_obs, n_states)`
        # temporary, BLAS-backed. At the coupled model's declared scale that
        # temporary is 80.7 MB per call. A matmul reduces in a different order,
        # so this costs the bitwise agreement with the conserved family and
        # lands a relative 4.0e-16 instead -- five orders inside the 1e-11 this
        # repository declares for a float64 comparison, which is the trade root
        # `CLAUDE.md` permits and this measurement is the price of (#649).
        mass = _weighted_mass(weights, offsets)
        mean = (weights.T @ values) / mass
        if self._tied:
            tied = solve_dispersion_tied(values, weights, mean, offsets)
            return Reestimate(
                NegativeBinomialEmission(
                    torch.full_like(mean, tied.value), mean, tied=True
                ),
                at_boundary=tied.at_boundary,
                iterations=tied.iterations,
                residual=tied.residual,
            )

        dispersion = torch.empty_like(mean)
        boundary = False
        iterations = 0
        residual = 0.0
        if offsets is None:
            # No exposure: one lockstep bisection over every state on the
            # distinct counts (issue #918), as the beta-binomial's (#892).
            batched = solve_dispersion_m_step(values, weights, [float(m) for m in mean])
            for state, solved in enumerate(batched):
                dispersion[state] = solved.value
            return Reestimate(
                NegativeBinomialEmission(dispersion, mean),
                at_boundary=any(one.at_boundary for one in batched),
                iterations=max(one.iterations for one in batched),
                residual=max(one.residual for one in batched),
            )
        # Under an exposure the rate differs per observation, so there is no
        # histogram to take for the log half. The compiled kernel sums the
        # digamma half on the tails and the rest per observation (issue #933);
        # the per-state torch solve below stays as its oracle.
        if mstep.M_STEP_BACKEND is Backend.RUST:
            try:
                exposed = solve_dispersion_exposed_rust(
                    values, weights, [float(m) for m in mean], offsets
                )
            except NoTails:
                exposed = None
            if exposed is not None:
                for state, solved in enumerate(exposed):
                    dispersion[state] = solved.value
                return Reestimate(
                    NegativeBinomialEmission(dispersion, mean),
                    at_boundary=any(one.at_boundary for one in exposed),
                    iterations=max(one.iterations for one in exposed),
                    residual=max(one.residual for one in exposed),
                )
        for state in range(self.n_states):
            # Formed once per state, outside the bisection: the solve is at
            # fixed `mu`, so `e * mu` does not move across its ~50 steps.
            rate = (
                float(mean[state]) if offsets is None else offsets * float(mean[state])
            )
            solved = solve_dispersion(values, weights[:, state], rate)
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


class PoissonEmission(EmissionFamily, CountEmissionFamily):
    """A count drawn from ``Poisson(mean[state])``: the equidispersed case.

    The family with no dispersion parameter. It is the ``r -> inf`` limit of
    :class:`NegativeBinomialEmission` and the ``n -> inf, p -> 0`` limit of
    :class:`BinomialEmission`, so it referees both against a second
    implementation rather than a hand-written expression.

    Parameters
    ----------
    mean : Values
        Per-state rate, shape ``(n_states,)``, strictly positive.

    Raises
    ------
    ValueError
        If a mean is not positive.
    """

    def __init__(self, mean: Values) -> None:
        self._mean = torch.as_tensor(mean, dtype=torch.float64).reshape(-1)
        if bool((self._mean <= 0.0).any()):
            msg = f"every mean must be positive, got {self._mean.tolist()}"
            raise ParameterDomainError(msg)

    @property
    def n_states(self) -> int:
        """Hidden states this family emits from."""
        return int(self._mean.shape[0])

    @property
    def is_discrete(self) -> bool:
        """True: the support is the non-negative integers."""
        return True

    @property
    def observation_dtype(self) -> torch.dtype:
        """Floating point, since ``lgamma`` consumes the counts."""
        return torch.float64

    @property
    def mean(self) -> torch.Tensor:
        """Per-state rate, shape ``(n_states,)``."""
        return self._mean

    @property
    def variance(self) -> torch.Tensor:
        """Equal to the mean. The defining property, and the one tested."""
        return self._mean

    def sample(
        self,
        states: np.ndarray,
        rng: np.random.Generator,
        covariate: ArrayLike | None = None,
    ) -> np.ndarray:
        """Draw one count per entry of ``states``."""
        refuse_covariate(self, covariate)
        return np.asarray(rng.poisson(self._mean.numpy()[states]))

    def log_density(
        self, observations: torch.Tensor, covariate: torch.Tensor | None = None
    ) -> torch.Tensor:
        """``y log(lambda) - lambda - log(y!)``."""
        refuse_covariate(self, covariate)
        counts = observations.unsqueeze(-1).to(self._mean.dtype)
        return counts * torch.log(self._mean) - self._mean - torch.lgamma(counts + 1.0)

    def bregman_divergence(self, observations: torch.Tensor) -> torch.Tensor:
        """``y log(y / lambda) - y + lambda``, the generalized I-divergence.

        The ``log(y!)`` of :meth:`log_density` is ``log b_phi(y)`` and drops
        out. Zero at ``lambda = y``, and ``lambda`` at ``y = 0``.

        Returns
        -------
        torch.Tensor
            Shape ``(..., n_states)``.
        """
        counts = observations.unsqueeze(-1).to(self._mean.dtype)
        return torch.xlogy(counts, counts / self._mean) - counts + self._mean

    def validate(self, observations: np.ndarray) -> None:
        """Raise if an observation is not a non-negative integer."""
        _validate_counts(observations)

    def reestimate(
        self,
        observations: ArrayLike,
        posterior: ArrayLike,
        covariate: ArrayLike | None = None,
    ) -> Reestimate[PoissonEmission]:
        """The posterior-weighted mean, in closed form."""
        refuse_covariate(self, covariate)
        posterior = as_tensor(posterior)
        values = (
            as_tensor(observations, self.observation_dtype)
            .reshape(-1)
            .to(posterior.dtype)
        )
        weights = posterior.reshape(-1, self.n_states)
        # `weights.T @ values` is the same number as the elementwise form
        # with no `(n_obs, n_states)` intermediate -- 80.7 MB at the declared
        # scale, where it is 8.55x faster, 54.6 ms to 6.4 ms. A matmul reduces
        # in a different order, so this lands a relative 1.3e-14 from the
        # elementwise fit: three orders inside the 1e-11 this repository
        # declares for a float64 comparison, which is the trade root
        # `CLAUDE.md` permits (#649, #651).
        mean = (weights.T @ values) / weights.sum(dim=0)
        return Reestimate(PoissonEmission(mean))

    def alignment_key(self) -> torch.Tensor:
        """The per-state rate, as a column: it is the whole family."""
        return self._mean.reshape(-1, 1)

    def named_parameters(self) -> Mapping[str, torch.Tensor]:
        """``mean``, the only parameter there is."""
        return {"mean": self._mean}


class BinomialEmission(EmissionFamily, CountEmissionFamily):
    """A count of successes in ``trials[state]`` attempts, each with ``p[state]``.

    The **under**-dispersed case: ``Var = n p (1 - p) < n p``. Every other
    count family here sits at or above equidispersion, so this one says
    whether the interface quietly assumes otherwise.

    The trial count is a declared constant per state, not a parameter: it is a
    property of how an observation was made, and estimating it from the data
    is a different problem.

    Parameters
    ----------
    trials : Values
        Per-state ``n``, shape ``(n_states,)``, positive integers.
    probability : Values
        Per-state ``p``, shape ``(n_states,)``, in ``(0, 1)``.

    Raises
    ------
    ValueError
        If the shapes disagree, a trial count is not a positive integer, or a
        probability is not strictly inside ``(0, 1)``.
    """

    def __init__(
        self,
        trials: Values,
        probability: Values,
    ) -> None:
        self._trials = torch.as_tensor(trials, dtype=torch.float64).reshape(-1)
        self._probability = torch.as_tensor(probability, dtype=torch.float64).reshape(
            -1
        )
        if self._trials.shape != self._probability.shape:
            msg = (
                f"trials and probability must have the same shape, got "
                f"{tuple(self._trials.shape)} and "
                f"{tuple(self._probability.shape)}"
            )
            raise ValueError(msg)
        if bool(((self._trials < 1) | (self._trials != self._trials.floor())).any()):
            msg = f"every trial count must be a positive integer, got {self._trials.tolist()}"
            raise ValueError(msg)
        if bool(((self._probability <= 0.0) | (self._probability >= 1.0)).any()):
            msg = (
                f"every probability must lie strictly in (0, 1), got "
                f"{self._probability.tolist()}"
            )
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
    def probability(self) -> torch.Tensor:
        """Per-state ``p``, shape ``(n_states,)``."""
        return self._probability

    @property
    def mean(self) -> torch.Tensor:
        """``n p``."""
        return self._trials * self._probability

    @property
    def variance(self) -> torch.Tensor:
        """``n p (1 - p)``, strictly below the mean."""
        return self._trials * self._probability * (1.0 - self._probability)

    def sample(
        self,
        states: np.ndarray,
        rng: np.random.Generator,
        covariate: ArrayLike | None = None,
    ) -> np.ndarray:
        """Draw one count per entry of ``states``."""
        refuse_covariate(self, covariate)
        return np.asarray(
            rng.binomial(
                self._trials.numpy()[states].astype(np.int64),
                self._probability.numpy()[states],
            )
        )

    def log_density(
        self, observations: torch.Tensor, covariate: torch.Tensor | None = None
    ) -> torch.Tensor:
        """``log C(n, y) + y log p + (n - y) log(1 - p)``."""
        refuse_covariate(self, covariate)
        counts = observations.unsqueeze(-1).to(self._trials.dtype)
        return (
            torch.lgamma(self._trials + 1.0)
            - torch.lgamma(counts + 1.0)
            - torch.lgamma(self._trials - counts + 1.0)
            + counts * torch.log(self._probability)
            + (self._trials - counts) * torch.log1p(-self._probability)
        )

    def bregman_divergence(self, observations: torch.Tensor) -> torch.Tensor:
        """``n`` times the Kullback-Leibler divergence of ``y / n`` from ``p``.

        ``y log(y / (n p)) + (n - y) log((n - y) / (n (1 - p)))``: the
        binomial coefficient is ``log b_phi(y)`` and drops out. Zero at
        ``p = y / n``, and unbounded as ``p`` leaves the support's interior.

        Returns
        -------
        torch.Tensor
            Shape ``(..., n_states)``.
        """
        counts = observations.unsqueeze(-1).to(self._trials.dtype)
        remaining = self._trials - counts
        return torch.xlogy(
            counts, counts / (self._trials * self._probability)
        ) + torch.xlogy(
            remaining, remaining / (self._trials * (1.0 - self._probability))
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
        self,
        observations: ArrayLike,
        posterior: ArrayLike,
        covariate: ArrayLike | None = None,
    ) -> Reestimate[BinomialEmission]:
        """``p = weighted mean / n``, in closed form; ``n`` is not estimated."""
        refuse_covariate(self, covariate)
        posterior = as_tensor(posterior)
        values = (
            as_tensor(observations, self.observation_dtype)
            .reshape(-1)
            .to(posterior.dtype)
        )
        weights = posterior.reshape(-1, self.n_states)
        # `weights.T @ values` is the same number as the elementwise form
        # with no `(n_obs, n_states)` intermediate -- 80.7 MB at the declared
        # scale, where it is 8.55x faster, 54.6 ms to 6.4 ms. A matmul reduces
        # in a different order, so this lands a relative 1.3e-14 from the
        # elementwise fit: three orders inside the 1e-11 this repository
        # declares for a float64 comparison, which is the trade root
        # `CLAUDE.md` permits (#649, #651).
        mean = (weights.T @ values) / weights.sum(dim=0)
        probability = (mean / self._trials).clamp(
            PROBABILITY_MARGIN, 1.0 - PROBABILITY_MARGIN
        )
        return Reestimate(BinomialEmission(self._trials, probability))

    def alignment_key(self) -> torch.Tensor:
        """The per-state mean and variance, both in observation units."""
        return torch.stack([self.mean, self.variance], dim=1)

    def named_parameters(self) -> Mapping[str, torch.Tensor]:
        """``probability``. The trial count is a constant, not a fitted value."""
        return {"probability": self._probability}


class BetaBinomialEmission(EmissionFamily, CountEmissionFamily):
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
    tied : bool
        One concentration ``a + b`` shared by every state, each keeping its
        own rate ``a / (a + b)`` (issue #933). Keyword-only, off by default.

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
        *,
        tied: bool = False,
    ) -> None:
        self._trials = torch.as_tensor(trials, dtype=torch.float64).reshape(-1)
        self._tied = tied
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
                raise ParameterDomainError(msg)
        _check_tied(tied, "concentration", self._alpha + self._beta, rtol=1e-12)

    @property
    def tied(self) -> bool:
        """Whether one concentration is shared by every state."""
        return self._tied

    @property
    def n_states(self) -> int:
        """Hidden states this family emits from.

        Read off ``alpha``, not ``trials``: the trial count may be supplied per
        observation and says nothing about how many states there are (#631).
        """
        return int(self._alpha.shape[0])

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

    def sample(
        self,
        states: np.ndarray,
        rng: np.random.Generator,
        covariate: ArrayLike | None = None,
    ) -> np.ndarray:
        """Draw a rate from the Beta, then a binomial count at that rate.

        ``covariate`` supplies one trial count per draw, shape ``(n_draws, 1)``;
        without it each draw takes its emitting state's declared count.
        """
        rate = rng.beta(self._alpha.numpy()[states], self._beta.numpy()[states])
        counts = (
            self._trials.numpy()[states]
            if covariate is None
            # Reshaped to the states, for the reason `NegativeBinomialEmission`
            # gives at its own draw (issue #658).
            else trial_count(as_tensor(covariate), self._trials)
            .reshape(states.shape)
            .numpy()
        )
        return np.asarray(rng.binomial(counts.astype(np.int64), rate))

    def log_density(
        self, observations: torch.Tensor, covariate: torch.Tensor | None = None
    ) -> torch.Tensor:
        """``log C(n, y) + log B(y + a, n - y + b) - log B(a, b)``.

        ``covariate`` is the trial count per observation, shape ``(..., 1)``;
        see :func:`trial_count` for why that axis is not optional. An
        observation above its own trial count scores ``-inf`` rather than
        raising, on the joint form's precedent: the support is a property of
        the pair, and a sequence carrying one impossible site is scored, not
        refused. A trial count of zero marks the successes unobserved: they
        score log 1 under every state, whatever count they carry (issue #933).
        """
        trials = trial_count(covariate, self._trials)
        counts = observations.unsqueeze(-1).to(self._trials.dtype)
        unobserved = trials == 0.0 if covariate is not None else None
        if unobserved is not None and bool(unobserved.any()):
            # An unobserved channel scores log 1 whatever count it carries
            # (issue #933); scored at zero successes first, so the discarded
            # branch is finite.
            counts = torch.where(unobserved, 0.0, counts)
        else:
            unobserved = None
        scores = _beta_binomial_log_density(counts, trials, self._alpha, self._beta)
        scores = torch.where(
            counts > trials, torch.full_like(scores, -torch.inf), scores
        )
        if unobserved is None:
            return scores
        return torch.where(unobserved, torch.zeros_like(scores), scores)

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
        self,
        observations: ArrayLike,
        posterior: ArrayLike,
        covariate: ArrayLike | None = None,
    ) -> Reestimate[BetaBinomialEmission]:
        """Alternating bisection for ``(a, b)``; see :func:`solve_beta_binomial`.

        Returns
        -------
        Reestimate
            The re-estimated family, whether a state's concentration reached
            the bound this data identifies it over, the iterations taken, and
            the relative change at the last step.
        """
        per_observation = covariate is not None
        posterior = as_tensor(posterior)
        values = (
            as_tensor(observations, self.observation_dtype)
            .reshape(-1)
            .to(posterior.dtype)
        )
        weights = posterior.reshape(-1, self.n_states)
        supplied = (
            validated_trials(as_tensor(covariate), self._trials).reshape(-1)
            if covariate is not None
            else self._trials
        )
        if per_observation and bool((supplied == 0.0).any()):
            # Dropped, as the negative binomial drops an unobserved total.
            observed = supplied > 0.0
            values, weights, supplied = (
                values[observed],
                weights[observed],
                supplied[observed],
            )

        alpha = torch.empty(self.n_states, dtype=torch.float64)
        beta = torch.empty(self.n_states, dtype=torch.float64)
        boundary = False
        converged = True
        iterations = 0
        residual = 0.0
        totals = [
            float(self._alpha[state] + self._beta[state])
            for state in range(self.n_states)
        ]
        if self._tied:
            tied = solve_beta_binomial_tied(
                values,
                weights,
                supplied
                if per_observation
                else [float(self._trials[state]) for state in range(self.n_states)],
                [
                    float(self._alpha[state]) / totals[state]
                    for state in range(self.n_states)
                ],
                totals[0],
            )
            return Reestimate(
                BetaBinomialEmission(self._trials, tied.alpha, tied.beta, tied=True),
                at_boundary=tied.at_boundary,
                converged=tied.converged,
                iterations=tied.iterations,
                residual=tied.residual,
            )
        batch = solve_beta_binomial_m_step(
            values,
            weights,
            supplied
            if per_observation
            else [float(self._trials[state]) for state in range(self.n_states)],
            [
                float(self._alpha[state]) / totals[state]
                for state in range(self.n_states)
            ],
            totals,
        )
        for state, solved in enumerate(batch):
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
        """``alpha`` and ``beta``. The trial count is conditioned on, never fitted."""
        return {"alpha": self._alpha, "beta": self._beta}


class CountPairEmission(EmissionFamily, CountEmissionFamily):
    """A total count and the successes within it, as one observation.

    The emission a coverage-and-allele-count assay produces: a depth ``n``
    from a negative binomial, and an allele count ``y`` from a beta-binomial
    over that depth. An observation is a pair in a trailing axis of length
    :data:`N_CHANNELS`, channel ``0`` the total and channel ``1`` the
    successes.

    **The two forms are different models, and neither is a default.** In the
    *independent* form the beta-binomial's trial count is a fixed parameter
    and the two log-densities are added, so ``n`` and ``y`` are independent
    given the state. In the *joint* form the trial count **is** the drawn
    total, so ``y | n`` is a beta-binomial over ``n`` and the channels are
    coupled: a deeper site carries a proportionally larger allele count. Which
    of the two a dataset came from is a claim about the assay, not a knob, so
    ``joint`` is keyword-only with no default --- a silent default here would
    pick a generative model on a caller's behalf (``CLAUDE.md``, Code
    Standards).

    A success count above the trial count is outside the support, and is
    scored at ``-inf`` rather than at the ``nan`` ``lgamma`` returns at a
    negative argument: a fit whose likelihood is ``nan`` is not a fit that
    lost.

    Parameters
    ----------
    dispersion, mean : Values
        The total channel's negative-binomial ``r`` and ``mu``, shape
        ``(n_states,)``.
    alpha, beta : Values
        The success channel's Beta parameters, shape ``(n_states,)``.
    trials : Values | None
        The independent form's fixed trial count per state, shape
        ``(n_states,)``, positive integers. ``None`` in the joint form, where
        the trial count is the observed total and a fixed one would be an
        unused parameter a reader would have to work out was ignored.
    joint : bool
        Whether the success channel's trials are the drawn total.

    Raises
    ------
    ValueError
        If the shapes disagree, a parameter is out of range, or ``trials`` is
        given in the joint form or omitted in the independent one.
    """

    #: Entries an observation carries: the total, then the successes within it.
    N_CHANNELS = 2

    def __init__(
        self,
        dispersion: Values,
        mean: Values,
        alpha: Values,
        beta: Values,
        trials: Values | None = None,
        *,
        joint: bool,
    ) -> None:
        if joint and trials is not None:
            msg = (
                "the joint form's trial count is the observed total; pass "
                "trials=None, or joint=False to fix it"
            )
            raise ValueError(msg)
        if not joint and trials is None:
            msg = "the independent form needs a fixed trial count per state"
            raise ValueError(msg)
        self._joint = joint
        self._total = NegativeBinomialEmission(dispersion, mean)
        self._alpha = torch.as_tensor(alpha, dtype=torch.float64).reshape(-1)
        self._beta = torch.as_tensor(beta, dtype=torch.float64).reshape(-1)
        # Built even in the joint form's absence, so the shape and positivity
        # checks are one implementation: a placeholder trial count of 1 is
        # never read, since the joint form scores against the observed total.
        placeholder = torch.ones_like(self._alpha)
        self._success = BetaBinomialEmission(
            placeholder if trials is None else trials, self._alpha, self._beta
        )
        if self._success.n_states != self._total.n_states:
            msg = (
                f"the two channels must have the same number of states, got "
                f"{self._total.n_states} and {self._success.n_states}"
            )
            raise ValueError(msg)

    @property
    def joint(self) -> bool:
        """Whether the success channel's trials are the drawn total."""
        return self._joint

    @property
    def n_states(self) -> int:
        """Hidden states this family emits from."""
        return self._total.n_states

    @property
    def is_discrete(self) -> bool:
        """True: the support is a pair of non-negative integers."""
        return True

    @property
    def observation_dtype(self) -> torch.dtype:
        """Floating point, as both channels' ``lgamma`` terms need."""
        return torch.float64

    @property
    def total(self) -> NegativeBinomialEmission:
        """The total channel, as the family it is."""
        return self._total

    @property
    def alpha(self) -> torch.Tensor:
        """The success channel's ``a``, shape ``(n_states,)``."""
        return self._alpha

    @property
    def beta(self) -> torch.Tensor:
        """The success channel's ``b``, shape ``(n_states,)``."""
        return self._beta

    @property
    def concentration(self) -> torch.Tensor:
        """``a + b``: how close the success channel is to a binomial."""
        return self._alpha + self._beta

    @property
    def rate(self) -> torch.Tensor:
        """``a / (a + b)``, the expected success fraction of a trial."""
        return self._alpha / self.concentration

    @property
    def trials(self) -> torch.Tensor | None:
        """The independent form's fixed trial count; ``None`` in the joint form."""
        return None if self._joint else self._success.trials

    @property
    def mean(self) -> torch.Tensor:
        """Per-state mean of each channel, shape ``(n_states, 2)``.

        The joint form's success mean is ``mu p``, the total's mean scaled by
        the rate, since ``E[y] = E[E[y | n]] = p E[n]``.
        """
        if not self._joint:
            return torch.stack([self._total.mean, self._success.mean], dim=1)
        return torch.stack([self._total.mean, self._total.mean * self.rate], dim=1)

    @property
    def variance(self) -> torch.Tensor:
        """Per-state variance of each channel, shape ``(n_states, 2)``.

        The joint form's success variance is the law of total variance over
        the drawn total: ``E[Var(y | n)] + Var(E[y | n])``, which is
        ``p (1 - p) (E[n**2] + M E[n]) / (1 + M) + p**2 Var(n)`` with
        ``M = a + b``. Both terms matter --- dropping the second understates
        the spread by the whole contribution of the varying depth.
        """
        if not self._joint:
            return torch.stack([self._total.variance, self._success.variance], dim=1)
        rate = self.rate
        total = self.concentration
        depth_mean = self._total.mean
        depth_variance = self._total.variance
        second = depth_variance + depth_mean**2
        success = (
            rate * (1.0 - rate) * (second + total * depth_mean) / (1.0 + total)
            + rate**2 * depth_variance
        )
        return torch.stack([depth_variance, success], dim=1)

    def sample(
        self,
        states: np.ndarray,
        rng: np.random.Generator,
        covariate: ArrayLike | None = None,
    ) -> np.ndarray:
        """Draw one ``(total, successes)`` pair per entry of ``states``.

        The total is drawn first and the successes second, so the two forms
        consume the generator in the same order and a fixture that switches
        form changes the data it is about rather than every draw in it.

        Returns
        -------
        np.ndarray
            Shape ``(n_draws, 2)``.
        """
        refuse_covariate(self, covariate)
        totals = self._total.sample(states, rng)
        if self._joint:
            rate = rng.beta(self._alpha.numpy()[states], self._beta.numpy()[states])
            successes = np.asarray(rng.binomial(totals.astype(np.int64), rate))
        else:
            successes = self._success.sample(states, rng)
        return np.stack([totals, successes], axis=-1)

    def log_density(
        self, observations: torch.Tensor, covariate: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Score every pair under every state.

        Parameters
        ----------
        observations : torch.Tensor
            Shape ``(..., 2)``.

        Returns
        -------
        torch.Tensor
            Shape ``(..., n_states)``.
        """
        refuse_covariate(self, covariate)
        values = observations.to(self._alpha.dtype)
        totals = values[..., 0].unsqueeze(-1)
        successes = values[..., 1].unsqueeze(-1)
        trials = totals if self._joint else self._success.trials
        # `lgamma` of the negative argument an unsupported pair produces is
        # `nan`, which propagates through the sum, so the pair is scored at
        # `-inf` instead. The trial count is raised to the success count
        # *inside* the density so the discarded branch is finite:
        # `torch.where` multiplies the branch it did not take by zero, and
        # `0 * nan` is `nan`, so a `nan` there reaches the gradient even
        # though it never reaches the value.
        supported = successes <= trials
        scored = self._total.log_density(values[..., 0]) + _beta_binomial_log_density(
            successes, torch.maximum(trials, successes), self._alpha, self._beta
        )
        return torch.where(supported, scored, torch.full_like(scored, -float("inf")))

    def bregman_divergence(self, observations: torch.Tensor) -> torch.Tensor:
        """The two channels' divergences, summed as their log-densities are.

        The channels are independent given the state, so the pair's divergence
        is the total's negative-binomial divergence and the successes'
        beta-binomial deviance added --- the same decomposition
        :meth:`log_density` makes, with the same ``-inf`` on an unsupported
        pair carried through as ``inf``, the divergence a pair no member of
        the family can produce is at.

        Parameters
        ----------
        observations : torch.Tensor
            Shape ``(..., 2)``.

        Returns
        -------
        torch.Tensor
            Shape ``(..., n_states)``.
        """
        values = observations.to(self._alpha.dtype)
        totals = values[..., 0].unsqueeze(-1)
        successes = values[..., 1].unsqueeze(-1)
        trials = totals if self._joint else self._success.trials
        supported = successes <= trials
        raised = torch.maximum(trials, successes)
        divergence = self._total.bregman_divergence(values[..., 0]) + _clamped_deviance(
            _beta_binomial_saturated(successes, raised, self.concentration),
            _beta_binomial_log_density(successes, raised, self._alpha, self._beta),
        )
        return torch.where(
            supported, divergence, torch.full_like(divergence, float("inf"))
        )

    def validate(self, observations: np.ndarray) -> None:
        """Raise if the pairs are not counts, or a success count is impossible.

        Raises
        ------
        ValueError
            If the trailing axis is not of length 2, an entry is not a
            non-negative integer, or a success count exceeds the total (joint
            form) or the state's largest fixed trial count (independent form).
        """
        values = np.asarray(observations)
        if values.ndim < 1 or values.shape[-1] != self.N_CHANNELS:
            msg = (
                f"a count pair has {self.N_CHANNELS} channels; got a trailing "
                f"axis of shape {tuple(values.shape)}"
            )
            raise ValueError(msg)
        _validate_counts(values)
        if self._joint:
            excess = int((values[..., 1] > values[..., 0]).sum())
            if excess:
                msg = (
                    f"the joint form's successes cannot exceed the total; "
                    f"{excess} pair(s) do"
                )
                raise ValueError(msg)
            return
        self._success.validate(values[..., 1])

    def reestimate(
        self,
        observations: ArrayLike,
        posterior: ArrayLike,
        covariate: ArrayLike | None = None,
    ) -> Reestimate[CountPairEmission]:
        """The M step: each channel's own, on the trials the form supplies.

        The total channel is :class:`NegativeBinomialEmission`'s dispersion
        solve, unchanged. The success channel is
        :class:`BetaBinomialEmission`'s alternating bisection, given the fixed
        trial count in the independent form and the *observed totals* in the
        joint one --- the whole difference between the two M steps, and why
        the solve takes a per-observation trial count.

        Returns
        -------
        Reestimate
            The re-estimated pair, converged only if both channels' solves
            were, at the boundary if either was, and reporting the larger
            iteration count and residual of the two.
        """
        refuse_covariate(self, covariate)
        posterior = as_tensor(posterior)
        values = (
            as_tensor(observations, self.observation_dtype)
            .reshape(-1, self.N_CHANNELS)
            .to(posterior.dtype)
        )
        weights = posterior.reshape(-1, self.n_states)
        totals, successes = values[:, 0], values[:, 1]

        depth = self._total.reestimate(totals, weights)
        if not self._joint:
            rate = self._success.reestimate(successes, weights)
            return Reestimate(
                CountPairEmission(
                    depth.emissions.dispersion,
                    depth.emissions.mean,
                    rate.emissions.alpha,
                    rate.emissions.beta,
                    self._success.trials,
                    joint=False,
                ),
                converged=depth.converged and rate.converged,
                at_boundary=depth.at_boundary or rate.at_boundary,
                iterations=max(depth.iterations, rate.iterations),
                residual=max(depth.residual, rate.residual),
            )

        alpha = torch.empty(self.n_states, dtype=torch.float64)
        beta = torch.empty(self.n_states, dtype=torch.float64)
        boundary = depth.at_boundary
        converged = depth.converged
        iterations = depth.iterations
        residual = depth.residual
        concentrations = [
            float(self._alpha[state] + self._beta[state])
            for state in range(self.n_states)
        ]
        batch = solve_beta_binomial_m_step(
            successes,
            weights,
            totals,
            [
                float(self._alpha[state]) / concentrations[state]
                for state in range(self.n_states)
            ],
            concentrations,
        )
        for state, solved in enumerate(batch):
            alpha[state] = solved.alpha
            beta[state] = solved.beta
            boundary = boundary or solved.at_boundary
            converged = converged and solved.converged
            iterations = max(iterations, solved.iterations)
            residual = max(residual, solved.residual)
        return Reestimate(
            CountPairEmission(
                depth.emissions.dispersion,
                depth.emissions.mean,
                alpha,
                beta,
                None,
                joint=True,
            ),
            converged=converged,
            at_boundary=boundary,
            iterations=iterations,
            residual=residual,
        )

    def alignment_key(self) -> torch.Tensor:
        """Both channels' mean and variance, shape ``(n_states, 4)``.

        :class:`NegativeBinomialEmission`'s two-moment signature, once per
        channel: two states agreeing on the depth and differing on the allele
        fraction are different states, and a key over the total alone would
        tie them.
        """
        return torch.cat([self.mean, self.variance], dim=1)

    def named_parameters(self) -> Mapping[str, torch.Tensor]:
        """``dispersion``, ``mean``, ``alpha`` and ``beta``.

        The independent form's trial count is a constant, as it is for
        :class:`BetaBinomialEmission`, and the joint form has none.
        """
        return {
            "dispersion": self._total.dispersion,
            "mean": self._total.mean,
            "alpha": self._alpha,
            "beta": self._beta,
        }


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
        + lgamma_shifted(counts, alpha)
        + lgamma_shifted(trials - counts, beta)
        - lgamma_shifted(trials, total)
        + torch.lgamma(total)
        - torch.lgamma(alpha)
        - torch.lgamma(beta)
    )


def lgamma_shifted(counts: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    """``lgamma(counts + shift)``, broadcast, with ``lgamma`` taken on the distinct counts only (issue #924).

    ``counts`` holds a few hundred distinct integers across thousands of
    observations, and the sum is formed per (observation, state), so
    evaluating on the distinct counts and gathering by index is the same
    number from far fewer ``lgamma`` calls: elementwise, so bitwise. Counts
    that do not broadcast against one ``shift`` per state take the direct
    form, and so
    does a ``shift`` autodiff is tracking: the gather's backward sums the
    gradient in another order, and an observed-information Hessian would no
    longer be the one it was.
    """
    if (
        counts.dim() == 0
        or counts.shape[-1] != 1
        or counts.numel() < 64
        or shift.dim() != 1
        or (shift.requires_grad and torch.is_grad_enabled())
    ):
        return torch.lgamma(counts + shift)
    distinct, inverse = torch.unique(counts, return_inverse=True)
    table = torch.lgamma(distinct.unsqueeze(-1) + shift.reshape(-1))
    return table[inverse.reshape(-1)].reshape(*counts.shape[:-1], -1)


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


def _weighted_mass(weights: torch.Tensor, offsets: torch.Tensor | None) -> torch.Tensor:
    """``sum_t w_t e_t`` per state: the denominator of the profiled mean.

    Three forms, and the middle one is why this is a function. Without an
    exposure it is the posterior mass. With a **constant** exposure it is
    ``e sum_t w_t`` exactly, because a constant factors out of the sum --- and
    forming it as a matmul instead leaves a one-ULP residue, which is enough to
    lose the bit-for-bit agreement with the conserved family at ``e = 1`` that
    issue #631 asks for. With a varying one it is ``weights.T @ offsets``: the
    same number as ``(weights * offsets.unsqueeze(-1)).sum(0)`` with no
    ``(n_obs, n_states)`` temporary, and BLAS-backed.

    Returns
    -------
    torch.Tensor
        Shape ``(n_states,)``.
    """
    if offsets is None:
        return weights.sum(dim=0)
    first = offsets.reshape(-1)[0]
    if bool((offsets == first).all()):
        return weights.sum(dim=0) * first
    return weights.T @ offsets
