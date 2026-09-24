"""The gradient-fit objectives: one ``Objective`` per emission family, its constraint map and starting point, and the metrics a tracked fit records.

Each scores through
:func:`~snakes_and_ladders.opt.hmm.forward.forward_log_likelihood_from_density`
and differentiates on the backend it is given. Imports
:mod:`~snakes_and_ladders.opt.hmm.forward` alone.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from snakes_and_ladders import oxisal
from snakes_and_ladders.backend import Backend, refuse_backend
from snakes_and_ladders.emissions import (
    BetaBinomialEmission,
    BinomialEmission,
    CategoricalEmission,
    EmissionFamily,
    GaussianEmission,
    NegativeBinomialEmission,
    PoissonEmission,
    identifiable_dispersion_bound,
    pooled_variance_floor,
)
from snakes_and_ladders.opt.constrain import (
    free_from_log_simplex,
    free_from_positive,
    free_from_probability,
    log_simplex,
    positive,
    probability,
)
from snakes_and_ladders.opt.hmm.forward import forward_log_likelihood_from_density
from snakes_and_ladders.opt.initialize import quantile_locations
from snakes_and_ladders.opt.objective import Objective, autograd_value_and_gradient

# How far apart the emission rows start, in unconstrained units. Large
# enough to leave the stationary point, small enough not to preselect an
# answer: at 1.0 a state favours its symbol by a factor of e.
_SYMMETRY_BREAK = 1.0


# Smallest mean a count state may start at. A quantile of the pooled counts
# is zero whenever a state's share of the data is mostly zeros, and log(0) is
# not a starting point; half a count is below any observable value and so
# commits to nothing.
_MINIMUM_COUNT_MEAN = 0.5


# How far a starting success rate is held from 0 and 1, where a logit is
# infinite. A quantile of the pooled counts hits either end whenever a state's
# share of the data is all failures or all successes.
_RATE_MARGIN = 1e-6


class _HmmObjective(Objective):
    """The part of an HMM objective that does not know what a state emits.

    The initial distribution and the transition matrix are simplex-valued
    whatever the observations are, and the forward recursion consumes a
    ``(n_sequences, length, n_states)`` block of per-site scores however they
    were produced. Everything else lives in the subclass that names a family.
    """

    def __init__(
        self,
        observations: np.ndarray,
        n_states: int,
        observation_dtype: torch.dtype,
        dtype: torch.dtype,
        covariate: np.ndarray | None = None,
        backend: Backend = Backend.JAX,
    ) -> None:
        refuse_backend(
            "an HMM objective's gradient", backend, (Backend.JAX, Backend.TORCH)
        )
        self._backend = backend
        self._jax: Callable[[np.ndarray], tuple[float, np.ndarray]] | None = None
        self._observations = torch.as_tensor(observations, dtype=observation_dtype)
        self._n_states = n_states
        self._dtype = dtype
        # Beside the observations, and for the same reason they are here: it is
        # fixed for the life of the objective, where the family is rebuilt from
        # `theta` on every call and so cannot hold it (issue #652). A family
        # that conditions on nothing refuses a covariate rather than ignoring
        # it, so this stays `None` unless a caller passes one.
        self._covariate = (
            None
            if covariate is None
            else torch.as_tensor(covariate, dtype=torch.float64)[..., None]
        )

    @property
    def observations(self) -> torch.Tensor:
        """The sequences being fitted, in the dtype the family scores them in.

        Public so an expectation-maximization run started from a ``theta`` of
        this objective fits the same data the gradient does
        (:func:`snakes_and_ladders.opt.starts.polish_by_baum_welch`, issue
        #894).
        """
        return self._observations

    @property
    def covariate(self) -> torch.Tensor | None:
        """The per-site covariate the family conditions on, or ``None``."""
        return self._covariate

    @property
    def _n_emission_parameters(self) -> int:
        """Free values the emission family occupies in ``theta``."""
        raise NotImplementedError  # pragma: no cover

    def emissions(self, theta: torch.Tensor) -> EmissionFamily:
        """The emission family ``theta``'s emission block encodes.

        Parameters
        ----------
        theta : torch.Tensor
            Unconstrained parameters.

        Returns
        -------
        EmissionFamily
            Differentiable with respect to ``theta``.
        """
        raise NotImplementedError  # pragma: no cover

    def _free_transitions_from(self, named: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """The unconstrained coordinates of the initial and transition blocks."""
        return torch.cat(
            [
                free_from_log_simplex(named["log_initial"]),
                free_from_log_simplex(named["log_transition"]).reshape(-1),
            ]
        )

    def theta_from(self, named: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """The unconstrained vector whose :meth:`constrain` is ``named``.

        The inverse of the constraint map, keyed as :meth:`constrain`
        returns. It lets a fit from *any* optimizer be given an interval: the
        observed information is a property of the objective at a point, and
        this turns a point stated in the model's own parameters into one the
        Hessian can be taken at (issue #268).

        Parameters
        ----------
        named : Mapping[str, torch.Tensor]
            Constrained parameters, under :meth:`constrain`'s own keys.

        Returns
        -------
        torch.Tensor
            ``theta`` such that ``constrain(theta)`` returns ``named``.
        """
        return torch.cat(
            [self._free_transitions_from(named), self._free_emissions_from(named)]
        )

    def _free_emissions_from(self, named: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """The emission block's unconstrained coordinates."""
        raise NotImplementedError  # pragma: no cover

    @property
    def _initial_slice(self) -> slice:
        return slice(0, self._n_states - 1)

    @property
    def _transition_slice(self) -> slice:
        start = self._n_states - 1
        return slice(start, start + self._n_states * (self._n_states - 1))

    @property
    def _emission_slice(self) -> slice:
        return slice(self._transition_slice.stop, self.n_parameters)

    @property
    def n_parameters(self) -> int:
        """Length of ``theta``: one free value per free parameter."""
        return (
            (self._n_states - 1)
            + self._n_states * (self._n_states - 1)
            + self._n_emission_parameters
        )

    def _transition_parameters(self, theta: torch.Tensor) -> dict[str, torch.Tensor]:
        """Log initial distribution and log transition matrix, from ``theta``."""
        m = self._n_states
        return {
            "log_initial": log_simplex(theta[self._initial_slice]),
            "log_transition": log_simplex(
                theta[self._transition_slice].reshape(m, m - 1)
            ),
        }

    def jax_energy(self) -> tuple[Callable[[Any, Any], Any], dict[str, Any]] | None:
        """The negative log-likelihood as a traceable JAX ``(theta, data)`` function and its data, under ``Backend.JAX``; ``None`` otherwise (issue #1008).

        What a compiled HMC chain runs inside its own loop
        (:class:`~snakes_and_ladders.sample.declared.DeclaredJaxEnergy`).
        """
        if self._backend is not Backend.JAX:
            return None
        from snakes_and_ladders.opt.hmm_jax import jax_energy

        return jax_energy(self)

    def value_and_gradient(
        self, theta: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The negative log-likelihood at ``theta`` and its gradient, detached (issue #1000).

        Under :data:`~snakes_and_ladders.backend.Backend.JAX`, the default,
        the compiled twin of :mod:`snakes_and_ladders.opt.hmm_jax` takes both
        at 0.07x--0.18x autograd's runtime at 10^4--10^5 positions; under
        :data:`~snakes_and_ladders.backend.Backend.TORCH` autograd through
        :meth:`__call__` does, and is the oracle the twin is pinned to at
        1e-10.
        """
        if self._backend is Backend.TORCH:
            return autograd_value_and_gradient(self, theta)
        if self._jax is None:
            from snakes_and_ladders.opt.hmm_jax import value_and_grad

            self._jax = value_and_grad(self)
        value_, gradient_ = self._jax(theta.detach().cpu().numpy())
        return (
            torch.tensor(value_, dtype=theta.dtype),
            torch.as_tensor(np.array(gradient_), dtype=theta.dtype),
        )

    def __call__(self, theta: torch.Tensor) -> torch.Tensor:
        """Negative log-likelihood of every observed sequence."""
        transitions = self._transition_parameters(theta)
        return -forward_log_likelihood_from_density(
            self.emissions(theta).log_density(
                self._observations, covariate=self._covariate
            ),
            transitions["log_initial"],
            transitions["log_transition"],
        )


def _free_transitions(
    initial: np.ndarray, transition: np.ndarray, dtype: torch.dtype
) -> torch.Tensor:
    """The unconstrained coordinates of a known initial distribution and transition."""
    return torch.cat(
        [
            free_from_log_simplex(torch.log(torch.as_tensor(initial, dtype=dtype))),
            free_from_log_simplex(
                torch.log(torch.as_tensor(transition, dtype=dtype))
            ).reshape(-1),
        ]
    )


class HmmObjective(_HmmObjective):
    """Negative log-likelihood of observed sequences, by the forward algorithm.

    Parameters
    ----------
    observations : np.ndarray
        Observed symbols, shape ``(n_sequences, sequence_length)``.
    n_states : int
        Hidden states.
    n_symbols : int
        Emission alphabet size.
    dtype : torch.dtype
        Precision of the computation; ``float64`` by default, since a
        finite-difference derivative check is meaningless in ``float32``.
    """

    def __init__(
        self,
        observations: np.ndarray,
        n_states: int,
        n_symbols: int,
        dtype: torch.dtype = torch.float64,
        covariate: np.ndarray | None = None,
        backend: Backend = Backend.JAX,
    ) -> None:
        super().__init__(observations, n_states, torch.long, dtype, covariate, backend)
        self._n_symbols = n_symbols

    @property
    def _n_emission_parameters(self) -> int:
        return self._n_states * (self._n_symbols - 1)

    def emissions(self, theta: torch.Tensor) -> CategoricalEmission:
        """The categorical family ``theta``'s emission block encodes."""
        return CategoricalEmission.from_log(
            log_simplex(
                theta[self._emission_slice].reshape(self._n_states, self._n_symbols - 1)
            )
        )

    def _free_emissions_from(self, named: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return free_from_log_simplex(named["log_emission"]).reshape(-1)

    def initial(self) -> torch.Tensor:
        """A start that is uninformative but **not** symmetric.

        The uniform point is a stationary point of the likelihood, not merely
        a poor guess: with all hidden states identical the gradient in the
        initial and transition parameters is exactly zero, and an optimizer
        started there stays there while the emission rows converge to the
        pooled symbol frequency (`tests/regression/test_opt_hmm.py` pins it).

        The emission rows are therefore tilted apart by a fixed amount, each
        state favouring a different symbol. Deterministic rather than random:
        a seeded jitter would make the fit depend on a second seed nobody
        declared.
        """
        theta = torch.zeros(self.n_parameters, dtype=self._dtype)
        free_emission = theta[self._emission_slice].reshape(
            self._n_states, self._n_symbols - 1
        )
        for state in range(self._n_states):
            free_emission[state, state % (self._n_symbols - 1)] = _SYMMETRY_BREAK
        return theta

    def constrain(self, theta: torch.Tensor) -> Mapping[str, torch.Tensor]:
        """Split ``theta`` into log initial, transition and emission matrices.

        Returned as *log* probabilities, which is what the forward recursion
        consumes; a caller comparing against truth exponentiates.
        """
        return {
            **self._transition_parameters(theta),
            **self.emissions(theta).named_parameters(),
        }

    def theta_from_truth(
        self, initial: np.ndarray, transition: np.ndarray, emission: np.ndarray
    ) -> torch.Tensor:
        """Place a known truth in the unconstrained coordinates.

        Parameters
        ----------
        initial : np.ndarray
            True initial distribution, shape ``(n_states,)``.
        transition : np.ndarray
            True transition matrix, shape ``(n_states, n_states)``.
        emission : np.ndarray
            True emission matrix, shape ``(n_states, n_symbols)``.

        Returns
        -------
        torch.Tensor
            ``theta`` such that ``constrain(theta)`` returns this truth.
        """
        return torch.cat(
            [
                _free_transitions(initial, transition, self._dtype),
                self._free_emissions_from(
                    {
                        "log_emission": torch.log(
                            torch.as_tensor(emission, dtype=self._dtype)
                        )
                    }
                ),
            ]
        )


class GaussianHmmObjective(_HmmObjective):
    """Negative log-likelihood of real-valued sequences from a Gaussian HMM.

    The same forward recursion over the same simplex-constrained transitions;
    only what a state emits differs. Two consequences the categorical instance
    never faced.

    **The objective may be negative for a *good* fit.** The emission term is a
    density, so the negative log-likelihood can be below zero.

    **The likelihood has no maximum.** Put one state's mean on a single
    observation and let its scale go to zero and the objective diverges, so
    ``fit`` reporting convergence means it satisfied the first-order condition
    somewhere, not that it found the supremum (``opt/CLAUDE.md``'s rule about
    ``converged``). :class:`GaussianEmission`'s variance floor makes an
    approach to that boundary visible in the EM oracle; a gradient fit is
    protected only by where it starts, which is why :meth:`initial` places the
    means on the data.

    Parameters
    ----------
    observations : np.ndarray
        Observed values, shape ``(n_sequences, sequence_length)``.
    n_states : int
        Hidden states.
    dtype : torch.dtype
        Precision of the computation; ``float64`` by default, since a
        finite-difference derivative check is meaningless in ``float32``.
    """

    def __init__(
        self,
        observations: np.ndarray,
        n_states: int,
        dtype: torch.dtype = torch.float64,
        covariate: np.ndarray | None = None,
        backend: Backend = Backend.JAX,
    ) -> None:
        super().__init__(
            observations, n_states, torch.float64, dtype, covariate, backend
        )
        self._variance_floor = pooled_variance_floor(observations)

    @property
    def _n_emission_parameters(self) -> int:
        return 2 * self._n_states

    @property
    def variance_floor(self) -> float:
        """The floor the EM oracle refuses at, derived from these observations."""
        return self._variance_floor

    def _mean_slice(self) -> slice:
        start = self._emission_slice.start
        return slice(start, start + self._n_states)

    def _log_scale_slice(self) -> slice:
        start = self._mean_slice().stop
        return slice(start, start + self._n_states)

    def emissions(self, theta: torch.Tensor) -> GaussianEmission:
        """The Gaussian family ``theta``'s emission block encodes.

        The mean is unconstrained and the scale reaches the optimizer through
        a log map, per ``opt/CLAUDE.md``: a positive parameter is kept
        positive by construction, never by projecting an iterate back.
        """
        return GaussianEmission(
            theta[self._mean_slice()],
            positive(theta[self._log_scale_slice()]),
            self._variance_floor,
        )

    def _free_emissions_from(self, named: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat(
            [
                named["mean"].reshape(-1),
                free_from_positive(named["scale"]).reshape(-1),
            ]
        )

    @property
    def gaussian_hmm_declaration(self) -> tuple[int, np.ndarray] | None:
        """``(m, observations)`` for a compiled chain, where :meth:`gradient` streams (issue #1008).

        What :meth:`gradient` computes through ``oxisal`` is what a compiled
        HMC chain evaluates itself (:mod:`snakes_and_ladders.sample.declared`):
        no covariate, ``float64``, sequences as rows.
        """
        if (
            self._covariate is not None
            or self._dtype != torch.float64
            or self._observations.dim() != 2
        ):
            return None
        return self._n_states, self._observations.numpy()

    def gradient(self, theta: torch.Tensor) -> torch.Tensor:
        """``d/dtheta`` of :meth:`__call__` by Fisher's identity, which ``hmc.gradient_at`` reads (issue #997).

        The score of the log-likelihood is the posterior expectation of the
        complete-data score, so with ``gamma`` and ``xi`` the posteriors of a
        state and of a pair, ``pi`` and ``A`` the initial distribution and
        the transitions, and ``N`` sequences:

        - an initial logit ``k >= 1``: ``sum gamma_1(k) - N pi_k``;
        - a transition logit ``(i, j >= 1)``: ``sum xi(i, j) - sum_j' xi(i, j') A_ij``;
        - a mean: ``sum_t gamma_t(s) (x_t - mu_s) / s_s^2``;
        - a log scale: ``sum_t gamma_t(s) ((x_t - mu_s)^2 / s_s^2 - 1)``;

        each summed in one streamed pass in
        ``oxisal.gaussian_hmm_statistics``, negated for the
        negative log-likelihood. Autograd through :meth:`__call__` is the
        oracle; a covariate, or any dtype but ``float64``, takes it.
        """
        m = self._n_states
        if (
            self._covariate is not None
            or self._dtype != torch.float64
            or self._observations.dim() != 2
        ):
            return autograd_value_and_gradient(self, theta)[1]
        free = theta.detach()
        transitions = self._transition_parameters(free)
        mean = free[self._mean_slice()].numpy()
        scale = np.exp(free[self._log_scale_slice()].numpy())
        first, pairs, moments, _ = oxisal.gaussian_hmm_statistics(
            np.ascontiguousarray(self._observations.numpy()),
            np.ascontiguousarray(transitions["log_initial"].numpy()),
            np.ascontiguousarray(transitions["log_transition"].numpy()).reshape(-1),
            np.ascontiguousarray(mean),
            np.ascontiguousarray(scale),
        )
        n_sequences = self._observations.shape[0]
        pairs = pairs.reshape(m, m)
        initial = np.exp(transitions["log_initial"].numpy())
        transition = np.exp(transitions["log_transition"].numpy())
        s0, s1, s2 = moments.reshape(m, 3).T
        score = np.concatenate(
            [
                (first - n_sequences * initial)[1:],
                (pairs - pairs.sum(axis=1, keepdims=True) * transition)[:, 1:].reshape(
                    -1
                ),
                s1 / scale**2,
                s2 / scale**2 - s0,
            ]
        )
        return torch.from_numpy(-score)

    def initial(self) -> torch.Tensor:
        """A start that is uninformative but **not** symmetric.

        Uniform transitions, for the reason the categorical instance gives.
        The means are placed at evenly spaced quantiles of the pooled
        observations, breaking exchangeability while committing to nothing
        about which state is which.

        On the data rather than at a fixed point, because a Gaussian mean far
        from every observation contributes a density that underflows: the
        state is then invisible to the E step and the fit reduces to one with
        fewer states. The scales start at the pooled standard deviation, the
        widest defensible value --- too *narrow* is the direction the
        likelihood is unbounded in.
        """
        theta = torch.zeros(self.n_parameters, dtype=self._dtype)
        values = self._observations.reshape(-1).to(self._dtype)
        theta[self._mean_slice()] = quantile_locations(values, self._n_states)
        theta[self._log_scale_slice()] = free_from_positive(values.std())
        return theta

    def constrain(self, theta: torch.Tensor) -> Mapping[str, torch.Tensor]:
        """Split ``theta`` into log transitions and the per-state mean and scale.

        The transitions are returned as log-probabilities, as for every HMM
        here; the emission parameters are returned in their own units, since
        a mean has no log form and a recovery test compares means.
        """
        return {
            **self._transition_parameters(theta),
            **self.emissions(theta).named_parameters(),
        }

    def theta_from_truth(
        self,
        initial: np.ndarray,
        transition: np.ndarray,
        mean: np.ndarray,
        scale: np.ndarray,
    ) -> torch.Tensor:
        """Place a known truth in the unconstrained coordinates.

        Parameters
        ----------
        initial : np.ndarray
            True initial distribution, shape ``(n_states,)``.
        transition : np.ndarray
            True transition matrix, shape ``(n_states, n_states)``.
        mean, scale : np.ndarray
            True per-state mean and standard deviation, shape ``(n_states,)``.

        Returns
        -------
        torch.Tensor
            ``theta`` such that ``constrain(theta)`` returns this truth.
        """
        return torch.cat(
            [
                _free_transitions(initial, transition, self._dtype),
                self._free_emissions_from(
                    {
                        "mean": torch.as_tensor(mean, dtype=self._dtype),
                        "scale": torch.as_tensor(scale, dtype=self._dtype),
                    }
                ),
            ]
        )


class _CountHmmObjective(_HmmObjective):
    """Shared scaffolding for the count instances: a start on the data.

    Every count family places its per-state location at evenly spaced
    quantiles of the pooled observations, for :class:`GaussianHmmObjective`'s
    reasons: a shared location is a stationary point, and a location far from
    every observation makes a state invisible to the E step.
    """

    def __init__(
        self,
        observations: np.ndarray,
        n_states: int,
        dtype: torch.dtype = torch.float64,
        covariate: np.ndarray | None = None,
        backend: Backend = Backend.JAX,
    ) -> None:
        super().__init__(
            observations, n_states, torch.float64, dtype, covariate, backend
        )

    def _location_quantiles(self) -> torch.Tensor:
        """Evenly spaced quantiles of the pooled observations, one per state."""
        return quantile_locations(
            self._observations.reshape(-1).to(self._dtype), self._n_states
        )

    def constrain(self, theta: torch.Tensor) -> Mapping[str, torch.Tensor]:
        """Log transitions, and the emission family's own named parameters."""
        return {
            **self._transition_parameters(theta),
            **self.emissions(theta).named_parameters(),
        }


class PoissonHmmObjective(_CountHmmObjective):
    """Negative log-likelihood of count sequences from a Poisson-emission HMM.

    One parameter per state and no dispersion, which makes it the control:
    whatever a richer count family's fit does that this one does not is the
    dispersion parameter and not the shared recursion.

    Parameters
    ----------
    observations : np.ndarray
        Observed counts, shape ``(n_sequences, sequence_length)``.
    n_states : int
        Hidden states.
    dtype : torch.dtype
        Precision of the computation; ``float64`` by default.
    """

    @property
    def _n_emission_parameters(self) -> int:
        return self._n_states

    def emissions(self, theta: torch.Tensor) -> PoissonEmission:
        """The Poisson family ``theta``'s emission block encodes."""
        return PoissonEmission(positive(theta[self._emission_slice]))

    def _free_emissions_from(self, named: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return free_from_positive(named["mean"]).reshape(-1)

    def initial(self) -> torch.Tensor:
        """Uniform transitions; rates at quantiles of the pooled counts."""
        theta = torch.zeros(self.n_parameters, dtype=self._dtype)
        theta[self._emission_slice] = free_from_positive(
            self._location_quantiles().clamp_min(_MINIMUM_COUNT_MEAN)
        )
        return theta

    def theta_from_truth(
        self, initial: np.ndarray, transition: np.ndarray, mean: np.ndarray
    ) -> torch.Tensor:
        """Place a known ``(pi, A, lambda)`` in the unconstrained coordinates."""
        return torch.cat(
            [
                _free_transitions(initial, transition, self._dtype),
                self._free_emissions_from(
                    {"mean": torch.as_tensor(mean, dtype=self._dtype)}
                ),
            ]
        )


class BinomialHmmObjective(_CountHmmObjective):
    """Negative log-likelihood of bounded counts from a binomial-emission HMM.

    The under-dispersed instance. The trial count is declared rather than
    fitted, so ``theta`` carries one free value per state and the constraint
    map is a logit: a probability has two boundaries, so positivity is not
    enough.

    Parameters
    ----------
    observations : np.ndarray
        Observed counts, shape ``(n_sequences, sequence_length)``.
    n_states : int
        Hidden states.
    trials : np.ndarray
        Declared trial count per state, shape ``(n_states,)``.
    dtype : torch.dtype
        Precision of the computation; ``float64`` by default.
    """

    def __init__(
        self,
        observations: np.ndarray,
        n_states: int,
        trials: np.ndarray,
        dtype: torch.dtype = torch.float64,
        covariate: np.ndarray | None = None,
        backend: Backend = Backend.JAX,
    ) -> None:
        super().__init__(observations, n_states, dtype, covariate, backend)
        self._trials = torch.as_tensor(trials, dtype=torch.float64).reshape(-1)

    @property
    def _n_emission_parameters(self) -> int:
        return self._n_states

    def emissions(self, theta: torch.Tensor) -> BinomialEmission:
        """The binomial family ``theta``'s emission block encodes."""
        return BinomialEmission(self._trials, probability(theta[self._emission_slice]))

    def _free_emissions_from(self, named: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return free_from_probability(named["probability"].reshape(-1))

    def initial(self) -> torch.Tensor:
        """Uniform transitions; success rates from quantiles of the counts."""
        theta = torch.zeros(self.n_parameters, dtype=self._dtype)
        rate = (self._location_quantiles() / self._trials).clamp(
            _RATE_MARGIN, 1.0 - _RATE_MARGIN
        )
        theta[self._emission_slice] = free_from_probability(rate)
        return theta

    def theta_from_truth(
        self, initial: np.ndarray, transition: np.ndarray, probability: np.ndarray
    ) -> torch.Tensor:
        """Place a known ``(pi, A, p)`` in the unconstrained coordinates."""
        return torch.cat(
            [
                _free_transitions(initial, transition, self._dtype),
                self._free_emissions_from(
                    {"probability": torch.as_tensor(probability, dtype=self._dtype)}
                ),
            ]
        )


class BetaBinomialHmmObjective(_CountHmmObjective):
    """Negative log-likelihood of bounded counts from a beta-binomial HMM.

    The over-dispersed counterpart of :class:`BinomialHmmObjective` on the
    same support, and the second instance whose EM counterpart's M step is a
    solve rather than a formula. The gradient fit is not --- autograd
    differentiates through ``lgamma`` --- so the pair shares the model and
    nothing else.

    Parameters
    ----------
    observations : np.ndarray
        Observed counts, shape ``(n_sequences, sequence_length)``.
    n_states : int
        Hidden states.
    trials : np.ndarray
        Declared trial count per state, shape ``(n_states,)``.
    dtype : torch.dtype
        Precision of the computation; ``float64`` by default.
    """

    def __init__(
        self,
        observations: np.ndarray,
        n_states: int,
        trials: np.ndarray,
        dtype: torch.dtype = torch.float64,
        covariate: np.ndarray | None = None,
        backend: Backend = Backend.JAX,
    ) -> None:
        super().__init__(observations, n_states, dtype, covariate, backend)
        self._trials = torch.as_tensor(trials, dtype=torch.float64).reshape(-1)

    @property
    def _n_emission_parameters(self) -> int:
        return 2 * self._n_states

    def _log_alpha_slice(self) -> slice:
        start = self._emission_slice.start
        return slice(start, start + self._n_states)

    def _log_beta_slice(self) -> slice:
        start = self._log_alpha_slice().stop
        return slice(start, start + self._n_states)

    def emissions(self, theta: torch.Tensor) -> BetaBinomialEmission:
        """The beta-binomial family ``theta``'s emission block encodes."""
        return BetaBinomialEmission(
            self._trials,
            positive(theta[self._log_alpha_slice()]),
            positive(theta[self._log_beta_slice()]),
        )

    def _free_emissions_from(self, named: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat(
            [
                free_from_positive(named["alpha"]).reshape(-1),
                free_from_positive(named["beta"]).reshape(-1),
            ]
        )

    def initial(self) -> torch.Tensor:
        """Uniform transitions; ``(a, b)`` from the quantile rates at unit concentration.

        The concentration starts at 1 --- the most overdispersed the family
        gets before the Beta becomes improper --- because ``a + b -> inf`` is
        the direction the likelihood goes flat in, and a start there would put
        the first E step inside the region this model's hazard lives in.
        """
        theta = torch.zeros(self.n_parameters, dtype=self._dtype)
        rate = (self._location_quantiles() / self._trials).clamp(
            _RATE_MARGIN, 1.0 - _RATE_MARGIN
        )
        theta[self._log_alpha_slice()] = torch.log(rate)
        theta[self._log_beta_slice()] = torch.log1p(-rate)
        return theta

    def theta_from_truth(
        self,
        initial: np.ndarray,
        transition: np.ndarray,
        alpha: np.ndarray,
        beta: np.ndarray,
    ) -> torch.Tensor:
        """Place a known ``(pi, A, a, b)`` in the unconstrained coordinates."""
        return torch.cat(
            [
                _free_transitions(initial, transition, self._dtype),
                self._free_emissions_from(
                    {
                        "alpha": torch.as_tensor(alpha, dtype=self._dtype),
                        "beta": torch.as_tensor(beta, dtype=self._dtype),
                    }
                ),
            ]
        )


class NegativeBinomialHmmObjective(_HmmObjective):
    """Negative log-likelihood of count sequences from a negative binomial HMM.

    The third emission family through the same recursion, and the one that
    says whether the seam is real: its EM counterpart's M step is a solve
    rather than a formula, the gradient fit is not, and the two disagree only
    if one is wrong.

    **The evidence is a probability again.** Counts are discrete, so
    ``log P(observations) <= 0`` holds here and does not for
    :class:`GaussianHmmObjective`.

    **The identifiability hazard is flatness.** Both parameters reach the
    optimizer through a log map, so neither can leave its feasible set, but no
    map fixes a likelihood flat in ``r`` past the point the data resolves it.
    :func:`snakes_and_ladders.emissions.identifiable_dispersion_bound` says
    where, and the EM oracle reports reaching it.

    Parameters
    ----------
    observations : np.ndarray
        Observed counts, shape ``(n_sequences, sequence_length)``.
    n_states : int
        Hidden states.
    dtype : torch.dtype
        Precision of the computation; ``float64`` by default, since a
        finite-difference derivative check is meaningless in ``float32``.
    """

    def __init__(
        self,
        observations: np.ndarray,
        n_states: int,
        dtype: torch.dtype = torch.float64,
        covariate: np.ndarray | None = None,
        backend: Backend = Backend.JAX,
    ) -> None:
        super().__init__(
            observations, n_states, torch.float64, dtype, covariate, backend
        )

    @property
    def _n_emission_parameters(self) -> int:
        return 2 * self._n_states

    def _log_dispersion_slice(self) -> slice:
        start = self._emission_slice.start
        return slice(start, start + self._n_states)

    def _log_mean_slice(self) -> slice:
        start = self._log_dispersion_slice().stop
        return slice(start, start + self._n_states)

    def emissions(self, theta: torch.Tensor) -> NegativeBinomialEmission:
        """The negative binomial family ``theta``'s emission block encodes."""
        return NegativeBinomialEmission(
            positive(theta[self._log_dispersion_slice()]),
            positive(theta[self._log_mean_slice()]),
        )

    def _free_emissions_from(self, named: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat(
            [
                free_from_positive(named["dispersion"]).reshape(-1),
                free_from_positive(named["mean"]).reshape(-1),
            ]
        )

    def initial(self) -> torch.Tensor:
        """A start that is uninformative but **not** symmetric.

        Uniform transitions, and per-state means at evenly spaced quantiles of
        the pooled counts, for the reasons the Gaussian instance gives. The
        dispersions start at the pooled method-of-moments value, or at the
        identifiable bound where the pooled counts are not overdispersed:
        starting *above* what the data can resolve would put the first E step
        in the flat region this model's hazard lives in.
        """
        theta = torch.zeros(self.n_parameters, dtype=self._dtype)
        values = self._observations.reshape(-1).to(self._dtype)
        means = quantile_locations(values, self._n_states).clamp_min(
            _MINIMUM_COUNT_MEAN
        )
        theta[self._log_mean_slice()] = free_from_positive(means)

        pooled_mean = float(values.mean())
        pooled_variance = float(values.var(unbiased=True))
        bound = identifiable_dispersion_bound(pooled_mean, float(values.numel()))
        moments = (
            pooled_mean**2 / (pooled_variance - pooled_mean)
            if pooled_variance > pooled_mean
            else bound
        )
        theta[self._log_dispersion_slice()] = math.log(min(moments, bound))
        return theta

    def constrain(self, theta: torch.Tensor) -> Mapping[str, torch.Tensor]:
        """Split ``theta`` into log transitions and the per-state ``(r, mu)``."""
        return {
            **self._transition_parameters(theta),
            **self.emissions(theta).named_parameters(),
        }

    def theta_from_truth(
        self,
        initial: np.ndarray,
        transition: np.ndarray,
        dispersion: np.ndarray,
        mean: np.ndarray,
    ) -> torch.Tensor:
        """Place a known truth in the unconstrained coordinates.

        Parameters
        ----------
        initial : np.ndarray
            True initial distribution, shape ``(n_states,)``.
        transition : np.ndarray
            True transition matrix, shape ``(n_states, n_states)``.
        dispersion, mean : np.ndarray
            True per-state ``r`` and ``mu``, shape ``(n_states,)``.

        Returns
        -------
        torch.Tensor
            ``theta`` such that ``constrain(theta)`` returns this truth.
        """
        return torch.cat(
            [
                _free_transitions(initial, transition, self._dtype),
                self._free_emissions_from(
                    {
                        "dispersion": torch.as_tensor(dispersion, dtype=self._dtype),
                        "mean": torch.as_tensor(mean, dtype=self._dtype),
                    }
                ),
            ]
        )


@dataclass(frozen=True)
class HmmMetrics:
    """What an HMM parameter vector means: the forward log-likelihood (issue #778).

    A :class:`snakes_and_ladders.track.Metrics` over ``theta``, satisfied
    structurally, for any family :class:`_HmmObjective` serves --- categorical,
    Gaussian, Poisson, binomial, beta-binomial, negative binomial --- because
    the number comes from the objective's own forward pass.

    ``log_likelihood`` is
    :func:`forward_log_likelihood_from_density` of ``theta``'s emissions,
    which :meth:`_HmmObjective.__call__` negates to minimize; it is read here
    with the sign the model states rather than the sign the optimizer wants.

    **The state accuracy the ticket listed is left out.**
    :func:`align_states` returns a permutation, not a number, and the decoder
    that would turn one into an accuracy ---
    :mod:`snakes_and_ladders.likelihood.forward_backward`, or
    :func:`snakes_and_ladders.search.spatio_sequential.label_accuracy` --- is
    outside what ``opt`` may import
    (``tests/regression/opt/test_opt_objective.py``). A state path is also not
    what a fit holds: ``theta`` is.

    Parameters
    ----------
    objective : _HmmObjective
        The objective being fitted, which owns the observations and the
        constraint map.
    """

    objective: _HmmObjective

    names: tuple[str, ...] = ("log_likelihood",)

    def __call__(self, theta: torch.Tensor) -> dict[str, float]:
        """``{"log_likelihood": -objective(theta)}``, computed without a graph."""
        with torch.no_grad():
            return {"log_likelihood": -float(self.objective(theta))}
