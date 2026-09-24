"""A discrete HMM: the second reference instance of ``Objective``.

One of issue #63's named cases, and unlike the Potts chain it has an
*independent fitting algorithm* -- Baum-Welch -- so a gradient fit can be
checked against something other than itself.

The forward recursion here and Felsenstein pruning are the same sum-product
computation on different graphs: a caterpillar tree carrying one observed
leaf per internal node *is* an HMM (``eq:forward`` of ``docs/tex/textbook.tex``,
derived in ``app:forward-backward``; Durbin et al., ch. 3; Koller & Friedman
for the general framing).

**Label switching.** The likelihood is invariant to permuting the hidden
states, so a fitted parameter set matches truth only up to a permutation. The
model is otherwise identifiable; every row is gauge-fixed by
:func:`snakes_and_ladders.opt.constrain.log_simplex`. A recovery test must align the
permutation before comparing.

Ground truth and data generation live in :mod:`snakes_and_ladders.sim.hmm`; this
module holds the fitting objective, its EM oracle, and the state-alignment
helper a recovery test needs.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from itertools import permutations
from typing import Any, Protocol

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
    refuse_collapsed,
)
from snakes_and_ladders.numerics import constant_chain_kernel
from snakes_and_ladders.opt.constrain import (
    free_from_log_simplex,
    free_from_positive,
    free_from_probability,
    log_simplex,
    positive,
    probability,
)
from snakes_and_ladders.opt.em import em_loop
from snakes_and_ladders.opt.initialize import quantile_locations
from snakes_and_ladders.opt.objective import Objective
from snakes_and_ladders.opt.termination import Termination
from snakes_and_ladders.ragged import Ragged

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
            point = theta.detach().clone().requires_grad_(True)
            value = self(point)
            (gradient,) = torch.autograd.grad(value, point)
            return value.detach(), gradient
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
            point = theta.detach().clone().requires_grad_(True)
            (grad,) = torch.autograd.grad(self(point), point)
            return grad
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


def forward_log_likelihood(
    observations: torch.Tensor,
    log_initial: torch.Tensor,
    log_transition: torch.Tensor,
    log_emission: torch.Tensor,
) -> torch.Tensor:
    """Total log-likelihood of ``observations`` by the forward recursion.

    Parameters
    ----------
    observations : torch.Tensor
        Integer symbols, shape ``(n_sequences, sequence_length)``.
    log_initial : torch.Tensor
        Log initial distribution, shape ``(m,)``.
    log_transition : torch.Tensor
        Log transition matrix, shape ``(m, m)``.
    log_emission : torch.Tensor
        Log emission matrix, shape ``(m, o)``.

    Returns
    -------
    torch.Tensor
        Scalar: the summed log-likelihood over sequences, differentiable
        with respect to every parameter.
    """
    return forward_log_likelihood_from_density(
        CategoricalEmission.from_log(log_emission).log_density(observations),
        log_initial,
        log_transition,
    )


def align_states(
    log_emission: torch.Tensor, reference: torch.Tensor
) -> tuple[int, ...]:
    """Permutation of fitted hidden states best matching ``reference``.

    The likelihood is invariant to relabelling the hidden states, so a
    recovery test has to choose a permutation. Emissions are the
    discriminating signal -- two states with the same emission distribution
    are the same state -- so it is the one minimizing total absolute emission
    difference, found by enumeration (``m!`` is small, and a greedy match can
    be wrong).

    Parameters
    ----------
    log_emission : torch.Tensor
        Fitted log emission matrix, shape ``(m, o)``.
    reference : torch.Tensor
        Emission matrix to align to, shape ``(m, o)``, as probabilities.

    Returns
    -------
    tuple[int, ...]
        ``order`` such that ``exp(log_emission)[list(order)]`` lines up with
        ``reference``.
    """
    return align_by_key(torch.exp(log_emission), reference)


def align_families(
    fitted: EmissionFamily, reference: EmissionFamily
) -> tuple[int, ...]:
    """Permutation of ``fitted``'s states best matching ``reference``'s.

    The same enumeration as :func:`align_states`, over whatever signature each
    family says distinguishes its states --- symbol probabilities for a
    categorical emission, means for a Gaussian one. The signature is the
    family's to define: a Gaussian fit has no emission matrix to align by.

    Parameters
    ----------
    fitted : EmissionFamily
        The fitted family.
    reference : EmissionFamily
        The family to align to.

    Returns
    -------
    tuple[int, ...]
        ``order`` such that state ``order[i]`` of ``fitted`` lines up with
        state ``i`` of ``reference``.
    """
    return align_by_key(fitted.alignment_key(), reference.alignment_key())


def align_by_key(fitted: torch.Tensor, reference: torch.Tensor) -> tuple[int, ...]:
    """The permutation minimizing total absolute distance between two key sets.

    Found by enumeration: ``m!`` is small, and a greedy match can be wrong.

    Parameters
    ----------
    fitted, reference : torch.Tensor
        Per-state signatures, shape ``(m, d)``.

    Returns
    -------
    tuple[int, ...]
        ``order`` such that ``fitted[list(order)]`` lines up with
        ``reference``.
    """
    best: tuple[int, ...] = ()
    best_cost = float("inf")
    for order in permutations(range(fitted.shape[0])):
        cost = float((fitted[list(order)] - reference).abs().sum())
        if cost < best_cost:
            best, best_cost = order, cost
    return best


def forward_log_likelihood_from_density(
    log_density: torch.Tensor,
    log_initial: torch.Tensor,
    log_transition: torch.Tensor,
) -> torch.Tensor:
    """Total log-likelihood from per-site emission scores already computed.

    The recursion that does not know what a state emits, so one forward
    algorithm serves a family over an alphabet and one over the reals: both
    arrive here as the same block of numbers.

    Parameters
    ----------
    log_density : torch.Tensor
        Emission scores, shape ``(n_sequences, length, n_states)``. A
        log-probability for a discrete family, a log-density otherwise.
    log_initial : torch.Tensor
        Log initial distribution, shape ``(m,)``.
    log_transition : torch.Tensor
        Log transition matrix. Either ``(m, m)``, one kernel for the whole
        chain, or ``(length - 1, m, m)``, one per step, so a kernel that is a
        function of position can be written down (issue #653). The step axis
        is shared across the batch: the kernel varies along the sequence, not
        between sequences.

    Returns
    -------
    torch.Tensor
        Scalar: the summed log-likelihood over sequences, differentiable with
        respect to every parameter. **Not** bounded above by zero where the
        emission family is continuous, since it then sums densities.

    Raises
    ------
    ValueError
        If ``log_transition`` is neither shape.

    Notes
    -----
    The constant form is expanded to the step axis with
    :meth:`torch.Tensor.expand`, a stride-zero view rather than a copy, and the
    choice between the two is hoisted out of the recursion, so the constant
    case indexes nothing and sees the same numbers in the same order --- the
    result is the one this function returned before the second shape was
    admitted, bitwise. The ``length``-fold memory the varying form costs is
    paid only by a caller who asks for it. The same reasoning and the
    measurement behind the hoist are in
    :func:`snakes_and_ladders.likelihood.forward_backward.step_kernels`, whose
    shape check this shares
    (:func:`snakes_and_ladders.numerics.constant_chain_kernel`, issue #857):
    ``opt`` may not import ``likelihood``, so the dispatch sits at the root
    rather than in either recursion.
    """
    length, n_states = log_density.shape[1], log_density.shape[2]
    constant: torch.Tensor | None = None
    if constant_chain_kernel(tuple(log_transition.shape), length, n_states):
        kernels = log_transition.expand(max(length - 1, 0), n_states, n_states)
        constant = log_transition
    else:
        kernels = log_transition
    alpha = log_initial.unsqueeze(0) + log_density[:, 0]
    for t in range(1, length):
        step = kernels[t - 1] if constant is None else constant
        alpha = (
            torch.logsumexp(alpha.unsqueeze(2) + step.unsqueeze(0), dim=1)
            + log_density[:, t]
        )
    return torch.logsumexp(alpha, dim=1).sum()


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


@dataclass(frozen=True)
class EmFit:
    """What one Baum-Welch run produced, and what it had to report.

    A tuple sufficed while every M step was a formula, and stopped when one
    became a solve (issue #229): a dispersion at the edge of what the data
    identifies is not an error, but a caller that cannot see it will build a
    Wald interval around a bound and call it an estimate.

    Parameters
    ----------
    log_initial, log_transition : torch.Tensor
        Fitted transition parameters, as log-probabilities.
    emissions : EmissionFamily
        The fitted emission family.
    log_likelihood : float
        The final log-likelihood. A density, and so possibly positive, where
        the family is continuous.
    emission_at_boundary : bool
        Whether any M step returned a parameter at the edge of the range this
        data identifies it over.
    termination : Termination | None
        Whether the outer loop met its relative tolerance and after how many
        EM iterations (issue #860). An unconverged emission M step is refused
        rather than reported, and still is: this answers for the outer loop.
    """

    log_initial: torch.Tensor
    log_transition: torch.Tensor
    emissions: EmissionFamily
    log_likelihood: float
    emission_at_boundary: bool = False
    termination: Termination | None = None


@dataclass(frozen=True)
class CategoricalFit:
    """What :func:`baum_welch` fitted, as log-probabilities.

    :class:`EmFit` is the general form, carrying a family rather than a
    matrix and two fields a categorical M step cannot fill: no categorical
    re-estimate sits at a boundary, and the outer loop's termination is
    :class:`EmFit`'s to report. This is the narrowing, and it carries what
    the four-tuple carried and nothing else (issue #865).

    Parameters
    ----------
    log_initial : torch.Tensor
        Shape ``(m,)``, the fitted initial distribution.
    log_transition : torch.Tensor
        Shape ``(m, m)``, the fitted transition kernel.
    log_emission : torch.Tensor
        Shape ``(m, n_symbols)``, the fitted emission matrix.
    log_likelihood : float
        The final log-likelihood.
    """

    log_initial: torch.Tensor
    log_transition: torch.Tensor
    log_emission: torch.Tensor
    log_likelihood: float

    def __iter__(self) -> Iterator[Any]:
        """The order callers unpack: the three parameters, then the value.

        ``Any`` and not a union: an unpacking gives every name the element
        type, so a union would mistype each of them.
        """
        yield from (
            self.log_initial,
            self.log_transition,
            self.log_emission,
            self.log_likelihood,
        )


def baum_welch(
    observations: np.ndarray,
    log_initial: torch.Tensor,
    log_transition: torch.Tensor,
    log_emission: torch.Tensor,
    max_iterations: int = 500,
    tolerance: float = 1e-12,
    backend: Backend = Backend.RUST,
) -> CategoricalFit:
    """Fit an HMM by expectation-maximization, with no autodiff involved.

    Baum-Welch is an independent fitting algorithm for the same model, so a
    gradient fit checked against it is checked against something other than
    itself. It shares no code with ``fit`` -- not the optimizer, not the
    parameterization (EM works directly in probabilities) -- only the model.

    Parameters
    ----------
    observations : np.ndarray
        Integer symbols, shape ``(n_sequences, sequence_length)``.
    log_initial, log_transition, log_emission : torch.Tensor
        Starting parameters, as log-probabilities.
    max_iterations : int
        Maximum EM iterations.
    tolerance : float
        Stop when the log-likelihood improves by less than this *relative*
        to its magnitude -- absolute would not transfer across data sizes
        (``DEV.md``, issue #111).
    backend : Backend
        :data:`~snakes_and_ladders.backend.Backend.RUST`, the default since
        issue #986, runs each E and M step in
        ``oxisal.categorical_em_step``: scaled messages
        streamed one sequence at a time into expected counts, so no array
        over every position is held. At 10^5 positions of three states one
        fit peaked at 62.6 MB on the batched route.
        :data:`~snakes_and_ladders.backend.Backend.PYTHON` is that route,
        :func:`baum_welch_family` in log space, and the oracle that pins the
        compiled one within 1e-10.

    Returns
    -------
    CategoricalFit
        Fitted log initial, log transition and log emission, and the final
        log-likelihood.
    """
    refuse_backend("baum_welch", backend, (Backend.PYTHON, Backend.RUST))
    if backend is Backend.RUST:
        return _streamed_baum_welch(
            observations,
            log_initial,
            log_transition,
            log_emission,
            max_iterations=max_iterations,
            tolerance=tolerance,
        )
    result = baum_welch_family(
        observations,
        log_initial,
        log_transition,
        CategoricalEmission.from_log(log_emission),
        max_iterations=max_iterations,
        tolerance=tolerance,
    )
    family = result.emissions
    if not isinstance(family, CategoricalEmission):  # pragma: no cover
        msg = f"expected a categorical M step, got {type(family).__name__}"
        raise TypeError(msg)
    return CategoricalFit(
        result.log_initial,
        result.log_transition,
        family.log_matrix,
        result.log_likelihood,
    )


def _ragged_e_step(
    emit: torch.Tensor,
    mask: torch.Tensor,
    lengths: np.ndarray,
    log_initial: torch.Tensor,
    log_transition: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """The E step in the compiled ragged kernel, returned in the padded layout (issue #933).

    ``emit`` is the masked ``(n, length, m)`` block; its live rows, in
    sequence order, are what the kernel walks. The log marginals are
    scattered back to the block with ``-inf`` at padding, as the torch
    recursion leaves them, so the M step reads one layout either way.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, float]
        Log gamma ``(n, length, m)``, log transition counts ``(m, m)``
        summed over sequences, and the log-likelihood.
    """
    values = np.ascontiguousarray(emit[mask].numpy(), dtype=np.float64)
    m = values.shape[1]
    gamma = np.empty_like(values)
    counts = np.empty((m, m), dtype=np.float64)
    evidence = np.empty(lengths.shape[0], dtype=np.float64)
    oxisal.ragged_posteriors(
        values,
        lengths,
        np.ascontiguousarray(log_initial.numpy(), dtype=np.float64),
        np.ascontiguousarray(log_transition.numpy(), dtype=np.float64),
        gamma,
        counts,
        evidence,
    )
    padded = torch.full(emit.shape, -float("inf"), dtype=emit.dtype)
    padded[mask] = torch.as_tensor(gamma)
    return padded, torch.as_tensor(counts), float(evidence.sum())


class CovariateUpdate(Protocol):
    """A covariate that depends on the parameters, recomputed before every E step (issue #933).

    Called with the current emission family, the posterior of the previous E
    step --- shape ``(n_sequences, length, m)``, zero at padded positions,
    and uniform over the states before the first --- and the covariate as
    :func:`baum_welch_family` holds it. It returns the covariate the next E
    step scores against and the M step after it conditions on, in the same
    layout. A normalizer that is a function of the parameters, as
    :class:`ExpectedRateNormalizer` is, enters the fit this way.
    """

    def __call__(
        self,
        emissions: EmissionFamily,
        posterior: torch.Tensor,
        covariate: torch.Tensor,
    ) -> torch.Tensor:
        """The covariate to score the next E step against."""
        ...


@dataclass(frozen=True)
class ExpectedRateNormalizer:
    """Divide each sequence's exposure by ``Z_c = sum_g lambda_g E[mu_{s_c(g)}]`` (issue #933).

    A consumer whose rate at position ``g`` of sequence ``c`` is ``lambda_g
    mu_s / Z_c`` normalizes each sequence to one: ``Z_c`` sums the unnormalized
    rates over the sequence's positions, and it depends on the states the
    sequence takes, so no fixed covariate can hold it. This is its plug-in
    form: the state at each position is averaged over the previous E step's
    posterior, so ``Z_c`` moves with the parameters and the posterior and the
    fit is expectation--conditional-maximization around it, the emission M
    step conditioning on ``Z`` held at its current value.

    ``mu_k`` is the family's first alignment coordinate: the mean, in
    observation units, of every count family and of a count pair's total
    channel. The rates are unchanged by scaling every ``mu_k`` together,
    since ``Z_c`` scales with them, so only their ratios are identified.

    Parameters
    ----------
    weights : torch.Tensor
        ``lambda``, shape ``(length,)`` for one per position or
        ``(n_sequences, length)`` for one per position of each sequence.
    channel : int | None
        The covariate channel holding the exposure, for a family whose
        covariate carries one per channel (a count pair's is channel 0);
        ``None`` for a single-channel family.
    """

    weights: torch.Tensor
    channel: int | None = None

    def normalizer(
        self, emissions: EmissionFamily, posterior: torch.Tensor
    ) -> torch.Tensor:
        """``Z_c`` per sequence, shape ``(n_sequences,)``."""
        means = emissions.alignment_key()[:, 0].to(posterior.dtype)
        expected = posterior @ means  # (n_sequences, length)
        return (self.weights.to(posterior.dtype) * expected).sum(dim=-1)

    def __call__(
        self,
        emissions: EmissionFamily,
        posterior: torch.Tensor,
        covariate: torch.Tensor,
    ) -> torch.Tensor:
        """``covariate`` with its exposure divided by each sequence's ``Z_c``."""
        divisor = self.normalizer(emissions, posterior)[:, None]
        updated = covariate.clone()
        if self.channel is None:
            updated[..., 0] = covariate[..., 0] / divisor
        else:
            updated[..., self.channel] = covariate[..., self.channel] / divisor
        return updated


def _streamed_baum_welch(
    observations: np.ndarray,
    log_initial: torch.Tensor,
    log_transition: torch.Tensor,
    log_emission: torch.Tensor,
    *,
    max_iterations: int,
    tolerance: float,
) -> CategoricalFit:
    """:func:`baum_welch` on the compiled step, driven by the same :func:`em_loop`."""
    # Borrowed where NumPy already holds int64 rows; a copy only otherwise.
    symbols = np.ascontiguousarray(observations, dtype=np.int64)
    m, n_symbols = log_emission.shape

    def step(
        state: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], float]:
        initial, transition, emission, log_likelihood = oxisal.categorical_em_step(
            symbols, *state
        )
        return (initial, transition, emission), log_likelihood

    def flat(tensor: torch.Tensor) -> np.ndarray:
        return np.ascontiguousarray(tensor.detach().numpy(), dtype=np.float64).reshape(
            -1
        )

    (initial, transition, emission), log_likelihood, _ = em_loop(
        step,
        (flat(log_initial), flat(log_transition), flat(log_emission)),
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    # Shaped in NumPy and wrapped without a copy: a first torch `reshape` in a
    # process costs 2.1 MB of resident memory, 60 times the fit's own.
    return CategoricalFit(
        torch.from_numpy(initial),
        torch.from_numpy(transition.reshape(m, m)),
        torch.from_numpy(emission.reshape(m, n_symbols)),
        log_likelihood,
    )


#: The count families :func:`baum_welch_family` streams through a table: one
#: channel, no covariate, and an M step that reads only weighted counts.
_TABLED = (
    PoissonEmission,
    BinomialEmission,
    NegativeBinomialEmission,
    BetaBinomialEmission,
)


#: The most cells a count table may span, ``(max count + 1) * (max covariate
#: + 1)``: its row lookup is four bytes a cell, so 4 MB at most.
_MOST_CELLS = 1 << 20


def _streams(
    observations: np.ndarray | Ragged,
    log_transition: torch.Tensor,
    emissions: EmissionFamily,
    covariate: np.ndarray | Ragged | None,
) -> bool:
    """Whether :func:`baum_welch_family` has a streamed step for this case (issue #997)."""
    m = emissions.n_states
    if (
        not isinstance(observations, np.ndarray)
        or observations.ndim != 2
        or observations.size == 0
        or log_transition.shape != (m, m)
    ):
        return False
    if type(emissions) is GaussianEmission:
        return covariate is None and emissions.n_channels == 1
    # Integer counts and an integer covariate: the table is indexed by them.
    if type(emissions) not in _TABLED or not _whole(observations):
        return False
    stride = 1
    if covariate is not None:
        if not (
            isinstance(covariate, np.ndarray)
            and covariate.shape == observations.shape
            and _whole(covariate)
        ):
            return False
        stride = int(covariate.max()) + 1
    return (int(observations.max()) + 1) * stride <= _MOST_CELLS


def _whole(values: np.ndarray) -> bool:
    """Whether ``values`` is an integer array with nothing below zero."""
    return np.issubdtype(values.dtype, np.integer) and int(values.min()) >= 0


def _direct_scoring(family: EmissionFamily) -> dict[str, Any]:
    """The compiled step's name and stacked parameters for ``family`` (issue #997)."""
    rows: list[torch.Tensor]
    if isinstance(family, PoissonEmission):
        name, rows = "poisson", [family.mean]
    elif isinstance(family, BinomialEmission):
        name, rows = "binomial", [family.trials, family.probability]
    elif isinstance(family, NegativeBinomialEmission):
        name, rows = "negative_binomial", [family.dispersion, family.mean]
    else:
        assert isinstance(family, BetaBinomialEmission)
        name, rows = "beta_binomial", [family.trials, family.alpha, family.beta]
    parameters = np.concatenate([row.detach().numpy().reshape(-1) for row in rows])
    return {"family": name, "parameters": parameters.astype(np.float64)}


def _streamed_family(
    observations: np.ndarray,
    log_initial: torch.Tensor,
    log_transition: torch.Tensor,
    emissions: EmissionFamily,
    *,
    max_iterations: int,
    tolerance: float,
    covariate: np.ndarray | None = None,
    with_table: bool = True,
    table_size: int | None = None,
    approx: bool = False,
    stirling_from: float = 10.0,
) -> EmFit:
    """:func:`baum_welch_family` on a compiled step that streams (issue #997).

    A one-channel Gaussian is scored and re-estimated in
    ``oxisal.gaussian_em_step``. A count family is scored in
    ``count_em_step``, term for term its ``log_density`` with a compiled
    ``lgamma``: once per cell for the ``table_size`` cells the data repeats
    most --- a cell is a distinct count, or a distinct pair of a count and its
    covariate --- and at every observation for the rest. The step returns each state's posterior
    weight on each cell, and the family's own ``reestimate`` runs on those,
    which is its M step exactly, the order of summation aside. No gradient is
    taken here, so torch's ``log_density`` is not needed on this route; it
    stays the batched oracle and the autodiff objectives' density.
    """
    m = emissions.n_states
    at_boundary = False

    def flat(tensor: torch.Tensor) -> np.ndarray:
        return np.ascontiguousarray(tensor.detach().numpy(), dtype=np.float64).reshape(
            -1
        )

    if isinstance(emissions, GaussianEmission):
        values = np.ascontiguousarray(observations, dtype=np.float64)
        floor = emissions.variance_floor

        def gaussian(
            state: tuple[np.ndarray, np.ndarray, EmissionFamily],
        ) -> tuple[tuple[np.ndarray, np.ndarray, EmissionFamily], float]:
            initial, transition, family = state
            assert isinstance(family, GaussianEmission)
            initial, transition, mean, variance, log_likelihood = (
                oxisal.gaussian_em_step(
                    values, initial, transition, flat(family.mean), flat(family.scale)
                )
            )
            refuse_collapsed(torch.from_numpy(variance), floor)
            fitted = GaussianEmission(mean, np.sqrt(variance), floor)
            return (initial, transition, fitted), log_likelihood

        step = gaussian
    else:
        # Borrowed where NumPy already holds int64 rows; a copy only otherwise.
        counts = np.ascontiguousarray(observations, dtype=np.int64)
        given = (
            None
            if covariate is None
            else np.ascontiguousarray(covariate, dtype=np.int64)
        )
        stride = 1 if given is None else int(given.max()) + 1
        cells, multiplicity = oxisal.count_cells(counts, given, stride)
        rows = np.zeros((int(counts.max()) + 1) * stride, dtype=np.uint32)
        rows[cells] = np.arange(cells.size, dtype=np.uint32)
        # The table holds the `table_size` cells that repeat most; ties go to
        # the lower cell, so the choice depends on the data alone.
        size = 0 if not with_table else cells.size if table_size is None else table_size
        in_table = np.zeros(cells.size, dtype=bool)
        in_table[np.argsort(-multiplicity, kind="stable")[: max(size, 0)]] = True
        support = torch.from_numpy((cells // stride).astype(np.float64))
        # The family broadcasts a covariate along its states, as
        # `baum_welch_family` hands it one.
        exposure = (
            None
            if given is None
            else torch.from_numpy((cells % stride).astype(np.float64)).unsqueeze(1)
        )

        def tabled(
            state: tuple[np.ndarray, np.ndarray, EmissionFamily],
        ) -> tuple[tuple[np.ndarray, np.ndarray, EmissionFamily], float]:
            nonlocal at_boundary
            initial, transition, family = state
            initial, transition, histogram, log_likelihood = oxisal.count_em_step(
                counts,
                given,
                stride,
                cells,
                rows,
                initial,
                transition,
                **_direct_scoring(family),
                tabled=in_table,
                approx=approx,
                stirling_from=stirling_from,
            )
            reestimate = family.reestimate(
                support, torch.from_numpy(histogram.reshape(-1, m)), covariate=exposure
            )
            if not reestimate.converged:
                msg = (
                    f"the emission M step did not settle after "
                    f"{reestimate.iterations} iterations, at a relative change "
                    f"of {reestimate.residual:.3e}"
                )
                raise ValueError(msg)
            at_boundary = at_boundary or reestimate.at_boundary
            return (initial, transition, reestimate.emissions), log_likelihood

        step = tabled

    (initial, transition, fitted), log_likelihood, termination = em_loop(
        step,
        (flat(log_initial), flat(log_transition), emissions),
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    return EmFit(
        log_initial=torch.from_numpy(initial),
        log_transition=torch.from_numpy(transition.reshape(m, m)),
        emissions=fitted,
        log_likelihood=log_likelihood,
        emission_at_boundary=at_boundary,
        termination=termination,
    )


def viterbi(
    observations: np.ndarray,
    log_initial: torch.Tensor,
    log_transition: torch.Tensor,
    emissions: EmissionFamily,
    backend: Backend = Backend.RUST,
) -> tuple[np.ndarray, float]:
    """The most probable hidden path of every sequence, and their total log-probability.

    Parameters
    ----------
    observations : np.ndarray
        Shape ``(n_sequences, length)``: symbols, real values or counts, as
        ``emissions`` scores them.
    log_initial, log_transition : torch.Tensor
        Log-probabilities, ``(m,)`` and ``(m, m)``.
    emissions : EmissionFamily
        The emission family.
    backend : Backend
        :data:`~snakes_and_ladders.backend.Backend.RUST`, the default,
        decodes the sequences in parallel in ``oxisal.
        hmm_viterbi`` where ``emissions`` is exactly a categorical, a
        one-channel Gaussian or a count family; any other family, and
        :data:`~snakes_and_ladders.backend.Backend.PYTHON`, take the NumPy
        recursion here, which is the oracle (issue #997).

    Returns
    -------
    tuple[np.ndarray, float]
        The paths, ``(n_sequences, length)`` ``int64``, and the sum over
        sequences of each path's joint log-probability. A tie goes to the
        lower state.
    """
    refuse_backend("viterbi", backend, (Backend.PYTHON, Backend.RUST))
    values = np.asarray(observations)
    compiled = _viterbi_family(emissions)
    if backend is Backend.RUST and compiled is not None and values.ndim == 2:
        name, parameters = compiled
        # Symbols and counts are read as the int64 NumPy holds them.
        dtype = np.int64 if np.issubdtype(values.dtype, np.integer) else np.float64
        states, log_probability = oxisal.hmm_viterbi(
            np.ascontiguousarray(values, dtype=dtype),
            np.ascontiguousarray(log_initial.detach().numpy(), dtype=np.float64),
            np.ascontiguousarray(
                log_transition.detach().numpy(), dtype=np.float64
            ).reshape(-1),
            name,
            parameters,
        )
        return states.reshape(values.shape), float(log_probability)
    emit = emissions.log_density(
        torch.as_tensor(values, dtype=emissions.observation_dtype)
    ).numpy()
    kernel = log_transition.detach().numpy()
    n_sequences, length = values.shape
    delta = log_initial.detach().numpy() + emit[:, 0]
    back = np.empty((n_sequences, length, delta.shape[1]), dtype=np.int64)
    for t in range(1, length):
        scores = delta[:, :, None] + kernel[None]
        back[:, t] = np.argmax(scores, axis=1)
        delta = np.take_along_axis(scores, back[:, t][:, None, :], axis=1)[:, 0]
        delta = delta + emit[:, t]
    states = np.empty((n_sequences, length), dtype=np.int64)
    states[:, -1] = np.argmax(delta, axis=1)
    for t in range(length - 1, 0, -1):
        states[:, t - 1] = np.take_along_axis(back[:, t], states[:, t, None], axis=1)[
            :, 0
        ]
    return states, float(delta.max(axis=1).sum())


def hmm_log_likelihood(
    observations: np.ndarray,
    log_initial: torch.Tensor,
    log_transition: torch.Tensor,
    emissions: EmissionFamily,
    backend: Backend = Backend.RUST,
) -> float:
    """The summed log-likelihood of every sequence at given parameters, with no gradient.

    The number hmmlearn's ``score`` and :func:`forward_log_likelihood_from_density`
    report. Returned as a ``float``, since no gradient is taken: a caller that
    needs one differentiates :func:`forward_log_likelihood_from_density`.

    Parameters
    ----------
    observations : np.ndarray
        Shape ``(n_sequences, length)``, as ``emissions`` scores them.
    log_initial, log_transition : torch.Tensor
        Log-probabilities, ``(m,)`` and ``(m, m)``.
    emissions : EmissionFamily
        The emission family.
    backend : Backend
        :data:`~snakes_and_ladders.backend.Backend.RUST`, the default, runs
        the scaled forward pass over the sequences in parallel in
        ``oxisal.hmm_score`` for the families :func:`viterbi`
        compiles, and forms no ``(n_sequences, length, m)`` array;
        :data:`~snakes_and_ladders.backend.Backend.PYTHON`, and any other
        family, take :func:`forward_log_likelihood_from_density`, the oracle
        (issue #997).

    Returns
    -------
    float
    """
    refuse_backend("hmm_log_likelihood", backend, (Backend.PYTHON, Backend.RUST))
    values = np.asarray(observations)
    compiled = _viterbi_family(emissions)
    if backend is Backend.RUST and compiled is not None and values.ndim == 2:
        name, parameters = compiled
        dtype = np.int64 if np.issubdtype(values.dtype, np.integer) else np.float64
        return float(
            oxisal.hmm_score(
                np.ascontiguousarray(values, dtype=dtype),
                np.ascontiguousarray(log_initial.detach().numpy(), dtype=np.float64),
                np.ascontiguousarray(
                    log_transition.detach().numpy(), dtype=np.float64
                ).reshape(-1),
                name,
                parameters,
            )
        )
    with torch.no_grad():
        return float(
            forward_log_likelihood_from_density(
                emissions.log_density(
                    torch.as_tensor(values, dtype=emissions.observation_dtype)
                ),
                log_initial,
                log_transition,
            )
        )


def _viterbi_family(emissions: EmissionFamily) -> tuple[str, np.ndarray] | None:
    """The compiled Viterbi's name and parameters for ``emissions``, or ``None``."""
    if type(emissions) is CategoricalEmission:
        return "categorical", np.ascontiguousarray(
            emissions.log_matrix.detach().numpy(), dtype=np.float64
        ).reshape(-1)
    if type(emissions) is GaussianEmission and emissions.n_channels == 1:
        stacked = torch.cat([emissions.mean, emissions.scale]).detach().numpy()
        return "gaussian", np.ascontiguousarray(stacked, dtype=np.float64)
    if type(emissions) in _TABLED:
        scoring = _direct_scoring(emissions)
        return str(scoring["family"]), scoring["parameters"]
    return None


def baum_welch_family(
    observations: np.ndarray | Ragged,
    log_initial: torch.Tensor,
    log_transition: torch.Tensor,
    emissions: EmissionFamily,
    max_iterations: int = 500,
    tolerance: float = 1e-12,
    covariate: np.ndarray | Ragged | None = None,
    *,
    update: CovariateUpdate | None = None,
    fit_transition: bool = True,
    backend: Backend = Backend.RUST,
    with_table: bool = True,
    table_size: int | None = None,
    approx: bool = False,
    stirling_from: float = 10.0,
) -> EmFit:
    """Baum-Welch over any emission family, with no autodiff involved.

    Takes a rectangular ``(n_sequences, length, ...)`` array as it always has,
    or a `Ragged` batch whose segments differ in length. There is one recursion
    underneath either: the rectangular form converts, and the route it replaced
    is conserved in `sandbox.rectangular_hmm` as the referee of that case
    (issue #666).

    The E step is the model: forward and backward messages in log space,
    identical whatever a state emits, and so is the M step for the initial
    distribution and the transitions, both simplex-valued for every family.
    Only the emission M step differs, and it is delegated to the family. The
    alternation around them is :func:`snakes_and_ladders.opt.em.em_loop`'s
    (issue #859); what this function computes in one iteration is below.

    Parameters
    ----------
    observations : np.ndarray
        Observations, shape ``(n_sequences, length)`` followed by whatever
        trailing axes the family's observation carries --- none for a scalar
        observation, a channel axis for a family over a pair of counts.
        Symbol indices or real values, as the family says.
    emissions : EmissionFamily
        Starting emission family.
    max_iterations : int
        Maximum EM iterations.
    tolerance : float
        Stop when the log-likelihood improves by less than this *relative*
        to its magnitude -- absolute would not transfer across data sizes
        (``DEV.md``, issue #111).
    log_initial, log_transition : torch.Tensor
        Starting parameters, as log-probabilities. ``log_transition`` is
        ``(m, m)``, one kernel for the whole chain; ``(length - 1, m, m)``, one
        per step (issue #658); or ``(n_sequences, length - 1, m, m)``, one per
        step of each sequence (issue #933), where a sequence's steps past its
        own end are never read. ``length`` is the longest sequence's.

        **A per-step kernel is conditioned on, not fitted.** It carries
        ``(length - 1) * m * (m - 1)`` free values against ``length - 1``
        transitions per sequence, so at the sequence counts this repository
        runs it is not identifiable and an M step that re-estimated it would
        return the posterior it was handed. Given one, this function holds it
        fixed and fits the initial distribution and the emissions --- the same
        standing a covariate has, and for the same reason. A kernel per
        sequence is held fixed on the same terms. Given a single matrix it
        fits that matrix, exactly as before.
    covariate : np.ndarray | None
        What each observation is scored against, carrying the observations'
        leading ``(n_sequences, length)`` -- an exposure for a rate family, a trial count for a
        bounded one (issue #652). It reaches both seams of the loop, the E
        step's scoring and the emission M step, because a fit that scores
        against an exposure and re-estimates without it is fitting two
        different models. ``None`` is the model this function had before.
        The family broadcasts a covariate along the states, so it wants a trailing singleton axis; the covariate is stored with the observations' own axes and the singleton is added here, where the observation layout is known. A caller should not have to carry a shape that exists for the family's broadcast.
    update : CovariateUpdate | None
        A covariate that depends on the parameters (issue #933): before every
        E step it maps the family, the previous posterior and ``covariate`` to
        the covariate that E step and the next M step use. It needs a
        ``covariate`` to update. The reported log-likelihood is each E step's,
        at the covariate it was scored against, so it is the objective of the
        plug-in fit and need not rise monotonically.
    fit_transition : bool
        Whether the M step re-estimates a single ``(m, m)`` transition. With
        ``False`` it is held, as a per-step or per-sequence kernel always is
        (issue #933).
    backend : Backend
        The E step's recursion. The default,
        :data:`~snakes_and_ladders.backend.Backend.RUST`, walks each sequence
        in the compiled ragged kernel (issue #933): 19.4x the torch recursion
        on 200 chains of 100-3,000 positions. It takes one kernel for the
        whole chain, so a per-step or per-sequence kernel keeps the torch
        recursion under either. :data:`~snakes_and_ladders.backend.Backend.PYTHON`
        is the padded torch recursion and the oracle, agreeing within 2e-12.

    backend : Backend
        :data:`~snakes_and_ladders.backend.Backend.RUST`, the default since
        issue #997, streams the E step one sequence at a time into
        sufficient statistics, where the family is exactly a one-channel
        :class:`GaussianEmission`, or a :class:`PoissonEmission`,
        :class:`BinomialEmission`, :class:`NegativeBinomialEmission` or
        :class:`BetaBinomialEmission` over integer counts with no covariate or
        an integer one, the observations are a rectangular array, and the
        kernel is one ``(m, m)`` matrix.
    with_table : bool
        How the streamed count step scores an observation, read only there.
        ``True``, the default, scores each occupied (count, covariate) cell
        once and gathers, which pays where counts repeat. ``False`` scores
        every observation, which pays where they do not. Both hand the same
        weighted cells to the family's M step.
    table_size : int | None
        How many cells the table holds under ``with_table``: the ones the
        data repeats most, which for counts are mostly the low ones; every
        other observation is scored where it stands. ``None``, the default,
        tables every occupied cell; ``0`` is ``with_table=False``.
    approx : bool
        Whether the streamed count step takes ``lgamma`` from the Stirling
        series at ``stirling_from`` and above --- one logarithm and one
        division, where the Lanczos sum below takes fourteen divisions --- and
        is read only there. ``False`` by default: the Lanczos sum throughout,
        within 1e-13 of the exact log-factorials (issue #999).
    stirling_from : float
        Where the Stirling series takes over under ``approx``. At the default
        10 it is within 4e-15 relative of the Lanczos sum; its first omitted
        term is ``3617 / (122400 x^13)``, 2.4e-11 absolute at 5.

    Returns
    -------
    EmFit
        The fitted parameters, the final log-likelihood, and whether any M
        step reported a parameter at the edge of what the data identifies.

    Raises
    ------
    ValueError
        If the family refuses its own re-estimate. A Gaussian family does so
        when a state's variance reaches its floor, which is an approach to a
        degenerate optimum rather than a convergence, and is reported as such
        rather than clamped away.
    """
    refuse_backend("the Baum-Welch E step", backend, (Backend.PYTHON, Backend.RUST))
    # The streamed step re-estimates every block and updates no covariate;
    # a fit that holds the transition or updates the exposure takes the
    # general route.
    if (
        backend is Backend.RUST
        and update is None
        and fit_transition
        and _streams(observations, log_transition, emissions, covariate)
    ):
        assert isinstance(observations, np.ndarray)
        assert covariate is None or isinstance(covariate, np.ndarray)
        return _streamed_family(
            observations,
            log_initial,
            log_transition,
            emissions,
            max_iterations=max_iterations,
            tolerance=tolerance,
            covariate=covariate,
            with_table=with_table,
            table_size=table_size,
            approx=approx,
            stirling_from=stirling_from,
        )
    # One batch form, and a rectangular argument converts to it (issue #666).
    # The segments are padded to the longest and masked; the mask is not a
    # convenience but the whole of the claim that padding never reaches a
    # likelihood, so it is applied in the E step *and*, through `gamma`, in the
    # emission M step. The conserved rectangular route in `sandbox` referees
    # the equal-length case bit for bit.
    batch = (
        observations
        if isinstance(observations, Ragged)
        else Ragged.from_rectangular(np.asarray(observations))
    )
    # Zero is a value every family admits and every padded position is masked
    # out of the arithmetic below, so the fill is arbitrary and never scored.
    block, present = batch.padded(fill=0)
    # The leading two axes are the sequence and the position. What follows them
    # is the family's own: none where an observation is a scalar, and one or
    # more where it is not --- a family over a pair of counts carries a channel
    # axis. Unpacking the whole shape refused every such family outright, so a
    # family could be made to condition on a covariate and still not be
    # fittable here (issue #658).
    data = torch.as_tensor(block, dtype=emissions.observation_dtype)
    mask = torch.as_tensor(present)
    n_sequences, length = batch.n_segments, batch.longest
    #: The last live position of each segment, which is where its chain ends.
    final = torch.as_tensor(batch.lengths, dtype=torch.long) - 1
    rows = torch.arange(n_sequences)
    steps = torch.arange(length)
    m = emissions.n_states
    varying = log_transition.shape != (m, m)
    per_sequence = log_transition.shape == (n_sequences, max(length - 1, 0), m, m)
    if (
        varying
        and not per_sequence
        and log_transition.shape
        != (
            max(length - 1, 0),
            m,
            m,
        )
    ):
        msg = (
            f"log_transition {tuple(log_transition.shape)} is neither ({m}, {m}), "
            f"({max(length - 1, 0)}, {m}, {m}) nor ({n_sequences}, "
            f"{max(length - 1, 0)}, {m}, {m}) for {n_sequences} chains of up to "
            f"{length} positions over {m} states"
        )
        raise ValueError(msg)
    kernels = (
        log_transition if varying else log_transition.expand(max(length - 1, 0), m, m)
    )
    # The trailing singleton the single-channel families broadcast over their
    # states with is added only where the covariate has no axes of its own; a
    # covariate carrying the family's axes is passed through, because the
    # singleton then belongs inside each of them and the family is what puts it
    # there (issue #658).
    exposure: torch.Tensor | None = None
    if covariate is not None:
        # The covariate is segmented exactly as the observations are, and is
        # padded with one rather than zero: it multiplies a rate or replaces a
        # trial count, and zero would be a division where the mask has not yet
        # removed the row (issue #658's neutral value, issue #666's padding).
        carried = (
            covariate
            if isinstance(covariate, Ragged)
            else Ragged.from_rectangular(np.asarray(covariate))
        )
        given, _ = carried.padded(fill=1)
        exposure = torch.as_tensor(given, dtype=torch.float64)
        if exposure.ndim == data.ndim == 2:
            exposure = exposure[..., None]
    if update is not None and exposure is None:
        msg = "a covariate update needs a covariate to update"
        raise ValueError(msg)
    compiled = backend is Backend.RUST and not varying
    lengths = np.asarray(batch.lengths, dtype=np.int64)
    # The posterior an update reads before the first E step: uniform over the
    # states, and zero at padded positions as every later one is.
    previous = mask.unsqueeze(2).expand(-1, -1, m).to(torch.float64) / m

    at_boundary = False

    def iterate(
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, EmissionFamily],
    ) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor, EmissionFamily], float]:
        """One E step, one M step, and the log-likelihood at the state given.

        The state is the initial distribution, the transition matrix, the
        per-step kernels it is expanded to and the emission family --- every
        parameter the recursion below reads and the M step rewrites.
        """
        nonlocal at_boundary, previous
        log_initial, log_transition, kernels, emissions = state
        scored = (
            exposure
            if update is None or exposure is None
            else update(emissions, previous, exposure)
        )
        # --- E step: forward and backward messages in log space ----------
        emit = emissions.log_density(data, covariate=scored)
        # A padded position scores log 1, so it adds nothing wherever it is
        # reached. Its `alpha` beyond the segment's end is still nonsense, which
        # is why the evidence is gathered at each segment's own last position
        # rather than read off the block's last column.
        emit = torch.where(mask.unsqueeze(2), emit, torch.zeros_like(emit))
        transition_counts: torch.Tensor | None
        if compiled:
            # The ragged kernel walks each sequence in place and returns the
            # log marginals, the log transition counts summed over sequences,
            # and each sequence's evidence (issue #933). It takes one kernel.
            gamma, transition_counts, log_likelihood = _ragged_e_step(
                emit, mask, lengths, log_initial, log_transition
            )
        else:
            alpha = torch.empty((n_sequences, length, m), dtype=log_initial.dtype)
            alpha[:, 0] = log_initial.unsqueeze(0) + emit[:, 0]
            for t in range(1, length):
                # One kernel for every sequence, or each sequence's own (#933).
                kernel = (
                    kernels[:, t - 1]
                    if per_sequence
                    else (kernels[t - 1] if varying else log_transition).unsqueeze(0)
                )
                alpha[:, t] = (
                    torch.logsumexp(alpha[:, t - 1].unsqueeze(2) + kernel, dim=1)
                    + emit[:, t]
                )
            beta = torch.zeros((n_sequences, length, m), dtype=log_initial.dtype)
            for t in range(length - 2, -1, -1):
                kernel = (
                    kernels[:, t]
                    if per_sequence
                    else (kernels[t] if varying else log_transition).unsqueeze(0)
                )
                onward = torch.logsumexp(
                    kernel + (emit[:, t + 1] + beta[:, t + 1]).unsqueeze(1),
                    dim=2,
                )
                # At or past a segment's last position the chain has ended: beta is
                # one, not whatever the next column carries. This is the backward
                # half of "the recursions restart at each boundary".
                ended = (t >= final).unsqueeze(1)
                beta[:, t] = torch.where(ended, torch.zeros_like(onward), onward)

            evidence = torch.logsumexp(alpha[rows, final], dim=1)
            log_likelihood = float(evidence.sum())

            gamma = torch.where(
                mask.unsqueeze(2),
                alpha + beta - evidence[:, None, None],
                torch.full_like(alpha, -float("inf")),
            )
            # A pair spans positions t and t+1, so it exists only where t is before
            # the segment's last position. The pair that would straddle a boundary
            # is not a transition the model took and is not counted as one.
            pairs = (steps[:-1] < final.unsqueeze(1))[:, :, None, None]
            xi = torch.where(
                pairs,
                alpha[:, :-1].unsqueeze(3)
                + (kernels if per_sequence else kernels.unsqueeze(0))
                + (emit[:, 1:] + beta[:, 1:]).unsqueeze(2)
                - evidence[:, None, None, None],
                torch.full(
                    (n_sequences, length - 1, m, m), -float("inf"), dtype=alpha.dtype
                ),
            )

            transition_counts = (
                None if varying else torch.logsumexp(xi.reshape(-1, m, m), dim=0)
            )

        # --- M step: normalized expected counts, then the family's own ---
        log_initial = torch.logsumexp(gamma[:, 0], dim=0) - torch.log(
            torch.tensor(float(n_sequences), dtype=gamma.dtype)
        )
        if fit_transition and transition_counts is not None:
            log_transition = transition_counts - torch.logsumexp(
                transition_counts, dim=1, keepdim=True
            )
            kernels = log_transition.expand(max(length - 1, 0), m, m)
        previous = torch.exp(gamma)
        step = emissions.reestimate(data, previous, covariate=scored)
        if not step.converged:
            msg = (
                f"the emission M step did not settle after {step.iterations} "
                f"iterations, at a relative change of {step.residual:.3e}: a "
                f"parameter read off iterations that never converged is not an "
                f"estimate, and a monotone outer likelihood would not have "
                f"shown it"
            )
            raise ValueError(msg)
        at_boundary = at_boundary or step.at_boundary
        return (log_initial, log_transition, kernels, step.emissions), log_likelihood

    (log_initial, log_transition, _, emissions), log_likelihood, termination = em_loop(
        iterate,
        (log_initial, log_transition, kernels, emissions),
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    return EmFit(
        log_initial=log_initial,
        log_transition=log_transition,
        emissions=emissions,
        log_likelihood=log_likelihood,
        emission_at_boundary=at_boundary,
        termination=termination,
    )
