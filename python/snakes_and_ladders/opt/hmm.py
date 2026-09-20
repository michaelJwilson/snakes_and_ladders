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
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import permutations

import numpy as np
import torch

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
from snakes_and_ladders.numerics import constant_chain_kernel
from snakes_and_ladders.opt.constrain import free_from_log_simplex, log_simplex
from snakes_and_ladders.opt.objective import Objective
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
    ) -> None:
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
    ) -> None:
        super().__init__(observations, n_states, torch.long, dtype, covariate)
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
        parts = [
            free_from_log_simplex(torch.log(torch.as_tensor(part, dtype=self._dtype)))
            for part in (initial, transition, emission)
        ]
        return torch.cat([parts[0], parts[1].reshape(-1), parts[2].reshape(-1)])


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
    ) -> None:
        super().__init__(observations, n_states, torch.float64, dtype, covariate)
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
            torch.exp(theta[self._log_scale_slice()]),
            self._variance_floor,
        )

    def _free_emissions_from(self, named: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat(
            [named["mean"].reshape(-1), torch.log(named["scale"]).reshape(-1)]
        )

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
        quantiles = (torch.arange(self._n_states, dtype=self._dtype) + 0.5) / (
            self._n_states
        )
        theta[self._mean_slice()] = torch.quantile(values, quantiles)
        theta[self._log_scale_slice()] = torch.log(values.std())
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
                free_from_log_simplex(
                    torch.log(torch.as_tensor(initial, dtype=self._dtype))
                ),
                free_from_log_simplex(
                    torch.log(torch.as_tensor(transition, dtype=self._dtype))
                ).reshape(-1),
                torch.as_tensor(mean, dtype=self._dtype).reshape(-1),
                torch.log(torch.as_tensor(scale, dtype=self._dtype)).reshape(-1),
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
    ) -> None:
        super().__init__(observations, n_states, torch.float64, dtype, covariate)

    def _location_quantiles(self) -> torch.Tensor:
        """Evenly spaced quantiles of the pooled observations, one per state."""
        values = self._observations.reshape(-1).to(self._dtype)
        quantiles = (torch.arange(self._n_states, dtype=self._dtype) + 0.5) / (
            self._n_states
        )
        return torch.quantile(values, quantiles)

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
        return PoissonEmission(torch.exp(theta[self._emission_slice]))

    def _free_emissions_from(self, named: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return torch.log(named["mean"]).reshape(-1)

    def initial(self) -> torch.Tensor:
        """Uniform transitions; rates at quantiles of the pooled counts."""
        theta = torch.zeros(self.n_parameters, dtype=self._dtype)
        theta[self._emission_slice] = torch.log(
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
                torch.log(torch.as_tensor(mean, dtype=self._dtype)).reshape(-1),
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
    ) -> None:
        super().__init__(observations, n_states, dtype, covariate)
        self._trials = torch.as_tensor(trials, dtype=torch.float64).reshape(-1)

    @property
    def _n_emission_parameters(self) -> int:
        return self._n_states

    def emissions(self, theta: torch.Tensor) -> BinomialEmission:
        """The binomial family ``theta``'s emission block encodes."""
        return BinomialEmission(
            self._trials, torch.sigmoid(theta[self._emission_slice])
        )

    def _free_emissions_from(self, named: Mapping[str, torch.Tensor]) -> torch.Tensor:
        rate = named["probability"].reshape(-1)
        return torch.log(rate) - torch.log1p(-rate)

    def initial(self) -> torch.Tensor:
        """Uniform transitions; success rates from quantiles of the counts."""
        theta = torch.zeros(self.n_parameters, dtype=self._dtype)
        rate = (self._location_quantiles() / self._trials).clamp(
            _RATE_MARGIN, 1.0 - _RATE_MARGIN
        )
        theta[self._emission_slice] = torch.log(rate) - torch.log1p(-rate)
        return theta

    def theta_from_truth(
        self, initial: np.ndarray, transition: np.ndarray, probability: np.ndarray
    ) -> torch.Tensor:
        """Place a known ``(pi, A, p)`` in the unconstrained coordinates."""
        rate = torch.as_tensor(probability, dtype=self._dtype).reshape(-1)
        return torch.cat(
            [
                _free_transitions(initial, transition, self._dtype),
                torch.log(rate) - torch.log1p(-rate),
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
    ) -> None:
        super().__init__(observations, n_states, dtype, covariate)
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
            torch.exp(theta[self._log_alpha_slice()]),
            torch.exp(theta[self._log_beta_slice()]),
        )

    def _free_emissions_from(self, named: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat(
            [
                torch.log(named["alpha"]).reshape(-1),
                torch.log(named["beta"]).reshape(-1),
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
                torch.log(torch.as_tensor(alpha, dtype=self._dtype)).reshape(-1),
                torch.log(torch.as_tensor(beta, dtype=self._dtype)).reshape(-1),
            ]
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
    ) -> None:
        super().__init__(observations, n_states, torch.float64, dtype, covariate)

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
            torch.exp(theta[self._log_dispersion_slice()]),
            torch.exp(theta[self._log_mean_slice()]),
        )

    def _free_emissions_from(self, named: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat(
            [
                torch.log(named["dispersion"]).reshape(-1),
                torch.log(named["mean"]).reshape(-1),
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
        quantiles = (torch.arange(self._n_states, dtype=self._dtype) + 0.5) / (
            self._n_states
        )
        means = torch.quantile(values, quantiles).clamp_min(_MINIMUM_COUNT_MEAN)
        theta[self._log_mean_slice()] = torch.log(means)

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
                free_from_log_simplex(
                    torch.log(torch.as_tensor(initial, dtype=self._dtype))
                ),
                free_from_log_simplex(
                    torch.log(torch.as_tensor(transition, dtype=self._dtype))
                ).reshape(-1),
                torch.log(torch.as_tensor(dispersion, dtype=self._dtype)).reshape(-1),
                torch.log(torch.as_tensor(mean, dtype=self._dtype)).reshape(-1),
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
    """

    log_initial: torch.Tensor
    log_transition: torch.Tensor
    emissions: EmissionFamily
    log_likelihood: float
    emission_at_boundary: bool = False


def baum_welch(
    observations: np.ndarray,
    log_initial: torch.Tensor,
    log_transition: torch.Tensor,
    log_emission: torch.Tensor,
    max_iterations: int = 500,
    tolerance: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
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

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]
        Fitted log initial, log transition and log emission, and the final
        log-likelihood.
    """
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
    return (
        result.log_initial,
        result.log_transition,
        family.log_matrix,
        result.log_likelihood,
    )


def baum_welch_family(
    observations: np.ndarray | Ragged,
    log_initial: torch.Tensor,
    log_transition: torch.Tensor,
    emissions: EmissionFamily,
    max_iterations: int = 500,
    tolerance: float = 1e-12,
    covariate: np.ndarray | Ragged | None = None,
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
    Only the emission M step differs, and it is delegated to the family.

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
        Starting parameters, as log-probabilities. ``log_transition`` is either
        ``(m, m)``, one kernel for the whole chain, or ``(length - 1, m, m)``,
        one per step (issue #658).

        **A per-step kernel is conditioned on, not fitted.** It carries
        ``(length - 1) * m * (m - 1)`` free values against ``length - 1``
        transitions per sequence, so at the sequence counts this repository
        runs it is not identifiable and an M step that re-estimated it would
        return the posterior it was handed. Given one, this function holds it
        fixed and fits the initial distribution and the emissions --- the same
        standing a covariate has, and for the same reason. Given a single
        matrix it fits that matrix, exactly as before.
    covariate : np.ndarray | None
        What each observation is scored against, carrying the observations'
        leading ``(n_sequences, length)`` -- an exposure for a rate family, a trial count for a
        bounded one (issue #652). It reaches both seams of the loop, the E
        step's scoring and the emission M step, because a fit that scores
        against an exposure and re-estimates without it is fitting two
        different models. ``None`` is the model this function had before.
        The family broadcasts a covariate along the states, so it wants a trailing singleton axis; the covariate is stored with the observations' own axes and the singleton is added here, where the observation layout is known. A caller should not have to carry a shape that exists for the family's broadcast.

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
    if varying and log_transition.shape != (max(length - 1, 0), m, m):
        msg = (
            f"log_transition {tuple(log_transition.shape)} is neither ({m}, {m}) "
            f"nor ({max(length - 1, 0)}, {m}, {m}) for a chain of {length} "
            f"positions over {m} states"
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

    previous = -float("inf")
    log_likelihood = previous
    at_boundary = False
    for _ in range(max_iterations):
        # --- E step: forward and backward messages in log space ----------
        emit = emissions.log_density(data, covariate=exposure)
        # A padded position scores log 1, so it adds nothing wherever it is
        # reached. Its `alpha` beyond the segment's end is still nonsense, which
        # is why the evidence is gathered at each segment's own last position
        # rather than read off the block's last column.
        emit = torch.where(mask.unsqueeze(2), emit, torch.zeros_like(emit))
        alpha = torch.empty((n_sequences, length, m), dtype=log_initial.dtype)
        alpha[:, 0] = log_initial.unsqueeze(0) + emit[:, 0]
        for t in range(1, length):
            kernel = kernels[t - 1] if varying else log_transition
            alpha[:, t] = (
                torch.logsumexp(
                    alpha[:, t - 1].unsqueeze(2) + kernel.unsqueeze(0), dim=1
                )
                + emit[:, t]
            )
        beta = torch.zeros((n_sequences, length, m), dtype=log_initial.dtype)
        for t in range(length - 2, -1, -1):
            kernel = kernels[t] if varying else log_transition
            onward = torch.logsumexp(
                kernel.unsqueeze(0) + (emit[:, t + 1] + beta[:, t + 1]).unsqueeze(1),
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
            + kernels.unsqueeze(0)
            + (emit[:, 1:] + beta[:, 1:]).unsqueeze(2)
            - evidence[:, None, None, None],
            torch.full(
                (n_sequences, length - 1, m, m), -float("inf"), dtype=alpha.dtype
            ),
        )

        # --- M step: normalized expected counts, then the family's own ---
        log_initial = torch.logsumexp(gamma[:, 0], dim=0) - torch.log(
            torch.tensor(float(n_sequences), dtype=gamma.dtype)
        )
        if not varying:
            transition_counts = torch.logsumexp(xi.reshape(-1, m, m), dim=0)
            log_transition = transition_counts - torch.logsumexp(
                transition_counts, dim=1, keepdim=True
            )
            kernels = log_transition.expand(max(length - 1, 0), m, m)
        step = emissions.reestimate(data, torch.exp(gamma), covariate=exposure)
        if not step.converged:
            msg = (
                f"the emission M step did not settle after {step.iterations} "
                f"iterations, at a relative change of {step.residual:.3e}: a "
                f"parameter read off iterations that never converged is not an "
                f"estimate, and a monotone outer likelihood would not have "
                f"shown it"
            )
            raise ValueError(msg)
        emissions = step.emissions
        at_boundary = at_boundary or step.at_boundary

        if abs(log_likelihood - previous) <= tolerance * abs(log_likelihood):
            break
        previous = log_likelihood

    return EmFit(
        log_initial=log_initial,
        log_transition=log_transition,
        emissions=emissions,
        log_likelihood=log_likelihood,
        emission_at_boundary=at_boundary,
    )
