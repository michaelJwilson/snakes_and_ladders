"""A discrete HMM: the second reference instance of ``Objective``.

Chosen over a second tree-like model for two reasons. It is one of issue
#63's named cases, and unlike the Potts chain it has an *independent fitting
algorithm* -- Baum-Welch -- so a gradient fit can be checked against
something other than itself (that check lands with the optimizer; this module
supplies the model it will be run on).

The forward recursion here and Felsenstein pruning are the same sum-product
computation on different graphs: a caterpillar tree carrying one observed
leaf per internal node *is* an HMM (Durbin et al., ch. 3; Koller & Friedman
for the general framing). The abstraction is expected to hold because the
three instances are one marginalization over three structures, not three
unrelated models sharing an optimizer.

**Label switching.** The likelihood is invariant to permuting the hidden
states, so a fitted parameter set matches truth only up to a permutation. The
model is otherwise identifiable; every row is gauge-fixed by
:func:`snakes_and_ladders.opt.constrain.log_simplex`. A recovery test must align the
permutation before comparing.

Ground truth and data generation live in :mod:`snakes_and_ladders.sim.hmm`; this module
holds only the fitting objective, its independent EM oracle, and the
state-alignment helper a recovery test needs.
"""

from __future__ import annotations

from collections.abc import Mapping
from itertools import permutations

import numpy as np
import torch

from snakes_and_ladders.emissions import (
    CategoricalEmission,
    EmissionFamily,
    GaussianEmission,
    pooled_variance_floor,
)
from snakes_and_ladders.opt.constrain import free_from_log_simplex, log_simplex

# How far apart the emission rows start, in unconstrained units. Large
# enough to leave the stationary point, small enough not to preselect an
# answer: at 1.0 a state favours its symbol by a factor of e.
_SYMMETRY_BREAK = 1.0


class _HmmObjective:
    """The part of an HMM objective that does not know what a state emits.

    The initial distribution and the transition matrix are simplex-valued
    whatever the observations are, and the forward recursion consumes a
    ``(n_sequences, length, n_states)`` block of per-site scores without
    caring how they were produced. Everything else is the emission family's,
    and lives in the subclass that names one.
    """

    def __init__(
        self,
        observations: np.ndarray,
        n_states: int,
        observation_dtype: torch.dtype,
        dtype: torch.dtype,
    ) -> None:
        self._observations = torch.as_tensor(observations, dtype=observation_dtype)
        self._n_states = n_states
        self._dtype = dtype

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
            self.emissions(theta).log_density(self._observations),
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
    ) -> None:
        super().__init__(observations, n_states, torch.long, dtype)
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

    def initial(self) -> torch.Tensor:
        """A start that is uninformative but **not** symmetric.

        The uniform point -- every distribution uniform -- is a stationary
        point of the likelihood, not merely a poor guess: with all hidden
        states identical, the gradient with respect to the initial and
        transition parameters is exactly zero, and an optimizer started there
        stays there forever while the emission rows converge to the pooled
        symbol frequency. `tests/regression/test_opt_hmm.py` pins that.

        So the emission rows are tilted apart by a fixed amount, each state
        favouring a different symbol. Deterministic rather than random: a
        seeded jitter would make the fit depend on a second seed nobody
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
    only what a state emits differs. Two things follow that the categorical
    instance never had to face.

    **The objective may be negative for a *good* fit.** The emission term is a
    density, so the evidence is a density and the negative log-likelihood this
    returns can be below zero. Nothing about that is pathological.

    **The likelihood has no maximum.** Put one state's mean on a single
    observation and let its scale go to zero and the objective diverges to
    minus infinity, so ``fit`` reporting convergence means it satisfied the
    first-order condition somewhere, not that it found the supremum ---
    ``opt/CLAUDE.md``'s rule about ``converged``, in a model where the
    distinction is not a technicality. :class:`GaussianEmission`'s variance
    floor is what makes an approach to that boundary visible in the EM oracle;
    a gradient fit is protected only by where it starts, which is why
    :meth:`initial` places the means on the data rather than at a point.

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
    ) -> None:
        super().__init__(observations, n_states, torch.float64, dtype)
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

    def initial(self) -> torch.Tensor:
        """A start that is uninformative but **not** symmetric.

        Uniform transitions, as for the categorical instance and for the same
        reason: the symmetric point is a stationary point, not a poor guess.
        The means are placed at evenly spaced quantiles of the pooled
        observations, which breaks the exchangeability the way the categorical
        tilt does while committing to nothing about which state is which.

        On the data rather than at a fixed point, because a Gaussian mean far
        from every observation contributes a density that underflows: the
        state is then invisible to the E step and the fit reduces to one with
        fewer states. The scales start at the pooled standard deviation, the
        widest defensible value --- a start that is too *narrow* is the
        direction the likelihood is unbounded in.
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


def forward_log_likelihood_from_density(
    log_density: torch.Tensor,
    log_initial: torch.Tensor,
    log_transition: torch.Tensor,
) -> torch.Tensor:
    """Total log-likelihood from per-site emission scores already computed.

    The recursion that does not know what a state emits. Splitting it out is
    what lets one forward algorithm serve a family over an alphabet and a
    family over the reals: the categorical case gathers a row of the emission
    matrix, a Gaussian case evaluates a density, and both arrive here as the
    same block of numbers.

    Parameters
    ----------
    log_density : torch.Tensor
        Emission scores, shape ``(n_sequences, length, n_states)``. A
        log-probability for a discrete family, a log-density otherwise.
    log_initial : torch.Tensor
        Log initial distribution, shape ``(m,)``.
    log_transition : torch.Tensor
        Log transition matrix, shape ``(m, m)``.

    Returns
    -------
    torch.Tensor
        Scalar: the summed log-likelihood over sequences, differentiable with
        respect to every parameter. **Not** bounded above by zero where the
        emission family is continuous, since it then sums densities.
    """
    alpha = log_initial.unsqueeze(0) + log_density[:, 0]
    for t in range(1, log_density.shape[1]):
        alpha = (
            torch.logsumexp(alpha.unsqueeze(2) + log_transition.unsqueeze(0), dim=1)
            + log_density[:, t]
        )
    return torch.logsumexp(alpha, dim=1).sum()


def align_states(
    log_emission: torch.Tensor, reference: torch.Tensor
) -> tuple[int, ...]:
    """Permutation of fitted hidden states best matching ``reference``.

    The likelihood is invariant to relabelling the hidden states, so a fitted
    parameter set matches truth only up to a permutation and a recovery test
    has to choose one. Emissions are the discriminating signal -- two states
    with the same emission distribution are genuinely the same state -- so
    the permutation is the one minimizing total absolute emission
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
    categorical emission, means for a Gaussian one. Which is why the signature
    is the family's to define: aligning a Gaussian fit by emission *matrix*
    would compare two things that do not exist, and aligning it by anything
    but the mean would let two states with different means look identical.

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


def baum_welch(
    observations: np.ndarray,
    log_initial: torch.Tensor,
    log_transition: torch.Tensor,
    log_emission: torch.Tensor,
    max_iterations: int = 500,
    tolerance: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Fit an HMM by expectation-maximization, with no autodiff involved.

    This is the point of having it: Baum-Welch is a genuinely independent
    fitting algorithm for the same model, so a gradient fit checked against
    it is checked against something other than itself. It shares no code with
    ``fit`` -- not the optimizer, not the parameterization (EM works directly
    in probabilities, with no unconstrained coordinates and no constraint
    map), only the model.

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
    fitted_initial, fitted_transition, family, log_likelihood = baum_welch_family(
        observations,
        log_initial,
        log_transition,
        CategoricalEmission.from_log(log_emission),
        max_iterations=max_iterations,
        tolerance=tolerance,
    )
    if not isinstance(family, CategoricalEmission):  # pragma: no cover
        msg = f"expected a categorical M step, got {type(family).__name__}"
        raise TypeError(msg)
    return fitted_initial, fitted_transition, family.log_matrix, log_likelihood


def baum_welch_family(
    observations: np.ndarray,
    log_initial: torch.Tensor,
    log_transition: torch.Tensor,
    emissions: EmissionFamily,
    max_iterations: int = 500,
    tolerance: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, EmissionFamily, float]:
    """Baum-Welch over any emission family, with no autodiff involved.

    The E step is the model: forward and backward messages in log space,
    identical whatever a state emits. The M step for the initial distribution
    and the transitions is likewise identical, since both are simplex-valued
    for every family. Only the emission M step differs, and it is delegated to
    the family rather than written here --- which is the seam this exists to
    justify.

    Parameters
    ----------
    observations : np.ndarray
        Observations, shape ``(n_sequences, length)``. Symbol indices or real
        values, as the family says.
    log_initial, log_transition : torch.Tensor
        Starting parameters, as log-probabilities.
    emissions : EmissionFamily
        Starting emission family.
    max_iterations : int
        Maximum EM iterations.
    tolerance : float
        Stop when the log-likelihood improves by less than this *relative*
        to its magnitude -- absolute would not transfer across data sizes
        (``DEV.md``, issue #111).

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, EmissionFamily, float]
        Fitted log initial, log transition, emission family, and the final
        log-likelihood.

    Raises
    ------
    ValueError
        If the family refuses its own re-estimate. A Gaussian family does so
        when a state's variance reaches its floor, which is an approach to a
        degenerate optimum rather than a convergence, and is reported as such
        rather than clamped away.
    """
    data = torch.as_tensor(observations, dtype=emissions.observation_dtype)
    n_sequences, length = data.shape
    m = emissions.n_states

    previous = -float("inf")
    log_likelihood = previous
    for _ in range(max_iterations):
        # --- E step: forward and backward messages in log space ----------
        emit = emissions.log_density(data)
        alpha = torch.empty((n_sequences, length, m), dtype=log_initial.dtype)
        alpha[:, 0] = log_initial.unsqueeze(0) + emit[:, 0]
        for t in range(1, length):
            alpha[:, t] = (
                torch.logsumexp(
                    alpha[:, t - 1].unsqueeze(2) + log_transition.unsqueeze(0), dim=1
                )
                + emit[:, t]
            )
        beta = torch.zeros((n_sequences, length, m), dtype=log_initial.dtype)
        for t in range(length - 2, -1, -1):
            beta[:, t] = torch.logsumexp(
                log_transition.unsqueeze(0)
                + (emit[:, t + 1] + beta[:, t + 1]).unsqueeze(1),
                dim=2,
            )

        evidence = torch.logsumexp(alpha[:, -1], dim=1)
        log_likelihood = float(evidence.sum())

        gamma = alpha + beta - evidence[:, None, None]
        xi = (
            alpha[:, :-1].unsqueeze(3)
            + log_transition.unsqueeze(0).unsqueeze(0)
            + (emit[:, 1:] + beta[:, 1:]).unsqueeze(2)
            - evidence[:, None, None, None]
        )

        # --- M step: normalized expected counts, then the family's own ---
        log_initial = torch.logsumexp(gamma[:, 0], dim=0) - torch.log(
            torch.tensor(float(n_sequences), dtype=gamma.dtype)
        )
        transition_counts = torch.logsumexp(xi.reshape(-1, m, m), dim=0)
        log_transition = transition_counts - torch.logsumexp(
            transition_counts, dim=1, keepdim=True
        )
        emissions = emissions.reestimate(data, torch.exp(gamma))

        if abs(log_likelihood - previous) <= tolerance * abs(log_likelihood):
            break
        previous = log_likelihood

    return log_initial, log_transition, emissions, log_likelihood
