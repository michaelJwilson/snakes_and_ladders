"""The gradient-fit objective over any emission family, its per-family starts, and the metrics a tracked fit records.

:class:`EmissionHmmObjective` serves any family on segments through
:func:`~sal.opt.hmm.forward.forward_log_likelihood_ragged`;
:func:`family_start` is the start each retired per-family objective took,
and those six names survive one release as deprecated constructors of it
(issue #1189). Imports :mod:`~sal.opt.hmm.forward` alone of this package.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from sal import oxisal
from sal.backend import Backend, refuse_backend
from sal.emissions import (
    BetaBinomialEmission,
    BinomialEmission,
    CategoricalEmission,
    EmissionFamily,
    GaussianEmission,
    NegativeBinomialEmission,
    PoissonEmission,
    flat_channels,
    identifiable_dispersion_bound,
    pooled_variance_floor,
)
from sal.opt.constrain import (
    domain_blocks,
    free_from_log_simplex,
    free_from_positive,
    free_from_probability,
    log_simplex,
    positive,
    probability,
)
from sal.opt.hmm.forward import forward_log_likelihood_ragged
from sal.opt.initialize import quantile_locations
from sal.opt.objective import Objective, autograd_value_and_gradient
from sal.ragged import Ragged

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


class EmissionHmmObjective(Objective):
    """Negative log-likelihood of an HMM over any emission family, on segments (issue #1169).

    :class:`~sal.opt.emission_mixture.EmissionMixtureObjective` with the
    weights replaced by a Markov chain: ``theta`` is the ``K - 1`` free
    initial logits, then ``K`` rows of ``K - 1`` free transition logits, each
    through :func:`~sal.opt.constrain.log_simplex`, then each parameter the family names, through the
    map its :meth:`~sal.emissions.EmissionFamily.parameter_domains` declares.
    The family rebuilds itself from the named parameters
    (:meth:`~sal.emissions.EmissionFamily.with_parameters`), so a joint count
    pair is as reachable as a Gaussian. Each segment restarts at the initial
    distribution, and no transition spans a boundary
    (:func:`~sal.opt.hmm.forward.forward_log_likelihood_ragged`).

    No parameter is held fixed here: :class:`~sal.opt.objective.Restricted`
    over :attr:`blocks` does that (issue #1168).

    Parameters
    ----------
    observations : Ragged | np.ndarray
        The segments end to end with their lengths, or a rectangular
        ``(n_sequences, length, ...)`` batch, read as equal-length segments
        (:meth:`~sal.ragged.Ragged.from_rectangular`). Trailing axes are the
        family's own: a count pair carries a channel axis of two.
    start : EmissionFamily
        The family :meth:`initial` starts at, with uniform initial and
        transition probabilities; its
        :meth:`~sal.emissions.EmissionFamily.named_parameters` name the
        emission blocks of ``theta``, and its constants (a trial count, the
        joint form, a variance floor) are every iterate's. A start whose
        states are exchangeable is a stationary point (``opt/CLAUDE.md``),
        so the caller breaks the symmetry, as a quantile start does.
    covariate : np.ndarray | torch.Tensor | None
        Per-observation covariate in the layout the family's ``log_density``
        documents, aligned with ``observations``: ``(total, ...)`` beside a
        :class:`~sal.ragged.Ragged`, ``(n_sequences, length, ...)`` beside a
        rectangular batch. ``None`` scores without one.
    backend : Backend
        Where :meth:`value_and_gradient` runs.
        :data:`~sal.backend.Backend.RUST`, the default, is the compiled E
        step and one backward pass, or the kernel
        :meth:`supported_gradient` names where it holds (issue #1248); :data:`~sal.backend.Backend.JAX` the
        twin of :mod:`sal.opt.hmm.jax`, the retired per-family objectives'
        default, for a family it covers on segments of any length
        (:func:`sal.opt.hmm.jax.twinned`), count pairs included (issue
        #1206); :data:`~sal.backend.Backend.TORCH` autograd through
        :meth:`__call__`, the oracle both are pinned to, and which declares
        no JAX energy. Measured at 10^6 positions, three states, the JAX
        twin's runtime against the compiled route's: 0.54x for a Gaussian,
        0.60x Poisson, 0.65x categorical (issue #1189); 0.39x beta-binomial,
        0.58x negative binomial with exposure since the twin takes each
        rising factorial once per distinct count (issue #1206, 4-core host,
        where the twin before it measured 1.61x and 2.57x); on the count
        pair of ``tests/benchmarks/test_hmm_jax_count_pair_bench.py``, 7,644
        rows in 88 unequal segments, K = 7, 0.24x--0.32x. The default stays
        ``RUST``: JAX is an optional dependency, so a default of JAX would
        make the route depend on the environment.

    Raises
    ------
    ValueError
        If ``start`` has fewer than two states, a parameter lies outside its
        domain, the covariate does not align with the observations, or
        ``backend`` is ``JAX`` for a family it has no twin of.
    """

    def __init__(
        self,
        observations: Ragged | np.ndarray,
        start: EmissionFamily,
        *,
        covariate: np.ndarray | torch.Tensor | None = None,
        backend: Backend = Backend.RUST,
    ) -> None:
        refuse_backend(
            "an HMM objective's gradient",
            backend,
            (Backend.RUST, Backend.JAX, Backend.TORCH),
        )
        self._backend = backend
        self._jax: Callable[[np.ndarray], tuple[float, np.ndarray]] | None = None
        self._kernel: oxisal.SupportedEnergy | None = None
        self._kernel_read = False
        if start.n_states < 2:
            msg = f"an HMM has at least two states, got {start.n_states}"
            raise ValueError(msg)
        leading = 1
        if not isinstance(observations, Ragged):
            observations = Ragged.from_rectangular(np.asarray(observations))
            leading = 2
        k = start.n_states
        named = {
            name: torch.as_tensor(value, dtype=torch.float64)
            for name, value in start.named_parameters().items()
        }
        self._blocks_at = domain_blocks(named, start.parameter_domains(), k * k - 1)
        self._n_parameters = max(
            (block.stop for block in self._blocks_at.values()), default=k * k - 1
        )
        self._k = k
        self._start = named
        self._start_family = start
        self._build = start.with_parameters
        self._lengths = observations.lengths
        self._lengths_array = np.asarray(self._lengths, dtype=np.int64)
        self._first = torch.as_tensor(observations.offsets[:-1], dtype=torch.long)
        self._observations = torch.as_tensor(
            observations.values, dtype=start.observation_dtype
        )
        total = self._observations.shape[0]
        self._covariate: torch.Tensor | None = None
        if covariate is not None:
            values = torch.as_tensor(covariate, dtype=torch.float64)
            if values.dim() < leading or values.shape[:leading].numel() != total:
                msg = (
                    f"covariate {tuple(values.shape)} does not align with "
                    f"{total} observations"
                )
                raise ValueError(msg)
            self._covariate = values.reshape(total, *values.shape[leading:])
        if backend is Backend.JAX:
            from sal.opt.hmm.jax import twinned

            if not twinned(self):
                msg = f"no JAX twin for an HMM of {type(start).__name__}"
                raise ValueError(msg)

    @property
    def observations(self) -> torch.Tensor:
        """The segments end to end, ``(total, ...)``, in the family's dtype."""
        return self._observations

    @property
    def lengths(self) -> tuple[int, ...]:
        """One length per segment."""
        return self._lengths

    @property
    def covariate(self) -> torch.Tensor | None:
        """The per-observation covariate, ``(total, ...)``, or ``None``."""
        return self._covariate

    @property
    def n_states(self) -> int:
        """``K``."""
        return self._k

    @property
    def start(self) -> EmissionFamily:
        """The family :meth:`initial` starts at: its type and constants are every iterate's (issue #1189)."""
        return self._start_family

    def rectangular(self) -> tuple[int, int] | None:
        """``(n_sequences, length)`` where every segment has one length; ``None`` otherwise."""
        if len(set(self._lengths)) != 1:
            return None
        return len(self._lengths), self._lengths[0]

    def supported_gradient(self) -> tuple[str, dict[str, object]] | None:
        """``oxisal``'s Gaussian HMM kernel on the sequences (:class:`~sal.sample.declared.SupportedGradient`, issues #1008, #1189, #1220).

        Supported where the kernel is this objective: a
        :class:`~sal.emissions.GaussianEmission` of scalar observations and
        no covariate, on segments of any length, handed over end to end
        beside their lengths (issue #1254). ``None`` otherwise.
        """
        if (
            type(self._start_family) is not GaussianEmission
            or self._covariate is not None
            or self._observations.dim() != 1
        ):
            return None
        return "gaussian_hmm", {
            "m": self._k,
            "observations": np.ascontiguousarray(self._observations.numpy()),
            "lengths": np.asarray(self._lengths, dtype=np.int64),
        }

    def gradient(self, theta: torch.Tensor) -> torch.Tensor:
        """``dU/dtheta``, detached: streamed where :meth:`supported_gradient` holds, :meth:`value_and_gradient`'s otherwise (issues #997, #1189, #1220).

        The kernel (``oxisal.SupportedEnergy``) sums the statistics
        ``oxisal.gaussian_hmm_statistics`` returns in one streamed pass and
        assembles Fisher's identity from them, with ``gamma`` and ``xi`` the
        posteriors of a state and of a pair, ``pi`` and ``A`` the chain, and
        ``N`` sequences:

        - an initial logit ``k >= 1``: ``sum gamma_1(k) - N pi_k``;
        - a transition logit ``(i, j >= 1)``: ``sum xi(i, j) - sum_j' xi(i, j') A_ij``;
        - a mean: ``sum_t gamma_t(s) (x_t - mu_s) / s_s^2``;
        - a log scale: ``sum_t gamma_t(s) ((x_t - mu_s)^2 / s_s^2 - 1)``;

        negated for the negative log-likelihood. What ``hmc.gradient_at``
        reads, and the arithmetic a compiled chain runs; the same call as
        :meth:`value_and_gradient` there, 5.6x faster than its compiled E
        step and backward pass at 10^6 positions. Autograd through
        :meth:`__call__` is the oracle.
        """
        kernel = self._supported_kernel()
        if kernel is None:
            return self.value_and_gradient(theta)[1]
        _, gradient = kernel.value_and_gradient(
            np.ascontiguousarray(theta.detach().numpy(), dtype=np.float64)
        )
        return torch.from_numpy(gradient)

    def _supported_kernel(self) -> oxisal.SupportedEnergy | None:
        """``oxisal.SupportedEnergy`` on :meth:`supported_gradient`, built on first use; ``None`` where it does not hold or ``backend`` is ``TORCH``."""
        if not self._kernel_read:
            self._kernel_read = True
            supported = self.supported_gradient()
            if supported is not None and self._backend is not Backend.TORCH:
                self._kernel = oxisal.SupportedEnergy(*supported, self.n_parameters)
        return self._kernel

    def jax_energy(self) -> tuple[Callable[[Any, Any], Any], dict[str, Any]] | None:
        """The negative log-likelihood as a traceable JAX ``(theta, data)`` function and its data, or ``None`` (issues #1008, #1189).

        What a compiled HMC chain runs inside its own loop
        (:class:`~sal.sample.declared.DeclaredJaxEnergy`), where
        :mod:`sal.opt.hmm.jax` has a twin of the start's family, on segments
        of any length (issue #1206), and the backend is not ``TORCH``;
        ``None`` otherwise.
        """
        from sal.opt.hmm.jax import jax_energy, twinned

        if self._backend is Backend.TORCH or not twinned(self):
            return None
        return jax_energy(self)

    @property
    def n_parameters(self) -> int:
        """Length of ``theta``: ``K^2 - 1`` free chain values and each named parameter's free coordinates."""
        return self._n_parameters

    @property
    def blocks(self) -> Mapping[str, slice]:
        """Each named parameter's coordinates in ``theta`` (:class:`~sal.opt.objective.DeclaredBlocks`).

        Keyed as :meth:`constrain` returns: the initial distribution, the
        transition matrix, then each parameter the family names.
        """
        k = self._k
        return {
            "log_initial": slice(0, k - 1),
            "log_transition": slice(k - 1, k * k - 1),
            **{
                name: slice(block.offset, block.stop)
                for name, block in self._blocks_at.items()
            },
        }

    def _chain(self, theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Log initial distribution and log transition matrix, from ``theta``."""
        k = self._k
        return (
            log_simplex(theta[: k - 1]),
            log_simplex(theta[k - 1 : k * k - 1].reshape(k, k - 1)),
        )

    def components(self, theta: torch.Tensor) -> EmissionFamily:
        """The family ``theta`` encodes, differentiable in ``theta``."""
        return self._build(
            {name: block.read(theta) for name, block in self._blocks_at.items()}
        )

    def initial(self) -> torch.Tensor:
        """Uniform initial and transition probabilities at ``start``'s parameters."""
        k = self._k
        uniform = -math.log(k)
        return self.theta_from(
            {
                "log_initial": torch.full((k,), uniform, dtype=torch.float64),
                "log_transition": torch.full((k, k), uniform, dtype=torch.float64),
                **self._start,
            }
        )

    def constrain(self, theta: torch.Tensor) -> Mapping[str, torch.Tensor]:
        """The log initial distribution, the log transitions and every named parameter."""
        log_initial, log_transition = self._chain(theta)
        return {
            "log_initial": log_initial,
            "log_transition": log_transition,
            **self.components(theta).named_parameters(),
        }

    def theta_from(self, named: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """The unconstrained vector whose :meth:`constrain` is ``named``."""
        parts = [
            free_from_log_simplex(torch.as_tensor(named["log_initial"])),
            free_from_log_simplex(torch.as_tensor(named["log_transition"])).reshape(-1),
        ]
        parts.extend(
            block.free_of(named[name]) for name, block in self._blocks_at.items()
        )
        return torch.cat([part.to(torch.float64) for part in parts])

    def theta_from_truth(
        self,
        initial: np.ndarray | torch.Tensor,
        transition: np.ndarray | torch.Tensor,
        **emission: np.ndarray | torch.Tensor,
    ) -> torch.Tensor:
        """:meth:`theta_from` of a known truth, the chain given as probabilities (issue #1189).

        Parameters
        ----------
        initial : np.ndarray | torch.Tensor
            True initial distribution, ``(K,)``.
        transition : np.ndarray | torch.Tensor
            True transition matrix, ``(K, K)``.
        **emission : np.ndarray | torch.Tensor
            Every parameter the family names, under
            :meth:`~sal.emissions.EmissionFamily.named_parameters`' keys and
            in its units: ``log_emission`` for a categorical family.

        Returns
        -------
        torch.Tensor
            ``theta`` such that :meth:`constrain` returns this truth.
        """
        return self.theta_from(
            {
                "log_initial": torch.log(torch.as_tensor(initial, dtype=torch.float64)),
                "log_transition": torch.log(
                    torch.as_tensor(transition, dtype=torch.float64)
                ),
                **{
                    key: torch.as_tensor(value, dtype=torch.float64)
                    for key, value in emission.items()
                },
            }
        )

    def _log_density(self, theta: torch.Tensor) -> torch.Tensor:
        """Every observation scored under every state, ``(total, K)``."""
        return self.components(theta).log_density(
            self._observations, covariate=self._covariate
        )

    def __call__(self, theta: torch.Tensor) -> torch.Tensor:
        """The negative log-likelihood summed over segments, by the differentiable forward recursion."""
        log_initial, log_transition = self._chain(theta)
        return -forward_log_likelihood_ragged(
            self._log_density(theta), self._lengths, log_initial, log_transition
        )

    def value_and_gradient(
        self, theta: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(U(theta), dU/dtheta)``, detached: the compiled E step, then one backward pass (issue #1169).

        ``oxisal.ragged_posteriors`` returns the log marginals ``gamma``, the
        log transition counts ``xi`` summed over segments and each segment's
        log evidence, with no graph. By Fisher's identity the score of the
        log-likelihood is the posterior expectation of the complete-data
        score, so with ``gamma``, ``xi`` and the first-position marginals
        ``gamma_0`` held at their values at ``theta``,

        ``d log L / d theta = d/d theta [sum gamma log p(x | z)
        + sum xi log A + sum gamma_0 log pi]``,

        one autograd pass through the emission and the two simplex maps,
        none through the recursion. The value is the summed log evidence,
        negated. Autograd through :meth:`__call__` is the oracle this is
        pinned to. ``backend`` routes it (the class's Parameters).

        Where :meth:`supported_gradient` holds and ``backend`` is
        ``RUST``, the value and gradient are :meth:`gradient`'s kernel's,
        one call: 0.58 ms against 11.9 ms by the route above, on a four-state
        Gaussian HMM of 200 sequences of 60 (issue #1248), on segments of any
        length since issue #1254.
        """
        if self._backend is Backend.TORCH:
            return autograd_value_and_gradient(self, theta)
        if self._backend is Backend.JAX:
            if self._jax is None:
                from sal.opt.hmm.jax import value_and_grad

                self._jax = value_and_grad(self)
            value_, gradient_ = self._jax(theta.detach().cpu().numpy())
            return (
                torch.tensor(value_, dtype=theta.dtype),
                torch.as_tensor(np.array(gradient_), dtype=theta.dtype),
            )
        kernel = self._supported_kernel()
        if kernel is not None:
            value, gradient = kernel.value_and_gradient(
                np.ascontiguousarray(theta.detach().numpy(), dtype=np.float64)
            )
            return torch.tensor(value, dtype=torch.float64), torch.from_numpy(gradient)
        point = theta.detach().clone().requires_grad_(True)
        log_initial, log_transition = self._chain(point)
        log_density = self._log_density(point)
        m = self._k
        values = np.ascontiguousarray(log_density.detach().numpy(), dtype=np.float64)
        gamma = np.empty_like(values)
        counts = np.empty((m, m), dtype=np.float64)
        evidence = np.empty(len(self._lengths), dtype=np.float64)
        oxisal.ragged_posteriors(
            values,
            self._lengths_array,
            np.ascontiguousarray(log_initial.detach().numpy(), dtype=np.float64),
            np.ascontiguousarray(log_transition.detach().numpy(), dtype=np.float64),
            gamma,
            counts,
            evidence,
        )
        posterior = torch.from_numpy(np.exp(gamma))
        surrogate = (
            (posterior * log_density).sum()
            + (torch.from_numpy(np.exp(counts)) * log_transition).sum()
            + (posterior[self._first].sum(dim=0) * log_initial).sum()
        )
        (derivative,) = torch.autograd.grad(surrogate, point)
        return (
            torch.tensor(-float(evidence.sum()), dtype=torch.float64),
            -derivative,
        )


def family_start(
    family: type,
    observations: np.ndarray | torch.Tensor,
    n_states: int,
    *,
    n_symbols: int | None = None,
    trials: np.ndarray | torch.Tensor | None = None,
) -> EmissionFamily:
    """The start the retired per-family objective of ``family`` took, as the family :class:`EmissionHmmObjective` starts at (issue #1189).

    Uninformative but **not** symmetric: with every state identical the
    gradient in the initial and transition parameters is exactly zero, and
    an optimizer started there stays there (``opt/CLAUDE.md``). Deterministic
    rather than random, since a seeded jitter would make a fit depend on a
    second seed nobody declared.

    * :class:`~sal.emissions.CategoricalEmission`: row ``s`` tilted towards
      symbol ``s mod (n_symbols - 1)`` by one unconstrained unit, a factor of
      ``e``.
    * :class:`~sal.emissions.GaussianEmission`: the means at evenly spaced
      quantiles of the pooled values
      (:func:`~sal.opt.initialize.quantile_locations`), every scale the
      pooled standard deviation --- too *narrow* is the direction the
      likelihood is unbounded in --- and the floor
      :func:`~sal.emissions.pooled_variance_floor` derives. Values all equal
      are :attr:`~sal.emissions.GaussianEmission.flat`, every scale
      ``sqrt`` of that floor (issue #1234).
    * :class:`~sal.emissions.PoissonEmission`: the rates at the quantiles,
      at least half a count, since ``log 0`` is not a start.
    * :class:`~sal.emissions.BinomialEmission`: the success rates at the
      quantiles over ``trials``, held ``1e-6`` from 0 and 1.
    * :class:`~sal.emissions.BetaBinomialEmission`: ``(a, b)`` at those
      rates at unit concentration, the most overdispersed the family gets
      before the Beta is improper; ``a + b -> inf`` is where the likelihood
      goes flat.
    * :class:`~sal.emissions.NegativeBinomialEmission`: the means as the
      Poisson's, every dispersion the pooled method-of-moments value or the
      identifiable bound (:func:`~sal.emissions.identifiable_dispersion_bound`),
      whichever is smaller: a start past what the data resolves sits in the
      flat region.

    Each value is the one the retired objective's ``components(initial())``
    returned, through the same constraint maps, so a fit started here repeats
    one started there.

    Parameters
    ----------
    family : type
        One of the six families above.
    observations : np.ndarray | torch.Tensor
        The observations, pooled whatever their layout.
    n_states : int
        ``K``.
    n_symbols : int | None
        The alphabet, for a categorical family only.
    trials : np.ndarray | torch.Tensor | None
        The declared trial count per state, ``(K,)``, for a binomial or
        beta-binomial family only.

    Raises
    ------
    ValueError
        If ``family`` is none of the six, or the argument it needs is missing.
    """
    if family is CategoricalEmission:
        if n_symbols is None:
            msg = "a categorical start needs n_symbols"
            raise ValueError(msg)
        free = torch.zeros(n_states, n_symbols - 1, dtype=torch.float64)
        for state in range(n_states):
            free[state, state % (n_symbols - 1)] = _SYMMETRY_BREAK
        return CategoricalEmission.from_log(log_simplex(free))
    values = torch.as_tensor(observations, dtype=torch.float64).reshape(-1)
    locations = quantile_locations(values, n_states)
    if family is GaussianEmission:
        floor = pooled_variance_floor(np.asarray(observations))
        flat = flat_channels(values.numpy())
        spread = values.std()
        if flat:
            spread = torch.tensor(math.sqrt(floor), dtype=torch.float64)
        return GaussianEmission(
            locations,
            positive(free_from_positive(spread).expand(n_states)),
            floor,
            flat=flat,
        )
    if family is PoissonEmission:
        return PoissonEmission(
            positive(free_from_positive(locations.clamp_min(_MINIMUM_COUNT_MEAN)))
        )
    if family is NegativeBinomialEmission:
        pooled_mean = float(values.mean())
        pooled_variance = float(values.var(unbiased=True))
        bound = identifiable_dispersion_bound(pooled_mean, float(values.numel()))
        moments = (
            pooled_mean**2 / (pooled_variance - pooled_mean)
            if pooled_variance > pooled_mean
            else bound
        )
        return NegativeBinomialEmission(
            positive(
                torch.full(
                    (n_states,), math.log(min(moments, bound)), dtype=torch.float64
                )
            ),
            positive(free_from_positive(locations.clamp_min(_MINIMUM_COUNT_MEAN))),
        )
    if family in (BinomialEmission, BetaBinomialEmission):
        if trials is None:
            msg = f"a {family.__name__} start needs trials"
            raise ValueError(msg)
        declared = torch.as_tensor(trials, dtype=torch.float64).reshape(-1)
        rate = (locations / declared).clamp(_RATE_MARGIN, 1.0 - _RATE_MARGIN)
        if family is BinomialEmission:
            return BinomialEmission(declared, probability(free_from_probability(rate)))
        return BetaBinomialEmission(
            declared, positive(torch.log(rate)), positive(torch.log1p(-rate))
        )
    msg = f"no per-family start for {family.__name__}"
    raise ValueError(msg)


class _RetiredHmmObjective(EmissionHmmObjective):
    """An :class:`EmissionHmmObjective` at :func:`family_start`, built under a retired per-family name (issue #1189).

    Deprecated: each subclass warns and is removed after the next release.
    ``theta``'s layout, :meth:`initial`, the value and the backend's
    gradient are the retired class's; ``observations`` and ``covariate`` are
    held end to end, as :class:`EmissionHmmObjective` holds them.
    """

    _kind: type

    def __init__(
        self,
        observations: np.ndarray,
        n_states: int,
        *,
        n_symbols: int | None = None,
        trials: np.ndarray | None = None,
        dtype: torch.dtype = torch.float64,
        covariate: np.ndarray | None = None,
        backend: Backend = Backend.JAX,
    ) -> None:
        name = type(self).__name__
        warnings.warn(
            f"{name} is deprecated and is removed after the next release; use "
            f"EmissionHmmObjective(observations, family_start("
            f"{self._kind.__name__}, observations, n_states, ...)) (issue #1189)",
            DeprecationWarning,
            stacklevel=3,
        )
        refuse_backend(
            "an HMM objective's gradient", backend, (Backend.JAX, Backend.TORCH)
        )
        if dtype != torch.float64:
            msg = f"{name} computes in float64 alone, got {dtype}"
            raise ValueError(msg)
        observations = np.asarray(observations)
        super().__init__(
            observations,
            family_start(
                self._kind, observations, n_states, n_symbols=n_symbols, trials=trials
            ),
            covariate=None
            if covariate is None
            else torch.as_tensor(covariate, dtype=torch.float64)[..., None],
            backend=backend,
        )


class HmmObjective(_RetiredHmmObjective):
    """Deprecated: ``EmissionHmmObjective(observations, family_start(CategoricalEmission, observations, n_states, n_symbols=n_symbols))`` (issue #1189)."""

    _kind = CategoricalEmission

    def __init__(
        self,
        observations: np.ndarray,
        n_states: int,
        n_symbols: int,
        dtype: torch.dtype = torch.float64,
        covariate: np.ndarray | None = None,
        backend: Backend = Backend.JAX,
    ) -> None:
        super().__init__(
            observations,
            n_states,
            n_symbols=n_symbols,
            dtype=dtype,
            covariate=covariate,
            backend=backend,
        )

    def theta_from_truth(  # type: ignore[override]
        self, initial: np.ndarray, transition: np.ndarray, emission: np.ndarray
    ) -> torch.Tensor:
        """``theta`` at a known ``(pi, A, B)``."""
        return super().theta_from_truth(
            initial,
            transition,
            log_emission=torch.log(torch.as_tensor(emission, dtype=torch.float64)),
        )


class GaussianHmmObjective(_RetiredHmmObjective):
    """Deprecated: ``EmissionHmmObjective(observations, family_start(GaussianEmission, observations, n_states))`` (issue #1189)."""

    _kind = GaussianEmission

    def __init__(
        self,
        observations: np.ndarray,
        n_states: int,
        dtype: torch.dtype = torch.float64,
        covariate: np.ndarray | None = None,
        backend: Backend = Backend.JAX,
    ) -> None:
        super().__init__(
            observations, n_states, dtype=dtype, covariate=covariate, backend=backend
        )

    @property
    def variance_floor(self) -> float:
        """The start family's floor, derived from these observations."""
        return float(self.start.variance_floor)  # type: ignore[attr-defined]

    def theta_from_truth(  # type: ignore[override]
        self,
        initial: np.ndarray,
        transition: np.ndarray,
        mean: np.ndarray,
        scale: np.ndarray,
    ) -> torch.Tensor:
        """``theta`` at a known ``(pi, A, mu, sigma)``."""
        return super().theta_from_truth(initial, transition, mean=mean, scale=scale)


class PoissonHmmObjective(_RetiredHmmObjective):
    """Deprecated: ``EmissionHmmObjective(observations, family_start(PoissonEmission, observations, n_states))`` (issue #1189)."""

    _kind = PoissonEmission

    def __init__(
        self,
        observations: np.ndarray,
        n_states: int,
        dtype: torch.dtype = torch.float64,
        covariate: np.ndarray | None = None,
        backend: Backend = Backend.JAX,
    ) -> None:
        super().__init__(
            observations, n_states, dtype=dtype, covariate=covariate, backend=backend
        )

    def theta_from_truth(  # type: ignore[override]
        self, initial: np.ndarray, transition: np.ndarray, mean: np.ndarray
    ) -> torch.Tensor:
        """``theta`` at a known ``(pi, A, lambda)``."""
        return super().theta_from_truth(initial, transition, mean=mean)


class BinomialHmmObjective(_RetiredHmmObjective):
    """Deprecated: ``EmissionHmmObjective(observations, family_start(BinomialEmission, observations, n_states, trials=trials))`` (issue #1189)."""

    _kind = BinomialEmission

    def __init__(
        self,
        observations: np.ndarray,
        n_states: int,
        trials: np.ndarray,
        dtype: torch.dtype = torch.float64,
        covariate: np.ndarray | None = None,
        backend: Backend = Backend.JAX,
    ) -> None:
        super().__init__(
            observations,
            n_states,
            trials=trials,
            dtype=dtype,
            covariate=covariate,
            backend=backend,
        )

    def theta_from_truth(  # type: ignore[override]
        self, initial: np.ndarray, transition: np.ndarray, probability: np.ndarray
    ) -> torch.Tensor:
        """``theta`` at a known ``(pi, A, p)``."""
        return super().theta_from_truth(initial, transition, probability=probability)


class BetaBinomialHmmObjective(_RetiredHmmObjective):
    """Deprecated: ``EmissionHmmObjective(observations, family_start(BetaBinomialEmission, observations, n_states, trials=trials))`` (issue #1189)."""

    _kind = BetaBinomialEmission

    def __init__(
        self,
        observations: np.ndarray,
        n_states: int,
        trials: np.ndarray,
        dtype: torch.dtype = torch.float64,
        covariate: np.ndarray | None = None,
        backend: Backend = Backend.JAX,
    ) -> None:
        super().__init__(
            observations,
            n_states,
            trials=trials,
            dtype=dtype,
            covariate=covariate,
            backend=backend,
        )

    def theta_from_truth(  # type: ignore[override]
        self,
        initial: np.ndarray,
        transition: np.ndarray,
        alpha: np.ndarray,
        beta: np.ndarray,
    ) -> torch.Tensor:
        """``theta`` at a known ``(pi, A, a, b)``."""
        return super().theta_from_truth(initial, transition, alpha=alpha, beta=beta)


class NegativeBinomialHmmObjective(_RetiredHmmObjective):
    """Deprecated: ``EmissionHmmObjective(observations, family_start(NegativeBinomialEmission, observations, n_states))`` (issue #1189)."""

    _kind = NegativeBinomialEmission

    def __init__(
        self,
        observations: np.ndarray,
        n_states: int,
        dtype: torch.dtype = torch.float64,
        covariate: np.ndarray | None = None,
        backend: Backend = Backend.JAX,
    ) -> None:
        super().__init__(
            observations, n_states, dtype=dtype, covariate=covariate, backend=backend
        )

    def theta_from_truth(  # type: ignore[override]
        self,
        initial: np.ndarray,
        transition: np.ndarray,
        dispersion: np.ndarray,
        mean: np.ndarray,
    ) -> torch.Tensor:
        """``theta`` at a known ``(pi, A, r, mu)``."""
        return super().theta_from_truth(
            initial, transition, dispersion=dispersion, mean=mean
        )


@dataclass(frozen=True)
class HmmMetrics:
    """What an HMM parameter vector means: the forward log-likelihood (issue #778).

    A :class:`sal.track.Metrics` over ``theta``, satisfied
    structurally, for any family :class:`EmissionHmmObjective` serves,
    because the number comes from the objective's own forward pass.

    ``log_likelihood`` is
    :func:`forward_log_likelihood_ragged` of ``theta``'s emissions,
    which :meth:`EmissionHmmObjective.__call__` negates to minimize; it is read here
    with the sign the model states rather than the sign the optimizer wants.

    **The state accuracy the ticket listed is left out.**
    :func:`align_states` returns a permutation, not a number, and the decoder
    that would turn one into an accuracy ---
    :mod:`sal.likelihood.forward_backward`, or
    :func:`sal.search.spatio_sequential.label_accuracy` --- is
    outside what ``opt`` may import
    (``tests/regression/opt/test_opt_objective.py``). A state path is also not
    what a fit holds: ``theta`` is.

    Parameters
    ----------
    objective : EmissionHmmObjective
        The objective being fitted, which owns the observations and the
        constraint map.
    """

    objective: EmissionHmmObjective

    names: tuple[str, ...] = ("log_likelihood",)

    def __call__(self, theta: torch.Tensor) -> dict[str, float]:
        """``{"log_likelihood": -objective(theta)}``, computed without a graph."""
        with torch.no_grad():
            return {"log_likelihood": -float(self.objective(theta))}
