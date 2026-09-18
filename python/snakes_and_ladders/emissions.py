"""What a hidden state emits, separated from how the model is fitted.

The emission is the only part of an HMM that knows what an observation
*is*. The recursions need three things from it: draw an observation given a
state, score one given a state, re-estimate from posterior state weights.
This module is that interface and its implementations;
``snakes_and_ladders.opt.hmm`` holds the recursions that consume it.

**Why it is here rather than in a module that owns a model.** ``opt/CLAUDE.md``
forbids ``snakes_and_ladders.opt`` from importing ``snakes_and_ladders.sim``, and
``tests/regression/opt/test_opt_objective.py`` asserts it, so a family defined
in ``sim`` would be unreachable from the objective that scores with it. A
family names no tree, no alignment and no lattice, so it sits beside
:mod:`snakes_and_ladders.numerics` on the same terms: importable from
anywhere, inverting no layering.

**A density is not a probability, and the difference is load-bearing.**
:meth:`EmissionFamily.log_density` returns a log-probability for a family over
a countable alphabet and a log *density* for one over the reals. Only the
first is bounded above by zero, so the evidence ``log P(observations)`` of a
continuous-emission HMM may be positive and an assertion that it is not fails
on correct code. :attr:`EmissionFamily.is_discrete` states which case a family
is, so a test can assert the bound exactly where it holds.

**An unbounded likelihood is a property of the model, not a bug in the fit.**
A Gaussian emission's likelihood has no maximum: put one state's mean on a
single observation and let its variance go to zero (Bishop, *Pattern
Recognition and Machine Learning*, section 9.2.1).
:class:`GaussianEmission` therefore carries an explicit variance floor and
**refuses** rather than clamps when a re-estimate reaches it, because a
clamped fit returns normally and its intervals mean nothing (issue #122).

**A covariate arrives shaped for the states, and that rule is stated here
because this is what enforces it.** :func:`trial_count` and :func:`exposure`
refuse a covariate whose last axis is not a singleton, so what reaches a
single-channel family carries one: the singleton is what broadcasts the
per-observation value along the states at the ``+``. A family whose
observation carries axes of its own --- the two-channel count pair --- takes
one covariate *per* axis instead, and the singleton belongs inside each
channel rather than after them. So a seam that slices a covariate adds the
singleton **only** where the covariate has no axes of its own; appending it to
a ``(S, V, 2)`` covariate makes ``(S, V, 2, 1)``, whose last axis names no
channel and broadcasts against the wrong one. Five functions across ``sim``
and ``likelihood`` slice a covariate and each obeys this; #670 is what a
fifth copy of the rule costs when one of them drifts, and issue #677 is why
they now point here rather than restate it.

Parameterization for an unconstrained optimizer belongs to
``snakes_and_ladders.opt.constrain``; here it would make
:mod:`snakes_and_ladders.sim` import ``snakes_and_ladders.opt`` transitively
to draw a sequence.
"""

from __future__ import annotations

import math
from abc import abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar, runtime_checkable

import numpy as np
import torch

from snakes_and_ladders.numerics_rust import sample_rows

#: What a family accepts for a parameter vector. A plain list is admitted
#: because a fixture states its truth in one; the constructor converts, and
#: validates what it converted.
type Values = np.ndarray | torch.Tensor | Sequence[float] | Sequence[int]

#: Nearest-neighbour spacing, in units of the pooled standard deviation, that
#: :func:`pooled_variance_floor` treats as the smallest scale a state can
#: occupy. ``n`` observations spread over a pooled standard deviation ``s``
#: sit at a typical spacing ``s / n``, so a state explaining a *neighbourhood*
#: has variance of at least that order and one collapsed onto a *single*
#: observation has variance heading to zero. Derived from the data rather than
#: fixed, so it transfers across fixture sizes.
COLLAPSE_EXPONENT = 2

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


#: The family a re-estimate returns, so a caller of a concrete family's M step
#: keeps that family's own parameters rather than the protocol's. Covariance is
#: sound because the record is frozen: nothing can write the narrower field
#: through the wider view.
FamilyT_co = TypeVar("FamilyT_co", covariant=True)


@dataclass(frozen=True)
class Reestimate(Generic[FamilyT_co]):
    """One emission M step, and whether what it returned is an answer.

    A closed-form M step needs none of this. An M step that is itself an
    optimization does: a silent inner failure surfaces as a non-monotone outer
    likelihood several EM iterations later, nowhere near its cause (issue
    #229).

    Parameters
    ----------
    emissions : FamilyT_co
        The re-estimated family, at its own type.
    converged : bool
        Whether an iterative M step settled; ``True`` by construction for a
        closed-form one. ``False`` is refusal-worthy, not a warning:
        ``likelihood/CLAUDE.md`` forbids returning a number read off
        iterations that never settled, and Baum-Welch raises on it.
    at_boundary : bool
        Whether a parameter reached the edge of the range this data can
        identify it over. **Not** an error: the estimate is a bound rather
        than a maximum, so a Wald interval around it summarizes nothing (issue
        #122). A caller counts these the way ``qa.opt_coverage`` counts a
        singular information matrix.
    iterations : int
        Iterations the inner solve took; ``0`` for a closed-form M step.
    residual : float
        The weighted score at the returned parameters, divided by the total
        weight, so it is comparable across data sizes. ``0.0`` for a
        closed-form M step, where the score is zero by construction.
    """

    emissions: FamilyT_co
    converged: bool = True
    at_boundary: bool = False
    iterations: int = 0
    residual: float = 0.0


class CovariateNotSupportedError(TypeError):
    """A family was given a per-observation covariate it cannot condition on.

    Refused rather than ignored. A covariate silently dropped is a model
    fitted to a different likelihood than the caller asked for, and the
    failure would show up as a recovery that is merely worse rather than as an
    error (root ``CLAUDE.md``: no silent behaviour changes).
    """


def trial_count(covariate: torch.Tensor | None, declared: torch.Tensor) -> torch.Tensor:
    """The trial count to score at: the covariate's, or the declared one.

    The covariate **overrides** ``declared`` rather than replacing it. A family
    keeps a per-state trial count so :attr:`BetaBinomialEmission.mean` and
    :attr:`~BetaBinomialEmission.variance` keep a value; a covariate says what
    the count actually was for each observation (issue #631).

    **The shape is the whole cost.** ``_beta_binomial_log_density`` is nine
    ``lgamma`` calls, and PyTorch evaluates each on its own operand's shape and
    broadcasts at the ``+``, not before. A covariate shaped ``(..., 1)``
    broadcasts along the state axis and leaves three terms at ``n_obs * K``,
    which is what the per-state count costs today. Materialised ``(..., K)`` by
    the caller, all nine become ``n_obs * K`` and the term count triples.

    Parameters
    ----------
    covariate : torch.Tensor | None
        One trial count per observation, trailing axis of length one.
    declared : torch.Tensor
        The family's per-state count, shape ``(n_states,)``.

    Returns
    -------
    torch.Tensor

    Raises
    ------
    ValueError
        If the covariate does not end in a singleton axis, or is not a
        positive integer count.
    """
    if covariate is None:
        return declared
    if covariate.ndim == 0 or covariate.shape[-1] != 1:
        msg = (
            f"a trial count per observation must end in a singleton axis so it "
            f"broadcasts along the states, got {tuple(covariate.shape)}; a "
            f"count materialised per state triples the lgamma term count"
        )
        raise ValueError(msg)
    return validated_trials(covariate, declared)


def exposure(covariate: torch.Tensor, declared: torch.Tensor) -> torch.Tensor:
    """The exposure to scale the rate by, shaped to broadcast along the states.

    The offset half of :func:`trial_count`, and the same layout rule for the
    same reason: shaped ``(..., 1)`` it broadcasts along the state axis, and
    materialised ``(..., n_states)`` by the caller it makes every term of the
    density ``n_obs * n_states`` (issue #631).

    Returns
    -------
    torch.Tensor

    Raises
    ------
    ValueError
        If the covariate does not end in a singleton axis, or is not positive.
    """
    if covariate.ndim == 0 or covariate.shape[-1] != 1:
        msg = (
            f"an exposure per observation must end in a singleton axis so it "
            f"broadcasts along the states, got {tuple(covariate.shape)}"
        )
        raise ValueError(msg)
    return validated_exposure(covariate, declared)


def validated_exposure(covariate: torch.Tensor, declared: torch.Tensor) -> torch.Tensor:
    """``covariate`` as exposures, in ``declared``'s dtype.

    The positivity check alone, without :func:`exposure`'s trailing-axis rule,
    for the M step --- which flattens over sequences and positions and never
    broadcasts along the states.

    Returns
    -------
    torch.Tensor

    Raises
    ------
    ValueError
        If an exposure is not strictly positive.
    """
    offsets = covariate.to(declared.dtype)
    if bool((offsets <= 0.0).any()):
        msg = "every exposure must be strictly positive"
        raise ValueError(msg)
    return offsets


def validated_trials(covariate: torch.Tensor, declared: torch.Tensor) -> torch.Tensor:
    """``covariate`` as trial counts, in ``declared``'s dtype.

    The support check alone, without :func:`trial_count`'s trailing-axis rule.
    An M step flattens its covariate over sequences and positions and never
    broadcasts it along the states, so that rule would refuse the shape the
    protocol documents for :meth:`EmissionFamily.reestimate`.

    Returns
    -------
    torch.Tensor

    Raises
    ------
    ValueError
        If a count is not a positive integer.
    """
    counts = covariate.to(declared.dtype)
    if bool(((counts < 1) | (counts != counts.floor())).any()):
        msg = "every trial count must be a positive integer"
        raise ValueError(msg)
    return counts


def refuse_covariate(family: object, covariate: torch.Tensor | None) -> None:
    """Raise unless ``covariate`` is ``None``.

    Public because an implementer outside this module calls it ---
    :class:`~snakes_and_ladders.sim.count_pairs.IndependentCountPair` is the
    eighth :class:`EmissionFamily` and does not live here.

    Raises
    ------
    CovariateNotSupportedError
        If a covariate is supplied.
    """
    if covariate is None:
        return
    name = type(family).__name__
    msg = (
        f"{name} conditions on no per-observation covariate; pass None. An "
        "exposure or a trial count per observation is issue #631, and lands on "
        "NegativeBinomialEmission and BetaBinomialEmission."
    )
    raise CovariateNotSupportedError(msg)


@runtime_checkable
class CountEmissionFamily(Protocol):
    """An emission family over the non-negative integers, which has moments.

    Every count family here states a closed form for both moments, and their
    *ratio* separates them: below one for the binomial, exactly one for the
    Poisson, above one for the negative binomial and the beta-binomial. The
    moments are a protocol so a test can range over the families and assert
    that bracketing.

    Under the seam rule: the count families' shared moments, which bracket
    equidispersion in five families and carry a per-channel pair (issue #399).
    """

    @property
    def mean(self) -> torch.Tensor:
        """Per-state mean, shape ``(n_states,)``.

        A family whose observation is a tuple states one moment per channel,
        shape ``(n_states, n_channels)``: :class:`CountPairEmission` emits a
        depth and an allele count, and a single mean over the two would be a
        number in no unit.
        """
        ...  # pragma: no cover

    @property
    def variance(self) -> torch.Tensor:
        """Per-state variance, shape ``(n_states,)``, or per channel."""
        ...  # pragma: no cover


@runtime_checkable
class EmissionFamily(Protocol):
    """The emission distribution of every hidden state, as one object.

    An implementation is immutable: :meth:`reestimate` returns a new family
    rather than updating in place, so an EM iterate cannot be aliased by the
    iterate before it.
    """

    @property
    def n_states(self) -> int:
        """Hidden states this family emits from."""
        ...  # pragma: no cover

    @property
    def is_discrete(self) -> bool:
        """Whether :meth:`log_density` returns a probability rather than a density.

        ``True`` for a family over a countable alphabet, where the evidence of
        a sequence is a probability and ``log P <= 0``. ``False`` for a family
        over the reals, where it is a density and that bound does not hold.
        """
        ...  # pragma: no cover

    @property
    def observation_dtype(self) -> torch.dtype:
        """Type an observation is carried in: integral or floating point."""
        ...  # pragma: no cover

    @abstractmethod
    def sample(
        self,
        states: np.ndarray,
        rng: np.random.Generator,
        covariate: torch.Tensor | None = None,
    ) -> np.ndarray:
        """Draw one observation per entry of ``states``.

        Parameters
        ----------
        states : np.ndarray
            Emitting state per draw, shape ``(n_draws,)``.
        rng : np.random.Generator
            Generator, passed in rather than seeded here (``sim/CLAUDE.md``).
        covariate : torch.Tensor | None
            One conditioning value per draw, shape ``(n_draws,)``. Conditioned
            on and never fitted. A family that cannot use one raises
            :class:`CovariateNotSupportedError` rather than dropping it
            (issue #631).

        Returns
        -------
        np.ndarray
            One observation per entry of ``states``.
        """
        ...  # pragma: no cover

    @abstractmethod
    def log_density(
        self, observations: torch.Tensor, covariate: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Score every observation under every state.

        Parameters
        ----------
        observations : torch.Tensor
            Observations of any leading shape ``(...)``.
        covariate : torch.Tensor | None
            One conditioning value per observation, shape ``(..., 1)`` so it
            broadcasts along the state axis rather than being materialised per
            state (issue #631). A family that cannot use one raises
            :class:`CovariateNotSupportedError`.

        Returns
        -------
        torch.Tensor
            Shape ``(..., n_states)``, differentiable with respect to this
            family's parameters. A log-probability where :attr:`is_discrete`,
            a log-density otherwise.
        """
        ...  # pragma: no cover

    @abstractmethod
    def bregman_divergence(self, observations: torch.Tensor) -> torch.Tensor:
        """``D_phi(T(y), mu_state)``: the divergence of this family's log-partition.

        The quantity ``Emission_Mixture++`` scores a candidate by
        (``eq:kmeanspp``; Banerjee et al., 2005). An exponential family writes
        ``p(y | theta) = exp(-D_phi(T(y), mu)) b_phi(T(y))``, so the divergence
        is the negative log density **less** ``-log b_phi(T(y))``, the log
        density's largest value over the family's own mean parameter, which
        depends on the observation alone. D-squared sampling normalizes its
        scores rather than shifting them, so that term is not a constant a
        seeding absorbs (issue #560).

        Parameters
        ----------
        observations : torch.Tensor
            Observations, of the shape :meth:`log_density` takes.

        Returns
        -------
        torch.Tensor
            Shape ``(..., n_states)``, non-negative, and zero at the state
            whose mean parameter is the observation.
        """
        ...  # pragma: no cover

    @abstractmethod
    def validate(self, observations: np.ndarray) -> None:
        """Raise if ``observations`` cannot have come from this family.

        Raises
        ------
        ValueError
            If an observation lies outside the family's support.
        """
        ...  # pragma: no cover

    @abstractmethod
    def reestimate(
        self,
        observations: torch.Tensor,
        posterior: torch.Tensor,
        covariate: torch.Tensor | None = None,
    ) -> Reestimate[EmissionFamily]:
        """The Baum-Welch M step for this family alone.

        Parameters
        ----------
        observations : torch.Tensor
            Observations, shape ``(n_sequences, length)``.
        posterior : torch.Tensor
            State posteriors ``P(state_t | observations)`` as probabilities,
            shape ``(n_sequences, length, n_states)``.
        covariate : torch.Tensor | None
            One conditioning value per observation, shape
            ``(n_sequences, length)``. A family that cannot use one raises
            :class:`CovariateNotSupportedError`.

        Returns
        -------
        Reestimate
            The re-estimated family, and what its M step had to report about
            producing it.
        """
        ...  # pragma: no cover

    @abstractmethod
    def alignment_key(self) -> torch.Tensor:
        """Per-state signature the hidden-state permutation is matched on.

        Shape ``(n_states, d)``. Label switching is unidentifiable in any HMM,
        so a recovery test aligns before comparing; the emission is the
        discriminating signal, since two states emitting identically are the
        same state. What "identically" means is the family's to say --- a row
        of symbol probabilities for a categorical emission, a mean for a
        Gaussian one --- which is why this is on the family and not in the
        aligner.
        """
        ...  # pragma: no cover

    @abstractmethod
    def named_parameters(self) -> Mapping[str, torch.Tensor]:
        """This family's parameters under the names the model states them in."""
        ...  # pragma: no cover


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
        covariate: torch.Tensor | None = None,
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
        observations: torch.Tensor,
        posterior: torch.Tensor,
        covariate: torch.Tensor | None = None,
    ) -> Reestimate[CategoricalEmission]:
        """Normalized expected symbol counts, in log space.

        Closed form, so the record it returns carries no convergence to
        report: this is the case the seam was designed against.
        """
        refuse_covariate(self, covariate)
        mask = torch.nn.functional.one_hot(
            observations.reshape(-1).to(torch.long), self.n_symbols
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
            raise ValueError(msg)
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
        covariate: torch.Tensor | None = None,
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
        :func:`snakes_and_ladders.opt.mixture.kmeans_plus_plus`: normalizing
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
        observations: torch.Tensor,
        posterior: torch.Tensor,
        covariate: torch.Tensor | None = None,
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
        weights = posterior.reshape(-1, self.n_states)
        mass = weights.sum(dim=0)
        values = observations.reshape(-1, self.n_channels).to(posterior.dtype)
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
        collapsed = variance <= self._variance_floor
        if bool(collapsed.any()):
            states = (
                torch.nonzero(collapsed.reshape(self.n_states, -1).any(dim=1))
                .reshape(-1)
                .tolist()
            )
            msg = (
                f"state(s) {states} re-estimated to variance "
                f"{variance[collapsed].tolist()}, at or below the floor "
                f"{self._variance_floor:.6g}: the Gaussian likelihood is "
                f"unbounded as a variance goes to zero, so this fit is "
                f"heading to a degenerate optimum rather than converging"
            )
            raise ValueError(msg)
        return Reestimate(
            GaussianEmission(mean, torch.sqrt(variance), self._variance_floor)
        )

    def alignment_key(self) -> torch.Tensor:
        """The per-state means, one row per state and one column per channel."""
        return self._mean.reshape(self.n_states, -1)

    def named_parameters(self) -> Mapping[str, torch.Tensor]:
        """``mean`` and ``scale``, the parameters the model is stated in."""
        return {"mean": self._mean, "scale": self._scale}


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
        covariate: torch.Tensor | None = None,
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
            rate = rate * exposure(covariate, self._mean).reshape(states.shape).numpy()
        return np.asarray(rng.negative_binomial(r, r / (r + rate)))

    def log_density(
        self, observations: torch.Tensor, covariate: torch.Tensor | None = None
    ) -> torch.Tensor:
        """The negative binomial log-probability of every count under every state.

        ``covariate`` is the exposure per observation, shape ``(..., 1)`` so it
        broadcasts along the state axis; the rate scored at is ``e_i mu_k``.
        Where the exposure is constant the two models are the same model,
        absorbing it as ``log mu - log(c)``, which is what makes the conserved
        family a referee for this one (issue #631).
        """
        counts = observations.unsqueeze(-1).to(self._mean.dtype)
        rate = self._mean
        if covariate is not None:
            rate = exposure(covariate, self._mean) * self._mean
        total = self._dispersion + rate
        return (
            torch.lgamma(counts + self._dispersion)
            - torch.lgamma(self._dispersion)
            - torch.lgamma(counts + 1.0)
            + self._dispersion * torch.log(self._dispersion / total)
            + counts * torch.log(rate / total)
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
        self,
        observations: torch.Tensor,
        posterior: torch.Tensor,
        covariate: torch.Tensor | None = None,
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
        offsets = (
            None
            if covariate is None
            else validated_exposure(covariate, self._mean).reshape(-1)
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

        dispersion = torch.empty_like(mean)
        boundary = False
        iterations = 0
        residual = 0.0
        for state in range(self.n_states):
            # Formed once per state, outside the bisection: the solve is at
            # fixed `mu`, so `e * mu` does not move across its ~50 steps.
            rate = (
                float(mean[state]) if offsets is None else offsets * float(mean[state])
            )
            solved = _solve_dispersion(values, weights[:, state], rate)
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
            raise ValueError(msg)

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
        covariate: torch.Tensor | None = None,
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
        observations: torch.Tensor,
        posterior: torch.Tensor,
        covariate: torch.Tensor | None = None,
    ) -> Reestimate[PoissonEmission]:
        """The posterior-weighted mean, in closed form."""
        refuse_covariate(self, covariate)
        values = observations.reshape(-1).to(posterior.dtype)
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
        covariate: torch.Tensor | None = None,
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
        observations: torch.Tensor,
        posterior: torch.Tensor,
        covariate: torch.Tensor | None = None,
    ) -> Reestimate[BinomialEmission]:
        """``p = weighted mean / n``, in closed form; ``n`` is not estimated."""
        refuse_covariate(self, covariate)
        values = observations.reshape(-1).to(posterior.dtype)
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
            _PROBABILITY_MARGIN, 1.0 - _PROBABILITY_MARGIN
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
        covariate: torch.Tensor | None = None,
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
            else trial_count(covariate, self._trials).reshape(states.shape).numpy()
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
        refused.
        """
        trials = trial_count(covariate, self._trials)
        counts = observations.unsqueeze(-1).to(self._trials.dtype)
        scores = _beta_binomial_log_density(counts, trials, self._alpha, self._beta)
        return torch.where(counts > trials, torch.full_like(scores, -torch.inf), scores)

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
        observations: torch.Tensor,
        posterior: torch.Tensor,
        covariate: torch.Tensor | None = None,
    ) -> Reestimate[BetaBinomialEmission]:
        """Alternating bisection for ``(a, b)``; see :func:`_solve_beta_binomial`.

        Returns
        -------
        Reestimate
            The re-estimated family, whether a state's concentration reached
            the bound this data identifies it over, the iterations taken, and
            the relative change at the last step.
        """
        per_observation = covariate is not None
        values = observations.reshape(-1).to(posterior.dtype)
        weights = posterior.reshape(-1, self.n_states)
        supplied = (
            validated_trials(covariate, self._trials).reshape(-1)
            if covariate is not None
            else self._trials
        )

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
                supplied if per_observation else float(self._trials[state]),
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
        covariate: torch.Tensor | None = None,
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
        observations: torch.Tensor,
        posterior: torch.Tensor,
        covariate: torch.Tensor | None = None,
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
        values = observations.reshape(-1, self.N_CHANNELS).to(posterior.dtype)
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
        for state in range(self.n_states):
            concentration = float(self._alpha[state] + self._beta[state])
            solved = _solve_beta_binomial(
                successes,
                weights[:, state],
                totals,
                float(self._alpha[state]) / concentration,
                concentration,
            )
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

    **A constant per-observation count reduces exactly**, rather than through
    that mean: it *is* a fixed trial count, and the weighted mean of a constant
    is only the constant to within rounding, which would move the bracket
    ``identifiable_concentration_bound`` returns for no reason.

    It does **not** make the M step bitwise, and the reason is data-dependent.
    The score functions sum ``w_i f(y_i, n_i)`` over a vector ``n`` rather than
    folding a scalar, so the summation order differs; on some draws the
    alternating bisection still lands on the same root and on others the fitted
    ``alpha`` moves a **relative 1.0e-06**, four orders outside
    ``_solve_beta_binomial``'s own 1e-10 tolerance. Issue #648 carries that.
    Scoring *is* bitwise --- one broadcast expression with no reduction --- and
    is the half of #631's bit-for-bit claim that holds today.
    """
    if isinstance(trials, torch.Tensor):
        first = trials.reshape(-1)[0]
        if bool((trials == first).all()):
            return float(first)
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
    values: torch.Tensor,
    weights: torch.Tensor,
    dispersion: float,
    mean: float | torch.Tensor,
) -> float:
    """The posterior-weighted score in ``r``, with the mean profiled out.

    ``sum_t w_t (digamma(y_t + r) - digamma(r)) + W log(r / (r + mu))``. The
    term in ``(mu - y_t) / (r + mu)`` that appears in the full derivative
    vanishes because ``mu`` is the weighted mean of ``y``, which is what
    "profiled out" buys.

    Under an exposure ``mean`` is the **rate** ``e_t mu`` per observation, and
    the last term stops factoring out of the sum: it is
    ``sum_t w_t log(r / (r + e_t mu))``. The profiling identity still holds,
    because the closed form for ``mu`` under an exposure is
    ``sum_t w_t y_t / sum_t w_t e_t`` (issue #631).
    """
    r = torch.tensor(dispersion, dtype=values.dtype)
    weighted = (weights * (torch.digamma(values + r) - torch.digamma(r))).sum()
    if isinstance(mean, torch.Tensor):
        return float(
            weighted + (weights * torch.log(dispersion / (dispersion + mean))).sum()
        )
    return float(weighted + weights.sum() * math.log(dispersion / (dispersion + mean)))


def _solve_dispersion(
    values: torch.Tensor,
    weights: torch.Tensor,
    mean: float | torch.Tensor,
    *,
    tolerance: float = 1e-12,
) -> _SolvedDispersion:
    """Maximize the weighted likelihood in ``r`` by bisection on ``log r``.

    ``mean`` is a per-state mean, or the **rate** ``e_t mu`` per observation
    under an exposure. It is formed once by the caller rather than rebuilt
    here, because the solve is at fixed ``mu`` and the product does not move
    across the bisection's steps.

    **That buys nothing measurable, and the measurement is the point.** Issue
    #631 predicted the repeated multiply would dominate. At ``n = 200,000`` and
    45 bisection steps it is **1.2 ms** against the two ``digamma`` calls per
    step at **97.8 ms**, in a solve of roughly **270 ms** --- under half a
    percent. Root ``CLAUDE.md``'s profile-first rule says a term that small
    pays for no optimization, so this form is kept for being the clearer one
    and not for a speed-up it does not deliver. The term that would pay is the
    ``digamma`` pair.
    """
    upper = identifiable_dispersion_bound(
        _effective_rate(mean, weights), float(weights.sum())
    )
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


def _effective_rate(mean: float | torch.Tensor, weights: torch.Tensor) -> float:
    """The one mean :func:`identifiable_dispersion_bound` is read at.

    A per-state mean is itself. A per-observation rate has no single value, so
    the bound is taken at the *posterior-weighted mean* rate, which is the
    construction :func:`_effective_trials` already uses one family over. A
    constant rate reduces exactly, for the reason given there.

    Returns
    -------
    float
    """
    if not isinstance(mean, torch.Tensor):
        return mean
    first = mean.reshape(-1)[0]
    if bool((mean == first).all()):
        return float(first)
    return float((weights * mean).sum() / weights.sum())


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
